use super::scott::ScottNumericalDistributionBuilder;
use super::truncnorm;
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution as RandDistribution, WeightedAliasIndex};
use std::collections::HashMap;

use super::truncnorm::NEG_HALF_LOG_2PI;

/// Previous continuous-normal kernels, retained until the new coefficients are computed.
/// Between parameters the buffers are always empty; `update_normal` swaps previous
/// contents in where lookups are needed and clears the slots when done, so the
/// buffers only ever carry spare capacity.
#[derive(Default)]
pub(crate) struct NormalBuffers {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub log_consts: Vec<f64>,
}

impl NormalBuffers {
    fn clear(&mut self) {
        self.mus.clear();
        self.sigmas.clear();
        self.log_consts.clear();
    }
}

/// Set-bit positions of `mask`, in ascending order.
#[doc(hidden)]
pub fn set_bits(mask: &[u64]) -> impl Iterator<Item = usize> + '_ {
    mask.iter().enumerate().flat_map(|(word, &bits)| {
        (0..64)
            .filter(move |&b| bits >> b & 1 == 1)
            .map(move |b| word * 64 + b)
    })
}

/// Refills `dst` with `n` values: retained positions keep their `old` value
/// remapped by `update`, added positions call `gather`.
fn remap_or_gather<T: Copy + Default>(
    dst: &mut Vec<T>,
    n: usize,
    update: &KernelUpdate,
    old: &[T],
    gather: impl Fn(usize) -> T,
) {
    dst.resize(n, T::default());
    for (k, slot) in dst.iter_mut().enumerate() {
        *slot = update.old_index(k).map_or_else(|| gather(k), |o| old[o]);
    }
}

/// Refills `dst` with `n` values under `update`: trusted updates reuse
/// retained values via `old` (which receives them by swap), otherwise all
/// positions are rebuilt from `gather`.
fn gather_column<T: Copy + Default>(
    dst: &mut Vec<T>,
    n: usize,
    update: &KernelUpdate,
    trusted: bool,
    old: &mut Vec<T>,
    gather: impl Fn(usize) -> T,
) {
    match update {
        // The buffer already holds the retained prefix; appended ids extend it.
        KernelUpdate::Append { retained } if trusted && dst.len() > *retained => {
            dst.truncate(*retained);
            dst.extend(update.added_positions(n).map(&gather));
        }
        // Retained values move to their new positions; added ones are gathered.
        KernelUpdate::Scatter { .. } if trusted && !dst.is_empty() => {
            std::mem::swap(dst, old);
            remap_or_gather(dst, n, update, old, gather);
        }
        _ => {
            std::mem::swap(dst, old);
            dst.clear();
            dst.extend((0..n).map(gather));
        }
    }
}

/// How the new observation ids relate to a previous model's ids. `positions`
/// maps each old observation to its new index (`usize::MAX` when removed) and
/// `indices` is the reverse; `retained` is the length of the common prefix,
/// which maps identically for pure appends.
pub(crate) enum KernelUpdate {
    Fresh,
    Append {
        retained: usize,
    },
    Scatter {
        positions: Vec<usize>,
        indices: Vec<Option<usize>>,
    },
}

impl KernelUpdate {
    /// Computes how `new_ids` relates to `old_ids` (both sorted ascending).
    pub fn between(old_ids: &[u32], new_ids: &[u32]) -> Self {
        if old_ids.is_empty() {
            return Self::Fresh;
        }
        if new_ids.starts_with(old_ids) {
            return Self::Append {
                retained: old_ids.len(),
            };
        }
        let mut indices = Vec::with_capacity(new_ids.len());
        let mut old = 0;
        for id in new_ids {
            while old < old_ids.len() && old_ids[old] < *id {
                old += 1;
            }
            indices.push((old_ids.get(old) == Some(id)).then_some(old));
        }
        let mut positions = vec![usize::MAX; old_ids.len()];
        for (new, old_index) in indices.iter().enumerate() {
            if let Some(old_index) = old_index {
                positions[*old_index] = new;
            }
        }
        Self::Scatter { positions, indices }
    }

    /// Old kernel index for new observation `k`, or `None` when `k` is added.
    pub fn old_index(&self, k: usize) -> Option<usize> {
        match self {
            Self::Fresh => None,
            Self::Append { retained } => (k < *retained).then_some(k),
            Self::Scatter { indices, .. } => indices.get(k).copied().flatten(),
        }
    }

    /// New positions not retained from the old kernels: `0..n` on a fresh
    /// build, the appended tail for appends, or the listed additions.
    pub fn added_positions(&self, n: usize) -> impl Iterator<Item = usize> + '_ {
        // Positions at or beyond `start` never have an old index.
        let start = match self {
            Self::Append { retained } => *retained,
            _ => 0,
        };
        (start..n).filter(|&k| self.old_index(k).is_none())
    }

    /// Number of observation kernels before this update.
    pub fn old_len(&self) -> usize {
        match self {
            Self::Fresh => 0,
            Self::Append { retained } => *retained,
            Self::Scatter { positions, .. } => positions.len(),
        }
    }
}

/// Inputs shared by the kernel-table update of one parameter: the column
/// values, which rows they occupy, and how the new rows map onto old kernels.
struct NormalUpdate<'a> {
    values: &'a [Option<f64>],
    order: &'a [usize],
    adj_low: f64,
    adj_high: f64,
    step: Option<f64>,
    log: bool,
    update: &'a KernelUpdate,
    /// Scott's-rule widths when set; `None` selects nearest-neighbor bins.
    scott: Option<&'a ScottNumericalDistributionBuilder<'a>>,
    /// Whether retained observations are guaranteed to hold identical values.
    trusted: bool,
    bins: &'a mut Option<BinState>,
    /// Kernel positions whose sigma may have changed.
    updated: &'a mut Vec<u64>,
    /// Previous kernel buffers, used for lookups then cleared for reuse.
    old: &'a mut NormalBuffers,
}

/// Recomputes the kernel log-coefficients `NEG_HALF_LOG_2PI - ln(sigma) - ln_mass`.
/// `updated` flags kernels whose sigma may have changed; `old` receives the
/// previous coefficients by swap so unchanged ones are reused.
fn update_log_consts(
    log_consts: &mut Vec<f64>,
    mus: &[f64],
    sigmas: &[f64],
    adj_low: f64,
    adj_high: f64,
    updated: &[u64],
    update: &KernelUpdate,
    old: &mut NormalBuffers,
) {
    let compute = |mu: f64, sigma: f64| {
        let ln_mass = truncnorm::log_diff_cdf((adj_low - mu) / sigma, (adj_high - mu) / sigma)
            .unwrap_or(f64::NEG_INFINITY);
        if ln_mass == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            NEG_HALF_LOG_2PI - sigma.ln() - ln_mass
        }
    };
    if let KernelUpdate::Append { retained } = update {
        // Appends keep the previous coefficients in place: recompute flagged
        // retained positions, evaluate the appended tail, restore the prior.
        // `updated` empty means flag information is unavailable, not verified.
        if !updated.is_empty() && log_consts.len() > *retained {
            let prior_const = log_consts[*retained];
            log_consts.truncate(*retained);
            for k in set_bits(updated).take_while(|&k| k < *retained) {
                log_consts[k] = compute(mus[k], sigmas[k]);
            }
            log_consts.extend((*retained..mus.len() - 1).map(|k| compute(mus[k], sigmas[k])));
            log_consts.push(prior_const);
            return;
        }
    }
    if old.log_consts.is_empty() {
        std::mem::swap(log_consts, &mut old.log_consts);
    }
    log_consts.clear();
    // `updated` marks kernels whose sigma may have changed; empty means no
    // flag information is available, so every position must be verified.
    let may_have_changed = |k: usize| {
        updated.is_empty() || updated.get(k / 64).is_some_and(|w| w >> (k % 64) & 1 == 1)
    };
    for k in 0..mus.len() {
        // The prior kernel always occupies the trailing slot.
        let old_index = update
            .old_index(k)
            .or((k + 1 == mus.len()).then_some(update.old_len()));
        let reused = old_index
            .filter(|&i| {
                !may_have_changed(k)
                    || ((old.mus.is_empty()
                        || old.mus.get(i).map(|m| m.to_bits()) == Some(mus[k].to_bits()))
                        && old.sigmas.get(i).map(|s| s.to_bits()) == Some(sigmas[k].to_bits()))
            })
            .and_then(|i| old.log_consts.get(i).copied())
            .filter(|c| !c.is_nan());
        log_consts.push(reused.unwrap_or_else(|| compute(mus[k], sigmas[k])));
    }
}

/// One search-space parameter's values in storage row order.
/// Entries at `order` positions must be `Some`; other entries are never read.
#[doc(hidden)]
pub struct ObservationColumn<'a> {
    pub name: &'a str,
    pub distribution: &'a crate::distribution::Distribution,
    pub values: &'a [Option<f64>],
}

/// Fixed-width bins over `(low, high]` tracking each bin's min/max extreme and
/// the observation index holding it, so bandwidth updates only rescan changed bins.
#[derive(Default, Clone)]
pub(crate) struct BinState {
    pub low: f64,
    pub high: f64,
    /// (min, min_idx, max, max_idx) per bin; `min > max` marks the bin empty.
    pub bins: Vec<(f64, usize, f64, usize)>,
}

/// Updates `mus`, `sigmas`, and `log_consts` in place from `ctx`. Retained
/// kernels keep their storage: buffers already holding the previous values are
/// extended in place, otherwise they swap contents with `ctx.old` for lookups.
fn update_normal(
    mus: &mut Vec<f64>,
    sigmas: &mut Vec<f64>,
    log_consts: &mut Vec<f64>,
    ctx: &mut NormalUpdate,
) {
    let (values, order, log) = (ctx.values, ctx.order, ctx.log);
    let n = order.len();
    let gather = |i: usize| -> f64 {
        let v = values[order[i]].expect("observation column lacks a selected value");
        if log {
            v.ln()
        } else {
            v
        }
    };
    gather_column(mus, n, ctx.update, ctx.trusted, &mut ctx.old.mus, gather);
    mus.push((ctx.adj_low + ctx.adj_high) / 2.0); // Add prior
    update_sigmas(&mus[..n], sigmas, ctx);
    if ctx.step.is_none() {
        update_log_consts(
            log_consts,
            mus,
            sigmas,
            ctx.adj_low,
            ctx.adj_high,
            ctx.updated,
            ctx.update,
            ctx.old,
        );
    }
    ctx.old.clear();
}

/// Fills `sigmas` with `obs.len() + 1` widths (observations then the prior):
/// Scott's rule or nearest-neighbor bins, else a sorted scan.
fn update_sigmas(obs: &[f64], sigmas: &mut Vec<f64>, ctx: &mut NormalUpdate) {
    let n = obs.len();
    if n == 0 {
        *ctx.bins = None;
        std::mem::swap(sigmas, &mut ctx.old.sigmas);
        sigmas.push(ctx.adj_high - ctx.adj_low);
    } else if let Some(scott) = ctx.scott {
        std::mem::swap(sigmas, &mut ctx.old.sigmas);
        ctx.updated.clear();
        *ctx.bins = None;
        scott.sigmas(obs, sigmas);
        sigmas.push(ctx.adj_high - ctx.adj_low + 1.0);
    } else if binned_sigmas(
        obs,
        ctx.adj_low,
        ctx.adj_high,
        sigmas,
        &mut ctx.old.sigmas,
        ctx.bins,
        ctx.update,
        (!ctx.trusted).then_some(&ctx.old.mus[..]),
        ctx.updated,
    ) {
        sigmas.push(ctx.adj_high - ctx.adj_low);
    } else {
        sorted_scan_sigmas(obs, sigmas, ctx);
    }
}

/// Fallback kernel widths when binning cannot apply: the larger gap to the
/// sorted neighbors, clamped to the search-space scale.
fn sorted_scan_sigmas(obs: &[f64], sigmas: &mut Vec<f64>, ctx: &mut NormalUpdate) {
    let n = obs.len();
    if ctx.old.sigmas.is_empty() {
        std::mem::swap(sigmas, &mut ctx.old.sigmas);
    }
    sigmas.clear();
    let mut sorted: Vec<(usize, f64)> = (0..n).map(|i| (i, obs[i])).collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    // Each kernel's width is the larger gap to its sorted neighbors.
    sigmas.resize(n, 0.0);
    if n == 1 {
        let v = sorted[0].1;
        sigmas[sorted[0].0] = (v - ctx.adj_low).max(ctx.adj_high - v);
    } else {
        for w in sorted.windows(2) {
            let gap = w[1].1 - w[0].1;
            sigmas[w[0].0] = sigmas[w[0].0].max(gap);
            sigmas[w[1].0] = sigmas[w[1].0].max(gap);
        }
    }
    sigmas.push(ctx.adj_high - ctx.adj_low);
    let maxsigma = ctx.adj_high - ctx.adj_low;
    let minsigma = maxsigma / (100.0_f64.min(n as f64 + 2.0));
    for s in sigmas.iter_mut() {
        *s = s.clamp(minsigma, maxsigma);
    }
}

/// Computes kernel widths from fixed-width bins. Only bin extremes can be a
/// kernel's farthest neighbor, so per-bin (min,max) pairs give every bandwidth.
/// When `update` allows reuse, `positions` remap bin indices and only `added`
/// observations are inserted; `sigmas` keeps its retained prefix and flagged
/// positions drop to the floor before the extreme sweeps. Returns `false` when
/// the caller must fall back to the sorted-scan formula.
pub(crate) fn binned_sigmas(
    obs: &[f64],
    low: f64,
    high: f64,
    sigmas: &mut Vec<f64>,
    old_sigmas: &mut Vec<f64>,
    cache: &mut Option<BinState>,
    update: &KernelUpdate,
    previous_mus: Option<&[f64]>,
    updated: &mut Vec<u64>,
) -> bool {
    updated.clear();
    if obs.len() < 2 {
        if old_sigmas.is_empty() && !sigmas.is_empty() {
            std::mem::swap(sigmas, old_sigmas);
        }
        *cache = None;
        return false;
    }

    let n_bins = 100.min(obs.len() + 2);
    let minsigma = (high - low) / n_bins as f64;

    let mut state = cache.take().unwrap_or_default();
    let reuse = reusable_bins(&mut state, n_bins, low, high, obs, update, previous_mus);
    let empty = (f64::INFINITY, 0_usize, f64::NEG_INFINITY, 0_usize);
    state.bins.resize(n_bins, empty);
    if reuse {
        updated.resize(obs.len().div_ceil(64), 0);
    } else {
        state.bins.fill(empty);
    }
    match update {
        // Retained kernels keep their sigmas; flagged ones reset to the floor.
        KernelUpdate::Append { retained } if reuse && sigmas.len() > *retained => {
            sigmas.truncate(*retained);
        }
        _ => {
            std::mem::swap(sigmas, old_sigmas);
            sigmas.clear();
        }
    }

    // Insert the observations new to this update (all of them on a rebuild),
    // updating bins and flagging kernels whose widths may change.
    let mut changed_bins = 0_u128;
    let mut insert = |i: usize| -> bool {
        let v = obs[i];
        if reuse {
            updated[i / 64] |= 1 << (i % 64);
        }
        if !(low..=high).contains(&v) {
            return false;
        }
        let index = (((v - low) / minsigma) as u32 as usize).min(n_bins - 1);
        let bin = &mut state.bins[index];
        let changes_min = v < bin.0 || (v == bin.0 && i < bin.1);
        let changes_max = v > bin.2 || (v == bin.2 && i >= bin.3);
        if reuse && (changes_min || changes_max) {
            changed_bins |= 1 << index;
            for old in [bin.1, bin.3] {
                updated[old / 64] |= 1 << (old % 64);
            }
        }
        if changes_min {
            (bin.0, bin.1) = (v, i);
        }
        if changes_max {
            (bin.2, bin.3) = (v, i);
        }
        true
    };
    let inserted = if reuse {
        update.added_positions(obs.len()).all(|i| insert(i))
    } else {
        (0..obs.len()).all(|i| insert(i))
    };
    if !inserted {
        updated.clear();
        return false;
    }

    // Only changed bins and their nearest non-empty neighbors can change widths.
    while changed_bins != 0 {
        let index = changed_bins.trailing_zeros() as usize;
        changed_bins &= changed_bins - 1;
        let neighbors = [
            state.bins[..index].iter().rfind(|b| b.0 <= b.2),
            Some(&state.bins[index]),
            state.bins[index + 1..].iter().find(|b| b.0 <= b.2),
        ];
        for bin in neighbors.into_iter().flatten() {
            for i in [bin.1, bin.3] {
                updated[i / 64] |= 1 << (i % 64);
            }
        }
    }
    sigmas.resize(obs.len(), minsigma);
    // Flagged retained kernels drop back to the width floor.
    for i in set_bits(updated) {
        sigmas[i] = minsigma;
    }
    let (mut prev_max, mut next_min) = (f64::NAN, f64::NAN);
    for bin in state.bins.iter().filter(|b| b.0 <= b.2) {
        sigmas[bin.1] = sigmas[bin.1].max(bin.0 - prev_max);
        prev_max = bin.2;
    }
    for bin in state.bins.iter().rev().filter(|b| b.0 <= b.2) {
        sigmas[bin.3] = sigmas[bin.3].max(next_min - bin.2);
        next_min = bin.0;
    }
    if n_bins == 100 && minsigma.is_finite() && minsigma > 0.0 {
        (state.low, state.high) = (low, high);
        *cache = Some(state);
    }
    true
}

/// Whether cached `state` applies to this update, remapping scatter bin
/// indices on success: same bounds and bins, retained values verified.
fn reusable_bins(
    state: &mut BinState,
    n_bins: usize,
    low: f64,
    high: f64,
    obs: &[f64],
    update: &KernelUpdate,
    previous_mus: Option<&[f64]>,
) -> bool {
    if matches!(update, KernelUpdate::Fresh)
        || !same_bounds(state.low, state.high, low, high)
        || state.bins.len() != n_bins
    {
        return false;
    }
    if let Some(previous_mus) = previous_mus {
        // Untrusted callers may change retained values; a changed kernel must
        // be rebuilt rather than remapped.
        let unchanged = (0..obs.len()).all(|k| {
            update
                .old_index(k)
                .map_or(true, |o| previous_mus.get(o) == Some(&obs[k]))
        });
        if !unchanged {
            return false;
        }
    }
    match update {
        KernelUpdate::Scatter { positions, .. } => {
            // A removed extreme can hide another retained observation in its
            // bin; without the next-in-bin order the bins must be rebuilt.
            state
                .bins
                .iter_mut()
                .filter(|bin| bin.0 <= bin.2)
                .all(|bin| {
                    let (lo, hi) = (positions[bin.1], positions[bin.3]);
                    (bin.1, bin.3) = (lo, hi);
                    lo != usize::MAX && hi != usize::MAX
                })
        }
        _ => true,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TruncNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    /// Precomputed: NEG_HALF_LOG_2PI - ln(sigma_k) - ln_mass_k, i.e. every term that does not
    /// depend on x. `-inf` for a kernel with no probability mass.
    log_consts: Vec<f64>,
}

impl TruncNormDistributions {
    /// Adds each kernel's log PDF at x into `acc`.
    pub(crate) fn accumulate_log_pdf(&self, x: f64, acc: &mut [f64]) {
        for (k, a) in acc.iter_mut().enumerate() {
            let z = (x - self.mus[k]) / self.sigmas[k];
            *a += self.log_consts[k] - 0.5 * z * z;
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TruncLogNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    /// Precomputed: NEG_HALF_LOG_2PI - ln(sigma_k) - ln_mass_k, i.e. every term that does not
    /// depend on x. `-inf` for a kernel with no probability mass.
    log_consts: Vec<f64>,
}

impl TruncLogNormDistributions {
    /// Adds each kernel's log PDF at ln_x into `acc`.
    pub(crate) fn accumulate_log_pdf(&self, ln_x: f64, acc: &mut [f64]) {
        for (k, a) in acc.iter_mut().enumerate() {
            let z = (ln_x - self.mus[k]) / self.sigmas[k];
            *a += self.log_consts[k] - 0.5 * z * z - ln_x;
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DiscreteTruncNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    pub step: f64,
    /// Precomputed: (low - mu_k) / sigma_k
    low_truncs: Vec<f64>,
    /// Precomputed: (high - mu_k) / sigma_k
    high_truncs: Vec<f64>,
    /// Precomputed: step / (2 * sigma_k)
    half_steps: Vec<f64>,
    /// Precomputed: log_diff_cdf(low_trunc_k, high_trunc_k)
    ln_denoms: Vec<f64>,
}

impl DiscreteTruncNormDistributions {
    pub(crate) fn new(mus: Vec<f64>, sigmas: Vec<f64>, low: f64, high: f64, step: f64) -> Self {
        let low_truncs: Vec<_> = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| (low - mu) / sigma)
            .collect();
        let high_truncs: Vec<_> = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| (high - mu) / sigma)
            .collect();
        let half_steps = sigmas.iter().map(|&sigma| step / (2.0 * sigma)).collect();
        let ln_denoms = low_truncs
            .iter()
            .zip(high_truncs.iter())
            .map(|(&a, &b)| truncnorm::log_diff_cdf(a, b).unwrap_or(f64::NEG_INFINITY))
            .collect();
        Self {
            mus,
            sigmas,
            low,
            high,
            step,
            low_truncs,
            high_truncs,
            half_steps,
            ln_denoms,
        }
    }

    /// Log PDF of the k-th kernel at x_val (without bounds check).
    pub(crate) fn log_pdf(&self, x_val: f64, k: usize) -> f64 {
        let ln_denom = self.ln_denoms[k];
        if ln_denom == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        let center = (x_val - self.mus[k]) / self.sigmas[k];
        let a = if x_val <= self.low {
            f64::NEG_INFINITY
        } else {
            center - self.half_steps[k]
        };
        let b = if x_val >= self.high {
            f64::INFINITY
        } else {
            center + self.half_steps[k]
        };
        if b <= self.low_truncs[k] || a >= self.high_truncs[k] {
            return f64::NEG_INFINITY;
        }
        let a_adj = a.max(self.low_truncs[k]);
        let b_adj = b.min(self.high_truncs[k]);
        if a_adj >= b_adj {
            return f64::NEG_INFINITY;
        }
        match truncnorm::log_diff_cdf(a_adj, b_adj) {
            Ok(ln_numer) => ln_numer - ln_denom,
            Err(_) => f64::NEG_INFINITY,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DiscreteTruncLogNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    pub step: f64,
    /// Precomputed: (ln(low) - mu_k) / sigma_k
    low_truncs: Vec<f64>,
    /// Precomputed: (ln(high) - mu_k) / sigma_k
    high_truncs: Vec<f64>,
    /// Precomputed: log_diff_cdf(low_trunc_k, high_trunc_k)
    ln_denoms: Vec<f64>,
}

impl DiscreteTruncLogNormDistributions {
    pub(crate) fn new(mus: Vec<f64>, sigmas: Vec<f64>, low: f64, high: f64, step: f64) -> Self {
        let ln_low = low.ln();
        let ln_high = high.ln();
        let low_truncs: Vec<_> = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| (ln_low - mu) / sigma)
            .collect();
        let high_truncs: Vec<_> = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| (ln_high - mu) / sigma)
            .collect();
        let ln_denoms = low_truncs
            .iter()
            .zip(high_truncs.iter())
            .map(|(&a, &b)| truncnorm::log_diff_cdf(a, b).unwrap_or(f64::NEG_INFINITY))
            .collect();
        Self {
            mus,
            sigmas,
            low,
            high,
            step,
            low_truncs,
            high_truncs,
            ln_denoms,
        }
    }

    /// Log PDF of the k-th kernel at x_val (without bounds check).
    pub(crate) fn log_pdf(&self, x_val: f64, k: usize) -> f64 {
        let ln_denom = self.ln_denoms[k];
        if ln_denom == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        let low_bound = (x_val - self.step / 2.0).max(f64::MIN_POSITIVE);
        let high_bound = x_val + self.step / 2.0;
        let a = if x_val <= self.low {
            f64::NEG_INFINITY
        } else {
            (low_bound.ln() - self.mus[k]) / self.sigmas[k]
        };
        let b = if x_val >= self.high {
            f64::INFINITY
        } else {
            (high_bound.ln() - self.mus[k]) / self.sigmas[k]
        };
        if b <= self.low_truncs[k] || a >= self.high_truncs[k] {
            return f64::NEG_INFINITY;
        }
        let a_adj = a.max(self.low_truncs[k]);
        let b_adj = b.min(self.high_truncs[k]);
        if a_adj >= b_adj {
            return f64::NEG_INFINITY;
        }
        match truncnorm::log_diff_cdf(a_adj, b_adj) {
            Ok(ln_numer) => ln_numer - ln_denom,
            Err(_) => f64::NEG_INFINITY,
        }
    }
}

fn categorical_probabilities(n_observed: usize, cardinality: usize) -> (f64, f64) {
    let prior_mass = 1.0 / (n_observed + 1) as f64;
    let denominator = 1.0 + cardinality as f64 * prior_mass;
    ((1.0 + prior_mass) / denominator, prior_mass / denominator)
}

#[derive(Debug, Clone)]
pub(crate) struct CategoricalDistributions {
    // Each observation contributes a uniform mass plus a spike at its category.
    // The final kernel is the uniform prior. Joint mixtures retain row identity.
    observed_categories: Vec<usize>,
    cardinality: usize,
    matching: f64,
    other: f64,
}

impl CategoricalDistributions {
    #[cfg(test)]
    pub(crate) fn new(observations: impl IntoIterator<Item = f64>, cardinality: usize) -> Self {
        assert!(cardinality > 0, "Categorical cardinality must be positive");
        let observations = observations.into_iter();
        let (lower, upper) = observations.size_hint();
        let mut observed_categories = Vec::with_capacity(upper.unwrap_or(lower));
        for value in observations {
            let category = value as usize;
            assert!(category < cardinality, "Observed category is out of range");
            observed_categories.push(category);
        }
        Self::with_categories(observed_categories, cardinality)
    }

    /// Builds from an already materialized category list, e.g. kept and appended
    /// categories on incremental updates.
    pub(crate) fn with_categories(observed_categories: Vec<usize>, cardinality: usize) -> Self {
        assert!(cardinality > 0, "Categorical cardinality must be positive");
        let (matching, other) = categorical_probabilities(observed_categories.len(), cardinality);
        Self {
            observed_categories,
            cardinality,
            matching,
            other,
        }
    }

    /// Updates `observed_categories` in place from the column at `order` rows.
    /// Trusted updates read only the positions `update` marks as added; retained
    /// categories are kept or moved without re-reading the column.
    pub(crate) fn update_categories(
        &mut self,
        values: &[Option<f64>],
        order: &[usize],
        update: &KernelUpdate,
        trusted: bool,
    ) {
        let gather =
            |i: usize| values[order[i]].expect("categorical lacks a selected value") as usize;
        let mut old = Vec::new();
        gather_column(
            &mut self.observed_categories,
            order.len(),
            update,
            trusted,
            &mut old,
            gather,
        );
        (self.matching, self.other) =
            categorical_probabilities(self.observed_categories.len(), self.cardinality);
    }

    fn probability(&self, kernel: usize, category: usize) -> f64 {
        match self.observed_categories.get(kernel) {
            Some(&observed) if observed == category => self.matching,
            Some(_) => self.other,
            None => 1.0 / self.cardinality as f64,
        }
    }

    fn accumulate_log_pdf(&self, category: usize, scores: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.observed_categories.len() + 1);
        let log_matching = self.matching.ln();
        let log_other = self.other.ln();
        for (&observed, score) in self.observed_categories.iter().zip(scores.iter_mut()) {
            *score += if observed == category {
                log_matching
            } else {
                log_other
            };
        }
        *scores.last_mut().unwrap() += (1.0 / self.cardinality as f64).ln();
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Distributions {
    TruncNorm(TruncNormDistributions),
    TruncLogNorm(TruncLogNormDistributions),
    DiscreteTruncNorm(DiscreteTruncNormDistributions),
    DiscreteTruncLogNorm(DiscreteTruncLogNormDistributions),
    Categorical(CategoricalDistributions),
}

/// `low`/`high` bounds compare by bits so the same search space reuses kernels.
fn same_bounds(old_low: f64, old_high: f64, low: f64, high: f64) -> bool {
    old_low.to_bits() == low.to_bits() && old_high.to_bits() == high.to_bits()
}

impl Distributions {
    /// Updates this parameter's kernels in place from `column` at `order` rows.
    /// Kernel arrays keep storage and unchanged coefficients; `update` tells which
    /// observation positions are retained, removed, or newly added. `spare`
    /// provides reusable buffers shared across the model's parameters.
    pub(crate) fn update(
        &mut self,
        column: &ObservationColumn,
        order: &[usize],
        update: &KernelUpdate,
        bins: &mut Option<BinState>,
        scott: Option<&ScottNumericalDistributionBuilder>,
        trusted: bool,
        updated: &mut Vec<u64>,
        spare: &mut NormalBuffers,
    ) {
        let (low, high, step, log) = match column.distribution {
            crate::distribution::Distribution::Float {
                low,
                high,
                step,
                log,
            } => (*low, *high, *step, *log),
            crate::distribution::Distribution::Int {
                low,
                high,
                step,
                log,
            } => (*low as f64, *high as f64, Some(*step as f64), *log),
            crate::distribution::Distribution::Categorical { cardinality } => {
                match self {
                    Self::Categorical(d) if d.cardinality == *cardinality => {
                        d.update_categories(column.values, order, update, trusted);
                    }
                    _ => {
                        let mut d =
                            CategoricalDistributions::with_categories(Vec::new(), *cardinality);
                        d.update_categories(column.values, order, update, false);
                        *self = Self::Categorical(d);
                    }
                }
                return;
            }
        };
        let (mut adj_low, mut adj_high) = match step {
            Some(s) => (low - s / 2.0, high + s / 2.0),
            None => (low, high),
        };
        if log {
            adj_low = adj_low.ln();
            adj_high = adj_high.ln();
        }
        let mut ctx = NormalUpdate {
            values: column.values,
            order,
            adj_low,
            adj_high,
            step,
            log,
            update,
            scott,
            trusted,
            bins,
            updated,
            old: spare,
        };
        if let Some((mus, sigmas, log_consts)) = self.kernels_mut(step, log, low, high) {
            update_normal(mus, sigmas, log_consts, &mut ctx);
            return;
        }
        // Shape or bounds changed: rebuild, reusing a matching discrete
        // variant's buffers for capacity.
        let (mut mus, mut sigmas) = self
            .take_matching_buffers(step, log, low, high)
            .unwrap_or_else(|| {
                *ctx.bins = None;
                (Vec::new(), Vec::new())
            });
        let mut log_consts = Vec::new();
        update_normal(&mut mus, &mut sigmas, &mut log_consts, &mut ctx);
        *self = match (step, log) {
            (None, false) => Self::TruncNorm(TruncNormDistributions {
                mus,
                sigmas,
                low,
                high,
                log_consts,
            }),
            (None, true) => Self::TruncLogNorm(TruncLogNormDistributions {
                mus,
                sigmas,
                low,
                high,
                log_consts,
            }),
            (Some(step), false) => Self::DiscreteTruncNorm(DiscreteTruncNormDistributions::new(
                mus, sigmas, low, high, step,
            )),
            (Some(step), true) => Self::DiscreteTruncLogNorm(
                DiscreteTruncLogNormDistributions::new(mus, sigmas, low, high, step),
            ),
        };
    }

    /// Kernel buffers movable into a rebuild: only a discrete variant with a
    /// matching shape (`step`, `log`) and identical bounds.
    fn take_matching_buffers(
        &mut self,
        step: Option<f64>,
        log: bool,
        low: f64,
        high: f64,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        match (&mut *self, step, log) {
            (Self::DiscreteTruncNorm(d), Some(s), false)
                if d.step == s && same_bounds(d.low, d.high, low, high) =>
            {
                Some((std::mem::take(&mut d.mus), std::mem::take(&mut d.sigmas)))
            }
            (Self::DiscreteTruncLogNorm(d), Some(s), true)
                if d.step == s && same_bounds(d.low, d.high, low, high) =>
            {
                Some((std::mem::take(&mut d.mus), std::mem::take(&mut d.sigmas)))
            }
            _ => None,
        }
    }

    /// Kernel buffers reusable in place: only a continuous variant with a
    /// matching shape (no `step`, matching `log`) and identical bounds.
    fn kernels_mut(
        &mut self,
        step: Option<f64>,
        log: bool,
        low: f64,
        high: f64,
    ) -> Option<(&mut Vec<f64>, &mut Vec<f64>, &mut Vec<f64>)> {
        match (&mut *self, step, log) {
            (Self::TruncNorm(d), None, false) if same_bounds(d.low, d.high, low, high) => {
                Some((&mut d.mus, &mut d.sigmas, &mut d.log_consts))
            }
            (Self::TruncLogNorm(d), None, true) if same_bounds(d.low, d.high, low, high) => {
                Some((&mut d.mus, &mut d.sigmas, &mut d.log_consts))
            }
            _ => None,
        }
    }

    /// Builds this parameter's distribution from `column` at `order` rows,
    /// reading every selected value. `spare` provides reusable write buffers
    /// shared across the model's parameters.
    pub(crate) fn new_from_column(
        column: &ObservationColumn,
        order: &[usize],
        update: &KernelUpdate,
        bins: &mut Option<BinState>,
        scott: Option<&ScottNumericalDistributionBuilder>,
        updated: &mut Vec<u64>,
        spare: &mut NormalBuffers,
    ) -> Self {
        let mut d = Self::TruncNorm(TruncNormDistributions {
            mus: Vec::new(),
            sigmas: Vec::new(),
            low: f64::NAN,
            high: f64::NAN,
            log_consts: Vec::new(),
        });
        d.update(column, order, update, bins, scott, false, updated, spare);
        d
    }

    pub(crate) fn n_kernels(&self) -> usize {
        match self {
            Self::TruncNorm(d) => d.mus.len(),
            Self::TruncLogNorm(d) => d.mus.len(),
            Self::DiscreteTruncNorm(d) => d.mus.len(),
            Self::DiscreteTruncLogNorm(d) => d.mus.len(),
            Self::Categorical(d) => d.observed_categories.len() + 1,
        }
    }
}

#[derive(Debug)]
pub(crate) struct MixtureOfProductDistribution {
    pub param_names: Vec<String>,          // Sorted param names
    pub distributions: Vec<Distributions>, // Sorted distributions
    pub log_weights: Vec<f64>,             // ln(w_i)
    pub log_sum_weights: f64,              // ln(sum weights)
    pub alias: WeightedAliasIndex<f64>,
    pub n_kernels: usize,
}

impl MixtureOfProductDistribution {
    pub fn new(entries: Vec<(String, Distributions)>, weights: Vec<f64>) -> Self {
        let (param_names, distributions) = entries.into_iter().unzip();
        let mut mixture = MixtureOfProductDistribution {
            param_names,
            distributions,
            log_weights: Vec::new(),
            log_sum_weights: 0.0,
            alias: WeightedAliasIndex::new(vec![1.0])
                .expect("unit weight is non-empty and non-negative"),
            n_kernels: 0,
        };
        mixture.set_weights(weights);
        mixture
    }

    /// Recomputes the sampling weights over the kernels: normalized ln weights,
    /// their log-sum, and the alias table. `weights` includes the prior's.
    pub(crate) fn set_weights(&mut self, weights: Vec<f64>) {
        self.log_sum_weights = weights.iter().sum::<f64>().max(0.0).ln();
        self.log_weights = weights.iter().map(|&w| w.max(0.0).ln()).collect();
        self.n_kernels = weights.len();
        self.alias =
            WeightedAliasIndex::new(weights).expect("weights must be non-empty and non-negative");
    }

    pub fn sample(&self, rng: &mut StdRng, size: usize) -> Vec<HashMap<String, f64>> {
        self.sample_ordered(rng, size)
            .into_iter()
            .map(|values| self.param_names.iter().cloned().zip(values).collect())
            .collect()
    }

    pub fn sample_ordered(&self, rng: &mut StdRng, size: usize) -> Vec<Vec<f64>> {
        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(size);

        for _ in 0..size {
            let k = self.alias.sample(rng); // Active kernel index
            let mut sample = Vec::with_capacity(self.param_names.len());
            for (param, dist) in self.param_names.iter().zip(self.distributions.iter()) {
                match dist {
                    Distributions::TruncNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let value = truncnorm::rvs(
                            rng,
                            (d.low - mu) / sigma,
                            (d.high - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        sample.push(value);
                    }
                    Distributions::TruncLogNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let log_value = truncnorm::rvs(
                            rng,
                            (d.low.ln() - mu) / sigma,
                            (d.high.ln() - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        sample.push(log_value.exp());
                    }
                    Distributions::DiscreteTruncNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let value = truncnorm::rvs(
                            rng,
                            (d.low - d.step / 2.0 - mu) / sigma,
                            (d.high + d.step / 2.0 - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        let discrete_value = (d.low + ((value - d.low) / d.step).round() * d.step)
                            .max(d.low)
                            .min(d.high);
                        sample.push(discrete_value);
                    }
                    Distributions::DiscreteTruncLogNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let log_value = truncnorm::rvs(
                            rng,
                            ((d.low - d.step / 2.0).max(f64::MIN_POSITIVE).ln() - mu) / sigma,
                            ((d.high + d.step / 2.0).max(f64::MIN_POSITIVE).ln() - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        let original = log_value.exp();
                        let discrete_value = (d.low
                            + ((original - d.low) / d.step).round() * d.step)
                            .max(d.low)
                            .min(d.high);
                        sample.push(discrete_value);
                    }
                    Distributions::Categorical(d) => {
                        let sum: f64 = (0..d.cardinality)
                            .map(|category| d.probability(k, category))
                            .sum();
                        assert!(
                            sum > 0.0,
                            "Categorical distribution has non-positive total probability for param {param}"
                        );

                        let u = rng.gen::<f64>() * sum;
                        let mut cum = 0.0;
                        let mut chosen = (d.cardinality - 1) as f64; // fallback
                        for category in 0..d.cardinality {
                            cum += d.probability(k, category);
                            if u <= cum {
                                chosen = category as f64;
                                break;
                            }
                        }
                        sample.push(chosen);
                    }
                }
            }
            samples.push(sample);
        }

        samples
    }

    pub fn log_pdf(&self, x: &HashMap<String, f64>) -> f64 {
        self.log_pdf_values(
            self.param_names.iter().map(|name| x.get(name).copied()),
            None,
        )
        .unwrap()
    }

    pub fn log_pdf_ordered(&self, values: &[f64]) -> f64 {
        if values.len() != self.distributions.len() {
            return f64::NEG_INFINITY;
        }
        self.log_pdf_values(values.iter().copied().map(Some), None)
            .unwrap()
    }

    /// Returns `None` only when the acquisition score cannot reach `best`.
    pub(crate) fn log_pdf_ordered_bounded(
        &self,
        values: &[f64],
        good: f64,
        best: f64,
    ) -> Option<f64> {
        if values.len() != self.distributions.len() {
            return Some(f64::NEG_INFINITY);
        }
        self.log_pdf_values(values.iter().copied().map(Some), Some((good, best)))
    }

    /// The weighted prior component bounds the mixture from below. Keep its arithmetic
    /// identical to the corresponding component in `log_pdf_values`, including normalization.
    pub(crate) fn log_pdf_prior_ordered(&self, values: &[f64]) -> f64 {
        if values.len() != self.distributions.len() {
            return f64::NEG_INFINITY;
        }
        let k = self.n_kernels - 1;
        let mut score = 0.0;
        for (dist, &x) in self.distributions.iter().zip(values) {
            score += match dist {
                Distributions::TruncNorm(d) if x >= d.low && x <= d.high => {
                    let z = (x - d.mus[k]) / d.sigmas[k];
                    d.log_consts[k] - 0.5 * z * z
                }
                Distributions::TruncLogNorm(d) if x > 0.0 && x >= d.low && x <= d.high => {
                    let ln_x = x.ln();
                    let z = (ln_x - d.mus[k]) / d.sigmas[k];
                    d.log_consts[k] - 0.5 * z * z - ln_x
                }
                Distributions::DiscreteTruncNorm(d) => d.log_pdf(x, k),
                Distributions::DiscreteTruncLogNorm(d) if x > 0.0 => d.log_pdf(x, k),
                Distributions::Categorical(d) if (x as usize) < d.cardinality => {
                    d.probability(k, x as usize).ln()
                }
                _ => return f64::NEG_INFINITY,
            };
        }
        (score + self.log_weights[k]) - self.log_sum_weights
    }

    fn log_pdf_values(
        &self,
        values: impl Iterator<Item = Option<f64>>,
        cutoff: Option<(f64, f64)>,
    ) -> Option<f64> {
        let n = self.n_kernels;
        let mut weighted_log_pdf = vec![0.0_f64; n];

        for (dist, value) in self.distributions.iter().zip(values) {
            let Some(x_val) = value else {
                return Some(f64::NEG_INFINITY);
            };

            match dist {
                Distributions::TruncNorm(d) => {
                    if x_val < d.low || x_val > d.high {
                        return Some(f64::NEG_INFINITY);
                    }
                    d.accumulate_log_pdf(x_val, &mut weighted_log_pdf);
                }
                Distributions::TruncLogNorm(d) => {
                    if x_val <= 0.0 || x_val < d.low || x_val > d.high {
                        return Some(f64::NEG_INFINITY);
                    }
                    d.accumulate_log_pdf(x_val.ln(), &mut weighted_log_pdf);
                }
                Distributions::DiscreteTruncNorm(d) => {
                    for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
                        if *weight == f64::NEG_INFINITY {
                            continue;
                        }
                        let lp = d.log_pdf(x_val, k);
                        if lp == f64::NEG_INFINITY {
                            *weight = f64::NEG_INFINITY;
                        } else {
                            *weight += lp;
                        }
                    }
                }
                Distributions::DiscreteTruncLogNorm(d) => {
                    if x_val <= 0.0 {
                        return Some(f64::NEG_INFINITY);
                    }
                    for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
                        if *weight == f64::NEG_INFINITY {
                            continue;
                        }
                        let lp = d.log_pdf(x_val, k);
                        if lp == f64::NEG_INFINITY {
                            *weight = f64::NEG_INFINITY;
                        } else {
                            *weight += lp;
                        }
                    }
                }
                Distributions::Categorical(d) => {
                    let xi = x_val as usize;
                    if xi >= d.cardinality {
                        return Some(f64::NEG_INFINITY);
                    }
                    d.accumulate_log_pdf(xi, &mut weighted_log_pdf);
                }
            }
        }

        // Add log weights
        for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
            let lw = self.log_weights[k];
            if lw.is_infinite() && lw.is_sign_negative() {
                *weight = f64::NEG_INFINITY;
            } else {
                *weight += lw;
            }
        }

        // Log-sum-exp across kernels
        let max = weighted_log_pdf
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if max.is_infinite() && max.is_sign_negative() {
            // All -inf -> return -inf

            return Some(f64::NEG_INFINITY);
        }
        // sum_exp >= 1 makes `max - log_sum_weights` a lower bound: once the
        // acquisition `good - poor` cannot reach `best`, skip the exp() calls.
        if cutoff.is_some_and(|(good, best)| {
            good.is_finite()
                && best.is_finite()
                && max.is_finite()
                && good - (max - self.log_sum_weights) < best
        }) {
            return None;
        }
        let sum_exp: f64 = weighted_log_pdf.iter().map(|v| (v - max).exp()).sum();
        // Weights are basically normalized, but sum to 1 may not hold due to float rounding error
        Some((max + sum_exp.ln()) - self.log_sum_weights) // Normalize by total weight to avoid float rounding error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn bounded_density_preserves_complete_scores_and_ties() {
        let mixture = MixtureOfProductDistribution::new(
            vec![(
                "x".to_string(),
                Distributions::TruncNorm(TruncNormDistributions {
                    mus: vec![-0.5, 0.0],
                    sigmas: vec![2.0, 1.0],
                    low: -1.0,
                    high: 1.0,
                    log_consts: vec![0.0, 0.0],
                }),
            )],
            vec![0.5, 0.5],
        );
        // `Some(v)` must bit-match `log_pdf_ordered`; `None` is only allowed
        // when the exact acquisition `good - v` stays below `best`. NaN and
        // infinite inputs must never cut off evaluation.
        for x in [-0.5, 0.0, 0.25, f64::NAN, f64::INFINITY] {
            let values = [x];
            let exact = mixture.log_pdf_ordered(&values);
            for good in [f64::NEG_INFINITY, -1.0, 0.0, 1.0, f64::INFINITY, f64::NAN] {
                for best in [f64::NEG_INFINITY, -1.0, 0.0, 1.0, f64::INFINITY, f64::NAN] {
                    match mixture.log_pdf_ordered_bounded(&values, good, best) {
                        Some(v) => assert_eq!(v.to_bits(), exact.to_bits(), "x={x}"),
                        None => {
                            assert!(good.is_finite() && best.is_finite() && exact.is_finite());
                            assert!(good - exact < best, "x={x} good={good} best={best}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unchanged_normalizers_are_reused_after_removing_an_observation() {
        // Two old observations [0.2, 0.4] and a prior; the scatter keeps only
        // the second, so both retained and prior coefficients are reused.
        let mut old = NormalBuffers {
            mus: vec![0.2, 0.4, 0.5],
            sigmas: vec![0.3; 3],
            log_consts: vec![10.0, 20.0, 30.0], // Markers distinguish reuse from recomputation.
        };
        let update = KernelUpdate::Scatter {
            positions: vec![usize::MAX, 0],
            indices: vec![Some(1)],
        };
        let mut constants = Vec::new();
        update_log_consts(
            &mut constants,
            &[0.4, 0.5],
            &[0.3, 0.3],
            0.0,
            1.0,
            &[],
            &update,
            &mut old,
        );
        assert_eq!(constants, [20.0, 30.0]); // Retained observation and prior.
    }

    #[test]
    fn test_mixture_of_product_distribution() {
        let truncnorm_dist = Distributions::TruncNorm(TruncNormDistributions {
            mus: vec![-0.5, 0.0], // mus[-1] is prior
            sigmas: vec![2.0, 1.0],
            low: -1.0,
            high: 1.0,
            log_consts: vec![0.0, 0.0],
        });
        let trunclognorm_dist = Distributions::TruncLogNorm(TruncLogNormDistributions {
            mus: vec![2.0, 3.0],
            sigmas: vec![2.0, 1.0],
            low: 1.0,
            high: 5.0,
            log_consts: vec![0.0, 0.0],
        });
        let discrete_truncnorm_dist = DiscreteTruncNormDistributions::new(
            vec![-0.5, 0.0], // ditto
            vec![1.0, 1.0],  // ditto
            -1.0,
            1.0,
            1.0,
        );
        let discrete_trunclognorm_dist = DiscreteTruncLogNormDistributions::new(
            vec![2.0, 3.0], // ditto
            vec![1.0, 1.0], // ditto
            1.0,
            5.0,
            1.0,
        );
        let categorical_dist = CategoricalDistributions::new([0.0], 2);
        let mut distributions = vec![
            ("param_truncnorm".to_string(), truncnorm_dist),
            ("param_trunclognorm".to_string(), trunclognorm_dist),
            (
                "param_discretetruncnorm".to_string(),
                Distributions::DiscreteTruncNorm(discrete_truncnorm_dist),
            ),
            (
                "param_discretetrunclognorm".to_string(),
                Distributions::DiscreteTruncLogNorm(discrete_trunclognorm_dist),
            ),
            (
                "param_categorical".to_string(),
                Distributions::Categorical(categorical_dist),
            ),
        ];
        distributions.sort_by(|a, b| a.0.cmp(&b.0));
        let mixture = MixtureOfProductDistribution::new(
            distributions,
            vec![0.5, 0.5], // weights.len() == mus.len()
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let samples = mixture.sample(&mut rng, 10);
        for sample in samples.iter() {
            let values: Vec<_> = mixture.param_names.iter().map(|key| sample[key]).collect();
            assert!(mixture.log_pdf_prior_ordered(&values) <= mixture.log_pdf(sample));
            let val_truncnorm = sample.get("param_truncnorm").unwrap();
            assert!(*val_truncnorm >= -1.0 && *val_truncnorm <= 1.0);
            let val_trunclognorm = sample.get("param_trunclognorm").unwrap();
            assert!(*val_trunclognorm >= 1.0 && *val_trunclognorm <= 5.0);
            let val_discretetruncnorm = sample.get("param_discretetruncnorm").unwrap();
            assert!(*val_discretetruncnorm >= -1.0 && *val_discretetruncnorm <= 1.0);
            let val_discretetrunclognorm = sample.get("param_discretetrunclognorm").unwrap();
            assert!(*val_discretetrunclognorm >= 1.0 && *val_discretetrunclognorm <= 5.0);
            let val_categorical = sample.get("param_categorical").unwrap();
            assert!(*val_categorical == 0.0 || *val_categorical == 1.0);
        }
    }

    #[test]
    fn categorical_kernels_keep_smoothed_probabilities_and_uniform_prior() {
        let distribution = CategoricalDistributions::new([1.0, 2.0], 3);
        let expected = [
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            [1.0 / 3.0; 3],
        ];
        for (category, _) in expected[0].iter().enumerate() {
            let mut scores = vec![0.0; 3];
            distribution.accumulate_log_pdf(category, &mut scores);
            for (kernel, (row, score)) in expected.iter().zip(scores).enumerate() {
                assert!((distribution.probability(kernel, category) - row[category]).abs() < 1e-15);
                assert!((score - row[category].ln()).abs() < 1e-15);
            }
        }
        for cardinality in [1, 3, 257] {
            let prior = CategoricalDistributions::new([], cardinality);
            let singleton = CategoricalDistributions::new([0.0; 10], 1);
            let mut scores = vec![0.0; 1];
            prior.accumulate_log_pdf(0, &mut scores);
            assert_eq!(prior.probability(0, 0), 1.0 / cardinality as f64);
            assert_eq!(scores[0], (1.0 / cardinality as f64).ln());
            assert_eq!(singleton.probability(0, 0), 1.0);
        }
    }

    #[test]
    fn categorical_joint_density_keeps_observation_correlations_and_weights() {
        let mixture = MixtureOfProductDistribution::new(
            vec![
                (
                    "x".into(),
                    Distributions::Categorical(CategoricalDistributions::new([0.0, 1.0], 2)),
                ),
                (
                    "y".into(),
                    Distributions::Categorical(CategoricalDistributions::new([0.0, 1.0], 2)),
                ),
            ],
            vec![1.0, 2.0, 0.5],
        );
        let matching = 4.0_f64 / 5.0;
        let other = 1.0_f64 / 5.0;
        for (x, y) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
            let likelihood = (if x == 0.0 { matching } else { other })
                * (if y == 0.0 { matching } else { other })
                + 2.0
                    * (if x == 1.0 { matching } else { other })
                    * (if y == 1.0 { matching } else { other })
                + 0.5 * 0.25;
            let actual = mixture.log_pdf(&HashMap::from([("x".into(), x), ("y".into(), y)]));
            assert!((actual - (likelihood / 3.5).ln()).abs() < 1e-14);
        }
        assert_eq!(
            mixture.log_pdf(&HashMap::from([("x".into(), 2.0), ("y".into(), 0.0)])),
            f64::NEG_INFINITY
        );
        assert_eq!(
            mixture.log_pdf(&HashMap::from([("x".into(), 0.0)])),
            f64::NEG_INFINITY
        );
        let mut rng = StdRng::seed_from_u64(42);
        let samples = mixture.sample(&mut rng, 20_000);
        let equal = samples
            .iter()
            .filter(|sample| sample["x"] == sample["y"])
            .count() as f64
            / samples.len() as f64;
        // Both observed kernels give P(x=y)=0.8^2+0.2^2; the prior gives 0.5.
        let expected_equal = (3.0 * (matching * matching + other * other) + 0.5 * 0.5) / 3.5;
        assert!((equal - expected_equal).abs() < 0.015);
    }
}
