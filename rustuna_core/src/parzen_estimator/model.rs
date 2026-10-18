use rand::rngs::StdRng;
use std::collections::HashMap;

use super::probability_distributions::CategoricalDistributions;
use super::probability_distributions::{
    Distributions, MixtureOfProductDistribution, NumericalParameters,
};
use crate::distribution::Distribution;
use crate::parzen_estimator::scott::ScottNumericalDistributionBuilder;

pub struct ParzenEstimator {
    mixuture_distribution: MixtureOfProductDistribution,
    observation_ids: Vec<u32>,
    bins: Vec<Option<BinState>>,
}

impl ParzenEstimator {
    pub fn new(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &HashMap<String, Distribution>,
        weights: &[f64],
        prior_weight: f64,
    ) -> Self {
        Self::with_builder(
            observations,
            search_space,
            weights,
            prior_weight,
            &DefaultNumericalDistributionBuilder,
            &DefaultCategoricalDistributionBuilder,
        )
    }

    pub fn with_scott(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &HashMap<String, Distribution>,
        weights: &[f64],
        prior_weight: f64,
    ) -> Self {
        Self::with_builder(
            observations,
            search_space,
            weights,
            prior_weight,
            &ScottNumericalDistributionBuilder::new(weights),
            &DefaultCategoricalDistributionBuilder,
        )
    }

    pub(crate) fn with_builder(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &HashMap<String, Distribution>,
        weights: &[f64],
        prior_weight: f64,
        num_builder: &impl NumericalDistributionBuilder,
        cat_builder: &impl CategoricalDistributionBuilder,
    ) -> Self {
        let n_observations = observations.values().next().map_or(0, |v| {
            assert!(
                observations.values().all(|w| w.len() == v.len()),
                "Observations have inconsistent lengths"
            );
            v.len()
        });
        assert_eq!(
            n_observations,
            weights.len(),
            "Number of observations and length of weights must be equal"
        );

        let mut keys: Vec<_> = search_space.keys().collect();
        keys.sort();

        Self::with_ordered_builder(
            keys.into_iter().map(|key| {
                (
                    key.as_str(),
                    &search_space[key],
                    observations
                        .get(key)
                        .map(Vec::as_slice)
                        .unwrap_or(&[])
                        .iter()
                        .copied(),
                )
            }),
            weights,
            prior_weight,
            num_builder,
            cat_builder,
            None,
        )
    }

    /// Builds a model from columns in strictly increasing parameter-name order.
    /// Each column and the strictly increasing observation IDs must match the weights.
    /// An optional previous model supplies buffers, fixed bins and unchanged density coefficients.
    pub fn from_ordered_observations<'a, I, V>(
        parameters: I,
        observation_ids: Vec<u32>,
        weights: &[f64],
        prior_weight: f64,
        previous: Option<Self>,
    ) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a Distribution, V)>,
        V: IntoIterator<Item = f64>,
    {
        assert_eq!(observation_ids.len(), weights.len());
        assert!(
            observation_ids.windows(2).all(|ids| ids[0] < ids[1]),
            "Observation IDs must be sorted and unique"
        );
        let mut indices = Vec::new();
        if let Some(model) = &previous {
            let old_ids = &model.observation_ids;
            let mut old = 0;
            for id in &observation_ids {
                while old < old_ids.len() && old_ids[old] < *id {
                    old += 1;
                }
                indices.push((old_ids.get(old) == Some(id)).then_some(old));
            }
        }
        let mut model = Self::with_ordered_builder(
            parameters
                .into_iter()
                .map(|(name, distribution, values)| (name, distribution, values.into_iter())),
            weights,
            prior_weight,
            &DefaultNumericalDistributionBuilder,
            &DefaultCategoricalDistributionBuilder,
            previous.map(|model| (model, indices.as_slice())),
        );
        model.observation_ids = observation_ids;
        assert!(
            model
                .mixuture_distribution
                .distributions
                .iter()
                .all(|distribution| distribution.n_kernels() == weights.len() + 1),
            "Observation columns and weights differ in length"
        );
        model
    }

    fn with_ordered_builder<'a, I, V>(
        parameters: I,
        weights: &[f64],
        prior_weight: f64,
        num_builder: &impl NumericalDistributionBuilder,
        cat_builder: &impl CategoricalDistributionBuilder,
        previous: Option<(Self, &[Option<usize>])>,
    ) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a Distribution, V)>,
        V: Iterator<Item = f64>,
    {
        let parameters = parameters.into_iter();
        let mut positions = Vec::new();
        let mut added = Vec::new();
        let (old_names, old_distributions, old_bins, indices) = match previous {
            Some((model, indices)) => {
                // Invert the shared ID mapping once for all parameter columns.
                positions.resize(model.observation_ids.len(), usize::MAX);
                for (new, old) in indices.iter().enumerate() {
                    if let Some(old) = old {
                        positions[*old] = new;
                    } else {
                        added.push(new);
                    }
                }
                let m = model.mixuture_distribution;
                (m.param_names, m.distributions, model.bins, indices)
            }
            None => (Vec::new(), Vec::new(), Vec::new(), &[][..]),
        };
        let mut old = old_names.into_iter().zip(old_distributions).zip(old_bins);
        let mut bin_states = Vec::new();
        let mut parameters_buffer = NumericalParameters::default();
        let mut constants = Vec::new();
        let mut distributions: Vec<(String, Distributions)> =
            Vec::with_capacity(parameters.size_hint().0);
        for (name, search_space, values) in parameters {
            if let Some((previous, _)) = distributions.last() {
                assert!(
                    previous.as_str() < name,
                    "Parameter names must be sorted and unique"
                );
            }
            let (reuse, mut bins) = old
                .next()
                .filter(|((old_name, _), _)| old_name == name)
                .map(|((_, distribution), bins)| (Some((distribution, indices)), bins))
                .unwrap_or_default();
            let distribution = match search_space {
                Distribution::Float { .. } | Distribution::Int { .. } => {
                    let (low, high, step, log) = match search_space {
                        Distribution::Float {
                            low,
                            high,
                            step,
                            log,
                        } => (*low, *high, *step, *log),
                        Distribution::Int {
                            low,
                            high,
                            step,
                            log,
                        } => (*low as f64, *high as f64, Some(*step as f64), *log),
                        _ => unreachable!(),
                    };
                    parameters_buffer.low = low;
                    parameters_buffer.high = high;
                    parameters_buffer.step = step;
                    parameters_buffer.log = log;
                    let previous = match &reuse {
                        Some((Distributions::TruncNorm(d), _)) => Some(d.mus.as_slice()),
                        Some((Distributions::TruncLogNorm(d), _)) => Some(d.mus.as_slice()),
                        _ => None,
                    };
                    num_builder.calculate_parameters(
                        values,
                        &mut parameters_buffer,
                        &mut bins,
                        previous.map(|mus| (mus, positions.as_slice(), added.as_slice())),
                    );
                    Distributions::from_numerical(&mut parameters_buffer, reuse, &mut constants)
                }
                Distribution::Categorical { .. } => {
                    bins = None;
                    cat_builder.calculate_categorical_distribution(values, search_space)
                }
            };
            distributions.push((name.to_owned(), distribution));
            bin_states.push(bins);
        }

        let weights_sum = {
            let s = weights.iter().sum::<f64>() + prior_weight;
            if s == 0.0 {
                (weights.len() + 1) as f64
            } else {
                s
            }
        };

        let weights_with_prior_weight = weights
            .iter()
            .chain(std::iter::once(&prior_weight))
            .map(|w| w / weights_sum)
            .collect();

        Self {
            observation_ids: Vec::new(),
            bins: bin_states,
            mixuture_distribution: MixtureOfProductDistribution::new(
                distributions,
                weights_with_prior_weight,
            ),
        }
    }

    pub fn sample(&self, rng: &mut StdRng, size: usize) -> Vec<HashMap<String, f64>> {
        self.mixuture_distribution.sample(rng, size)
    }

    /// Returns the parameter names used by ordered samples and scores.
    pub fn parameter_names(&self) -> &[String] {
        &self.mixuture_distribution.param_names
    }

    /// Samples values in [`Self::parameter_names`] order.
    pub fn sample_ordered(&self, rng: &mut StdRng, size: usize) -> Vec<Vec<f64>> {
        self.mixuture_distribution.sample_ordered(rng, size)
    }

    pub fn log_pdf(&self, x: &HashMap<String, f64>) -> f64 {
        self.mixuture_distribution.log_pdf(x)
    }

    /// Scores values in [`Self::parameter_names`] order.
    /// Returns negative infinity if the value count does not match the model.
    pub fn log_pdf_ordered(&self, values: &[f64]) -> f64 {
        self.mixuture_distribution.log_pdf_ordered(values)
    }

    /// Maximizes `self.log_pdf_ordered(x) - poor.log_pdf_ordered(x)`, keeping the first tie.
    /// For joint models, prior-based upper bounds skip candidates that cannot win.
    pub fn best_candidate_index(&self, poor: &Self, samples: &[Vec<f64>]) -> usize {
        let mut ranked = Vec::new();
        if self.parameter_names().len() > 1 {
            ranked = samples
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    let good = self.log_pdf_ordered(x);
                    let bound = good - poor.mixuture_distribution.log_pdf_prior_ordered(x);
                    // Unusable bounds must always receive a full evaluation.
                    let upper = if bound.is_finite() {
                        bound
                    } else {
                        f64::INFINITY
                    };
                    (i, good, upper)
                })
                .collect::<Vec<_>>();
            ranked.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
        }
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for position in 0..samples.len() {
            let (i, good, upper) = if ranked.is_empty() {
                // A single prior is usually too weak a bound for independent TPE.
                (
                    position,
                    self.log_pdf_ordered(&samples[position]),
                    f64::INFINITY,
                )
            } else {
                ranked[position]
            };
            if upper < best_val {
                break;
            }
            let acquisition = good - poor.log_pdf_ordered(&samples[i]);
            if acquisition > best_val || (acquisition == best_val && i < best_idx) {
                best_val = acquisition;
                best_idx = i;
            }
        }
        best_idx
    }
}

pub(crate) trait NumericalDistributionBuilder {
    fn calculate_parameters(
        &self,
        observations: impl Iterator<Item = f64>,
        parameters: &mut NumericalParameters,
        bins: &mut Option<BinState>,
        previous: Option<(&[f64], &[usize], &[usize])>,
    );
}

pub(crate) trait CategoricalDistributionBuilder {
    fn calculate_categorical_distribution(
        &self,
        observations: impl Iterator<Item = f64>,
        search_space: &Distribution,
    ) -> Distributions;
}

pub(crate) struct DefaultNumericalDistributionBuilder;
pub(crate) struct DefaultCategoricalDistributionBuilder;

/// Fixed-width bin extremes, indexed in the previous model's observation order.
#[derive(Default)]
pub(crate) struct BinState {
    low: f64,
    high: f64,
    bins: Vec<(f64, usize, f64, usize)>,
}

/// Each observation's larger distance to its two sorted neighbors, floored at `minsigma`, or
/// `None` if fewer than two observations are given or some of them are outside the search space.
///
/// A neighbor nearer than `minsigma` cannot lift that floor, and binning at that width hides
/// exactly those: within a bin only the extremes can have a farther neighbor, and it is the
/// adjacent non-empty bin's extreme. The per-bin extremes therefore give every bandwidth.
fn binned_sigmas(
    obs: &[f64],
    low: f64,
    high: f64,
    sigmas: &mut Vec<f64>,
    cache: &mut Option<BinState>,
    previous: Option<(&[f64], &[usize], &[usize])>,
    updated: &mut Vec<u64>,
) -> Option<()> {
    updated.clear();
    if obs.len() < 2 {
        // No inter-observation neighbor exists; the caller falls back to endpoint distances.
        *cache = None;
        return None;
    }

    let n_bins = 100.min(obs.len() + 2);
    let minsigma = (high - low) / n_bins as f64;

    let mut state = cache.take().unwrap_or_default();
    let (mus, positions, added) = previous.unwrap_or_default();
    let reuse = state.low.to_bits() == low.to_bits()
        && state.high.to_bits() == high.to_bits()
        && state.bins.len() == n_bins
        && mus.len() == positions.len() + 1
        && positions
            .iter()
            .enumerate()
            .all(|(old, &new)| obs.get(new).map(|v| v.to_bits()) == Some(mus[old].to_bits()));
    let empty = (f64::INFINITY, 0_usize, f64::NEG_INFINITY, 0_usize);
    state.bins.resize(n_bins, empty);
    if reuse {
        updated.resize(obs.len().div_ceil(64), 0);
        for bin in state.bins.iter_mut().filter(|bin| bin.0 <= bin.2) {
            bin.1 = positions[bin.1];
            bin.3 = positions[bin.3];
        }
    } else {
        state.bins.fill(empty);
    }
    let mut changed_bins = 0_u128;
    let added = if reuse { added } else { &[] };
    let rebuild_len = if reuse { 0 } else { obs.len() };
    for i in added.iter().copied().chain(0..rebuild_len) {
        let v = obs[i];
        if reuse {
            updated[i / 64] |= 1 << (i % 64);
        }
        if !(low..=high).contains(&v) {
            // The observation is outside the search space.
            updated.clear();
            return None;
        }
        let index = (((v - low) / minsigma) as u32 as usize).min(n_bins - 1);
        let bin = &mut state.bins[index];
        let changes_min = v < bin.0 || (v == bin.0 && i < bin.1);
        let changes_max = v > bin.2 || (v == bin.2 && i >= bin.3);
        if reuse && (changes_min || changes_max) {
            changed_bins |= 1 << index;
            // A displaced extreme returns to the width floor.
            for old in [bin.1, bin.3] {
                updated[old / 64] |= 1 << (old % 64);
            }
        }
        // Insertions can precede retained observations: keep the original tie order.
        if changes_min {
            (bin.0, bin.1) = (v, i);
        }
        if changes_max {
            (bin.2, bin.3) = (v, i);
        }
    }

    sigmas.resize(obs.len(), minsigma);
    sigmas.fill(minsigma);
    let (mut prev_max, mut next_min) = (f64::NAN, f64::NAN);
    for bin in state.bins.iter().filter(|b| b.0 <= b.2) {
        sigmas[bin.1] = sigmas[bin.1].max(bin.0 - prev_max);
        prev_max = bin.2;
    }
    for bin in state.bins.iter().rev().filter(|b| b.0 <= b.2) {
        sigmas[bin.3] = sigmas[bin.3].max(next_min - bin.2);
        next_min = bin.0;
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
    if n_bins == 100 && minsigma.is_finite() && minsigma > 0.0 && previous.is_some() {
        (state.low, state.high) = (low, high);
        *cache = Some(state);
    }
    Some(())
}

impl NumericalDistributionBuilder for DefaultNumericalDistributionBuilder {
    fn calculate_parameters(
        &self,
        observations: impl Iterator<Item = f64>,
        parameters: &mut NumericalParameters,
        bins: &mut Option<BinState>,
        previous: Option<(&[f64], &[usize], &[usize])>,
    ) {
        // Currently, we assume consider_prior=True, consider_endpoints=False, and consider_magic_clip=True.
        let NumericalParameters {
            mus,
            sigmas,
            low,
            high,
            step,
            log,
            updated,
        } = parameters;
        updated.clear();
        let (low, high, step_opt, log) = (*low, *high, *step, *log);
        let (mut adj_low, mut adj_high) = match step_opt {
            Some(s) => (low - s / 2.0, high + s / 2.0),
            None => (low, high),
        };

        mus.clear();
        mus.extend(observations);
        if log {
            adj_low = adj_low.ln();
            adj_high = adj_high.ln();
            for mu in mus.iter_mut() {
                *mu = mu.ln();
            }
        }
        mus.push((adj_low + adj_high) / 2.0); // Add prior

        sigmas.clear();
        sigmas.reserve(mus.len());
        if mus.len() == 1 {
            // Case: prior only
            *bins = None;
            sigmas.push(adj_high - adj_low);
        } else if binned_sigmas(
            &mus[..mus.len() - 1],
            adj_low,
            adj_high,
            sigmas,
            bins,
            previous,
            updated,
        )
        .is_some()
        {
            // Sigma for prior
            sigmas.push(adj_high - adj_low);
        } else {
            let m = mus.len() - 1; // exclude prior
            let mut idx_vals: Vec<(usize, f64)> = (0..m).map(|i| (i, mus[i])).collect();
            idx_vals.sort_by(|a, b| a.1.total_cmp(&b.1));
            let sorted_obs: Vec<f64> = idx_vals.iter().map(|&(_, v)| v).collect();

            // consider_endpoints=False: boundary observations use only the neighbor distance.
            // When m == 1, no inter-observation neighbor exists, so fall back to endpoint distances.
            sigmas.resize(m, 0.0);
            for (j, &(orig_idx, _)) in idx_vals.iter().enumerate() {
                sigmas[orig_idx] = if m == 1 {
                    (sorted_obs[0] - adj_low).max(adj_high - sorted_obs[0])
                } else if j == 0 {
                    sorted_obs[1] - sorted_obs[0]
                } else if j == m - 1 {
                    sorted_obs[m - 1] - sorted_obs[m - 2]
                } else {
                    (sorted_obs[j] - sorted_obs[j - 1]).max(sorted_obs[j + 1] - sorted_obs[j])
                };
            }
            // Sigma for prior
            sigmas.push(adj_high - adj_low);

            // Clamp (minsigma, maxsigma)
            let maxsigma = adj_high - adj_low;
            let minsigma = (adj_high - adj_low) / (100.0_f64.min(1.0 + sigmas.len() as f64));
            for s in sigmas.iter_mut() {
                *s = s.clamp(minsigma, maxsigma);
            }
        }
    }
}

impl CategoricalDistributionBuilder for DefaultCategoricalDistributionBuilder {
    fn calculate_categorical_distribution(
        &self,
        observations: impl Iterator<Item = f64>,
        search_space: &Distribution,
    ) -> Distributions {
        let cardinality = match search_space {
            Distribution::Categorical { cardinality } => *cardinality,
            _ => unreachable!("Invalid distribution type for categorical calculation"),
        };

        Distributions::Categorical(CategoricalDistributions::new(observations, cardinality))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn candidate_pruning_matches_exhaustive_scoring() {
        let space = Distribution::new_float(-5.0, 5.0, None, false);
        let model = |values: &[f64], prior| {
            ParzenEstimator::from_ordered_observations(
                ["x", "y"].map(|name| (name, &space, values.iter().copied())),
                (0..values.len() as u32).collect(),
                &vec![1.0; values.len()],
                prior,
                None,
            )
        };
        let good = model(&[-3.0, -2.0, -1.0], 1.0);
        for prior in [0.0, 1.0] {
            let poor = model(&[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], prior);
            let mut samples = good.sample_ordered(&mut StdRng::seed_from_u64(42), 24);
            let invalid = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY].map(|v| vec![v; 2]);
            assert_eq!(good.best_candidate_index(&poor, &invalid), 0);
            samples.splice(0..0, invalid);
            let (mut index, mut best) = (0, f64::NEG_INFINITY);
            for (i, x) in samples.iter().enumerate() {
                let score = good.log_pdf_ordered(x) - poor.log_pdf_ordered(x);
                if score > best {
                    (index, best) = (i, score);
                }
            }
            assert_eq!(good.best_candidate_index(&poor, &samples), index);
            let ties = [vec![4.9; 2], vec![0.0; 2]];
            assert_eq!(poor.best_candidate_index(&poor, &ties), 0);
        }
    }

    #[test]
    fn reused_normalizers_match_rebuilt_models_across_history_changes() {
        let mut previous = None;
        let mut rng = StdRng::seed_from_u64(931);
        // Mix a tight cluster with sparse observations to exercise clipped and changing widths.
        let values: Vec<f64> = (0..150)
            .map(|i| rng.gen_range(if i % 3 == 0 { 0.55..0.57 } else { 0.6..5.0 }))
            .collect();
        for (step, n) in [1, 2, 30, 97, 98, 99, 140, 139, 130, 145, 145]
            .into_iter()
            .enumerate()
        {
            let mut ids: Vec<u32> = (0..n).collect();
            if step >= 7 {
                ids.retain(|&id| id != 17);
            }
            let observations: Vec<f64> = ids
                .iter()
                .map(|&id| {
                    if step >= 8 && id == 3 {
                        0.56
                    } else {
                        values[id as usize]
                    }
                })
                .collect();
            // Keep the width unchanged when shifting bounds, to exercise coefficient invalidation.
            let shift = if step >= 9 { 0.25 } else { 0.0 };
            let space = Distribution::new_float(0.25 + shift, 5.25 + shift, None, step >= 10);
            let other_space = Distribution::new_float(0.1, 7.0, None, true);
            let weights: Vec<f64> = (0..ids.len()).map(|i| (i + step + 1) as f64).collect();
            let parameters = || {
                [
                    ("x", &space, observations.iter().copied()),
                    ("y", &other_space, observations.iter().copied()),
                ]
            };
            let fresh = ParzenEstimator::from_ordered_observations(
                parameters(),
                ids.clone(),
                &weights,
                1.0,
                None,
            );
            let cached = ParzenEstimator::from_ordered_observations(
                parameters(),
                ids,
                &weights,
                1.0,
                previous.take(),
            );
            let mut a = StdRng::seed_from_u64(42);
            let mut b = StdRng::seed_from_u64(42);
            let samples = fresh.sample_ordered(&mut a, 100);
            assert_eq!(samples, cached.sample_ordered(&mut b, 100));
            for x in &samples {
                assert_eq!(
                    fresh.log_pdf_ordered(x).to_bits(),
                    cached.log_pdf_ordered(x).to_bits(),
                    "step={step}"
                );
            }
            previous = Some(cached);
        }
    }

    #[test]
    fn binned_sigmas_matches_a_naive_scan() {
        // The larger distance to the two neighbors in sorted order, floored.
        let naive = |obs: &[f64], minsigma: f64| {
            let n = obs.len();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| obs[a].total_cmp(&obs[b]));
            let mut want = vec![0.0; n];
            for (j, &i) in order.iter().enumerate() {
                // A missing neighbor leaves NaN, which `max` discards.
                let gap = |k: usize| (obs[order[k]] - obs[i]).abs();
                let left = if j > 0 { gap(j - 1) } else { f64::NAN };
                let right = if j + 1 < n { gap(j + 1) } else { f64::NAN };
                want[i] = left.max(right).max(minsigma);
            }
            want
        };

        let (low, high) = (-10.0, 10.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for m in [2, 3, 7, 64, 500] {
            for spread in [1.0, 1e-4, 1e-9] {
                // Endpoints, a duplicate, and a cluster far tighter than one bin, so that the
                // neighbor rule and its tie-breaking are both exercised.
                let mut obs: Vec<f64> = (0..m).map(|_| rng.gen_range(-spread..spread)).collect();
                (obs[0], obs[m - 1], obs[m / 2]) = (low, high, obs[m / 3]);

                let minsigma = (high - low) / (100.0_f64.min(2.0 + m as f64));
                let (mut binned, mut updated) = (Vec::new(), Vec::new());
                binned_sigmas(&obs, low, high, &mut binned, &mut None, None, &mut updated).unwrap();
                assert_eq!(binned, naive(&obs, minsigma), "m = {m}, spread = {spread}");
            }
        }
    }

    #[test]
    fn fixed_bins_preserve_extreme_ties_after_insertions() {
        let mut obs = vec![0.0; 100];
        obs[..4].copy_from_slice(&[-4.0, -4.0, 4.0, 4.0]);
        let (mut cache, mut old_mus) = (None, Vec::new());
        for (index, value) in [
            (100, 0.0),
            (0, -4.0),
            (5, 4.0),
            (103, -0.0),
            (52, 2.0),
            (6, -2.0),
        ] {
            let positions: Vec<_> = (0..obs.len())
                .map(|i| i + usize::from(i >= index))
                .collect();
            obs.insert(index, value);
            let build = |sigmas, cache, previous| {
                binned_sigmas(&obs, -5.0, 5.0, sigmas, cache, previous, &mut Vec::new()).unwrap();
            };
            let (mut fresh, mut cached, mut no_cache) = (Vec::new(), Vec::new(), None);
            build(&mut fresh, &mut no_cache, None);
            let previous = (&old_mus[..], &positions[..], &[index][..]);
            build(&mut cached, &mut cache, Some(previous));
            assert_eq!(cached, fresh, "insertion at {index}");
            assert!(cache.is_some());
            old_mus = obs.clone();
            old_mus.push(0.0); // Prior.
        }
    }

    #[test]
    fn build_parzen_estimator() {
        let mut observations = HashMap::<String, Vec<f64>>::new();
        observations.insert("a".to_string(), vec![0.1, 0.4, 0.35]);
        observations.insert("c".to_string(), vec![0.05, 0.3, 0.7]);
        observations.insert("b".to_string(), vec![1.0, 3.0, 4.0]);
        observations.insert("d".to_string(), vec![2.0, 4.0, 5.0]);
        observations.insert("e".to_string(), vec![0.0, 1.0, 2.0]);

        let mut search_space = HashMap::<String, Distribution>::new();
        search_space.insert(
            "a".to_string(),
            Distribution::new_float(0.01, 1.0, None, false),
        );
        search_space.insert(
            "b".to_string(),
            Distribution::new_float(0.01, 1.0, None, true),
        );
        search_space.insert("c".to_string(), Distribution::new_int(1, 5, 1, false));
        search_space.insert("d".to_string(), Distribution::new_int(1, 5, 1, true));
        search_space.insert("e".to_string(), Distribution::new_categorical(3));

        let parzen_estimator =
            ParzenEstimator::new(&observations, &search_space, &[0.2, 0.5, 0.3], 1.0);
        let streamed = ParzenEstimator::from_ordered_observations(
            ["a", "b", "c", "d", "e"].map(|name| {
                (
                    name,
                    &search_space[name],
                    observations[name].iter().copied(),
                )
            }),
            vec![0, 1, 2],
            &[0.2, 0.5, 0.3],
            1.0,
            None,
        );
        let mut rng = StdRng::seed_from_u64(42);
        let samples = parzen_estimator.sample(&mut rng, 10);
        let mut ordered_rng = StdRng::seed_from_u64(42);
        let ordered = parzen_estimator.sample_ordered(&mut ordered_rng, 10);
        let mut streamed_rng = StdRng::seed_from_u64(42);
        assert_eq!(ordered, streamed.sample_ordered(&mut streamed_rng, 10));
        assert_eq!(
            parzen_estimator.parameter_names(),
            ["a", "b", "c", "d", "e"]
        );
        for (sample, values) in samples.iter().zip(&ordered) {
            assert_eq!(
                *values,
                ["a", "b", "c", "d", "e"].map(|name| sample[name]).to_vec()
            );
            assert_eq!(
                parzen_estimator.log_pdf(sample).to_bits(),
                parzen_estimator.log_pdf_ordered(values).to_bits()
            );
            assert_eq!(
                parzen_estimator.log_pdf_ordered(values).to_bits(),
                streamed.log_pdf_ordered(values).to_bits()
            );
        }
        assert_eq!(
            rand::RngCore::next_u64(&mut rng),
            rand::RngCore::next_u64(&mut ordered_rng)
        );
        assert_eq!(parzen_estimator.log_pdf_ordered(&[]), f64::NEG_INFINITY);
        assert_eq!(
            parzen_estimator.log_pdf_ordered(&[0.0; 6]),
            f64::NEG_INFINITY
        );
        let mut invalid = ordered[0].clone();
        invalid[0] = -1.0;
        assert_eq!(
            parzen_estimator.log_pdf_ordered(&invalid),
            f64::NEG_INFINITY
        );
        assert_eq!(samples.len(), 10);
        for sample in samples.iter() {
            let a = sample.get("a").unwrap();
            assert!(*a >= 0.01 && *a <= 1.0);
            let b = sample.get("b").unwrap();
            assert!(*b >= 0.01 && *b <= 1.0);
            let c = sample.get("c").unwrap();
            assert!(*c == 1.0 || *c == 2.0 || *c == 3.0 || *c == 4.0 || *c == 5.0);
            let d = sample.get("d").unwrap();
            assert!(*d == 1.0 || *d == 2.0 || *d == 3.0 || *d == 4.0 || *d == 5.0);
            let e = sample.get("e").unwrap();
            assert!(*e == 0.0 || *e == 1.0 || *e == 2.0);
        }
    }

    #[test]
    fn ordered_observations_keep_the_uniform_categorical_prior() {
        let distribution = Distribution::new_categorical(3);
        let model = ParzenEstimator::from_ordered_observations(
            [("x", &distribution, std::iter::empty())],
            Vec::new(),
            &[],
            1.0,
            None,
        );
        for category in 0..3 {
            assert_eq!(
                model.log_pdf_ordered(&[category as f64]),
                (1.0_f64 / 3.0).ln()
            );
        }
    }

    #[test]
    #[should_panic(expected = "Observation columns and weights differ in length")]
    fn ordered_observations_reject_inconsistent_weights() {
        ParzenEstimator::from_ordered_observations(
            [("x", &Distribution::new_categorical(3), [1.0, 2.0])],
            vec![0],
            &[1.0],
            1.0,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "Parameter names must be sorted and unique")]
    fn ordered_observations_reject_unsorted_parameters() {
        let distribution = Distribution::new_categorical(3);
        ParzenEstimator::from_ordered_observations(
            [("b", &distribution, [0.0]), ("a", &distribution, [1.0])],
            vec![0],
            &[1.0],
            1.0,
            None,
        );
    }
}
