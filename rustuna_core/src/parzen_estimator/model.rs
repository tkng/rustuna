use rand::rngs::StdRng;
use std::collections::HashMap;
use std::iter::once;

#[cfg(test)]
use super::probability_distributions::binned_sigmas;
use super::probability_distributions::{
    BinState, Distributions, KernelUpdate, MixtureOfProductDistribution, NormalBuffers,
    ObservationColumn,
};
use crate::distribution::Distribution;
use crate::parzen_estimator::scott::ScottNumericalDistributionBuilder;

pub struct ParzenEstimator {
    mixuture_distribution: MixtureOfProductDistribution,
    observation_ids: Vec<u32>,
    bins: Vec<Option<BinState>>,
}

impl Default for ParzenEstimator {
    fn default() -> Self {
        Self {
            mixuture_distribution: MixtureOfProductDistribution::new(Vec::new(), vec![1.0]),
            observation_ids: Vec::new(),
            bins: Vec::new(),
        }
    }
}

impl ParzenEstimator {
    pub fn new(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &HashMap<String, Distribution>,
        weights: &[f64],
        prior_weight: f64,
    ) -> Self {
        Self::with_builder(observations, search_space, weights, prior_weight, None)
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
            Some(&ScottNumericalDistributionBuilder::new(weights)),
        )
    }

    fn with_builder(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &HashMap<String, Distribution>,
        weights: &[f64],
        prior_weight: f64,
        scott: Option<&ScottNumericalDistributionBuilder>,
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
        let owned: Vec<Vec<f64>> = keys
            .iter()
            .map(|key| {
                observations
                    .get(key.as_str())
                    .map(|v| v.to_vec())
                    .unwrap_or_default()
            })
            .collect();
        let order: Vec<usize> = (0..n_observations).collect();
        let mut model = Self::default();
        model.update_with(
            &order,
            (0..n_observations as u32).collect(),
            weights,
            prior_weight,
            keys.iter()
                .zip(&owned)
                .map(|(key, values)| ObservationColumn {
                    name: key.as_str(),
                    distribution: &search_space[key.as_str()],
                    values,
                }),
            scott,
            false,
        );
        model
    }

    /// Builds a model from columns in strictly increasing parameter-name order.
    /// Each column and the strictly increasing observation IDs must match the weights.
    /// An optional previous model supplies kernels, bins and unchanged coefficients.
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
        let owned: Vec<(String, &Distribution, Vec<f64>)> = parameters
            .into_iter()
            .map(|(name, distribution, values)| {
                let values: Vec<f64> = values.into_iter().collect();
                assert_eq!(
                    values.len(),
                    observation_ids.len(),
                    "Observation columns and weights differ in length"
                );
                (name.to_owned(), distribution, values)
            })
            .collect();
        let order: Vec<usize> = (0..observation_ids.len()).collect();
        let mut model = previous.unwrap_or_default();
        model.update_with(
            &order,
            observation_ids,
            weights,
            prior_weight,
            owned
                .iter()
                .map(|(name, distribution, values)| ObservationColumn {
                    name,
                    distribution: *distribution,
                    values,
                }),
            None,
            false,
        );
        model
    }

    /// Rebuilds this model from `columns` at the rows given by `order`, in strictly
    /// increasing parameter-name order. Kernels for observations already present in
    /// this model are reused; only newly added observations are read from the columns.
    #[doc(hidden)]
    pub fn update<'a, I>(
        &mut self,
        order: &[usize],
        observation_ids: Vec<u32>,
        weights: &[f64],
        prior_weight: f64,
        columns: I,
    ) where
        I: IntoIterator<Item = ObservationColumn<'a>>,
    {
        self.update_with(
            order,
            observation_ids,
            weights,
            prior_weight,
            columns,
            None,
            true,
        );
    }

    fn update_with<'a, I>(
        &mut self,
        order: &[usize],
        observation_ids: Vec<u32>,
        weights: &[f64],
        prior_weight: f64,
        columns: I,
        scott: Option<&ScottNumericalDistributionBuilder>,
        trusted: bool,
    ) where
        I: IntoIterator<Item = ObservationColumn<'a>>,
    {
        assert_eq!(order.len(), observation_ids.len());
        assert!(
            observation_ids.windows(2).all(|ids| ids[0] < ids[1]),
            "Observation IDs must be sorted and unique"
        );
        assert_eq!(weights.len(), observation_ids.len());
        let update = KernelUpdate::between(&self.observation_ids, &observation_ids);
        let m = &mut self.mixuture_distribution;
        let mut old = std::mem::take(&mut m.param_names)
            .into_iter()
            .zip(std::mem::take(&mut m.distributions))
            .zip(std::mem::take(&mut self.bins));
        let mut updated = Vec::new();
        let mut spare = NormalBuffers::default();
        let mut n_kernels = None;
        let mut last_name = "";
        for column in columns {
            assert!(
                column.name > last_name,
                "Parameter names must be sorted and unique"
            );
            last_name = column.name;
            let (name, distribution, bins) =
                match old.next().filter(|((name, _), _)| name == column.name) {
                    Some(((name, mut d), mut bins)) => {
                        d.update(
                            &column,
                            order,
                            &update,
                            &mut bins,
                            scott,
                            trusted,
                            &mut updated,
                            &mut spare,
                        );
                        (name, d, bins)
                    }
                    None => (
                        column.name.to_owned(),
                        Distributions::new_from_column(
                            &column,
                            order,
                            &update,
                            &mut None,
                            scott,
                            &mut updated,
                            &mut spare,
                        ),
                        None,
                    ),
                };
            match n_kernels {
                None => n_kernels = Some(distribution.n_kernels()),
                Some(k) => assert_eq!(
                    k,
                    distribution.n_kernels(),
                    "Observation columns and weights differ in length"
                ),
            }
            m.param_names.push(name);
            m.distributions.push(distribution);
            self.bins.push(bins);
        }
        assert!(
            n_kernels.is_none_or(|k| k == observation_ids.len() + 1),
            "Observation columns and weights differ in length"
        );
        let sum_w = {
            let s = weights.iter().sum::<f64>() + prior_weight;
            if s == 0.0 {
                (weights.len() + 1) as f64
            } else {
                s
            }
        };
        m.set_weights(
            weights
                .iter()
                .copied()
                .chain(once(prior_weight))
                .map(|w| w / sum_w)
                .collect(),
        );
        self.observation_ids = observation_ids;
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

    /// Samples into a caller-owned buffer so row allocations are reused.
    #[doc(hidden)]
    pub fn sample_ordered_into(&self, rng: &mut StdRng, size: usize, samples: &mut Vec<Vec<f64>>) {
        self.mixuture_distribution
            .sample_ordered_into(rng, size, samples)
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
            let Some(poor_density) =
                poor.mixuture_distribution
                    .log_pdf_ordered_bounded(&samples[i], good, best_val)
            else {
                continue;
            };
            let acquisition = good - poor_density;
            if acquisition > best_val || (acquisition == best_val && i < best_idx) {
                best_val = acquisition;
                best_idx = i;
            }
        }
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    /// Repeated `update` calls after `mutate` must match a fresh rebuild
    /// of the same observation set bit-for-bit.
    fn updates_match_rebuilt(
        dists: &[Distribution],
        initial: usize,
        rounds: usize,
        seed: u64,
        mutate: impl Fn(&mut Vec<Vec<f64>>, &mut Vec<u32>, &mut StdRng),
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let names: Vec<String> = (0..dists.len()).map(|i| format!("p{i}")).collect();
        let mut obs: Vec<Vec<f64>> = dists
            .iter()
            .map(|_| (0..initial).map(|_| rng.gen_range(-5.0..5.0)).collect())
            .collect();
        let mut ids: Vec<u32> = (0..initial as u32).collect();
        let mut model = ParzenEstimator::from_ordered_observations(
            names
                .iter()
                .zip(dists)
                .zip(&obs)
                .map(|((n, d), v)| (n.as_str(), d, v.iter().copied())),
            ids.clone(),
            &vec![1.0; initial],
            1.0,
            None,
        );
        let mut columns: Vec<Vec<f64>> = vec![vec![f64::NAN; initial + 2 * rounds]; dists.len()];
        for round in 0..rounds {
            mutate(&mut obs, &mut ids, &mut rng);
            let n = ids.len();
            let weights = vec![1.0; n];
            let order: Vec<usize> = (0..n).collect();
            for (values, column) in obs.iter().zip(&mut columns) {
                assert_eq!(values.len(), n);
                for (cell, &v) in column.iter_mut().zip(values) {
                    *cell = v;
                }
            }
            model.update(
                &order,
                ids.clone(),
                &weights,
                1.0,
                names
                    .iter()
                    .zip(dists)
                    .zip(&columns)
                    .map(|((name, distribution), values)| ObservationColumn {
                        name,
                        distribution,
                        values,
                    }),
            );
            let fresh = ParzenEstimator::from_ordered_observations(
                names
                    .iter()
                    .zip(dists)
                    .zip(&obs)
                    .map(|((n, d), v)| (n.as_str(), d, v.iter().copied())),
                ids.clone(),
                &weights,
                1.0,
                None,
            );
            for point in [-4.0, -1.0, 0.0, 2.0, 4.9] {
                let x = vec![point; dists.len()];
                assert_eq!(
                    model.mixuture_distribution.log_pdf_ordered(&x).to_bits(),
                    fresh.mixuture_distribution.log_pdf_ordered(&x).to_bits(),
                    "round={round} point={point}"
                );
            }
        }
    }

    #[test]
    fn columnar_append_matches_rebuilt_model() {
        let dist = Distribution::Float {
            low: -5.0,
            high: 5.0,
            step: None,
            log: false,
        };
        for m in [3_usize, 60, 150] {
            updates_match_rebuilt(&[dist.clone()], m, 6, 7, |obs, ids, rng| {
                obs[0].push(rng.gen_range(-5.0..5.0));
                ids.push(ids.len() as u32);
            });
        }
    }

    #[test]
    fn columnar_scatter_matches_rebuilt_model() {
        // Remove the middle id and append two new ids, like the good/poor
        // reshuffling of real TPE updates.
        let dist = Distribution::Float {
            low: -5.0,
            high: 5.0,
            step: None,
            log: false,
        };
        updates_match_rebuilt(&[dist.clone(), dist], 120, 8, 11, |obs, ids, rng| {
            let mid = obs[0].len() / 2;
            for values in obs.iter_mut() {
                values.remove(mid);
                values.push(rng.gen_range(-5.0..5.0));
                values.push(rng.gen_range(-5.0..5.0));
            }
            ids.remove(mid);
            let next = *ids.last().unwrap() + 1;
            ids.extend([next, next + 1]);
        });
    }

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
                binned_sigmas(
                    &obs,
                    low,
                    high,
                    &mut binned,
                    &mut Vec::new(),
                    &mut None,
                    &KernelUpdate::Fresh,
                    None,
                    &mut updated,
                );
                assert_eq!(binned, naive(&obs, minsigma), "m = {m}, spread = {spread}");
            }
        }
    }

    #[test]
    fn fixed_bins_preserve_extreme_ties_after_insertions() {
        let mut obs = vec![0.0; 100];
        obs[..4].copy_from_slice(&[-4.0, -4.0, 4.0, 4.0]);
        let mut cache = None;
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
            let indices: Vec<Option<usize>> = (0..obs.len())
                .map(|i| (i != index).then_some(i - usize::from(i > index)))
                .collect();
            let update = KernelUpdate::Scatter { positions, indices };
            let build = |sigmas, cache, update: &KernelUpdate| {
                binned_sigmas(
                    &obs,
                    -5.0,
                    5.0,
                    sigmas,
                    &mut Vec::new(),
                    cache,
                    update,
                    None,
                    &mut Vec::new(),
                );
            };
            let (mut fresh, mut cached, mut no_cache) = (Vec::new(), Vec::new(), None);
            build(&mut fresh, &mut no_cache, &KernelUpdate::Fresh);
            build(&mut cached, &mut cache, &update);
            assert_eq!(cached, fresh, "insertion at {index}");
            assert!(cache.is_some());
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
