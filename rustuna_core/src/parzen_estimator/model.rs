use rand::rngs::StdRng;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
use super::probability_distributions::binned_sigmas;
use super::probability_distributions::{
    BinState, Distributions, KernelUpdate, MixtureOfProductDistribution, NormalBuffers,
    ObservationColumn,
};
use super::SharedState;
use crate::distribution::Distribution;
use crate::parzen_estimator::scott::ScottNumericalDistributionBuilder;

/// Incrementally fitted mixture model shared by TPE and importance evaluation.
/// Retains density coefficients and update state, but no component-selection table.
pub struct ParzenEstimator {
    mixuture_distribution: MixtureOfProductDistribution,
    observation_ids: Arc<[u32]>,
    bins: Vec<Option<BinState>>,
}

impl Default for ParzenEstimator {
    fn default() -> Self {
        Self {
            mixuture_distribution: MixtureOfProductDistribution::new(Vec::new(), vec![1.0]),
            observation_ids: Vec::new().into(),
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
        let ids: Vec<u32> = (0..n_observations as u32).collect();
        let mut model = Self::default();
        model.update_with(
            &SharedState::new(&order, &ids, weights, prior_weight),
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
            &SharedState::new(&order, &observation_ids, weights, prior_weight),
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
    /// Returns the normalized weights used for this update so candidate generation
    /// can consume them. Density-only callers can discard these temporary values.
    #[doc(hidden)]
    pub fn update<'a, I>(
        &mut self,
        order: &[usize],
        observation_ids: Vec<u32>,
        weights: &[f64],
        prior_weight: f64,
        columns: I,
    ) -> Arc<[f64]>
    where
        I: IntoIterator<Item = ObservationColumn<'a>>,
    {
        self.update_shared(
            &SharedState::new(order, &observation_ids, weights, prior_weight),
            columns,
        )
    }

    /// Like [`Self::update`], but adopts row positions, observation IDs and
    /// density coefficients shared with other models. Models fitted on
    /// identical rows and weights retain one allocation of the shared state
    /// instead of one copy each. `shared` must describe the same rows and
    /// weights this update would consume; callers establish that by sharing
    /// only states built from identical inputs, e.g. parameters whose presence
    /// masks select the same rows of an unchanged observation set.
    #[doc(hidden)]
    pub fn update_shared<'a, I>(&mut self, shared: &SharedState, columns: I) -> Arc<[f64]>
    where
        I: IntoIterator<Item = ObservationColumn<'a>>,
    {
        self.update_with(shared, columns, None, true)
    }

    /// Whether `self` and `other` hold the same shared state allocation, i.e.
    /// both were updated with the same [`SharedState`] instance.
    #[doc(hidden)]
    pub fn shares_state_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.observation_ids, &other.observation_ids)
            && Arc::ptr_eq(
                &self.mixuture_distribution.log_weights,
                &other.mixuture_distribution.log_weights,
            )
    }

    fn update_with<'a, I>(
        &mut self,
        shared: &SharedState,
        columns: I,
        scott: Option<&ScottNumericalDistributionBuilder>,
        trusted: bool,
    ) -> Arc<[f64]>
    where
        I: IntoIterator<Item = ObservationColumn<'a>>,
    {
        let order = &shared.order[..];
        let update = KernelUpdate::between(&self.observation_ids, &shared.observation_ids);
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
            n_kernels.is_none_or(|k| k == shared.observation_ids.len() + 1),
            "Observation columns and weights differ in length"
        );
        m.log_weights = shared.log_weights.clone();
        m.log_sum_weights = shared.log_sum_weights;
        m.n_kernels = shared.weights.len();
        self.observation_ids = shared.observation_ids.clone();
        shared.weights.clone()
    }

    /// Parameter names in the order used by component draws and density values.
    pub fn parameter_names(&self) -> &[String] {
        &self.mixuture_distribution.param_names
    }

    /// Number of mixture components, including the trailing prior.
    pub fn component_count(&self) -> usize {
        self.mixuture_distribution.n_kernels
    }

    /// Draws one point from a caller-selected component into a reusable row.
    /// Values follow [`Self::parameter_names`]; component selection belongs to the caller.
    #[inline]
    pub fn draw_component(&self, rng: &mut StdRng, component: usize, values: &mut Vec<f64>) {
        self.mixuture_distribution
            .draw_component(rng, component, values)
    }

    pub fn log_pdf(&self, x: &HashMap<String, f64>) -> f64 {
        self.mixuture_distribution.log_pdf(x)
    }

    /// Scores values in [`Self::parameter_names`] order.
    /// Returns negative infinity if the value count does not match the model.
    pub fn log_pdf_ordered(&self, values: &[f64]) -> f64 {
        self.mixuture_distribution.log_pdf_ordered(values)
    }

    /// Log density of the weighted prior component; a lower bound on the mixture.
    pub fn log_pdf_prior_ordered(&self, values: &[f64]) -> f64 {
        self.mixuture_distribution.log_pdf_prior_ordered(values)
    }

    /// Evaluates the mixture unless `stop` rejects a finite lower bound on its log density.
    /// Returns `None` only when `stop` requests early termination.
    pub fn log_pdf_ordered_with_lower_bound(
        &self,
        values: &[f64],
        stop: impl FnOnce(f64) -> bool,
    ) -> Option<f64> {
        self.mixuture_distribution
            .log_pdf_ordered_with_lower_bound(values, stop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn draw(model: &ParzenEstimator, rng: &mut StdRng, component: usize) -> Vec<f64> {
        let mut values = Vec::new();
        model.draw_component(rng, component, &mut values);
        values
    }

    /// Models updated through a [`SharedState`] reuse its allocation and behave
    /// bit-for-bit like models built by [`ParzenEstimator::update`].
    #[test]
    fn shared_updates_match_plain_updates_bit_for_bit() {
        let space = Distribution::new_float(-5.0, 5.0, None, false);
        let x_values = [0.1, 0.2, 0.4];
        let y_values = [4.0, -2.0, 0.0];
        let shared = SharedState::new(&[0, 1, 2], &[10, 20, 30], &[1.0, 2.0, 1.0], 1.0);
        let mut a = ParzenEstimator::default();
        let a_weights = a.update_shared(
            &shared,
            [ObservationColumn {
                name: "x",
                distribution: &space,
                values: &x_values,
            }],
        );
        let mut b = ParzenEstimator::default();
        b.update_shared(
            &shared,
            [ObservationColumn {
                name: "y",
                distribution: &space,
                values: &y_values,
            }],
        );
        assert!(a.shares_state_with(&b));
        assert_eq!(a_weights[..], shared.weights[..]);

        let mut plain = ParzenEstimator::default();
        let plain_weights = plain.update(
            &[0, 1, 2],
            vec![10, 20, 30],
            &[1.0, 2.0, 1.0],
            1.0,
            [ObservationColumn {
                name: "x",
                distribution: &space,
                values: &x_values,
            }],
        );
        // Equal contents but a private allocation, so not shared in the Arc sense.
        assert!(!a.shares_state_with(&plain));
        assert_eq!(a_weights[..], plain_weights[..]);
        assert_eq!(a.parameter_names(), plain.parameter_names());
        for x in [-4.5, 0.0, 0.25, 3.3] {
            assert_eq!(
                a.log_pdf_ordered(&[x]).to_bits(),
                plain.log_pdf_ordered(&[x]).to_bits()
            );
        }
        let (mut rng_a, mut rng_b) = (StdRng::seed_from_u64(7), StdRng::seed_from_u64(7));
        for k in 0..a.component_count() {
            assert_eq!(
                draw(&a, &mut rng_a, k),
                draw(&plain, &mut rng_b, k),
                "component {k}"
            );
        }
    }

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
            for i in 0..100 {
                let component = i % fresh.component_count();
                let x = draw(&fresh, &mut a, component);
                assert_eq!(x, draw(&cached, &mut b, component));
                assert_eq!(
                    fresh.log_pdf_ordered(&x).to_bits(),
                    cached.log_pdf_ordered(&x).to_bits(),
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
        let mut streamed_rng = StdRng::seed_from_u64(42);
        assert_eq!(
            parzen_estimator.parameter_names(),
            ["a", "b", "c", "d", "e"]
        );
        let mut samples = Vec::new();
        for i in 0..10 {
            let component = i % parzen_estimator.component_count();
            let values = draw(&parzen_estimator, &mut rng, component);
            assert_eq!(values, draw(&streamed, &mut streamed_rng, component));
            let sample: HashMap<String, f64> = parzen_estimator
                .parameter_names()
                .iter()
                .cloned()
                .zip(values.iter().copied())
                .collect();
            let expected = parzen_estimator.log_pdf_ordered(&values).to_bits();
            assert_eq!(parzen_estimator.log_pdf(&sample).to_bits(), expected);
            assert_eq!(streamed.log_pdf_ordered(&values).to_bits(), expected);
            samples.push(sample);
        }
        assert_eq!(
            rand::RngCore::next_u64(&mut rng),
            rand::RngCore::next_u64(&mut streamed_rng)
        );
        assert_eq!(parzen_estimator.log_pdf_ordered(&[]), f64::NEG_INFINITY);
        assert_eq!(
            parzen_estimator.log_pdf_ordered(&[0.0; 6]),
            f64::NEG_INFINITY
        );
        assert_eq!(
            parzen_estimator.log_pdf_ordered(&[-1.0; 5]),
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
