use rand::rngs::StdRng;
use std::{collections::HashMap, vec};

use super::probability_distributions::CategoricalDistributions;
use super::probability_distributions::{
    Distributions, MixtureOfProductDistribution, NumericalParameters,
};
use crate::distribution::Distribution;
use crate::parzen_estimator::scott::ScottNumericalDistributionBuilder;

pub struct ParzenEstimator {
    mixuture_distribution: MixtureOfProductDistribution,
    observation_ids: Vec<u32>,
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
    /// An optional previous model supplies buffers and unchanged density coefficients.
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
        let (old_names, old_distributions, indices) = match previous {
            Some((model, indices)) => {
                let m = model.mixuture_distribution;
                (m.param_names, m.distributions, indices)
            }
            None => (Vec::new(), Vec::new(), &[][..]),
        };
        let mut old = old_names.into_iter().zip(old_distributions);
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
            let reuse = old
                .next()
                .filter(|(old_name, _)| old_name == name)
                .map(|(_, distribution)| (distribution, indices));
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
                    num_builder.calculate_parameters(values, &mut parameters_buffer);
                    Distributions::from_numerical(&mut parameters_buffer, reuse, &mut constants)
                }
                Distribution::Categorical { .. } => {
                    cat_builder.calculate_categorical_distribution(values, search_space)
                }
            };
            distributions.push((name.to_owned(), distribution));
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
}

pub(crate) trait NumericalDistributionBuilder {
    fn calculate_parameters(
        &self,
        observations: impl Iterator<Item = f64>,
        parameters: &mut NumericalParameters,
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

/// Each observation's larger distance to its two sorted neighbors, floored at `minsigma`, or
/// `None` if fewer than two observations are given or some of them are outside the search space.
///
/// A neighbor nearer than `minsigma` cannot lift that floor, and binning at that width hides
/// exactly those: within a bin only the extremes can have a farther neighbor, and it is the
/// adjacent non-empty bin's extreme. The per-bin extremes therefore give every bandwidth.
fn binned_sigmas(obs: &[f64], low: f64, high: f64, sigmas: &mut Vec<f64>) -> Option<()> {
    if obs.len() < 2 {
        // No inter-observation neighbor exists; the caller falls back to endpoint distances.
        return None;
    }

    let n_bins = 100.min(obs.len() + 2);
    let minsigma = (high - low) / n_bins as f64;

    let mut bins = vec![(f64::INFINITY, 0_usize, f64::NEG_INFINITY, 0_usize); n_bins];
    for (i, &v) in obs.iter().enumerate() {
        if !(low..=high).contains(&v) {
            // The observation is outside the search space.
            return None;
        }
        let bin = &mut bins[(((v - low) / minsigma) as usize).min(n_bins - 1)];
        if v < bin.0 {
            (bin.0, bin.1) = (v, i);
        }
        if v >= bin.2 {
            (bin.2, bin.3) = (v, i);
        }
    }

    sigmas.resize(obs.len(), minsigma);
    sigmas.fill(minsigma);
    let (mut prev_max, mut next_min) = (f64::NAN, f64::NAN);
    for bin in bins.iter().filter(|b| b.0 <= b.2) {
        sigmas[bin.1] = sigmas[bin.1].max(bin.0 - prev_max);
        prev_max = bin.2;
    }
    for bin in bins.iter().rev().filter(|b| b.0 <= b.2) {
        sigmas[bin.3] = sigmas[bin.3].max(next_min - bin.2);
        next_min = bin.0;
    }
    Some(())
}

impl NumericalDistributionBuilder for DefaultNumericalDistributionBuilder {
    fn calculate_parameters(
        &self,
        observations: impl Iterator<Item = f64>,
        parameters: &mut NumericalParameters,
    ) {
        // Currently, we assume consider_prior=True, consider_endpoints=False, and consider_magic_clip=True.
        let NumericalParameters {
            mus,
            sigmas,
            low,
            high,
            step,
            log,
        } = parameters;
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
            sigmas.push(adj_high - adj_low);
        } else if binned_sigmas(&mus[..mus.len() - 1], adj_low, adj_high, sigmas).is_some() {
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
    fn reused_normalizers_match_rebuilt_models_across_history_changes() {
        let mut previous = None;
        let mut rng = StdRng::seed_from_u64(931);
        // A tight cluster lets a mean change while its clipped bandwidth stays unchanged.
        let values: Vec<f64> = (0..150).map(|_| rng.gen_range(0.55..0.57)).collect();
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
                let mut binned = Vec::new();
                binned_sigmas(&obs, low, high, &mut binned).unwrap();
                assert_eq!(binned, naive(&obs, minsigma), "m = {m}, spread = {spread}");
            }
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
