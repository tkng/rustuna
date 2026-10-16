use crate::distribution::Distribution;
use crate::parzen_estimator::model::NumericalDistributionBuilder;
use crate::parzen_estimator::probability_distributions::{
    DiscreteTruncLogNormDistributions, DiscreteTruncNormDistributions, Distributions,
    TruncLogNormDistributions, TruncNormDistributions,
};

pub(crate) struct ScottNumericalDistributionBuilder<'a> {
    weights: &'a [f64],
}

impl<'a> ScottNumericalDistributionBuilder<'a> {
    pub(crate) fn new(weights: &'a [f64]) -> Self {
        Self { weights }
    }
}

impl<'a> NumericalDistributionBuilder for ScottNumericalDistributionBuilder<'a> {
    fn calculate_numerical_distribution(
        &self,
        observations: impl Iterator<Item = f64>,
        search_space: &Distribution,
    ) -> Distributions {
        // NOTE: Since the Optuna TPE bandwidth selection is too wide for this analysis, we use the Scott's rule:
        // David W. Scott. 1992. Multivariate Density Estimation: Theory, Practice, and Visualization. John Wiley & Sons.
        let (low, high, step, log) = match search_space {
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => (*low as f64, *high as f64, Some(*step as f64), *log),
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => (*low, *high, *step, *log),
            _ => panic!("Unsupported distribution type for ScottNumericalDistributionBuilder"),
        };
        let weights_cum = self
            .weights
            .iter()
            .scan(0.0, |acc, &w| {
                *acc += w;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        let weights_sum = weights_cum.last().cloned().unwrap_or(1.0);
        let observations: Vec<f64> = if log {
            observations.map(|v| v.ln()).collect()
        } else {
            observations.collect()
        };
        let n_observations = observations.len();
        let (low, high) = if log {
            (low.ln(), high.ln())
        } else {
            (low, high)
        };

        let mean_est = observations
            .iter()
            .zip(self.weights.iter())
            .map(|(v, w)| v * w)
            .sum::<f64>()
            / weights_sum;
        let sigma_est = {
            let var = observations
                .iter()
                .zip(self.weights.iter())
                .map(|(v, w)| w * (v - mean_est).powi(2))
                .sum::<f64>()
                / (weights_sum - 1.0).max(1.0);
            var.sqrt()
        };

        let q1_idx = weights_cum.partition_point(|&v| v < (weights_sum / 4.0).floor());
        let q3_idx = weights_cum.partition_point(|&v| v <= (weights_sum * 3.0 / 4.0).floor());
        let iqr = observations[q3_idx.min(n_observations - 1)] - observations[q1_idx];

        let sigma_est = 1.059 * sigma_est.min(iqr / 1.34) * weights_sum.powf(-0.2);
        // To avoid numerical errors. 0.5/1.64 means 1.64sigma (=90%) will fit in the target grid.
        let sigma_min = 0.5 / 1.64;
        let mus_with_prior = observations
            .into_iter()
            .chain(std::iter::once((low + high) / 2.0));
        let sigmas = std::iter::repeat_n(sigma_est.max(sigma_min), n_observations);
        let sigmas_with_prior = sigmas.chain(std::iter::once(high - low + 1.0));

        match (step, log) {
            (None, false) => Distributions::TruncNorm(TruncNormDistributions::new(
                mus_with_prior.collect(),
                sigmas_with_prior.collect(),
                low,
                high,
            )),
            (None, true) => Distributions::TruncLogNorm(TruncLogNormDistributions::new(
                mus_with_prior.collect(),
                sigmas_with_prior.collect(),
                low,
                high,
            )),
            (Some(step), false) => {
                Distributions::DiscreteTruncNorm(DiscreteTruncNormDistributions::new(
                    mus_with_prior.collect(),
                    sigmas_with_prior.collect(),
                    low,
                    high,
                    step,
                ))
            }
            (Some(step), true) => {
                Distributions::DiscreteTruncLogNorm(DiscreteTruncLogNormDistributions::new(
                    mus_with_prior.collect(),
                    sigmas_with_prior.collect(),
                    low,
                    high,
                    step,
                ))
            }
        }
    }
}
