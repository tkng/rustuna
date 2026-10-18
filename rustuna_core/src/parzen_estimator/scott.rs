use crate::parzen_estimator::model::{BinState, NumericalDistributionBuilder};
use crate::parzen_estimator::probability_distributions::NumericalParameters;

pub(crate) struct ScottNumericalDistributionBuilder<'a> {
    weights: &'a [f64],
}

impl<'a> ScottNumericalDistributionBuilder<'a> {
    pub(crate) fn new(weights: &'a [f64]) -> Self {
        Self { weights }
    }
}

impl<'a> NumericalDistributionBuilder for ScottNumericalDistributionBuilder<'a> {
    fn calculate_parameters(
        &self,
        observations: impl Iterator<Item = f64>,
        parameters: &mut NumericalParameters,
        _bins: &mut Option<BinState>,
        _previous: Option<(&[f64], &[usize], &[usize])>,
    ) {
        // NOTE: Since the Optuna TPE bandwidth selection is too wide for this analysis, we use the Scott's rule:
        // David W. Scott. 1992. Multivariate Density Estimation: Theory, Practice, and Visualization. John Wiley & Sons.
        let NumericalParameters {
            mus,
            sigmas,
            low,
            high,
            log,
            ..
        } = parameters;
        let weights_cum = self
            .weights
            .iter()
            .scan(0.0, |acc, &w| {
                *acc += w;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        let weights_sum = weights_cum.last().cloned().unwrap_or(1.0);
        mus.clear();
        if *log {
            mus.extend(observations.map(|v| v.ln()));
            *low = low.ln();
            *high = high.ln();
        } else {
            mus.extend(observations);
        }
        let observations = mus.as_slice();
        let n_observations = observations.len();

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
        mus.push((*low + *high) / 2.0);
        sigmas.clear();
        sigmas.extend(std::iter::repeat_n(
            sigma_est.max(sigma_min),
            n_observations,
        ));
        sigmas.push(*high - *low + 1.0);
    }
}
