pub(crate) struct ScottNumericalDistributionBuilder<'a> {
    weights: &'a [f64],
}

impl<'a> ScottNumericalDistributionBuilder<'a> {
    pub(crate) fn new(weights: &'a [f64]) -> Self {
        Self { weights }
    }

    /// Computes the shared Scott's-rule kernel width into `sigmas` for the
    /// already log-transformed `observations`; the caller appends the prior width.
    // NOTE: Since the Optuna TPE bandwidth selection is too wide for this analysis, we use the Scott's rule:
    // David W. Scott. 1992. Multivariate Density Estimation: Theory, Practice, and Visualization. John Wiley & Sons.
    pub(crate) fn sigmas(&self, observations: &[f64], sigmas: &mut Vec<f64>) {
        let weights_cum = self
            .weights
            .iter()
            .scan(0.0, |acc, &w| {
                *acc += w;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        let weights_sum = weights_cum.last().cloned().unwrap_or(1.0);
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
        sigmas.clear();
        sigmas.extend(std::iter::repeat_n(
            sigma_est.max(sigma_min),
            n_observations,
        ));
    }
}
