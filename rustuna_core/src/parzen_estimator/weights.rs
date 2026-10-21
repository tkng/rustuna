use std::sync::Arc;

/// Normalized mixture weights, with the prior component in the last slot.
///
/// Model updates create these values after fitting kernels and use them to
/// compute log densities. Candidate generation can then consume the returned
/// values to build a selection table, without retaining linear weights in the
/// model or reconstructing them from logs.
pub fn normalized_weights(observations: &[f64], prior: f64) -> Vec<f64> {
    let mut sum = observations.iter().sum::<f64>() + prior;
    if sum == 0.0 {
        sum = (observations.len() + 1) as f64;
    }
    // Validate while normalizing so density-only updates need no extra scan.
    let mut valid = true;
    let mut has_mass = false;
    let values: Vec<f64> = observations
        .iter()
        .copied()
        .chain(std::iter::once(prior))
        .map(|weight| {
            let weight = weight / sum;
            valid &= weight.is_finite() && weight >= 0.0;
            has_mass |= weight > 0.0;
            weight
        })
        .collect();
    assert!(
        valid && has_mass,
        "mixture weights must be finite, non-negative and have positive total mass"
    );
    values
}

/// Parameter-independent inputs of one model update: kernel row positions,
/// the trials behind them, and their mixture coefficients.
///
/// Estimators fitted on identical rows and weights — independent TPE
/// parameters with equal presence masks — produce identical contents, so one
/// instance can be shared instead of retaining O(parameters × history) copies.
/// Whether an existing instance describes a given update is the caller's
/// responsibility; `ParzenEstimator::update` always builds a private one.
#[derive(Clone, Debug)]
pub struct SharedState {
    /// Row positions into the observation columns, one per observation ID.
    pub(crate) order: Arc<[usize]>,
    /// Trial numbers backing the fitted kernels, sorted strictly increasing.
    pub(crate) observation_ids: Arc<[u32]>,
    /// Normalized mixture weights, with the prior component in the last slot.
    pub(crate) weights: Arc<[f64]>,
    pub(crate) log_weights: Arc<[f64]>,
    pub(crate) log_sum_weights: f64,
}

impl SharedState {
    /// Normalizes `weights` and derives the log-density coefficients.
    /// `observation_ids` must be sorted and unique, and `order` and `weights`
    /// must have one element per ID.
    pub fn new(
        order: &[usize],
        observation_ids: &[u32],
        weights: &[f64],
        prior_weight: f64,
    ) -> Self {
        assert!(
            observation_ids.windows(2).all(|ids| ids[0] < ids[1]),
            "Observation IDs must be sorted and unique"
        );
        assert_eq!(
            order.len(),
            observation_ids.len(),
            "Observation columns and weights differ in length"
        );
        assert_eq!(
            weights.len(),
            observation_ids.len(),
            "Observation columns and weights differ in length"
        );
        let weights: Arc<[f64]> = normalized_weights(weights, prior_weight).into();
        Self {
            order: order.into(),
            observation_ids: observation_ids.into(),
            log_weights: weights.iter().map(|&w| w.max(0.0).ln()).collect(),
            log_sum_weights: weights.iter().sum::<f64>().max(0.0).ln(),
            weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_include_the_prior_and_allow_zero_mass_components() {
        assert_eq!(
            normalized_weights(&[0.0, 2.0, 1.0], 1.0),
            [0.0, 0.5, 0.25, 0.25]
        );
        assert_eq!(normalized_weights(&[], 1.0), [1.0]);
        assert_eq!(normalized_weights(&[1.0], 0.0), [1.0, 0.0]);
        assert_eq!(
            normalized_weights(&[0.0, 1.0, 0.0], 0.0),
            [0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(normalized_weights(&[], f64::from_bits(1)), [1.0]);
    }

    #[test]
    fn invalid_weights_are_rejected_before_model_or_candidate_construction() {
        for (observations, prior) in [
            (vec![], 0.0),
            (vec![0.0], 0.0),
            (vec![-1.0], 2.0),
            (vec![f64::NAN], 1.0),
            (vec![1.0, f64::NAN], 1.0),
            (vec![1.0, -0.5], 1.0),
            (vec![1.0], f64::NAN),
            (vec![1.0], f64::INFINITY),
            (vec![f64::MAX, f64::MAX], 1.0),
        ] {
            assert!(std::panic::catch_unwind(|| normalized_weights(&observations, prior)).is_err());
        }
    }

    #[test]
    fn shared_state_normalizes_and_derives_log_coefficients() {
        let state = SharedState::new(&[2, 5, 7], &[10, 20, 30], &[1.0, 2.0, 1.0], 1.0);
        assert_eq!(state.order[..], [2, 5, 7]);
        assert_eq!(state.observation_ids[..], [10, 20, 30]);
        assert_eq!(state.weights[..], [0.2, 0.4, 0.2, 0.2]);
        assert_eq!(
            state.log_weights[..],
            [0.2_f64.ln(), 0.4_f64.ln(), 0.2_f64.ln(), 0.2_f64.ln()]
        );
        assert_eq!(state.log_sum_weights, 1.0_f64.ln());
    }

    #[test]
    fn shared_state_rejects_unsorted_or_mismatched_inputs() {
        for (order, ids, weights) in [
            // Unsorted IDs.
            (&[0, 1][..], &[5, 3][..], &[1.0, 1.0][..]),
            // `order`/`weights` and ID length mismatches.
            (&[0][..], &[5, 6][..], &[1.0, 1.0][..]),
            (&[0, 1][..], &[5, 6][..], &[1.0][..]),
        ] {
            assert!(
                std::panic::catch_unwind(|| SharedState::new(order, ids, weights, 1.0)).is_err()
            );
        }
    }
}
