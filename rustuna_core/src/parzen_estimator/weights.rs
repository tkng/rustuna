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
}
