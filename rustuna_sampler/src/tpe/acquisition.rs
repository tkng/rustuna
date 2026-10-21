use rustuna_core::internal::parzen_estimator::ParzenEstimator;

/// Selects the largest log density ratio, preserving the first candidate on ties.
/// Prior bounds and partial mixture evaluation avoid scoring candidates that cannot win.
pub(super) fn best_candidate_index(
    good: &ParzenEstimator,
    poor: &ParzenEstimator,
    samples: &[Vec<f64>],
) -> usize {
    let joint = good.parameter_names().len() > 1;
    let mut ranked: Vec<_> = samples
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let density = good.log_pdf_ordered(x);
            // A single prior is usually too weak a bound for independent TPE,
            // and unusable bounds must always receive a full evaluation.
            let upper = joint
                .then(|| density - poor.log_pdf_prior_ordered(x))
                .filter(|b| b.is_finite())
                .unwrap_or(f64::INFINITY);
            (i, density, upper)
        })
        .collect();
    if joint {
        ranked.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    }
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (i, density, upper) in ranked {
        if upper < best_val {
            break;
        }
        let Some(poor_density) =
            poor.log_pdf_ordered_with_lower_bound(&samples[i], |lower_bound| {
                density.is_finite() && best_val.is_finite() && density - lower_bound < best_val
            })
        else {
            continue;
        };
        let acquisition = density - poor_density;
        if acquisition > best_val || (acquisition == best_val && i < best_idx) {
            best_val = acquisition;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::super::candidates::generate_candidates;
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use rustuna_core::distribution::Distribution;
    use rustuna_core::internal::parzen_estimator::normalized_weights;

    #[test]
    fn candidate_pruning_matches_exhaustive_scoring() {
        for names in [&["x"][..], &["x", "y"][..]] {
            let space = Distribution::new_float(-5.0, 5.0, None, false);
            let model = |values: &[f64], prior| {
                ParzenEstimator::from_ordered_observations(
                    names
                        .iter()
                        .map(|&name| (name, &space, values.iter().copied())),
                    (0..values.len() as u32).collect(),
                    &vec![1.0; values.len()],
                    prior,
                    None,
                )
            };
            let good = model(&[-3.0, -2.0, -1.0], 1.0);
            for prior in [0.0, 1.0] {
                let poor = model(&[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], prior);
                let mut samples = Vec::new();
                generate_candidates(
                    &good,
                    &normalized_weights(&[1.0; 3], 1.0),
                    &mut StdRng::seed_from_u64(42),
                    24,
                    &mut samples,
                );
                let invalid =
                    [f64::NAN, f64::INFINITY, f64::NEG_INFINITY].map(|v| vec![v; names.len()]);
                assert_eq!(best_candidate_index(&good, &poor, &invalid), 0);
                samples.splice(0..0, invalid);
                let (mut index, mut best) = (0, f64::NEG_INFINITY);
                for (i, x) in samples.iter().enumerate() {
                    let score = good.log_pdf_ordered(x) - poor.log_pdf_ordered(x);
                    if score > best {
                        (index, best) = (i, score);
                    }
                }
                assert_eq!(best_candidate_index(&good, &poor, &samples), index);
                let ties = [vec![4.9; names.len()], vec![0.0; names.len()]];
                assert_eq!(best_candidate_index(&poor, &poor, &ties), 0);
            }
        }
    }
}
