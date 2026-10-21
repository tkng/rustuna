use rand::rngs::StdRng;
use rand_distr::{Distribution, WeightedAliasIndex};
use rustuna_core::internal::parzen_estimator::ParzenEstimator;

/// Generates candidates in parameter-name order, reusing caller-owned rows.
/// `weights` are the normalized mixture weights returned by [`ParzenEstimator::update`];
/// the selection table exists only for this call and the model retains no sampling state.
pub(super) fn generate_candidates(
    model: &ParzenEstimator,
    weights: &[f64],
    rng: &mut StdRng,
    size: usize,
    samples: &mut Vec<Vec<f64>>,
) {
    assert_eq!(model.component_count(), weights.len());
    samples.resize_with(size, Vec::new);
    let selection = WeightedAliasIndex::new(weights.to_vec())
        .expect("mixture weights are validated before candidate generation");
    for sample in samples {
        let component = selection.sample(rng);
        model.draw_component(rng, component, sample);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, SeedableRng};
    use rustuna_core::distribution::Distribution as SearchDistribution;
    use rustuna_core::internal::parzen_estimator::{normalized_weights, ObservationColumn};

    #[test]
    fn discrete_candidates_and_rng_consumption_preserve_seeded_sequences() {
        let distributions = [
            SearchDistribution::new_float(-5.0, 5.0, None, false),
            SearchDistribution::new_float(1.0, 11.0, None, true),
            SearchDistribution::new_float(-5.0, 5.0, Some(1.0), false),
            SearchDistribution::new_int(1, 10, 1, true),
            SearchDistribution::new_categorical(257),
        ];
        let names = ["p0", "p1", "p2", "p3", "p4"];
        let columns: Vec<Vec<f64>> = (0..5)
            .map(|parameter| {
                (0..120)
                    .map(|i| match parameter {
                        0 => -4.5 + (i * 17 % 90) as f64 / 10.0,
                        1 | 3 => 1.0 + (i % 10) as f64,
                        2 => (i % 11) as f64 - 5.0,
                        _ => (i % 257) as f64,
                    })
                    .collect()
            })
            .collect();
        let observation_weights: Vec<f64> = (0..120).map(|i| (i + 1) as f64).collect();
        // Recorded before candidate generation was extracted. Check integer-valued
        // candidates by bits and RNG consumption for every shape. Floating-point
        // sample bits can depend on the target's math library.
        let rng_states = [
            (7, 0xf34819c216632098_u64, 0xead243dbcb073a91_u64),
            (42, 0x2c3f52181acb1073_u64, 0xbc7fe4eab9d1ba44_u64),
        ];
        for parameters in [&[0][..], &[1], &[2], &[3], &[4], &[0, 1, 2, 3, 4]] {
            for prior in [0.0, 1.0] {
                for (seed, univariate_rng, multivariate_rng) in rng_states {
                    let mut model = ParzenEstimator::default();
                    let weights = model.update(
                        &(0..120).collect::<Vec<_>>(),
                        (0..120).collect(),
                        &observation_weights,
                        prior,
                        parameters.iter().map(|&p| ObservationColumn {
                            name: names[p],
                            distribution: &distributions[p],
                            values: &columns[p],
                            categories: None,
                        }),
                    );
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut samples = Vec::new();
                    generate_candidates(&model, &weights, &mut rng, 48, &mut samples);
                    let hash = samples
                        .iter()
                        .flatten()
                        .fold(0xcbf29ce484222325_u64, |hash, value| {
                            (hash ^ value.to_bits()).wrapping_mul(0x100000001b3)
                        });
                    let expected_hash = match (parameters, seed) {
                        ([2], 7) => Some(0xf38d945a1cd8d6e5_u64),
                        ([2], 42) if prior == 0.0 => Some(0xfaa9945a1cd8d6e5),
                        ([2], 42) => Some(0xbed5945a1cd8d6e5),
                        ([3], 7) => Some(0x2653945a1cd8d6e5),
                        ([3], 42) => Some(0x8cc7945a1cd8d6e5),
                        ([4], 7) => Some(0xa49f545a1cd8d6e5),
                        ([4], 42) => Some(0xf06c345a1cd8d6e5),
                        ([0] | [1] | [0, 1, 2, 3, 4], _) => None,
                        _ => unreachable!("missing candidate hash reference"),
                    };
                    if let Some(expected_hash) = expected_hash {
                        assert_eq!(
                            hash, expected_hash,
                            "parameters={parameters:?}, prior={prior}, seed={seed}"
                        );
                    }
                    let expected_rng = if parameters.len() == 1 {
                        univariate_rng
                    } else {
                        multivariate_rng
                    };
                    assert_eq!(
                        rng.next_u64(),
                        expected_rng,
                        "parameters={parameters:?}, prior={prior}, seed={seed}"
                    );
                }
            }
        }
    }

    fn correlated_categorical_model() -> ParzenEstimator {
        let distribution = SearchDistribution::new_categorical(2);
        ParzenEstimator::from_ordered_observations(
            ["x", "y"].map(|name| (name, &distribution, [0.0, 1.0])),
            vec![0, 1],
            &[1.0, 2.0],
            0.5,
            None,
        )
    }

    fn weights() -> Vec<f64> {
        normalized_weights(&[1.0, 2.0], 0.5)
    }

    #[test]
    fn candidate_rows_are_overwritten_and_reused_when_the_batch_shrinks() {
        let model = correlated_categorical_model();
        let mut samples = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);
        generate_candidates(&model, &weights(), &mut rng, 6, &mut samples);
        let first_two = samples[..2].to_vec();
        let pointers: Vec<_> = samples[..2].iter().map(|row| row.as_ptr()).collect();
        let mut rng = StdRng::seed_from_u64(42);
        generate_candidates(&model, &weights(), &mut rng, 2, &mut samples);
        assert_eq!(samples, first_two);
        assert_eq!(
            samples.iter().map(|row| row.as_ptr()).collect::<Vec<_>>(),
            pointers
        );
        let mut untouched_rng = rng.clone();
        generate_candidates(&model, &weights(), &mut rng, 0, &mut samples);
        assert!(samples.is_empty());
        assert_eq!(rng.next_u64(), untouched_rng.next_u64());
        generate_candidates(&model, &weights(), &mut rng, 9, &mut samples);
        assert_eq!(samples.len(), 9);
        assert!(samples
            .iter()
            .all(|row| row.len() == 2 && row.iter().all(|x| *x == 0.0 || *x == 1.0)));
    }

    #[test]
    fn joint_candidates_preserve_component_weights_and_correlations() {
        let model = correlated_categorical_model();
        let mut samples = Vec::new();
        generate_candidates(
            &model,
            &weights(),
            &mut StdRng::seed_from_u64(42),
            20_000,
            &mut samples,
        );
        let equal = samples
            .iter()
            .filter(|sample| sample[0] == sample[1])
            .count() as f64
            / samples.len() as f64;
        // The observed kernels have P(x=y)=0.8^2+0.2^2; the uniform prior has 0.5.
        let expected_equal = (3.0 * (0.8 * 0.8 + 0.2 * 0.2) + 0.5 * 0.5) / 3.5;
        assert!((equal - expected_equal).abs() < 0.015);
    }
}
