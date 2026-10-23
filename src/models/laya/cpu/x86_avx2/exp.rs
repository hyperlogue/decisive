pub(super) use crate::avx2_exp::negative4;
use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn negative8(input: __m256) -> __m256 {
    let x = _mm256_max_ps(input, _mm256_set1_ps(-104.));
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(std::f32::consts::LOG2_E),
    ));
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145751953125), x);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428606765330187e-6), r);
    let mut p = _mm256_set1_ps(1. / 5040.);
    for coefficient in [1. / 720., 1. / 120., 1. / 24., 1. / 6., 0.5, 1., 1.] {
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(coefficient));
    }
    let exponent = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127 + 64));
    let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(exponent));
    let result = _mm256_mul_ps(
        _mm256_mul_ps(p, scale),
        _mm256_set1_ps(f32::from_bits((127 - 64) << 23)),
    );
    _mm256_blendv_ps(
        result,
        _mm256_add_ps(input, input),
        _mm256_cmp_ps::<_CMP_UNORD_Q>(input, input),
    )
}

#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)]
pub(super) unsafe fn negative4_polynomial(input: __m128) -> __m128 {
    let x = _mm256_cvtps_pd(_mm_max_ps(input, _mm_set1_ps(-104.)));
    let n = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_pd(
        x,
        _mm256_set1_pd(std::f64::consts::LOG2_E),
    ));
    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(0.6931471803691238), x);
    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(1.9082149292705877e-10), r);
    let r2 = _mm256_mul_pd(r, r);
    let r4 = _mm256_mul_pd(r2, r2);
    let r8 = _mm256_mul_pd(r4, r4);
    let g0 = _mm256_add_pd(r, _mm256_set1_pd(1.));
    let g1 = _mm256_fmadd_pd(r, _mm256_set1_pd(1. / 6.), _mm256_set1_pd(0.5));
    let g2 = _mm256_fmadd_pd(r, _mm256_set1_pd(1. / 120.), _mm256_set1_pd(1. / 24.));
    let g3 = _mm256_fmadd_pd(r, _mm256_set1_pd(1. / 5040.), _mm256_set1_pd(1. / 720.));
    let g4 = _mm256_fmadd_pd(r, _mm256_set1_pd(1. / 362880.), _mm256_set1_pd(1. / 40320.));
    let g5 = _mm256_fmadd_pd(
        r,
        _mm256_set1_pd(1. / 39916800.),
        _mm256_set1_pd(1. / 3628800.),
    );
    let h0 = _mm256_fmadd_pd(g1, r2, g0);
    let h1 = _mm256_fmadd_pd(g3, r2, g2);
    let h2 = _mm256_fmadd_pd(g5, r2, g4);
    let j0 = _mm256_fmadd_pd(h1, r4, h0);
    let j1 = _mm256_fmadd_pd(_mm256_set1_pd(1. / 479001600.), r4, h2);
    let p = _mm256_fmadd_pd(j1, r8, j0);
    let exponent = _mm256_add_epi64(
        _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(n)),
        _mm256_set1_epi64x(1023),
    );
    let scale = _mm256_castsi256_pd(_mm256_slli_epi64::<52>(exponent));
    let result = _mm256_cvtpd_ps(_mm256_mul_pd(p, scale));
    _mm_blendv_ps(
        result,
        _mm_add_ps(input, input),
        _mm_cmpunord_ps(input, input),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn double_polynomial_handles_special_values_and_float_range() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut state = 4531u32;
        for block in 0..250_000 {
            let values: [f32; 4] = std::array::from_fn(|lane| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if block == 0 {
                    [0., -0., f32::NAN, f32::NEG_INFINITY][lane]
                } else if block == 1 {
                    [-f32::MAX, -104., -103., -f32::MIN_POSITIVE][lane]
                } else if block % 2 == 0 {
                    -(state as f64 / u32::MAX as f64 * 104.) as f32
                } else {
                    f32::from_bits(0x8000_0000 | (state % 0x42d0_0001))
                }
            });
            let mut output = [0.; 4];
            unsafe {
                _mm_storeu_ps(
                    output.as_mut_ptr(),
                    negative4_polynomial(_mm_loadu_ps(values.as_ptr())),
                );
            }
            for (x, y) in values.into_iter().zip(output) {
                if x.is_nan() {
                    assert!(y.is_nan());
                } else {
                    assert_eq!(y.to_bits(), ((x as f64).exp() as f32).to_bits(), "x={x:?}");
                }
            }
        }
    }
    #[test]
    fn polynomial_exp_bounds_rounding_and_handles_special_values() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut state = 4531u32;
        for block in 0..125_000 {
            let values: [f32; 8] = std::array::from_fn(|lane| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if block == 0 {
                    [
                        -0.,
                        0.,
                        -f32::MIN_POSITIVE,
                        -104.,
                        -105.,
                        f32::NEG_INFINITY,
                        f32::NAN,
                        -f32::MAX,
                    ][lane]
                } else if block % 2 == 0 {
                    -(state as f64 / u32::MAX as f64 * 104.) as f32
                } else {
                    f32::from_bits(0x8000_0000 | (state % 0x42d0_0001))
                }
            });
            let mut actual = [0.; 8];
            unsafe {
                _mm256_storeu_ps(
                    actual.as_mut_ptr(),
                    negative8(_mm256_loadu_ps(values.as_ptr())),
                );
            }
            for (x, y) in values.into_iter().zip(actual) {
                if x.is_nan() {
                    assert!(y.is_nan());
                    continue;
                }
                let expected = (x as f64).exp() as f32;
                assert!(
                    expected.to_bits().abs_diff(y.to_bits()) <= 1,
                    "x={x:?} y={y:?} expected={expected:?}"
                );
            }
        }
    }
    #[test]
    fn negative_exp_stays_within_one_ulp_of_double_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut state = 4531u32;
        let mut checked = 0;
        for block in 0..250_000 {
            let mut values = [0.; 4];
            for (lane, v) in values.iter_mut().enumerate() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                *v = if block % 2 == 0 {
                    -(state as f64 / u32::MAX as f64 * 104.) as f32
                } else {
                    f32::from_bits(0x8000_0000 | (state % 0x42d0_0001))
                };
                if block < 32 {
                    *v = [-0., -f32::MIN_POSITIVE, -104., f32::NEG_INFINITY][lane];
                }
            }
            let mut got = [0.; 4];
            unsafe {
                _mm_storeu_ps(got.as_mut_ptr(), negative4(_mm_loadu_ps(values.as_ptr())));
            }
            for (&x, &actual) in values.iter().zip(&got) {
                let expected = (x as f64).exp() as f32;
                let ulp = expected.to_bits().abs_diff(actual.to_bits());
                assert!(
                    ulp <= 1,
                    "x={x:?} got={actual:?} expected={expected:?} ulp={ulp}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 1_000_000);
    }
}
