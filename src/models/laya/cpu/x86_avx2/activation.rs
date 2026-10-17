//! SIMD evaluation of the small/medium branches of libm 0.2.16 erff.
//! The polynomial order and separate FP32 adds/multiplies match libm; other
//! lanes use libm itself. No approximate reciprocal, FMA contraction or fast exp.
// Coefficients/branch formulas derived from libm's FreeBSD s_erff.c port:
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn polynomial(x: __m256, coefficients: &[f32]) -> __m256 {
    let mut y = _mm256_set1_ps(*coefficients.last().unwrap());
    for &v in coefficients[..coefficients.len() - 1].iter().rev() {
        y = _mm256_add_ps(_mm256_set1_ps(v), _mm256_mul_ps(x, y));
    }
    y
}
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gated(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let sign_mask = _mm256_set1_ps(-0.0);
        let tiny = _mm256_set1_ps(f32::from_bits(0x31800000));
        let small = _mm256_set1_ps(0.84375);
        let medium = _mm256_set1_ps(1.25);
        let scale = _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2);
        let end = gate.len() / 8 * 8;
        for i in (0..end).step_by(8) {
            let g = _mm256_loadu_ps(gate.as_ptr().add(i));
            let x = _mm256_mul_ps(g, scale);
            let ax = _mm256_andnot_ps(sign_mask, x);
            let use_small = _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_GE_OQ>(ax, tiny),
                _mm256_cmp_ps::<_CMP_LT_OQ>(ax, small),
            );
            let use_medium = _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_GE_OQ>(ax, small),
                _mm256_cmp_ps::<_CMP_LT_OQ>(ax, medium),
            );
            let used = _mm256_movemask_ps(_mm256_or_ps(use_small, use_medium)) as u32;
            if used == 0 {
                for j in i..i + 8 {
                    out[j] = super::super::gelu(gate[j]) * up[j];
                }
                continue;
            }
            let z = _mm256_mul_ps(x, x);
            let r = polynomial(
                z,
                &[
                    1.2837916613e-01,
                    -3.2504209876e-01,
                    -2.8481749818e-02,
                    -5.7702702470e-03,
                    -2.3763017452e-05,
                ],
            );
            let s = polynomial(
                z,
                &[
                    1.0,
                    3.9791721106e-01,
                    6.5022252500e-02,
                    5.0813062117e-03,
                    1.3249473704e-04,
                    -3.9602282413e-06,
                ],
            );
            let mut erf = _mm256_add_ps(x, _mm256_mul_ps(x, _mm256_div_ps(r, s)));
            if _mm256_movemask_ps(use_medium) != 0 {
                let z = _mm256_sub_ps(ax, one);
                let p = polynomial(
                    z,
                    &[
                        -2.3621185683e-03,
                        4.1485610604e-01,
                        -3.7220788002e-01,
                        3.1834661961e-01,
                        -1.1089469492e-01,
                        3.5478305072e-02,
                        -2.1663755178e-03,
                    ],
                );
                let q = polynomial(
                    z,
                    &[
                        1.0,
                        1.0642088205e-01,
                        5.4039794207e-01,
                        7.1828655899e-02,
                        1.2617121637e-01,
                        1.3637083583e-02,
                        1.1984500103e-02,
                    ],
                );
                let erfc = _mm256_sub_ps(
                    _mm256_set1_ps(1.0_f32 - 8.4506291151e-01),
                    _mm256_div_ps(p, q),
                );
                let value = _mm256_xor_ps(_mm256_sub_ps(one, erfc), _mm256_and_ps(x, sign_mask));
                erf = _mm256_blendv_ps(erf, value, use_medium);
            }
            let y = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_mul_ps(half, g), _mm256_add_ps(one, erf)),
                _mm256_loadu_ps(up.as_ptr().add(i)),
            );
            _mm256_storeu_ps(out.as_mut_ptr().add(i), y);
            let mut fallback = (!used) & 255;
            while fallback != 0 {
                let lane = fallback.trailing_zeros() as usize;
                out[i + lane] = super::super::gelu(gate[i + lane]) * up[i + lane];
                fallback &= fallback - 1;
            }
        }
        for i in end..gate.len() {
            out[i] = super::super::gelu(gate[i]) * up[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_scalar_branches_and_float_edges() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut input: Vec<_> = (0..100_003)
            .map(|i| (i as f32 - 50_000.0) / 5000.)
            .collect();
        let mut seed = 1_u32;
        for _ in 0..100_003 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            input.push(f32::from_bits(seed));
        }
        input.extend([
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
        ]);
        for edge in [f32::from_bits(0x31800000), 0.84375, 1.25, 6.0] {
            for bits in edge.to_bits() - 8..=edge.to_bits() + 8 {
                input.push(f32::from_bits(bits) * std::f32::consts::SQRT_2);
                input.push(-f32::from_bits(bits) * std::f32::consts::SQRT_2);
            }
        }
        let up: Vec<_> = (0..input.len())
            .map(|i| (i % 113) as f32 * 0.07 - 3.)
            .collect();
        let mut actual = vec![0.0; input.len()];
        unsafe { gated(&input, &up, &mut actual) };
        for (i, &a) in actual.iter().enumerate() {
            let e = super::super::super::gelu(input[i]) * up[i];
            assert!(
                a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
                "{i}: x={} a={a} e={e}",
                input[i]
            );
        }
    }
}
