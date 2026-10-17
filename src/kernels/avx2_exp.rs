//! AVX2/FMA exponential primitives; model dispatch and vector loops stay local.
//! Adapted to Rust AVX2/FMA from Arm optimized-routines expf.c / exp2f_data.c,
//! revision 59afb6fed4275e2fc249df49fce43ee5e75af178. The source algorithm has
//! a reported 0.502 ULP nearest-rounding error; our adaptation is tested separately.
//! FP64 range reduction preserves subnormal results; overflow/NaN lanes are handled.
//
// Copyright (c) 2017-2025, Arm Limited.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
use std::arch::x86_64::*;

const TABLE: [u64; 32] = [
    0x3ff0000000000000,
    0x3fefd9b0d3158574,
    0x3fefb5586cf9890f,
    0x3fef9301d0125b51,
    0x3fef72b83c7d517b,
    0x3fef54873168b9aa,
    0x3fef387a6e756238,
    0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb,
    0x3feedea64c123422,
    0x3feece086061892d,
    0x3feebfdad5362a27,
    0x3feeb42b569d4f82,
    0x3feeab07dd485429,
    0x3feea47eb03a5585,
    0x3feea09e667f3bcd,
    0x3fee9f75e8ec5f74,
    0x3feea11473eb0187,
    0x3feea589994cce13,
    0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5,
    0x3feec49182a3f090,
    0x3feed503b23e255d,
    0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47,
    0x3fef199bdd85529c,
    0x3fef3720dcef9069,
    0x3fef5818dcfba487,
    0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da,
    0x3fefd0765b6e4540,
];

// Caller verifies AVX2/FMA and nonpositive inputs (NaN/-inf are handled).
#[cfg(any(feature = "laya", feature = "decima"))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn negative4(input: __m128) -> __m128 {
    unsafe {
        let x = _mm256_cvtps_pd(_mm_max_ps(input, _mm_set1_ps(-104.)));
        let z = _mm256_mul_pd(x, _mm256_set1_pd(f64::from_bits(0x40471547652b82fe)));
        let shift = _mm256_set1_pd(f64::from_bits(0x4338000000000000));
        let rounded = _mm256_add_pd(z, shift);
        let ki = _mm256_castpd_si256(rounded);
        let r = _mm256_sub_pd(z, _mm256_sub_pd(rounded, shift));
        let index = _mm256_and_si256(ki, _mm256_set1_epi64x(31));
        let table = _mm256_i64gather_epi64::<8>(TABLE.as_ptr().cast(), index);
        let scale = _mm256_castsi256_pd(_mm256_add_epi64(table, _mm256_slli_epi64::<47>(ki)));
        let polynomial = _mm256_fmadd_pd(
            _mm256_set1_pd(f64::from_bits(0x3ebc6af84b912394)),
            r,
            _mm256_set1_pd(f64::from_bits(0x3f2ebfce50fac4f3)),
        );
        let linear = _mm256_fmadd_pd(
            _mm256_set1_pd(f64::from_bits(0x3f962e42ff0c52d6)),
            r,
            _mm256_set1_pd(1.),
        );
        let value = _mm256_fmadd_pd(polynomial, _mm256_mul_pd(r, r), linear);
        let result = _mm256_cvtpd_ps(_mm256_mul_pd(value, scale));
        // MAXPS selects its second operand for NaN; restore NaNs before reduction.
        _mm_blendv_ps(
            result,
            _mm_add_ps(input, input),
            _mm_cmpunord_ps(input, input),
        )
    }
}

// Caller verifies AVX2/FMA; overflow, underflow and NaN lanes are handled.
#[cfg(any(test, feature = "kev", feature = "jev-style", feature = "pplx-flash"))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp4(input: __m128) -> __m128 {
    unsafe {
        let x = _mm256_cvtps_pd(_mm_min_ps(
            _mm_max_ps(input, _mm_set1_ps(-104.)),
            _mm_set1_ps(89.),
        ));
        let z = _mm256_mul_pd(x, _mm256_set1_pd(f64::from_bits(0x40471547652b82fe)));
        let shift = _mm256_set1_pd(f64::from_bits(0x4338000000000000));
        let rounded = _mm256_add_pd(z, shift);
        let ki = _mm256_castpd_si256(rounded);
        let r = _mm256_sub_pd(z, _mm256_sub_pd(rounded, shift));
        let index = _mm256_and_si256(ki, _mm256_set1_epi64x(31));
        let table = _mm256_i64gather_epi64::<8>(TABLE.as_ptr().cast(), index);
        let scale = _mm256_castsi256_pd(_mm256_add_epi64(table, _mm256_slli_epi64::<47>(ki)));
        let polynomial = _mm256_fmadd_pd(
            _mm256_set1_pd(f64::from_bits(0x3ebc6af84b912394)),
            r,
            _mm256_set1_pd(f64::from_bits(0x3f2ebfce50fac4f3)),
        );
        let linear = _mm256_fmadd_pd(
            _mm256_set1_pd(f64::from_bits(0x3f962e42ff0c52d6)),
            r,
            _mm256_set1_pd(1.),
        );
        let value = _mm256_fmadd_pd(polynomial, _mm256_mul_pd(r, r), linear);
        let result = _mm256_cvtpd_ps(_mm256_mul_pd(value, scale));
        // MAXPS selects its second operand for NaN; restore NaNs before reduction.
        _mm_blendv_ps(
            result,
            _mm_add_ps(input, input),
            _mm_cmpunord_ps(input, input),
        )
    }
}

#[cfg(any(feature = "kev", feature = "jev-style", feature = "pplx-flash"))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn sigmoid8(x: __m256) -> __m256 {
    unsafe {
        let lo = _mm256_castps256_ps128(x);
        let hi = _mm256_extractf128_ps::<1>(x);
        let one = _mm_set1_ps(1.);
        let sign = _mm_set1_ps(-0.);
        let lo = _mm_div_ps(one, _mm_add_ps(one, exp4(_mm_xor_ps(lo, sign))));
        let hi = _mm_div_ps(one, _mm_add_ps(one, exp4(_mm_xor_ps(hi, sign))));
        _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo), hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_float_domain_matches_scalar_exp_within_one_ulp() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut seed = 571u32;
        for block in 0..250000 {
            let values: [f32; 4] = std::array::from_fn(|lane| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                if block == 0 {
                    [f32::NEG_INFINITY, -0., f32::INFINITY, f32::NAN][lane]
                } else if block % 2 == 0 {
                    (seed as f64 / u32::MAX as f64 * 193. - 104.) as f32
                } else {
                    f32::from_bits(seed % 0x7f800000) * if lane % 2 == 0 { -1. } else { 1. }
                }
            });
            let mut output = [0.; 4];
            unsafe {
                _mm_storeu_ps(output.as_mut_ptr(), exp4(_mm_loadu_ps(values.as_ptr())));
            }
            for (x, y) in values.into_iter().zip(output) {
                if x.is_nan() {
                    assert!(y.is_nan());
                } else {
                    assert!(y.to_bits().abs_diff(x.exp().to_bits()) <= 1, "exp({x})={y}");
                }
            }
        }
    }
}
