//! Four-lane negative expf for attention, with FP64 range reduction/polynomial.
//! Adapted to Rust AVX2/FMA from Arm optimized-routines expf.c / exp2f_data.c,
//! revision 59afb6fed4275e2fc249df49fce43ee5e75af178. The source algorithm has
//! a reported 0.502 ULP nearest-rounding error; our adaptation is tested separately.
//! This is not a general exp API: inputs are nonpositive softmax differences.
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
use crate::avx2_exp::negative4;
use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        super::cpu::softmax(x);
        return;
    }
    unsafe {
        let end = x.len() / 4 * 4;
        let m = _mm_set1_ps(max);
        for i in (0..end).step_by(4) {
            let a = _mm_sub_ps(_mm_loadu_ps(x.as_ptr().add(i)), m);
            _mm_storeu_ps(x.as_mut_ptr().add(i), negative4(a));
        }
        for v in &mut x[end..] {
            *v = (*v - max).exp();
        }
        let sum = x.iter().sum::<f32>();
        let d = _mm256_set1_ps(sum);
        let end = x.len() / 8 * 8;
        for i in (0..end).step_by(8) {
            _mm256_storeu_ps(
                x.as_mut_ptr().add(i),
                _mm256_div_ps(_mm256_loadu_ps(x.as_ptr().add(i)), d),
            );
        }
        for v in &mut x[end..] {
            *v /= sum;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_softmax_bounds_and_tails() {
        if !super::super::cpu::packed_supported() {
            return;
        }
        for n in [1, 3, 4, 7, 8, 31, 64, 127, 512] {
            for range in [0., 0.01, 1., 10., 100.] {
                let mut actual: Vec<f32> = (0..n)
                    .map(|i| ((i * 331 % 127) as f32 / 63. - 1.) * range)
                    .collect();
                let mut expected = actual.clone();
                super::super::cpu::softmax(&mut expected);
                unsafe {
                    softmax(&mut actual);
                }
                for (a, b) in actual.iter().zip(&expected) {
                    assert!((a - b).abs() < 3e-7, "{n} {range}: {a} {b}");
                }
            }
        }
        for mut row in [
            vec![f32::NEG_INFINITY; 8],
            vec![0., f32::NAN, 1., 2.],
            vec![0., f32::INFINITY, 1., 2.],
        ] {
            unsafe {
                softmax(&mut row);
            }
            assert!(row.iter().all(|v| v.is_nan()));
        }
        let mut seed = 4531_u32;
        for _ in 0..250_000 {
            let x: [f32; 4] = std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                -(seed as f64 / u32::MAX as f64 * 110.) as f32
            });
            let mut y = [0.; 4];
            unsafe {
                _mm_storeu_ps(y.as_mut_ptr(), negative4(_mm_loadu_ps(x.as_ptr())));
            }
            for (x, y) in x.into_iter().zip(y) {
                assert!(y.to_bits().abs_diff(((x as f64).exp() as f32).to_bits()) <= 1);
            }
        }
    }
}
