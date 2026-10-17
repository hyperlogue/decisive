//! Decima-local SIMD128 erf-based GELU. Adapted from RTen 0.27.0's
//! rten-vecmath/src/{erf,exp}.rs, revision
//! bc1b852cb0ef70a6e8616e884c026f3ce3577c4f, under its MIT option.
//! Modifications: direct Wasm intrinsics, unfused arithmetic, saturated erf
//! inputs and model-local bounds. This approximates erf, not tanh GELU.
// Copyright (c) 2026 RTen project contributors.
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
use core::arch::wasm32::*;

#[inline]
fn scalar(x: f32) -> f32 {
    0.5 * x * (1. + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

// Requires -x² in [-36, 0]; no overflow or subnormal path.
#[inline]
fn gaussian_exp(x: v128) -> v128 {
    let magic = f32x4_splat(12582912.);
    let j = f32x4_sub(
        f32x4_add(f32x4_mul(x, f32x4_splat(std::f32::consts::LOG2_E)), magic),
        magic,
    );
    let r = f32x4_add(f32x4_mul(j, f32x4_splat(-6.93145752e-1)), x);
    let r = f32x4_add(f32x4_mul(j, f32x4_splat(-1.42860677e-6)), r);
    let mut p = f32x4_splat(1.37805939e-3);
    for c in [
        8.37312452e-3,
        4.16695364e-2,
        1.66664720e-1,
        4.99999851e-1,
        1.,
        1.,
    ] {
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(c));
    }
    let scale = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(j), i32x4_splat(127)), 23);
    f32x4_mul(p, scale)
}

#[inline]
fn four(gate: &[f32]) -> v128 {
    unsafe {
        let g = v128_load(gate.as_ptr().cast());
        let x = f32x4_mul(g, f32x4_splat(std::f32::consts::FRAC_1_SQRT_2));
        // erf rounds to ±1 here; saturate before squaring to avoid overflow.
        let a = f32x4_min(f32x4_abs(x), f32x4_splat(6.));
        let t = f32x4_div(
            f32x4_splat(1.),
            f32x4_add(f32x4_splat(1.), f32x4_mul(a, f32x4_splat(0.3275911))),
        );
        let mut p = f32x4_splat(1.061405429);
        for c in [-1.453152027, 1.421413741, -0.284496736, 0.254829592] {
            p = f32x4_add(f32x4_mul(p, t), f32x4_splat(c));
        }
        let exp = gaussian_exp(f32x4_neg(f32x4_mul(a, a)));
        let erf = f32x4_sub(f32x4_splat(1.), f32x4_mul(f32x4_mul(p, t), exp));
        let erf = v128_xor(erf, v128_and(x, f32x4_splat(-0.)));
        f32x4_mul(
            f32x4_mul(f32x4_splat(0.5), g),
            f32x4_add(f32x4_splat(1.), erf),
        )
    }
}

#[cfg(feature = "kernel-checks")]
pub(super) fn check() -> anyhow::Result<serde_json::Value> {
    let mut input: Vec<_> = (0..200003).map(|i| (i as f32 - 100000.) / 10000.).collect();
    let mut seed = 1u32;
    for _ in 0..200003 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        input.push(f32::from_bits(seed));
    }
    input.extend([
        0.,
        -0.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
    ]);
    for edge in [f32::from_bits(0x31800000), 0.84375, 1.25, 6.] {
        for bits in edge.to_bits() - 8..=edge.to_bits() + 8 {
            input.push(f32::from_bits(bits) * std::f32::consts::SQRT_2);
            input.push(-f32::from_bits(bits) * std::f32::consts::SQRT_2);
        }
    }
    let mut count = 0;
    let mut maximum_error = 0_f32;
    for row in input.chunks_exact(4) {
        let mut actual = [0_f32; 4];
        unsafe {
            v128_store(actual.as_mut_ptr().cast(), four(row));
        }
        for (&a, &x) in actual.iter().zip(row) {
            let e = scalar(x);
            if a.is_finite() && e.is_finite() {
                maximum_error = maximum_error.max((a - e).abs());
            }
            anyhow::ensure!(
                a == e || (a.is_nan() && e.is_nan()) || (a - e).abs() <= 1e-6,
                "GELU mismatch: {x} -> {a} vs {e}"
            );
            count += 1;
        }
    }
    Ok(serde_json::json!({"values":count,"maximum_absolute_error":maximum_error,"bound":1e-6}))
}

pub(super) fn apply(x: &mut [f32]) {
    let end = x.len() / 4 * 4;
    for i in (0..end).step_by(4) {
        let y = four(&x[i..i + 4]);
        unsafe {
            v128_store(x.as_mut_ptr().add(i).cast(), y);
        }
    }
    for v in &mut x[end..] {
        *v = scalar(*v);
    }
}
