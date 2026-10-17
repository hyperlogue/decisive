//! SIMD evaluation of the small/medium branches of libm 0.2.16 erff.
//! Arithmetic order matches the scalar implementation; other lanes fall back
//! to libm. Keep this model-local so numerical choices remain independent.
// Coefficients/branch formulas derived from libm's FreeBSD s_erff.c port:
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
use core::arch::wasm32::*;

#[inline]
fn polynomial(x: v128, coefficients: &[f32]) -> v128 {
    let mut y = f32x4_splat(*coefficients.last().unwrap());
    for &v in coefficients[..coefficients.len() - 1].iter().rev() {
        y = f32x4_add(f32x4_splat(v), f32x4_mul(x, y));
    }
    y
}
#[inline]
fn scalar(x: f32) -> f32 {
    0.5 * x * (1. + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}
#[inline]
fn four(gate: &[f32]) -> v128 {
    unsafe {
        let g = v128_load(gate.as_ptr().cast());
        let x = f32x4_mul(g, f32x4_splat(std::f32::consts::FRAC_1_SQRT_2));
        let ax = f32x4_abs(x);
        let small = v128_and(
            f32x4_ge(ax, f32x4_splat(f32::from_bits(0x31800000))),
            f32x4_lt(ax, f32x4_splat(0.84375)),
        );
        let medium = v128_and(
            f32x4_ge(ax, f32x4_splat(0.84375)),
            f32x4_lt(ax, f32x4_splat(1.25)),
        );
        let used = i32x4_bitmask(v128_or(small, medium));
        if used == 0 {
            return f32x4(
                scalar(gate[0]),
                scalar(gate[1]),
                scalar(gate[2]),
                scalar(gate[3]),
            );
        }
        let z = f32x4_mul(x, x);
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
                1.,
                3.9791721106e-01,
                6.5022252500e-02,
                5.0813062117e-03,
                1.3249473704e-04,
                -3.9602282413e-06,
            ],
        );
        let mut erf = f32x4_add(x, f32x4_mul(x, f32x4_div(r, s)));
        if i32x4_bitmask(medium) != 0 {
            let z = f32x4_sub(ax, f32x4_splat(1.));
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
                    1.,
                    1.0642088205e-01,
                    5.4039794207e-01,
                    7.1828655899e-02,
                    1.2617121637e-01,
                    1.3637083583e-02,
                    1.1984500103e-02,
                ],
            );
            let erfc = f32x4_sub(f32x4_splat(1.0_f32 - 8.4506291151e-01), f32x4_div(p, q));
            let value = v128_xor(
                f32x4_sub(f32x4_splat(1.), erfc),
                v128_and(x, f32x4_splat(-0.)),
            );
            erf = v128_bitselect(value, erf, medium);
        }
        let y = f32x4_mul(
            f32x4_mul(f32x4_splat(0.5), g),
            f32x4_add(f32x4_splat(1.), erf),
        );
        if used == 15 {
            return y;
        }
        let mut result = [0.; 4];
        v128_store(result.as_mut_ptr().cast(), y);
        let mut fallback = (!used) & 15;
        while fallback != 0 {
            let lane = fallback.trailing_zeros() as usize;
            result[lane] = scalar(gate[lane]);
            fallback &= fallback - 1;
        }
        v128_load(result.as_ptr().cast())
    }
}

#[cfg(feature = "kernel-checks")]
pub(super) fn check() -> anyhow::Result<usize> {
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
    for row in input.chunks_exact(4) {
        let mut actual = [0_f32; 4];
        unsafe {
            v128_store(actual.as_mut_ptr().cast(), four(row));
        }
        for (&a, &x) in actual.iter().zip(row) {
            let e = scalar(x);
            anyhow::ensure!(
                a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
                "GELU mismatch: {x} -> {a} vs {e}"
            );
            count += 1;
        }
    }
    Ok(count)
}

pub(super) fn gated(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());
    let end = gate.len() / 4 * 4;
    for i in (0..end).step_by(4) {
        unsafe {
            v128_store(
                out.as_mut_ptr().add(i).cast(),
                f32x4_mul(four(&gate[i..i + 4]), v128_load(up.as_ptr().add(i).cast())),
            );
        }
    }
    for i in end..gate.len() {
        out[i] = scalar(gate[i]) * up[i];
    }
}
