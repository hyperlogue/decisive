use core::arch::wasm32::*;
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

#[inline]
fn negative2(input: v128) -> v128 {
    let x = f64x2_promote_low_f32x4(f32x4_max(input, f32x4_splat(-104.)));
    let z = f64x2_mul(x, f64x2_splat(f64::from_bits(0x40471547652b82fe)));
    let shift = f64x2_splat(f64::from_bits(0x4338000000000000));
    let rounded = f64x2_add(z, shift);
    let r = f64x2_sub(z, f64x2_sub(rounded, shift));
    let first = (i64x2_extract_lane::<0>(rounded) as u64 & 31) as usize;
    let second = (i64x2_extract_lane::<1>(rounded) as u64 & 31) as usize;
    let scale = i64x2_add(
        i64x2(TABLE[first] as i64, TABLE[second] as i64),
        i64x2_shl(rounded, 47),
    );
    let polynomial = f64x2_add(
        f64x2_mul(f64x2_splat(f64::from_bits(0x3ebc6af84b912394)), r),
        f64x2_splat(f64::from_bits(0x3f2ebfce50fac4f3)),
    );
    let linear = f64x2_add(
        f64x2_mul(f64x2_splat(f64::from_bits(0x3f962e42ff0c52d6)), r),
        f64x2_splat(1.),
    );
    let value = f64x2_add(f64x2_mul(polynomial, f64x2_mul(r, r)), linear);
    f32x4_demote_f64x2_zero(f64x2_mul(value, scale))
}
#[inline]
fn precise_negative4(input: v128) -> v128 {
    i32x4_shuffle::<0, 1, 4, 5>(
        negative2(input),
        negative2(i32x4_shuffle::<2, 3, 2, 3>(input, input)),
    )
}
#[inline]
fn negative4(input: v128) -> v128 {
    if i32x4_bitmask(f32x4_lt(input, f32x4_splat(-80.))) != 0 {
        return precise_negative4(input);
    }
    let magic = f32x4_splat(12582912.);
    let j = f32x4_sub(
        f32x4_add(
            f32x4_mul(input, f32x4_splat(std::f32::consts::LOG2_E)),
            magic,
        ),
        magic,
    );
    let r = f32x4_add(f32x4_mul(j, f32x4_splat(-6.93145752e-1)), input);
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
pub(super) fn softmax(x: &mut [f32]) {
    let end = x.len() / 4 * 4;
    unsafe {
        let mut maxima = f32x4_splat(f32::NEG_INFINITY);
        for i in (0..end).step_by(4) {
            maxima = f32x4_max(maxima, v128_load(x.as_ptr().add(i).cast()));
        }
        let mut max = f32x4_extract_lane::<0>(maxima)
            .max(f32x4_extract_lane::<1>(maxima))
            .max(f32x4_extract_lane::<2>(maxima))
            .max(f32x4_extract_lane::<3>(maxima));
        for &v in &x[end..] {
            max = max.max(v);
        }
        if !max.is_finite() {
            super::cpu::softmax(x);
            return;
        }
        let m = f32x4_splat(max);
        let mut sums = f32x4_splat(0.);
        for i in (0..end).step_by(4) {
            let v = negative4(f32x4_sub(v128_load(x.as_ptr().add(i).cast()), m));
            sums = f32x4_add(sums, v);
            v128_store(x.as_mut_ptr().add(i).cast(), v);
        }
        let mut sum = f32x4_extract_lane::<0>(sums)
            + f32x4_extract_lane::<1>(sums)
            + f32x4_extract_lane::<2>(sums)
            + f32x4_extract_lane::<3>(sums);
        for v in &mut x[end..] {
            *v = (*v - max).exp();
            sum += *v;
        }
        let divisor = f32x4_splat(sum);
        for i in (0..end).step_by(4) {
            v128_store(
                x.as_mut_ptr().add(i).cast(),
                f32x4_div(v128_load(x.as_ptr().add(i).cast()), divisor),
            );
        }
        for v in &mut x[end..] {
            *v /= sum;
        }
    }
}

#[cfg(feature = "kernel-checks")]
pub(super) fn check() -> anyhow::Result<serde_json::Value> {
    use anyhow::ensure;
    let mut seed = 4531u32;
    let mut max_ulp = 0;
    let mut checked = 0;
    for range in [80., 110.] {
        for _ in 0..250_000 {
            let x: [f32; 4] = std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                -(seed as f64 / u32::MAX as f64 * range) as f32
            });
            let mut y = [0_f32; 4];
            unsafe {
                v128_store(
                    y.as_mut_ptr().cast(),
                    negative4(v128_load(x.as_ptr().cast())),
                );
            }
            for (x, y) in x.into_iter().zip(y) {
                let ulp = y.to_bits().abs_diff(((x as f64).exp() as f32).to_bits());
                ensure!(ulp <= 1, "Wasm exp error: {x} -> {y}, {ulp} ULP");
                max_ulp = max_ulp.max(ulp);
                checked += 1;
            }
        }
    }
    let mut max_error = 0f32;
    for n in [1, 3, 4, 7, 8, 31, 64, 127, 512, 2048] {
        for range in [0., 0.01, 1., 10., 100.] {
            let mut actual: Vec<f32> = (0..n)
                .map(|i| ((i * 331 % 127) as f32 / 63. - 1.) * range)
                .collect();
            let mut expected = actual.clone();
            super::cpu::softmax(&mut expected);
            softmax(&mut actual);
            for (a, b) in actual.iter().zip(&expected) {
                let error = (a - b).abs();
                ensure!(error < 3e-7, "Wasm softmax drift {n}: {a} != {b}");
                max_error = max_error.max(error);
            }
        }
    }
    for mut row in [
        vec![f32::NEG_INFINITY; 8],
        vec![0., f32::NAN, 1., 2.],
        vec![0., f32::INFINITY, 1., 2.],
    ] {
        softmax(&mut row);
        ensure!(
            row.iter().all(|v| v.is_nan()),
            "non-finite softmax failed to propagate"
        );
    }
    Ok(
        serde_json::json!({"exp_values":checked,"maximum_exp_ulp":max_ulp,"maximum_softmax_error":max_error}),
    )
}
