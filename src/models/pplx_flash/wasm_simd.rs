use std::arch::wasm32::*;

pub(super) fn pack(data: &[u8], k: usize, n: usize) -> Vec<u16> {
    let mut packed = vec![0; n.div_ceil(8) * 8 * k];
    for panel in 0..n.div_ceil(8) {
        for j in 0..k {
            for c in 0..8.min(n - panel * 8) {
                let at = ((panel * 8 + c) * k + j) * 2;
                packed[(panel * k + j) * 8 + c] = u16::from_le_bytes([data[at], data[at + 1]]);
            }
        }
    }
    packed
}
pub(super) fn project(x: &[f32], w: &[u16], y: &mut [f32], k: usize, n: usize) {
    let count = x.len() / k;
    for panel in 0..n.div_ceil(8) {
        let full = count / 4 * 4;
        for row in (0..full).step_by(4) {
            unsafe {
                rows::<4>(x, w, y, k, n, panel, row);
            }
        }
        unsafe {
            match count - full {
                1 => rows::<1>(x, w, y, k, n, panel, full),
                2 => rows::<2>(x, w, y, k, n, panel, full),
                3 => rows::<3>(x, w, y, k, n, panel, full),
                _ => {}
            }
        }
    }
}
#[cfg_attr(not(feature = "pplx-wasm-fma"), inline(always))]
#[cfg_attr(feature = "pplx-wasm-fma", inline)]
#[cfg_attr(feature = "pplx-wasm-fma", target_feature(enable = "relaxed-simd"))]
unsafe fn rows<const R: usize>(
    x: &[f32],
    w: &[u16],
    y: &mut [f32],
    k: usize,
    n: usize,
    panel: usize,
    row: usize,
) {
    unsafe {
        let mut lo = [f32x4_splat(0.); R];
        let mut hi = lo;
        for j in 0..k {
            let b = v128_load(w.as_ptr().add((panel * k + j) * 8).cast());
            let b0 = i32x4_shl(u32x4_extend_low_u16x8(b), 16);
            let b1 = i32x4_shl(u32x4_extend_high_u16x8(b), 16);
            for r in 0..R {
                let a = f32x4_splat(*x.get_unchecked((row + r) * k + j));
                #[cfg(feature = "pplx-wasm-fma")]
                {
                    lo[r] = f32x4_relaxed_madd(a, b0, lo[r]);
                    hi[r] = f32x4_relaxed_madd(a, b1, hi[r]);
                }
                #[cfg(not(feature = "pplx-wasm-fma"))]
                {
                    lo[r] = f32x4_add(lo[r], f32x4_mul(a, b0));
                    hi[r] = f32x4_add(hi[r], f32x4_mul(a, b1));
                }
            }
        }
        for r in 0..R {
            let dst = (row + r) * n + panel * 8;
            if panel * 8 + 8 <= n {
                v128_store(y.as_mut_ptr().add(dst).cast(), lo[r]);
                v128_store(y.as_mut_ptr().add(dst + 4).cast(), hi[r]);
            } else {
                let mut tail = [0.; 8];
                v128_store(tail.as_mut_ptr().cast(), lo[r]);
                v128_store(tail.as_mut_ptr().add(4).cast(), hi[r]);
                y[dst..dst + n - panel * 8].copy_from_slice(&tail[..n - panel * 8]);
            }
        }
    }
}

#[cfg(feature = "kernel-checks")]
pub(super) fn checks() -> anyhow::Result<serde_json::Value> {
    let mut maximum = 0f32;
    let mut values = 0;
    for (m, k, n) in [
        (1, 7, 1),
        (3, 17, 17),
        (4, 128, 32),
        (7, 256, 65),
        (9, 2560, 128),
    ] {
        let bytes: Vec<u8> = (0..k * n)
            .flat_map(|i| {
                half::bf16::from_f32(((i * 17 % 113) as f32 - 56.) / 64.)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let x: Vec<_> = (0..m * k)
            .map(|i| ((i * 23 % 127) as f32 - 63.) / 128.)
            .collect();
        let w = pack(&bytes, k, n);
        let mut actual = vec![0.; m * n];
        project(&x, &w, &mut actual, k, n);
        for r in 0..m {
            for c in 0..n {
                let expected = (0..k)
                    .map(|j| {
                        let at = (c * k + j) * 2;
                        let b = f32::from_bits(
                            (u16::from_le_bytes([bytes[at], bytes[at + 1]]) as u32) << 16,
                        );
                        x[r * k + j] as f64 * b as f64
                    })
                    .sum::<f64>() as f32;
                maximum = maximum.max((actual[r * n + c] - expected).abs());
                values += 1;
            }
        }
    }
    anyhow::ensure!(maximum <= 1e-5, "Wasm projection mismatch {maximum}");
    Ok(
        serde_json::json!({"values":values,"max_absolute_error":maximum,"tile":"4 tokens x 8 outputs","weight_storage":"bf16"}),
    )
}
