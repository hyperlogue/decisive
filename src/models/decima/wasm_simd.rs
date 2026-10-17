use core::arch::wasm32::*;
#[cfg(not(feature = "decima-wasm-fma"))]
const PANEL: usize = 12;
#[cfg(feature = "decima-wasm-fma")]
const PANEL: usize = 8;
const PARTS: usize = PANEL / 4;

#[inline]
#[cfg_attr(feature = "decima-wasm-fma", target_feature(enable = "relaxed-simd"))]
unsafe fn multiply_add(a: v128, b: v128, sum: v128) -> v128 {
    #[cfg(feature = "decima-wasm-fma")]
    {
        f32x4_relaxed_madd(a, b, sum)
    }
    #[cfg(not(feature = "decima-wasm-fma"))]
    {
        f32x4_add(sum, f32x4_mul(a, b))
    }
}

pub fn pack(w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut dst = vec![0.; n.div_ceil(PANEL) * PANEL * k];
    for c in 0..n {
        for j in 0..k {
            dst[(c / PANEL * k + j) * PANEL + c % PANEL] = w[c * k + j];
        }
    }
    dst
}
#[cfg(any(feature = "webgpu", feature = "kernel-checks"))]
pub(super) fn unpack(packed: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut weights = vec![0.; k * n];
    for c in 0..n {
        for j in 0..k {
            weights[c * k + j] = packed[(c / PANEL * k + j) * PANEL + c % PANEL];
        }
    }
    weights
}

#[cfg(feature = "decima-wasm-threads")]
pub(super) fn multiply_parallel(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    tile: usize,
) -> bool {
    use rayon::prelude::*;
    let count = x.len() / k;
    let tasks = super::wasm_threads::count().min(count.div_ceil(24));
    if tasks <= 1 || count * k * n < 2_000_000 {
        return false;
    }
    let rows = count.div_ceil(tasks).div_ceil(tile) * tile;
    y.par_chunks_mut(rows * n)
        .enumerate()
        .for_each(|(chunk, out)| {
            let first = chunk * rows;
            let count = out.len() / n;
            multiply(&x[first * k..(first + count) * k], w, out, k, n, bias, tile);
        });
    true
}

pub fn multiply(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    tile: usize,
) {
    assert!(k > 0 && n > 0 && x.len() % k == 0 && y.len() == x.len() / k * n);
    assert_eq!(w.len(), n.div_ceil(PANEL) * PANEL * k);
    assert!(bias.is_none_or(|b| b.len() == n));
    assert!((1..=6).contains(&tile));
    let total_rows = x.len() / k;
    let cache_rows = (65536 / k).clamp(tile, 64) / tile * tile;
    for base in (0..total_rows).step_by(cache_rows) {
        let end = (base + cache_rows).min(total_rows);
        for panel in 0..n.div_ceil(PANEL) {
            for first in (base..end).step_by(tile) {
                let rows = tile.min(end - first);
                unsafe {
                    match rows {
                        1 => rows_kernel::<1>(x, w, y, k, n, bias, first, panel),
                        2 => rows_kernel::<2>(x, w, y, k, n, bias, first, panel),
                        3 => rows_kernel::<3>(x, w, y, k, n, bias, first, panel),
                        4 => rows_kernel::<4>(x, w, y, k, n, bias, first, panel),
                        5 => rows_kernel::<5>(x, w, y, k, n, bias, first, panel),
                        6 => rows_kernel::<6>(x, w, y, k, n, bias, first, panel),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
}
#[target_feature(enable = "simd128")]
#[cfg_attr(feature = "decima-wasm-fma", target_feature(enable = "relaxed-simd"))]
unsafe fn rows_kernel<const R: usize>(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    first: usize,
    panel: usize,
) {
    unsafe {
        let mut accum = [[f32x4_splat(0.); PARTS]; R];
        for j in 0..k {
            let b = w.as_ptr().add((panel * k + j) * PANEL);
            let weights: [v128; PARTS] =
                std::array::from_fn(|part| v128_load(b.add(part * 4).cast()));
            for r in 0..R {
                let a = f32x4_splat(*x.get_unchecked((first + r) * k + j));
                for part in 0..PARTS {
                    accum[r][part] = multiply_add(a, weights[part], accum[r][part]);
                }
            }
        }
        for r in 0..R {
            let col = panel * PANEL;
            if col + PANEL <= n {
                let dst = y.as_mut_ptr().add((first + r) * n + col);
                for part in 0..PARTS {
                    let bias = bias.map_or(f32x4_splat(0.), |b| {
                        v128_load(b.as_ptr().add(col + part * 4).cast())
                    });
                    v128_store(dst.add(part * 4).cast(), f32x4_add(accum[r][part], bias));
                }
            } else {
                let mut tail = [0.; PANEL];
                for part in 0..PARTS {
                    v128_store(tail.as_mut_ptr().add(part * 4).cast(), accum[r][part]);
                }
                for c in col..n {
                    y[(first + r) * n + c] = tail[c - col] + bias.map_or(0., |b| b[c]);
                }
            }
        }
    }
}

#[cfg(feature = "kernel-checks")]
pub(super) fn check() -> anyhow::Result<usize> {
    let mut checked = 0;
    for (k, n) in [(1, 1), (7, 3), (35, 17), (128, 64), (384, 384)] {
        let weight: Vec<f32> = (0..k * n)
            .map(|i| ((i * 41 % 127) as f32 - 63.) / 128.)
            .collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 / 71.).collect();
        let packed = pack(&weight, k, n);
        anyhow::ensure!(unpack(&packed, k, n) == weight, "Wasm weight export");
        for m in [0, 1, 3, 5, 6, 7, 13, 65, 129] {
            for tile in 1..=6 {
                let x: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 17 % 61) as f32 - 30.) / 64.)
                    .collect();
                let mut guard = vec![999.; m * n + 2];
                multiply(
                    &x,
                    &packed,
                    &mut guard[1..m * n + 1],
                    k,
                    n,
                    Some(&bias),
                    tile,
                );
                #[cfg(feature = "decima-wasm-threads")]
                {
                    let mut parallel = vec![0.; m * n];
                    if multiply_parallel(&x, &packed, &mut parallel, k, n, Some(&bias), tile) {
                        anyhow::ensure!(
                            parallel
                                .iter()
                                .zip(&guard[1..m * n + 1])
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "Wasm parallel projection changed output bits"
                        );
                        checked += m * n;
                    }
                }
                anyhow::ensure!(
                    guard[0] == 999. && guard[m * n + 1] == 999.,
                    "Wasm output overwrite"
                );
                for r in 0..m {
                    for c in 0..n {
                        let mut expected = 0.;
                        for j in 0..k {
                            expected += x[r * k + j] * weight[c * k + j];
                        }
                        expected += bias[c];
                        anyhow::ensure!(
                            guard[1 + r * n + c].to_bits() == expected.to_bits(),
                            "Wasm projection {m}x{k}x{n}, tile {tile}: row {r} col {c}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    Ok(checked)
}
