use core::arch::wasm32::*;
#[inline(always)]
fn multiply_add(a: v128, b: v128, sum: v128) -> v128 {
    f32x4_add(sum, f32x4_mul(a, b))
}

#[cfg(feature = "fp16")]
use half::f16;

#[cfg(feature = "fp16")]
pub fn pack_half(w: &[f16], k: usize, n: usize) -> Vec<f16> {
    let mut dst = vec![f16::ZERO; n.div_ceil(8) * 8 * k];
    for c in 0..n {
        for j in 0..k {
            dst[(c / 8 * k + j) * 8 + c % 8] = w[c * k + j];
        }
    }
    dst
}

#[inline]
#[cfg(feature = "fp16")]
fn expand_half(bits: v128) -> v128 {
    let sign = i32x4_shl(v128_and(bits, i32x4_splat(0x8000)), 16);
    let magnitude = v128_and(bits, i32x4_splat(0x7fff));
    let exponent = v128_and(bits, i32x4_splat(0x7c00));
    let normal = i32x4_add(i32x4_shl(magnitude, 13), i32x4_splat(0x38000000));
    let special = v128_or(i32x4_shl(magnitude, 13), i32x4_splat(0x7f800000));
    let small = f32x4_mul(f32x4_convert_u32x4(magnitude), f32x4_splat(1. / 16777216.));
    let value = v128_bitselect(small, normal, i32x4_eq(exponent, i32x4_splat(0)));
    let value = v128_bitselect(special, value, i32x4_eq(exponent, i32x4_splat(0x7c00)));
    v128_or(value, sign)
}

#[cfg(feature = "fp16")]
fn decode_half(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    let full = src.len() / 8 * 8;
    for i in (0..full).step_by(8) {
        unsafe {
            let bits = v128_load(src.as_ptr().add(i).cast());
            v128_store(
                dst.as_mut_ptr().add(i).cast(),
                expand_half(u32x4_extend_low_u16x8(bits)),
            );
            v128_store(
                dst.as_mut_ptr().add(i + 4).cast(),
                expand_half(u32x4_extend_high_u16x8(bits)),
            );
        }
    }
    for i in full..src.len() {
        dst[i] = src[i].to_f32();
    }
}

#[cfg(feature = "fp16")]
pub fn multiply_half(
    x: &[f32],
    w: &[f16],
    y: &mut [f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    tile: usize,
    workspace: &mut Vec<f32>,
) {
    assert!(k > 0 && n > 0 && x.len() % k == 0 && y.len() == x.len() / k * n);
    assert_eq!(w.len(), n.div_ceil(8) * 8 * k);
    assert!(bias.is_none_or(|b| b.len() == n));
    assert!((1..=6).contains(&tile));
    workspace.resize(k * 8, 0.);
    let total_rows = x.len() / k;
    let cache_rows = (65536 / k).clamp(tile, 64) / tile * tile;
    for base in (0..total_rows).step_by(cache_rows) {
        let end = (base + cache_rows).min(total_rows);
        for panel in 0..n.div_ceil(8) {
            decode_half(&w[panel * k * 8..][..k * 8], workspace);
            for first in (base..end).step_by(tile) {
                unsafe {
                    match tile.min(end - first) {
                        1 => rows_kernel::<1>(x, workspace, y, k, n, bias, first, panel),
                        2 => rows_kernel::<2>(x, workspace, y, k, n, bias, first, panel),
                        3 => rows_kernel::<3>(x, workspace, y, k, n, bias, first, panel),
                        4 => rows_kernel::<4>(x, workspace, y, k, n, bias, first, panel),
                        5 => rows_kernel::<5>(x, workspace, y, k, n, bias, first, panel),
                        6 => rows_kernel::<6>(x, workspace, y, k, n, bias, first, panel),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
}

pub fn pack(w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut dst = vec![0.; n.div_ceil(8) * 8 * k];
    for c in 0..n {
        for j in 0..k {
            dst[(c / 8 * k + j) * 8 + c % 8] = w[c * k + j];
        }
    }
    dst
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
    assert_eq!(w.len(), n.div_ceil(8) * 8 * k);
    assert!(bias.is_none_or(|b| b.len() == n));
    assert!((1..=6).contains(&tile));
    let total_rows = x.len() / k;
    let cache_rows = (65536 / k).clamp(tile, 64) / tile * tile;
    for base in (0..total_rows).step_by(cache_rows) {
        let end = (base + cache_rows).min(total_rows);
        for panel in 0..n.div_ceil(8) {
            let weights = &w[panel * k * 8..][..k * 8];
            for first in (base..end).step_by(tile) {
                let rows = tile.min(end - first);
                unsafe {
                    match rows {
                        1 => rows_kernel::<1>(x, weights, y, k, n, bias, first, panel),
                        2 => rows_kernel::<2>(x, weights, y, k, n, bias, first, panel),
                        3 => rows_kernel::<3>(x, weights, y, k, n, bias, first, panel),
                        4 => rows_kernel::<4>(x, weights, y, k, n, bias, first, panel),
                        5 => rows_kernel::<5>(x, weights, y, k, n, bias, first, panel),
                        6 => rows_kernel::<6>(x, weights, y, k, n, bias, first, panel),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
}
#[target_feature(enable = "simd128")]
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
        {
            let mut accum = [[f32x4_splat(0.); 2]; R];
            for j in 0..k {
                let b = w.as_ptr().add(j * 8);
                let b0 = v128_load(b.cast());
                let b1 = v128_load(b.add(4).cast());
                for r in 0..R {
                    let a = f32x4_splat(*x.get_unchecked((first + r) * k + j));
                    accum[r][0] = multiply_add(a, b0, accum[r][0]);
                    accum[r][1] = multiply_add(a, b1, accum[r][1]);
                }
            }
            for r in 0..R {
                let col = panel * 8;
                if col + 8 <= n {
                    let dst = y.as_mut_ptr().add((first + r) * n + col);
                    v128_store(
                        dst.cast(),
                        f32x4_add(
                            accum[r][0],
                            bias.map_or(f32x4_splat(0.), |b| v128_load(b.as_ptr().add(col).cast())),
                        ),
                    );
                    v128_store(
                        dst.add(4).cast(),
                        f32x4_add(
                            accum[r][1],
                            bias.map_or(f32x4_splat(0.), |b| {
                                v128_load(b.as_ptr().add(col + 4).cast())
                            }),
                        ),
                    );
                } else {
                    let mut tail = [0.; 8];
                    v128_store(tail.as_mut_ptr().cast(), accum[r][0]);
                    v128_store(tail.as_mut_ptr().add(4).cast(), accum[r][1]);
                    for c in col..n {
                        y[(first + r) * n + c] = tail[c - col] + bias.map_or(0., |b| b[c]);
                    }
                }
            }
        }
    }
}

pub(super) fn attention(
    qkv: &[f32],
    out: &mut [f32],
    heads: usize,
    dim: usize,
    window: Option<usize>,
    tile: usize,
    scratch: &mut Vec<super::attention_scratch::HeadScratch>,
    queries: Option<&[usize]>,
) {
    let d = heads * dim;
    let n = qkv.len() / (3 * d);
    let nq = queries.map_or(n, <[usize]>::len);
    assert_eq!(out.len(), nq * d);
    assert!(queries.is_none() || window.is_none());
    assert!(queries.is_none_or(|q| q.iter().all(|&i| i < n)));
    scratch.resize_with(1, Default::default);
    let s = &mut scratch[0];
    let scale = (dim as f32).sqrt().recip();
    for h in 0..heads {
        s.key.resize(n.div_ceil(8) * 8 * dim, 0.);
        s.key.fill(0.);
        for c in 0..n {
            for j in 0..dim {
                s.key[(c / 8 * dim + j) * 8 + c % 8] = qkv[c * 3 * d + d + h * dim + j];
            }
        }
        for first in (0..nq).step_by(tile) {
            let rows = tile.min(nq - first);
            let lo = window.map_or(0, |w| first.saturating_sub(w) / 8 * 8);
            let hi = window.map_or(n, |w| {
                (first + rows + w).div_ceil(8).saturating_mul(8).min(n)
            });
            let keys = hi - lo;
            s.query.resize(rows * dim, 0.);
            for r in 0..rows {
                let index = queries.map_or(first + r, |q| q[first + r]);
                s.query[r * dim..(r + 1) * dim]
                    .copy_from_slice(&qkv[index * 3 * d + h * dim..][..dim]);
            }
            s.scores.resize(rows * keys, 0.);
            multiply(
                &s.query,
                &s.key[lo / 8 * 8 * dim..lo / 8 * 8 * dim + keys.div_ceil(8) * 8 * dim],
                &mut s.scores,
                dim,
                keys,
                None,
                4,
            );
            for (r, score) in s.scores.chunks_exact_mut(keys).enumerate() {
                let begin = window.map_or(0, |w| (first + r).saturating_sub(w) - lo);
                let end = window.map_or(keys, |w| (first + r + w + 1).min(n) - lo);
                score[..begin].fill(f32::NEG_INFINITY);
                score[end..].fill(f32::NEG_INFINITY);
                for x in &mut score[begin..end] {
                    *x *= scale;
                }
                super::softmax_wasm::softmax(score);
            }
            if window.is_some() || first == 0 {
                s.value.resize(dim.div_ceil(8) * 8 * keys, 0.);
                s.value.fill(0.);
                for c in 0..dim {
                    for j in 0..keys {
                        s.value[(c / 8 * keys + j) * 8 + c % 8] =
                            qkv[(lo + j) * 3 * d + 2 * d + h * dim + c];
                    }
                }
            }
            s.values.resize(rows * dim, 0.);
            multiply(&s.scores, &s.value, &mut s.values, keys, dim, None, 4);
            for r in 0..rows {
                out[(first + r) * d + h * dim..][..dim]
                    .copy_from_slice(&s.values[r * dim..][..dim]);
            }
        }
    }
}

#[cfg(feature = "kernel-checks")]
pub(super) fn check() -> anyhow::Result<usize> {
    let mut checked = 0;
    #[cfg(feature = "fp16")]
    {
        let half: Vec<f16> = (0..=65535).map(f16::from_bits).collect();
        let mut singles = vec![0.; half.len()];
        decode_half(&half, &mut singles);
        for (a, b) in half.iter().zip(&singles) {
            anyhow::ensure!(
                if a.is_nan() {
                    b.is_nan()
                } else {
                    a.to_f32().to_bits() == b.to_bits()
                },
                "half conversion {:04x}",
                a.to_bits()
            );
        }
    }
    for (k, n) in [(1, 1), (7, 3), (35, 17), (128, 64), (384, 384)] {
        let weight: Vec<f32> = (0..k * n)
            .map(|i| ((i * 41 % 127) as f32 - 63.) / 128.)
            .collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 / 71.).collect();
        let packed = pack(&weight, k, n);
        #[cfg(feature = "fp16")]
        let packed_half = pack_half(
            &weight
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
            k,
            n,
        );
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
                anyhow::ensure!(
                    guard[0] == 999. && guard[m * n + 1] == 999.,
                    "Wasm output overwrite"
                );
                #[cfg(feature = "fp16")]
                {
                    let mut half_guard = vec![999.; m * n + 2];
                    multiply_half(
                        &x,
                        &packed_half,
                        &mut half_guard[1..m * n + 1],
                        k,
                        n,
                        Some(&bias),
                        tile,
                        &mut Vec::new(),
                    );
                    anyhow::ensure!(
                        half_guard == guard,
                        "packed half projection {m}x{k}x{n}, tile {tile}"
                    );
                }
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
    // Prune queries only; selected outputs still attend to every allowed key/value.
    for n in [1, 7, 33, 129, 513] {
        let (heads, dim) = (2, 16);
        let d = heads * dim;
        let qkv: Vec<_> = (0..n * 3 * d)
            .map(|i| ((i * 31 % 131) as f32 - 65.) / 71.)
            .collect();
        for tile in [16, 32, 64] {
            let mut scratch = vec![];
            let mut full = vec![0.; n * d];
            attention(&qkv, &mut full, heads, dim, None, tile, &mut scratch, None);
            let indices = [n - 1, 0, n / 2, n - 1, 0];
            let mut selected = vec![0.; indices.len() * d];
            attention(
                &qkv,
                &mut selected,
                heads,
                dim,
                None,
                tile,
                &mut scratch,
                Some(&indices),
            );
            for (row, &index) in selected.chunks_exact(d).zip(&indices) {
                anyhow::ensure!(
                    row == &full[index * d..][..d],
                    "selected Wasm attention differs at {n}, {tile}, {index}"
                );
            }
        }
    }
    Ok(checked)
}
