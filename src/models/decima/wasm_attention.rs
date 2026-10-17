use core::arch::wasm32::*;
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

fn multiply(x: &[f32], w: &[f32], y: &mut [f32], k: usize, n: usize) {
    let tile = 4;
    assert!(k > 0 && n > 0 && x.len() % k == 0 && y.len() == x.len() / k * n);
    assert_eq!(w.len(), n.div_ceil(PANEL) * PANEL * k);
    let total_rows = x.len() / k;
    let cache_rows = (65536 / k).clamp(tile, 64) / tile * tile;
    for base in (0..total_rows).step_by(cache_rows) {
        let end = (base + cache_rows).min(total_rows);
        for panel in 0..n.div_ceil(PANEL) {
            for first in (base..end).step_by(tile) {
                let rows = tile.min(end - first);
                unsafe {
                    match rows {
                        1 => rows_kernel::<1>(x, w, y, k, n, first, panel),
                        2 => rows_kernel::<2>(x, w, y, k, n, first, panel),
                        3 => rows_kernel::<3>(x, w, y, k, n, first, panel),
                        4 => rows_kernel::<4>(x, w, y, k, n, first, panel),
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
                    v128_store(
                        dst.add(part * 4).cast(),
                        f32x4_add(accum[r][part], f32x4_splat(0.)),
                    );
                }
            } else {
                let mut tail = [0.; PANEL];
                for part in 0..PARTS {
                    v128_store(tail.as_mut_ptr().add(part * 4).cast(), accum[r][part]);
                }
                for c in col..n {
                    y[(first + r) * n + c] = tail[c - col] + 0.;
                }
            }
        }
    }
}

pub(super) fn apply(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    heads: usize,
    out: &mut Vec<f32>,
    scores: &mut Vec<f32>,
    work: &mut Vec<f32>,
    tile: usize,
) {
    assert!(heads > 0 && d % heads == 0 && !q.is_empty() && !k.is_empty());
    assert!(q.len() % d == 0 && k.len() % d == 0 && v.len() == k.len());
    let (nq, nk, hd) = (q.len() / d, k.len() / d, d / heads);
    #[cfg(feature = "decima-wasm-threads")]
    if apply_parallel(q, k, v, d, heads, out, work, tile) {
        scores.clear();
        return;
    }
    let rows = tile.max(1).min(nq);
    let key_len = nk.div_ceil(PANEL) * PANEL * hd;
    let value_len = hd.div_ceil(PANEL) * PANEL * nk;
    work.resize(key_len + value_len + 2 * rows * hd, 0.);
    let (key, remaining) = work.split_at_mut(key_len);
    let (value, remaining) = remaining.split_at_mut(value_len);
    let (query, context) = remaining.split_at_mut(rows * hd);
    out.resize(nq * d, 0.);
    let scale = 1. / (hd as f32).sqrt();
    for h in 0..heads {
        key.fill(0.);
        value.fill(0.);
        for i in 0..nk {
            for j in 0..hd {
                key[(i / PANEL * hd + j) * PANEL + i % PANEL] = k[i * d + h * hd + j];
                value[(j / PANEL * nk + i) * PANEL + j % PANEL] = v[i * d + h * hd + j];
            }
        }
        for first in (0..nq).step_by(rows) {
            let count = rows.min(nq - first);
            let query = &mut query[..count * hd];
            let context = &mut context[..count * hd];
            for r in 0..count {
                query[r * hd..][..hd].copy_from_slice(&q[(first + r) * d + h * hd..][..hd]);
            }
            scores.resize(count * nk, 0.);
            multiply(query, key, scores, hd, nk);
            for row in scores.chunks_exact_mut(nk) {
                for v in row.iter_mut() {
                    *v *= scale;
                }
                super::softmax_wasm::softmax(row);
            }
            multiply(scores, value, context, nk, hd);
            for r in 0..count {
                out[(first + r) * d + h * hd..][..hd].copy_from_slice(&context[r * hd..][..hd]);
            }
        }
    }
}

#[cfg(feature = "decima-wasm-threads")]
fn apply_parallel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    heads: usize,
    out: &mut Vec<f32>,
    head_outputs: &mut Vec<f32>,
    tile: usize,
) -> bool {
    use rayon::prelude::*;
    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<(Vec<f32>, Vec<f32>)> = const {
            RefCell::new((Vec::new(), Vec::new()))
        };
    }
    let workers = super::wasm_threads::count().min(heads);
    let (nq, nk, hd) = (q.len() / d, k.len() / d, d / heads);
    if workers <= 1 || nq * nk < 4096 {
        return false;
    }
    let group = heads.div_ceil(workers);
    head_outputs.resize(nq * d, 0.);
    head_outputs
        .par_chunks_mut(group * nq * hd)
        .enumerate()
        .for_each(|(chunk, output)| {
            SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let (scores, work) = &mut *scratch;
                for (local, output) in output.chunks_exact_mut(nq * hd).enumerate() {
                    head_to_contiguous(
                        q,
                        k,
                        v,
                        d,
                        chunk * group + local,
                        hd,
                        output,
                        scores,
                        work,
                        tile,
                    );
                }
            });
        });
    out.resize(nq * d, 0.);
    for h in 0..heads {
        let src = &head_outputs[h * nq * hd..][..nq * hd];
        for r in 0..nq {
            out[r * d + h * hd..][..hd].copy_from_slice(&src[r * hd..][..hd]);
        }
    }
    true
}

#[cfg(feature = "decima-wasm-threads")]
fn head_to_contiguous(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    h: usize,
    hd: usize,
    out: &mut [f32],
    scores: &mut Vec<f32>,
    work: &mut Vec<f32>,
    tile: usize,
) {
    let (nq, nk) = (q.len() / d, k.len() / d);
    let rows = tile.max(1).min(nq);
    let key_len = nk.div_ceil(PANEL) * PANEL * hd;
    let value_len = hd.div_ceil(PANEL) * PANEL * nk;
    work.resize(key_len + value_len + 2 * rows * hd, 0.);
    let (key, remaining) = work.split_at_mut(key_len);
    let (value, remaining) = remaining.split_at_mut(value_len);
    let (query, context) = remaining.split_at_mut(rows * hd);
    key.fill(0.);
    value.fill(0.);
    for i in 0..nk {
        for j in 0..hd {
            key[(i / PANEL * hd + j) * PANEL + i % PANEL] = k[i * d + h * hd + j];
            value[(j / PANEL * nk + i) * PANEL + j % PANEL] = v[i * d + h * hd + j];
        }
    }
    let scale = 1. / (hd as f32).sqrt();
    for first in (0..nq).step_by(rows) {
        let count = rows.min(nq - first);
        let query = &mut query[..count * hd];
        let context = &mut context[..count * hd];
        for r in 0..count {
            query[r * hd..][..hd].copy_from_slice(&q[(first + r) * d + h * hd..][..hd]);
        }
        scores.resize(count * nk, 0.);
        multiply(query, key, scores, hd, nk);
        for row in scores.chunks_exact_mut(nk) {
            for v in row.iter_mut() {
                *v *= scale;
            }
            super::softmax_wasm::softmax(row);
        }
        multiply(scores, value, context, nk, hd);
        out[first * hd..][..count * hd].copy_from_slice(context);
    }
}

#[cfg(feature = "kernel-checks")]
pub(super) fn check() -> anyhow::Result<serde_json::Value> {
    let mut checked = 0;
    let mut maximum = 0_f32;
    for (nq, nk, heads, hd) in [
        (1, 1, 1, 1),
        (7, 11, 2, 5),
        (33, 65, 3, 32),
        (129, 512, 2, 64),
        (512, 512, 1, 32),
    ] {
        let d = heads * hd;
        let values = |n, seed| {
            (0..n)
                .map(|i| ((i * 31 + seed) % 127) as f32 / 63. - 1.)
                .collect::<Vec<_>>()
        };
        let q = values(nq * d, 17);
        let k = values(nk * d, 31);
        let v = values(nk * d, 53);
        let mut reference = vec![0_f32; nq * d];
        let mut row = vec![0.; nk];
        for i in 0..nq {
            for h in 0..heads {
                for j in 0..nk {
                    let mut score = 0.;
                    for c in 0..hd {
                        score += q[i * d + h * hd + c] * k[j * d + h * hd + c];
                    }
                    row[j] = score * (1. / (hd as f32).sqrt());
                }
                super::cpu::softmax(&mut row);
                for c in 0..hd {
                    let mut value = 0.;
                    for j in 0..nk {
                        value += row[j] * v[j * d + h * hd + c];
                    }
                    reference[i * d + h * hd + c] = value;
                }
            }
        }
        let mut out = vec![];
        let mut scores = vec![];
        let mut work = vec![];
        for tile in [1, 7, 16, 32, 64] {
            apply(&q, &k, &v, d, heads, &mut out, &mut scores, &mut work, tile);
            for (&a, &b) in out.iter().zip(&reference) {
                let error = (a - b).abs();
                maximum = maximum.max(error);
                anyhow::ensure!(
                    error <= 5e-6,
                    "Wasm attention {nq}x{nk}, dim {hd}, tile {tile}: {a} vs {b}"
                );
                checked += 1;
            }
            anyhow::ensure!(
                scores.len() <= tile.min(nq) * nk,
                "attention score tile exceeded"
            );
        }
    }
    Ok(serde_json::json!({"values":checked,"maximum_error":maximum}))
}
