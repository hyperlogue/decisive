use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Output(*mut f32);
unsafe impl Send for Output {}
unsafe impl Sync for Output {}
impl Output {
    unsafe fn copy(self, start: usize, values: &[f32]) {
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), self.0.add(start), values.len()) };
    }
}

pub(super) fn apply(
    qkv: &[f32],
    out: &mut [f32],
    heads: usize,
    dim: usize,
    window: Option<usize>,
    tile: usize,
    scratch: &mut Vec<super::attention_scratch::HeadScratch>,
) {
    apply_inner(qkv, out, heads, dim, window, tile, scratch, None);
}
pub(super) fn apply_selected(
    qkv: &[f32],
    out: &mut [f32],
    heads: usize,
    dim: usize,
    tile: usize,
    scratch: &mut Vec<super::attention_scratch::HeadScratch>,
    queries: &[usize],
) {
    apply_inner(qkv, out, heads, dim, None, tile, scratch, Some(queries));
}
fn apply_inner(
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
    let m = out.len() / d;
    assert_eq!(qkv.len(), n * 3 * d);
    assert_eq!(out.len(), m * d);
    if let Some(queries) = queries {
        assert!(window.is_none() && queries.len() == m && queries.iter().all(|&q| q < n));
    } else {
        assert_eq!(m, n);
    }
    let output = Output(out.as_mut_ptr());
    scratch.resize_with(heads, Default::default);
    scratch.par_iter_mut().enumerate().for_each(|(head, work)| {
        let scores = &mut work.scores;
        let values = &mut work.values;
        values.resize(tile.min(m) * dim, 0.0);
        for start in (0..m).step_by(tile) {
            let rows = (m - start).min(tile);
            let lo = window.map_or(0, |w| start.saturating_sub(w));
            let hi = window.map_or(n, |w| (start + rows + w).min(n));
            let keys = hi - lo;
            scores.resize(rows * keys, 0.0);
            let (query, query_stride) = if let Some(queries) = queries {
                work.query.resize(rows * dim, 0.);
                for (row, &index) in work
                    .query
                    .chunks_exact_mut(dim)
                    .zip(&queries[start..start + rows])
                {
                    row.copy_from_slice(&qkv[index * 3 * d + head * dim..][..dim]);
                }
                (work.query.as_ptr(), dim)
            } else {
                (qkv[start * 3 * d + head * dim..].as_ptr(), 3 * d)
            };
            unsafe {
                matrixmultiply::sgemm(
                    rows,
                    dim,
                    keys,
                    (dim as f32).sqrt().recip(),
                    query,
                    query_stride as isize,
                    1,
                    qkv.as_ptr().add(lo * 3 * d + d + head * dim),
                    1,
                    (3 * d) as isize,
                    0.0,
                    scores.as_mut_ptr(),
                    keys as isize,
                    1,
                );
            }
            for (row, score) in scores.chunks_exact_mut(keys).enumerate() {
                if let Some(w) = window {
                    let begin = (start + row).saturating_sub(w) - lo;
                    let end = (start + row + w + 1).min(n) - lo;
                    score[..begin].fill(f32::NEG_INFINITY);
                    score[end..].fill(f32::NEG_INFINITY);
                }
                super::softmax(score);
            }
            unsafe {
                matrixmultiply::sgemm(
                    rows,
                    keys,
                    dim,
                    1.0,
                    scores.as_ptr(),
                    keys as isize,
                    1,
                    qkv.as_ptr().add(lo * 3 * d + 2 * d + head * dim),
                    (3 * d) as isize,
                    1,
                    0.0,
                    values.as_mut_ptr(),
                    dim as isize,
                    1,
                );
                for row in 0..rows {
                    output.copy((start + row) * d + head * dim, &values[row * dim..][..dim]);
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn windows_and_query_tails_match_scalar_attention() {
        for tile in [16, 32, 64, 128, 256] {
            for n in [17, 31, 65, 257] {
                for dim in [7, 16] {
                    for window in [None, Some(0), Some(8), Some(32)] {
                        let heads = 3;
                        let d = heads * dim;
                        let qkv: Vec<_> = (0..n * 3 * d).map(|i| (i as f32 * 0.17).sin()).collect();
                        let mut actual = vec![0.0; n * d];
                        apply(&qkv, &mut actual, heads, dim, window, tile, &mut Vec::new());
                        for t in 0..n {
                            let lo = window.map_or(0, |w| t.saturating_sub(w));
                            let hi = window.map_or(n, |w| (t + w + 1).min(n));
                            for h in 0..heads {
                                let mut p: Vec<_> = (lo..hi)
                                    .map(|j| {
                                        (0..dim)
                                            .map(|i| {
                                                qkv[t * 3 * d + h * dim + i]
                                                    * qkv[j * 3 * d + d + h * dim + i]
                                            })
                                            .sum::<f32>()
                                            / (dim as f32).sqrt()
                                    })
                                    .collect();
                                super::super::softmax(&mut p);
                                for i in 0..dim {
                                    let expected: f32 = (lo..hi)
                                        .zip(&p)
                                        .map(|(j, p)| p * qkv[j * 3 * d + 2 * d + h * dim + i])
                                        .sum();
                                    assert!(
                                        (actual[t * d + h * dim + i] - expected).abs() < 2e-6,
                                        "n={n} dim={dim} window={window:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
