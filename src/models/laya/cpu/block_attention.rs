use super::{AttentionScratch, attention_scratch::HeadScratch};
#[cfg(feature = "cpu")]
use rayon::prelude::*;
use std::ops::Range;

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct SoftmaxState {
    pub maximum: f32,
    pub denominator: f32,
}
#[derive(Clone, Copy)]
struct Output(*mut f32);
unsafe impl Send for Output {}
unsafe impl Sync for Output {}
impl Output {
    fn ptr(self) -> *mut f32 {
        self.0
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_block(
    qkv: &[f32],
    out: &mut [f32],
    states: &mut Vec<SoftmaxState>,
    heads: usize,
    dim: usize,
    queries: Range<usize>,
    keys: Range<usize>,
    tile: usize,
    scratch: &mut AttentionScratch,
) {
    assert!(heads > 0 && dim > 0 && tile > 0);
    let d = heads * dim;
    let n = qkv.len() / (3 * d);
    assert_eq!(qkv.len(), n * 3 * d);
    assert!(queries.start < queries.end && queries.end <= n);
    assert!(keys.start < keys.end && keys.end <= n);
    let m = queries.len();
    assert_eq!(out.len(), m * d);
    states.resize(m * heads, SoftmaxState::default());
    scratch.heads.resize_with(heads, Default::default);
    let output = Output(out.as_mut_ptr());
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    let packed = scratch.packed_avx2 && dim == 64 && super::packed_supported();
    let run = |(head, work): (usize, &mut HeadScratch)| {
        work.normalizers.resize(m * 2, 0.);
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if packed {
            unsafe {
                super::x86_avx2::block_attention_head(
                    qkv,
                    output.ptr(),
                    d,
                    head,
                    queries.clone(),
                    keys.clone(),
                    tile,
                    work,
                );
            }
            return;
        }
        let k = keys.len();
        for start in (0..m).step_by(tile) {
            let rows = (m - start).min(tile);
            work.scores.resize(rows * k, 0.);
            unsafe {
                matrixmultiply::sgemm(
                    rows,
                    dim,
                    k,
                    (dim as f32).sqrt().recip(),
                    qkv.as_ptr()
                        .add((queries.start + start) * 3 * d + head * dim),
                    (3 * d) as isize,
                    1,
                    qkv.as_ptr().add(keys.start * 3 * d + d + head * dim),
                    1,
                    (3 * d) as isize,
                    0.,
                    work.scores.as_mut_ptr(),
                    k as isize,
                    1,
                );
            }
            for (row, scores) in work.scores.chunks_exact_mut(k).enumerate() {
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.;
                for x in scores.iter_mut() {
                    *x = (*x - maximum).exp();
                    sum += *x;
                }
                for x in scores {
                    *x /= sum;
                }
                work.normalizers[(start + row) * 2] = maximum;
                work.normalizers[(start + row) * 2 + 1] = sum;
            }
            unsafe {
                matrixmultiply::sgemm(
                    rows,
                    k,
                    dim,
                    1.,
                    work.scores.as_ptr(),
                    k as isize,
                    1,
                    qkv.as_ptr().add(keys.start * 3 * d + 2 * d + head * dim),
                    (3 * d) as isize,
                    1,
                    0.,
                    output.ptr().add(start * d + head * dim),
                    d as isize,
                    1,
                );
            }
        }
    };
    #[cfg(feature = "cpu")]
    scratch.heads.par_iter_mut().enumerate().for_each(run);
    #[cfg(not(feature = "cpu"))]
    scratch.heads.iter_mut().enumerate().for_each(run);
    for (head, work) in scratch.heads.iter().enumerate() {
        for row in 0..m {
            states[row * heads + head] = SoftmaxState {
                maximum: work.normalizers[row * 2],
                denominator: work.normalizers[row * 2 + 1],
            };
        }
    }
}
