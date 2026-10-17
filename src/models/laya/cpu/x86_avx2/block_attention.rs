use super::packed_attention::{pack_kv, pack_rows, product};
use crate::laya::cpu::attention_scratch::HeadScratch;
use std::ops::Range;

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn head(
    qkv: &[f32],
    out: *mut f32,
    d: usize,
    head: usize,
    queries: Range<usize>,
    keys: Range<usize>,
    tile: usize,
    work: &mut HeadScratch,
) {
    let m = queries.len();
    let k = keys.len();
    work.key.resize(k.div_ceil(16) * 16 * 64, 0.);
    work.value.resize(k * 64, 0.);
    unsafe {
        pack_kv(
            &qkv[keys.start * 3 * d..keys.end * 3 * d],
            d,
            head,
            k,
            &mut work.key,
            &mut work.value,
        );
    }
    for start in (0..m).step_by(tile) {
        let rows = (m - start).min(tile);
        work.query.resize(rows.div_ceil(6) * 6 * 64, 0.);
        work.scores.resize(rows * k, 0.);
        unsafe {
            pack_rows(
                &qkv[(queries.start + start) * 3 * d + head * 64..],
                rows,
                64,
                3 * d,
                &mut work.query,
            );
            product(
                &work.query,
                &work.key,
                work.scores.as_mut_ptr(),
                rows,
                64,
                k,
                k,
                64,
                0.125,
            );
        }
        for (row, scores) in work.scores.chunks_exact_mut(k).enumerate() {
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = super::softmax::table_scale(scores, false);
            for value in scores {
                *value /= sum;
            }
            work.normalizers[(start + row) * 2] = maximum;
            work.normalizers[(start + row) * 2 + 1] = sum;
        }
        work.query.resize(rows.div_ceil(6) * 6 * k, 0.);
        unsafe {
            pack_rows(&work.scores, rows, k, k, &mut work.query);
            product(
                &work.query,
                &work.value,
                out.add(start * d + head * 64),
                rows,
                k,
                64,
                d,
                k,
                1.,
            );
        }
    }
}
