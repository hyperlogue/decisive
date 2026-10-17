use super::{
    Output,
    packed_attention::{pack_kv, pack_rows, product},
};
use crate::laya::cpu::{AttentionMath, attention_scratch::HeadScratch};
use rayon::prelude::*;
use std::arch::x86_64::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply(
    qkv: &[f32],
    out: &mut [f32],
    heads: usize,
    query_tile: usize,
    key_tile: usize,
    scratch: &mut Vec<HeadScratch>,
    math: AttentionMath,
) {
    assert!(crate::laya::cpu::packed_supported() && heads > 0);
    assert!(key_tile >= 16 && key_tile.is_multiple_of(16) && query_tile > 0);
    let d = heads * 64;
    let n = qkv.len() / (3 * d);
    assert!(n > 0 && qkv.len() == n * 3 * d && out.len() == n * d);
    let output = Output(out.as_mut_ptr());
    scratch.resize_with(heads, Default::default);
    scratch.par_iter_mut().enumerate().for_each(|(head, work)| {
        work.key.resize(n.div_ceil(16) * 16 * 64, 0.);
        work.value.resize(n * 64, 0.);
        unsafe {
            pack_kv(qkv, d, head, n, &mut work.key, &mut work.value);
        }
        for start in (0..n).step_by(query_tile) {
            let rows = (n - start).min(query_tile);
            work.query.resize(rows.div_ceil(6) * 6 * 64, 0.);
            work.values.resize(rows * 64, 0.);
            work.online.resize(rows * 2, 0.);
            unsafe {
                pack_rows(
                    &qkv[start * 3 * d + head * 64..],
                    rows,
                    64,
                    3 * d,
                    &mut work.query,
                );
            }
            for lo in (0..n).step_by(key_tile) {
                let keys = (n - lo).min(key_tile);
                work.scores.resize(rows * keys, 0.);
                work.probabilities.resize(rows.div_ceil(6) * 6 * keys, 0.);
                work.normalizers.resize(rows * 2, 0.);
                unsafe {
                    product(
                        &work.query,
                        &work.key[lo * 64..],
                        work.scores.as_mut_ptr(),
                        rows,
                        64,
                        keys,
                        keys,
                        64,
                        0.125,
                    );
                }
                for (row, scores) in work.scores.chunks_exact_mut(keys).enumerate() {
                    let (max, sum) = super::softmax::raw_stats(scores, math);
                    work.normalizers[row * 2] = max;
                    work.normalizers[row * 2 + 1] = sum;
                }
                unsafe {
                    pack_rows(&work.scores, rows, keys, keys, &mut work.probabilities);
                    product(
                        &work.probabilities,
                        &work.value[lo * 16..],
                        work.values.as_mut_ptr(),
                        rows,
                        keys,
                        64,
                        64,
                        n,
                        1.,
                    );
                }
                for row in 0..rows {
                    let block_max = work.normalizers[row * 2] as f64;
                    let block_sum = work.normalizers[row * 2 + 1] as f64;
                    let dst = unsafe { output.ptr().add((start + row) * d + head * 64) };
                    if lo == 0 {
                        work.online[row * 2] = block_max;
                        work.online[row * 2 + 1] = block_sum;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                work.values.as_ptr().add(row * 64),
                                dst,
                                64,
                            );
                        }
                    } else {
                        let old_max = work.online[row * 2];
                        let max = old_max.max(block_max);
                        let old_scale = (old_max - max).exp();
                        let block_scale = (block_max - max).exp();
                        work.online[row * 2] = max;
                        work.online[row * 2 + 1] =
                            work.online[row * 2 + 1] * old_scale + block_sum * block_scale;
                        unsafe {
                            combine(dst, &work.values[row * 64..][..64], old_scale, block_scale);
                        }
                    }
                }
            }
            for row in 0..rows {
                unsafe {
                    divide(
                        output.ptr().add((start + row) * d + head * 64),
                        work.online[row * 2 + 1],
                    );
                }
            }
        }
    });
}
#[target_feature(enable = "avx2,fma")]
unsafe fn combine(dst: *mut f32, fresh: &[f32], old_scale: f64, new_scale: f64) {
    unsafe {
        let a = _mm256_set1_pd(old_scale);
        let b = _mm256_set1_pd(new_scale);
        for i in (0..64).step_by(4) {
            let old = _mm256_cvtps_pd(_mm_loadu_ps(dst.add(i)));
            let new = _mm256_cvtps_pd(_mm_loadu_ps(fresh.as_ptr().add(i)));
            _mm_storeu_ps(
                dst.add(i),
                _mm256_cvtpd_ps(_mm256_fmadd_pd(new, b, _mm256_mul_pd(old, a))),
            );
        }
    }
}
#[target_feature(enable = "avx2")]
unsafe fn divide(dst: *mut f32, denominator: f64) {
    unsafe {
        let divisor = _mm256_set1_pd(denominator);
        for i in (0..64).step_by(4) {
            let x = _mm256_cvtps_pd(_mm_loadu_ps(dst.add(i)));
            _mm_storeu_ps(dst.add(i), _mm256_cvtpd_ps(_mm256_div_pd(x, divisor)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn online_global_attention_matches_dense_with_tails_and_reused_scratch() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        let heads = 2;
        let d = heads * 64;
        let mut scratch = Vec::new();
        for n in [17, 64, 129, 512, 1025, 2049, 129] {
            for amplitude in [0.1, 1.7, 6.] {
                let input: Vec<_> = (0..n * 3 * d)
                    .map(|i| (i as f32 * 0.017).sin() * amplitude)
                    .collect();
                let mut expected = vec![0.; n * d];
                super::super::packed_attention(
                    &input,
                    &mut expected,
                    heads,
                    None,
                    64,
                    &mut Vec::new(),
                    true,
                    AttentionMath::TableExp,
                    false,
                );
                for key_tile in [64, 128, 256, 512] {
                    let mut actual = vec![f32::NAN; n * d + 8];
                    apply(
                        &input,
                        &mut actual[..n * d],
                        heads,
                        64,
                        key_tile,
                        &mut scratch,
                        AttentionMath::TableExpSum,
                    );
                    assert!(actual[..n * d].iter().all(|v| v.is_finite()));
                    let error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f32::max);
                    assert!(
                        error <= 2e-5,
                        "n={n} amplitude={amplitude} key_tile={key_tile} error={error}"
                    );
                    assert!(actual[n * d..].iter().all(|x| x.is_nan()));
                }
            }
        }
    }
}
