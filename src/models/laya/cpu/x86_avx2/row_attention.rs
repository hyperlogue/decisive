use super::*;
use crate::laya::cpu::attention_scratch::HeadScratch;

pub(crate) fn apply(
    qkv: &[f32],
    out: &mut [f32],
    heads: usize,
    query_tile: usize,
    scratch: &mut Vec<HeadScratch>,
) {
    assert!(
        crate::laya::cpu::packed_supported() && heads > 0 && query_tile > 0 && query_tile % 8 == 0
    );
    let d = heads * 64;
    let n = qkv.len() / (3 * d);
    assert_eq!(qkv.len(), n * 3 * d);
    assert_eq!(out.len(), n * d);
    assert!(n > 0);
    let output = Output(out.as_mut_ptr());
    scratch.resize_with(heads, Default::default);
    scratch.par_iter_mut().enumerate().for_each(|(head, work)| {
        let key_len = n.div_ceil(8) * 8 * 64;
        work.key.resize(key_len, 0.);
        work.value.resize(n * 64, 0.);
        unsafe {
            pack_kv(qkv, d, head, n, &mut work.key, &mut work.value);
        }
        for start in (0..n).step_by(query_tile) {
            let rows = (n - start).min(query_tile);
            let blocks = rows.div_ceil(8);
            work.query.resize(blocks * 8 * 64, 0.);
            work.scores.resize(blocks * 8 * n, 0.);
            unsafe {
                pack_query(
                    &qkv[start * 3 * d + head * 64..],
                    rows,
                    3 * d,
                    &mut work.query,
                );
                query_key(&work.query, &work.key, &mut work.scores, blocks, n);
                for scores in work.scores.chunks_exact_mut(n * 8) {
                    normalize(scores, n);
                }
                probability_value(
                    &work.scores,
                    &work.value,
                    output.ptr().add(start * d + head * 64),
                    rows,
                    n,
                    d,
                );
            }
        }
    });
}
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transpose(v: [__m256; 8]) -> [__m256; 8] {
    let a = [
        _mm256_unpacklo_ps(v[0], v[1]),
        _mm256_unpackhi_ps(v[0], v[1]),
        _mm256_unpacklo_ps(v[2], v[3]),
        _mm256_unpackhi_ps(v[2], v[3]),
        _mm256_unpacklo_ps(v[4], v[5]),
        _mm256_unpackhi_ps(v[4], v[5]),
        _mm256_unpacklo_ps(v[6], v[7]),
        _mm256_unpackhi_ps(v[6], v[7]),
    ];
    let b = [
        _mm256_shuffle_ps::<0x44>(a[0], a[2]),
        _mm256_shuffle_ps::<0xee>(a[0], a[2]),
        _mm256_shuffle_ps::<0x44>(a[1], a[3]),
        _mm256_shuffle_ps::<0xee>(a[1], a[3]),
        _mm256_shuffle_ps::<0x44>(a[4], a[6]),
        _mm256_shuffle_ps::<0xee>(a[4], a[6]),
        _mm256_shuffle_ps::<0x44>(a[5], a[7]),
        _mm256_shuffle_ps::<0xee>(a[5], a[7]),
    ];
    [
        _mm256_permute2f128_ps::<0x20>(b[0], b[4]),
        _mm256_permute2f128_ps::<0x20>(b[1], b[5]),
        _mm256_permute2f128_ps::<0x20>(b[2], b[6]),
        _mm256_permute2f128_ps::<0x20>(b[3], b[7]),
        _mm256_permute2f128_ps::<0x31>(b[0], b[4]),
        _mm256_permute2f128_ps::<0x31>(b[1], b[5]),
        _mm256_permute2f128_ps::<0x31>(b[2], b[6]),
        _mm256_permute2f128_ps::<0x31>(b[3], b[7]),
    ]
}
#[target_feature(enable = "avx2")]
unsafe fn pack_kv(
    src: &[f32],
    d: usize,
    head: usize,
    n: usize,
    key: &mut [f32],
    value: &mut [f32],
) {
    unsafe {
        for t in 0..n {
            for i in (0..64).step_by(8) {
                let v = _mm256_loadu_ps(src.as_ptr().add(t * 3 * d + head * 64 + 2 * d + i));
                _mm256_storeu_ps(value.as_mut_ptr().add(i / 8 * n * 8 + t * 8), v);
            }
        }
        for t in (0..n).step_by(8) {
            for i in (0..64).step_by(8) {
                let rows = std::array::from_fn(|r| {
                    if t + r < n {
                        _mm256_loadu_ps(src.as_ptr().add((t + r) * 3 * d + head * 64 + d + i))
                    } else {
                        _mm256_setzero_ps()
                    }
                });
                let columns = transpose(rows);
                for j in 0..8 {
                    _mm256_storeu_ps(key.as_mut_ptr().add(t * 64 + (i + j) * 8), columns[j]);
                }
            }
        }
    }
}
#[target_feature(enable = "avx2")]
unsafe fn pack_query(src: &[f32], rows: usize, stride: usize, out: &mut [f32]) {
    unsafe {
        for row in (0..rows).step_by(8) {
            for i in (0..64).step_by(8) {
                let values = std::array::from_fn(|r| {
                    if row + r < rows {
                        _mm256_loadu_ps(src.as_ptr().add((row + r) * stride + i))
                    } else {
                        _mm256_setzero_ps()
                    }
                });
                let columns = transpose(values);
                for j in 0..8 {
                    _mm256_storeu_ps(out.as_mut_ptr().add(row * 64 + (i + j) * 8), columns[j]);
                }
            }
        }
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn query_key(q: &[f32], k: &[f32], scores: &mut [f32], blocks: usize, n: usize) {
    unsafe {
        for key in (0..n).step_by(8) {
            for block in 0..blocks {
                let mut sum = [_mm256_setzero_ps(); 8];
                for i in 0..64 {
                    let w = _mm256_loadu_ps(k.as_ptr().add(key * 64 + i * 8));
                    for r in 0..8 {
                        sum[r] = _mm256_fmadd_ps(
                            _mm256_broadcast_ss(&*q.as_ptr().add(block * 64 * 8 + i * 8 + r)),
                            w,
                            sum[r],
                        );
                    }
                }
                for v in &mut sum {
                    *v = _mm256_fmadd_ps(_mm256_set1_ps(0.125), *v, _mm256_setzero_ps());
                }
                let columns = transpose(sum);
                for j in 0..(n - key).min(8) {
                    _mm256_storeu_ps(
                        scores.as_mut_ptr().add(block * n * 8 + (key + j) * 8),
                        columns[j],
                    );
                }
            }
        }
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn normalize(scores: &mut [f32], n: usize) {
    unsafe {
        let mut max = _mm256_set1_ps(f32::NEG_INFINITY);
        for key in 0..n {
            max = _mm256_max_ps(max, _mm256_loadu_ps(scores.as_ptr().add(key * 8)));
        }
        let mut sum = _mm256_setzero_ps();
        let vector_keys = n / 4 * 4;
        for key in 0..n {
            let x = _mm256_sub_ps(_mm256_loadu_ps(scores.as_ptr().add(key * 8)), max);
            let e = if key < vector_keys {
                let lo = super::exp::negative4(_mm256_castps256_ps128(x));
                let hi = super::exp::negative4(_mm256_extractf128_ps::<1>(x));
                _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo), hi)
            } else {
                let mut lanes = [0.; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), x);
                for lane in &mut lanes {
                    *lane = lane.exp();
                }
                _mm256_loadu_ps(lanes.as_ptr())
            };
            sum = _mm256_add_ps(sum, e);
            _mm256_storeu_ps(scores.as_mut_ptr().add(key * 8), e);
        }
        for key in 0..n {
            _mm256_storeu_ps(
                scores.as_mut_ptr().add(key * 8),
                _mm256_div_ps(_mm256_loadu_ps(scores.as_ptr().add(key * 8)), sum),
            );
        }
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn probability_value(
    p: &[f32],
    v: &[f32],
    out: *mut f32,
    rows: usize,
    n: usize,
    stride: usize,
) {
    unsafe {
        for panel in 0..8 {
            for row in (0..rows).step_by(8) {
                for begin in (0..n).step_by(256) {
                    let mut sum = [_mm256_setzero_ps(); 8];
                    for key in begin..(begin + 256).min(n) {
                        let w = _mm256_loadu_ps(v.as_ptr().add(panel * n * 8 + key * 8));
                        for r in 0..8 {
                            sum[r] = _mm256_fmadd_ps(
                                _mm256_broadcast_ss(&*p.as_ptr().add(row * n + key * 8 + r)),
                                w,
                                sum[r],
                            );
                        }
                    }
                    for r in 0..(rows - row).min(8) {
                        let dst = out.add((row + r) * stride + panel * 8);
                        let previous = if begin == 0 {
                            _mm256_setzero_ps()
                        } else {
                            _mm256_loadu_ps(dst)
                        };
                        _mm256_storeu_ps(
                            dst,
                            _mm256_fmadd_ps(_mm256_set1_ps(1.), sum[r], previous),
                        );
                    }
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_parallel_softmax_matches_serial_row_reduction() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        let mut scratch = Vec::new();
        for (n, heads, tile) in [
            (17, 1, 16),
            (513, 1, 64),
            (2048, 1, 64),
            (2049, 1, 64),
            (33, 3, 16),
        ] {
            let d = heads * 64;
            let input: Vec<_> = (0..n * d * 3)
                .map(|i| (i as f32 * 0.017).sin() * 1.7)
                .collect();
            let mut expected = vec![0.; n * d];
            super::super::packed_attention::apply(
                &input,
                &mut expected,
                heads,
                None,
                tile,
                &mut Vec::new(),
                true,
                crate::laya::cpu::AttentionMath::TableExp,
                false,
            );
            let mut actual = vec![-17.; n * d + 2];
            apply(&input, &mut actual[1..n * d + 1], heads, tile, &mut scratch);
            let max = expected
                .iter()
                .zip(&actual[1..])
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(
                expected
                    .iter()
                    .zip(&actual[1..])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "n={n} heads={heads} max={max}"
            );
            assert_eq!(actual[0], -17.);
            assert_eq!(actual[n * d + 1], -17.);
        }
    }
}
