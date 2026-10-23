use super::*;
use crate::laya::cpu::attention_scratch::HeadScratch;

pub(crate) fn apply(
    qkv: &[f32],
    out: &mut [f32],
    heads: usize,
    window: Option<usize>,
    query_tile: usize,
    scratch: &mut Vec<HeadScratch>,
    skip_masked: bool,
    math: crate::laya::cpu::AttentionMath,
    packed_normalization: bool,
) {
    apply_inner(
        qkv,
        out,
        heads,
        window,
        query_tile,
        scratch,
        skip_masked,
        math,
        packed_normalization,
        None,
    );
}
pub(crate) fn apply_selected(
    qkv: &[f32],
    out: &mut [f32],
    heads: usize,
    query_tile: usize,
    scratch: &mut Vec<HeadScratch>,
    math: crate::laya::cpu::AttentionMath,
    packed_normalization: bool,
    queries: &[usize],
) {
    apply_inner(
        qkv,
        out,
        heads,
        None,
        query_tile,
        scratch,
        true,
        math,
        packed_normalization,
        Some(queries),
    );
}
fn apply_inner(
    qkv: &[f32],
    out: &mut [f32],
    heads: usize,
    window: Option<usize>,
    query_tile: usize,
    scratch: &mut Vec<HeadScratch>,
    skip_masked: bool,
    math: crate::laya::cpu::AttentionMath,
    packed_normalization: bool,
    queries: Option<&[usize]>,
) {
    assert!(crate::laya::cpu::packed_supported() && heads > 0);
    let d = heads * 64;
    let n = qkv.len() / (3 * d);
    let m = out.len() / d;
    assert!(n > 0);
    assert_eq!(out.len(), m * d);
    assert_eq!(qkv.len(), n * 3 * d);
    if let Some(queries) = queries {
        assert!(window.is_none() && queries.len() == m && queries.iter().all(|&q| q < n));
    } else {
        assert_eq!(m, n);
    }
    assert!(query_tile > 0 && query_tile.is_multiple_of(16));
    assert!(window.is_none_or(|w| w.is_multiple_of(16)));
    let output = Output(out.as_mut_ptr());
    scratch.resize_with(heads, Default::default);
    scratch.par_iter_mut().enumerate().for_each(|(head, work)| {
        let key_len = n.div_ceil(16) * 16 * 64;
        if work.key.len() < key_len {
            work.key.resize(key_len, 0.);
        }
        if work.value.len() < n * 64 {
            work.value.resize(n * 64, 0.);
        }
        unsafe {
            pack_kv(qkv, d, head, n, &mut work.key, &mut work.value);
        }
        for start in (0..m).step_by(query_tile) {
            let rows = (m - start).min(query_tile);
            let lo = window.map_or(0, |w| start.saturating_sub(w));
            let hi = window.map_or(n, |w| (start + rows + w).min(n));
            let keys = hi - lo;
            let query_len = rows.div_ceil(6) * 6 * 64;
            if work.query.len() < query_len {
                work.query.resize(query_len, 0.);
            }
            let q = &mut work.query[..query_len];
            if let Some(queries) = queries {
                for (r, &index) in queries[start..start + rows].iter().enumerate() {
                    let src = &qkv[index * 3 * d + head * 64..][..64];
                    for i in 0..64 {
                        q[r / 6 * 6 * 64 + i * 6 + r % 6] = src[i];
                    }
                }
            } else {
                unsafe {
                    pack_rows(&qkv[start * 3 * d + head * 64..], rows, 64, 3 * d, q);
                }
            }
            let score_len = rows * keys;
            if work.scores.len() < score_len {
                work.scores.resize(score_len, 0.);
            }
            let scores = &mut work.scores[..score_len];
            unsafe {
                product(
                    q,
                    &work.key[lo * 64..key_len],
                    scores.as_mut_ptr(),
                    rows,
                    64,
                    keys,
                    keys,
                    64,
                    0.125,
                );
            }
            let probability_len = rows.div_ceil(6) * 6 * keys;
            if work.query.len() < probability_len {
                work.query.resize(probability_len, 0.);
            }
            let p = &mut work.query[..probability_len];
            if matches!(math, crate::laya::cpu::AttentionMath::TableExpRows)
                && window.is_none()
                && rows >= 12
            {
                unsafe {
                    pack_rows(scores, rows, keys, keys, p);
                }
                super::packed_softmax::normalize(p, rows, keys);
            } else {
                work.normalizers.resize(rows, 1.);
                for (r, scores) in scores.chunks_exact_mut(keys).enumerate() {
                    if let Some(w) = window {
                        let begin = (start + r).saturating_sub(w) - lo;
                        let end = (start + r + w + 1).min(n) - lo;
                        if skip_masked {
                            scores[..begin].fill(0.);
                            scores[end..].fill(0.);
                            work.normalizers[r] = normalize_for_pack(
                                &mut scores[begin..end],
                                math,
                                packed_normalization,
                            );
                            continue;
                        }
                        scores[..begin].fill(f32::NEG_INFINITY);
                        scores[end..].fill(f32::NEG_INFINITY);
                    }
                    work.normalizers[r] = normalize_for_pack(scores, math, packed_normalization);
                }
                unsafe {
                    if packed_normalization {
                        pack_normalized(scores, rows, keys, &work.normalizers, p);
                    } else {
                        pack_rows(scores, rows, keys, keys, p);
                    }
                }
            }
            unsafe {
                product(
                    p,
                    &work.value[lo * 16..n * 64],
                    output.ptr().add(start * d + head * 64),
                    rows,
                    keys,
                    64,
                    d,
                    n,
                    1.0,
                );
            }
        }
    });
}
#[inline]
fn normalize(scores: &mut [f32], math: crate::laya::cpu::AttentionMath) {
    match math {
        crate::laya::cpu::AttentionMath::Scalar => crate::laya::cpu::softmax(scores),
        crate::laya::cpu::AttentionMath::ExactSimd => super::softmax::exact(scores),
        crate::laya::cpu::AttentionMath::TableExp => super::softmax::table_exp(scores),
        crate::laya::cpu::AttentionMath::TableExpSum => super::softmax::table_exp_sum(scores),
        crate::laya::cpu::AttentionMath::TableExpRows => super::softmax::table_exp(scores),
        crate::laya::cpu::AttentionMath::PolynomialExpSum => {
            super::softmax::polynomial_exp_sum(scores)
        }
    }
}
#[inline]
fn normalize_for_pack(
    scores: &mut [f32],
    math: crate::laya::cpu::AttentionMath,
    packed: bool,
) -> f32 {
    if packed {
        match math {
            crate::laya::cpu::AttentionMath::TableExp => {
                return super::softmax::table_scale(scores, false);
            }
            crate::laya::cpu::AttentionMath::TableExpSum => {
                return super::softmax::table_scale(scores, true);
            }
            crate::laya::cpu::AttentionMath::PolynomialExpSum => {
                return super::softmax::polynomial_scale(scores);
            }
            _ => {}
        }
    }
    normalize(scores, math);
    1.
}
#[target_feature(enable = "avx2")]
unsafe fn pack_normalized(src: &[f32], rows: usize, k: usize, sums: &[f32], dst: &mut [f32]) {
    unsafe {
        for row in (0..rows).step_by(6) {
            let count = (rows - row).min(6);
            let end = if count == 6 { k / 8 * 8 } else { 0 };
            if count == 6 {
                let scale: [__m256; 6] = std::array::from_fn(|r| _mm256_set1_ps(sums[row + r]));
                for i in (0..end).step_by(8) {
                    let v = std::array::from_fn(|r| {
                        _mm256_div_ps(
                            _mm256_loadu_ps(src.as_ptr().add((row + r) * k + i)),
                            scale[r],
                        )
                    });
                    super::input::interleave(v, dst.as_mut_ptr().add(row * k + i * 6));
                }
            }
            for i in end..k {
                for r in 0..count {
                    dst[row * k + i * 6 + r] = src[(row + r) * k + i] / sums[row + r];
                }
            }
        }
    }
}
#[target_feature(enable = "avx2")]
pub(super) unsafe fn pack_rows(src: &[f32], rows: usize, k: usize, stride: usize, dst: &mut [f32]) {
    unsafe {
        for row in (0..rows).step_by(6) {
            let count = (rows - row).min(6);
            let end = if count == 6 { k / 8 * 8 } else { 0 };
            for i in (0..end).step_by(8) {
                let v = std::array::from_fn(|r| {
                    _mm256_loadu_ps(src.as_ptr().add((row + r) * stride + i))
                });
                super::input::interleave(v, dst.as_mut_ptr().add(row * k + i * 6));
            }
            for i in end..k {
                for r in 0..count {
                    dst[row * k + i * 6 + r] = src[(row + r) * stride + i];
                }
            }
        }
    }
}
#[target_feature(enable = "avx2")]
pub(super) unsafe fn pack_kv(
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
                _mm256_storeu_ps(value.as_mut_ptr().add(i / 16 * n * 16 + t * 16 + i % 16), v);
            }
        }
        let end = n / 8 * 8;
        for t in (0..end).step_by(8) {
            for i in (0..64).step_by(8) {
                let v: [__m256; 8] = std::array::from_fn(|r| {
                    _mm256_loadu_ps(src.as_ptr().add((t + r) * 3 * d + head * 64 + d + i))
                });
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
                for j in 0..4 {
                    let dst = key
                        .as_mut_ptr()
                        .add(t / 16 * 16 * 64 + (i + j) * 16 + t % 16);
                    _mm256_storeu_ps(dst, _mm256_permute2f128_ps::<0x20>(b[j], b[j + 4]));
                    _mm256_storeu_ps(
                        dst.add(4 * 16),
                        _mm256_permute2f128_ps::<0x31>(b[j], b[j + 4]),
                    );
                }
            }
        }
        for t in end..n {
            for i in 0..64 {
                key[t / 16 * 16 * 64 + i * 16 + t % 16] = src[t * 3 * d + head * 64 + d + i];
            }
        }
    }
}
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn product(
    a: &[f32],
    b: &[f32],
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    stride: usize,
    b_stride: usize,
    scale: f32,
) {
    assert_eq!(a.len(), m.div_ceil(6) * 6 * k);
    assert!(k > 0 && n > 0 && b.len() >= (n.div_ceil(16) - 1) * b_stride * 16 + k * 16);
    for panel in 0..n.div_ceil(16) {
        for row in (0..m).step_by(6) {
            for begin in (0..k).step_by(256) {
                unsafe {
                    let a = a.as_ptr().add(row * k + begin * 6);
                    let b = b.as_ptr().add(panel * b_stride * 16 + begin * 16);
                    let c = c.add(row * stride + panel * 16);
                    let width = (n - panel * 16).min(16);
                    let count = (k - begin).min(256);
                    macro_rules! run {
                        ($r:literal) => {
                            tile::<$r>(a, b, c, count, stride, width, scale, begin != 0)
                        };
                    }
                    match (m - row).min(6) {
                        1 => run!(1),
                        2 => run!(2),
                        3 => run!(3),
                        4 => run!(4),
                        5 => run!(5),
                        6 => run!(6),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
}
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile<const ROWS: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    stride: usize,
    width: usize,
    scale: f32,
    accumulate: bool,
) {
    unsafe {
        let mut lo = [_mm256_setzero_ps(); ROWS];
        let mut hi = lo;
        for t in 0..k {
            let w0 = _mm256_loadu_ps(b.add(t * 16));
            let w1 = _mm256_loadu_ps(b.add(t * 16 + 8));
            for r in 0..ROWS {
                let x = _mm256_broadcast_ss(&*a.add(t * 6 + r));
                lo[r] = _mm256_fmadd_ps(x, w0, lo[r]);
                hi[r] = _mm256_fmadd_ps(x, w1, hi[r]);
            }
        }
        let scale = _mm256_set1_ps(scale);
        let zero = _mm256_setzero_ps();
        let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let low_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(width as i32), lanes);
        let high_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(width as i32 - 8), lanes);
        for r in 0..ROWS {
            let mut old_lo = zero;
            let mut old_hi = zero;
            if accumulate {
                old_lo = _mm256_maskload_ps(c.add(r * stride), low_mask);
                if width > 8 {
                    old_hi = _mm256_maskload_ps(c.add(r * stride + 8), high_mask);
                }
            }
            lo[r] = _mm256_fmadd_ps(scale, lo[r], old_lo);
            hi[r] = _mm256_fmadd_ps(scale, hi[r], old_hi);
        }
        if width == 16 {
            for r in 0..ROWS {
                _mm256_storeu_ps(c.add(r * stride), lo[r]);
                _mm256_storeu_ps(c.add(r * stride + 8), hi[r]);
            }
        } else {
            for r in 0..ROWS {
                _mm256_maskstore_ps(c.add(r * stride), low_mask, lo[r]);
                if width > 8 {
                    _mm256_maskstore_ps(c.add(r * stride + 8), high_mask, hi[r]);
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_probability_packing_preserves_bits_and_vector_sum_is_bounded() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        let mut scratch = Vec::new();
        for n in [17, 129, 512, 2049, 31] {
            for window in [None, Some(64)] {
                let qkv: Vec<_> = (0..n * 64 * 3)
                    .map(|i| (i as f32 * 0.071).sin() * 1.7)
                    .collect();
                let mut expected = vec![0.; n * 64];
                apply(
                    &qkv,
                    &mut expected,
                    1,
                    window,
                    64,
                    &mut Vec::new(),
                    true,
                    crate::laya::cpu::AttentionMath::TableExp,
                    false,
                );
                let mut actual = vec![-13.; n * 64 + 2];
                apply(
                    &qkv,
                    &mut actual[1..n * 64 + 1],
                    1,
                    window,
                    64,
                    &mut scratch,
                    true,
                    crate::laya::cpu::AttentionMath::TableExp,
                    true,
                );
                assert!(
                    expected
                        .iter()
                        .zip(&actual[1..])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "n={n} window={window:?}"
                );
                apply(
                    &qkv,
                    &mut actual[1..n * 64 + 1],
                    1,
                    window,
                    64,
                    &mut scratch,
                    true,
                    crate::laya::cpu::AttentionMath::TableExpSum,
                    true,
                );
                let error = expected
                    .iter()
                    .zip(&actual[1..])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f32::max);
                assert!(error <= 1e-5, "n={n} window={window:?} error={error}");
                assert_eq!(actual[0], -13.);
                assert_eq!(actual[n * 64 + 1], -13.);
            }
        }
    }
    #[test]
    fn packed_attention_matches_existing_tiled_path() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        let mut scratch = Vec::new();
        for heads in [1, 3] {
            for query_tile in [16, 64, 256] {
                for n in [512, 257, 64, 17, 31, 32, 33, 1, 7, 16, 63, 97] {
                    for window in [None, Some(0), Some(16), Some(64)] {
                        let d = heads * 64;
                        let qkv: Vec<_> = (0..n * d * 3)
                            .map(|i| (i as f32 * 0.017).sin() * 0.7)
                            .collect();
                        let mut expected = vec![0.; n * d];
                        crate::laya::cpu::tiled_attention::apply(
                            &qkv,
                            &mut expected,
                            heads,
                            64,
                            window,
                            query_tile,
                            &mut Vec::new(),
                        );
                        let mut got = vec![f32::NAN; n * d + 2];
                        got[0] = -17.;
                        got[n * d + 1] = 31.;
                        apply(
                            &qkv,
                            &mut got[1..n * d + 1],
                            heads,
                            window,
                            query_tile,
                            &mut scratch,
                            false,
                            crate::laya::cpu::AttentionMath::Scalar,
                            false,
                        );
                        assert!(
                            got[1..n * d + 1]
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "n={n} heads={heads} tile={query_tile} window={window:?}, max={}",
                            got[1..n * d + 1]
                                .iter()
                                .zip(&expected)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0., f32::max)
                        );
                        assert_eq!(got[0], -17.);
                        assert_eq!(got[n * d + 1], 31.);
                    }
                }
            }
        }
    }

    #[test]
    fn long_attention_and_sparse_softmax_preserve_bits() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        let mut scratch = Vec::new();
        for n in [512, 1024, 2048, 2049, 513, 1025, 17] {
            let heads = 1;
            let d = 64;
            let qkv: Vec<_> = (0..n * d * 3)
                .map(|i| (i as f32 * 0.017).sin() * 0.7)
                .collect();
            for window in [None, Some(0), Some(64)] {
                for query_tile in [16, 64, 256] {
                    let mut expected = vec![0.; n * d];
                    crate::laya::cpu::tiled_attention::apply(
                        &qkv,
                        &mut expected,
                        heads,
                        64,
                        window,
                        query_tile,
                        &mut Vec::new(),
                    );
                    let mut got = vec![f32::NAN; n * d + 2];
                    got[0] = -17.;
                    got[n * d + 1] = 31.;
                    apply(
                        &qkv,
                        &mut got[1..n * d + 1],
                        heads,
                        window,
                        query_tile,
                        &mut scratch,
                        true,
                        crate::laya::cpu::AttentionMath::ExactSimd,
                        false,
                    );
                    assert!(
                        got[1..n * d + 1]
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "n={n} tile={query_tile} window={window:?}, max={}",
                        got[1..n * d + 1]
                            .iter()
                            .zip(&expected)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0., f32::max)
                    );
                    assert_eq!(got[0], -17.);
                    assert_eq!(got[n * d + 1], 31.);
                }
            }
        }
    }
}
