use crate::laya::cpu::Norm;
use rayon::prelude::*;
use std::arch::x86_64::*;

pub(super) fn plain_rows(x: &[f32], k: usize, out: &mut [f32]) {
    let rows = x.len() / k;
    assert!(rows <= 6 && x.len() == rows * k && out.len() == 6 * k);
    let end = if rows == 6 { k / 8 * 8 } else { 0 };
    if end > 0 {
        unsafe {
            plain(x, k, out, end);
        }
    }
    for i in end..k {
        for row in 0..rows {
            out[i * 6 + row] = x[row * k + i];
        }
    }
}

pub(crate) fn prepare(x: &[f32], k: usize, output: &mut Vec<f32>, norm: Option<&Norm>) -> usize {
    assert!(crate::laya::cpu::packed_supported());
    assert!(k > 0 && x.len() % k == 0);
    if let Some(n) = norm {
        assert_eq!(n.weight.len(), k);
        assert!(n.bias.as_ref().is_none_or(|b| b.len() == k));
    }
    let rows = x.len() / k;
    let len = rows.div_ceil(6) * 6 * k;
    if output.len() < len {
        output.resize(len, 0.);
    }
    let run = |(src, dst): (&[f32], &mut [f32])| {
        let rows = src.len() / k;
        let mut mean = [0.; 6];
        let mut scale = [1.; 6];
        if let Some(norm) = norm {
            for (r, row) in src.chunks_exact(k).enumerate() {
                let m = row.iter().map(|&v| v as f64).sum::<f64>() / k as f64;
                let var = row.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / k as f64;
                mean[r] = m as f32;
                scale[r] = (var as f32 + norm.eps).sqrt().recip();
            }
        }
        let end = if rows == 6 { k / 8 * 8 } else { 0 };
        if end > 0 {
            unsafe {
                if let Some(norm) = norm {
                    normalized(
                        src,
                        k,
                        dst,
                        &norm.weight,
                        norm.bias.as_deref(),
                        &mean,
                        &scale,
                        end,
                    );
                } else {
                    plain(src, k, dst, end);
                }
            }
        }
        for i in end..k {
            for r in 0..rows {
                dst[i * 6 + r] = if let Some(norm) = norm {
                    (src[r * k + i] - mean[r]) * scale[r] * norm.weight[i]
                        + norm.bias.as_ref().map_or(0., |b| b[i])
                } else {
                    src[r * k + i]
                };
            }
        }
    };
    let dst = &mut output[..len];
    if rows >= 96 {
        x.par_chunks(6 * k)
            .zip(dst.par_chunks_mut(6 * k))
            .with_min_len(2)
            .for_each(run);
    } else {
        x.chunks(6 * k).zip(dst.chunks_mut(6 * k)).for_each(run);
    }
    len
}
#[target_feature(enable = "avx2")]
unsafe fn plain(x: &[f32], k: usize, out: &mut [f32], end: usize) {
    unsafe {
        for i in (0..end).step_by(8) {
            let v = std::array::from_fn(|r| _mm256_loadu_ps(x.as_ptr().add(r * k + i)));
            interleave(v, out.as_mut_ptr().add(i * 6));
        }
    }
}
#[target_feature(enable = "avx2")]
unsafe fn normalized(
    x: &[f32],
    k: usize,
    out: &mut [f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    mean: &[f32; 6],
    scale: &[f32; 6],
    end: usize,
) {
    unsafe {
        for i in (0..end).step_by(8) {
            let w = _mm256_loadu_ps(weight.as_ptr().add(i));
            let b = bias.map_or(_mm256_setzero_ps(), |b| _mm256_loadu_ps(b.as_ptr().add(i)));
            let v = std::array::from_fn(|r| {
                let x = _mm256_loadu_ps(x.as_ptr().add(r * k + i));
                _mm256_add_ps(
                    _mm256_mul_ps(
                        _mm256_mul_ps(
                            _mm256_sub_ps(x, _mm256_set1_ps(mean[r])),
                            _mm256_set1_ps(scale[r]),
                        ),
                        w,
                    ),
                    b,
                )
            });
            interleave(v, out.as_mut_ptr().add(i * 6));
        }
    }
}
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn interleave(v: [__m256; 6], out: *mut f32) {
    unsafe {
        let a = _mm256_unpacklo_ps(v[0], v[1]);
        let b = _mm256_unpackhi_ps(v[0], v[1]);
        let c = _mm256_unpacklo_ps(v[2], v[3]);
        let d = _mm256_unpackhi_ps(v[2], v[3]);
        let p = [
            _mm256_shuffle_ps::<0x44>(a, c),
            _mm256_shuffle_ps::<0xee>(a, c),
            _mm256_shuffle_ps::<0x44>(b, d),
            _mm256_shuffle_ps::<0xee>(b, d),
        ];
        for i in 0..4 {
            _mm_storeu_ps(out.add(i * 6), _mm256_castps256_ps128(p[i]));
            _mm_storeu_ps(out.add((i + 4) * 6), _mm256_extractf128_ps::<1>(p[i]));
        }
        for (g, p) in [
            _mm256_unpacklo_ps(v[4], v[5]),
            _mm256_unpackhi_ps(v[4], v[5]),
        ]
        .iter()
        .enumerate()
        {
            let lo = _mm_castps_si128(_mm256_castps256_ps128(*p));
            let hi = _mm_castps_si128(_mm256_extractf128_ps::<1>(*p));
            _mm_storel_epi64(out.add(g * 12 + 4).cast(), lo);
            _mm_storel_epi64(out.add(g * 12 + 10).cast(), _mm_srli_si128::<8>(lo));
            _mm_storel_epi64(out.add(g * 12 + 28).cast(), hi);
            _mm_storel_epi64(out.add(g * 12 + 34).cast(), _mm_srli_si128::<8>(hi));
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interleaved_inputs_preserve_bits_and_retained_storage() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        let mut got = Vec::new();
        for k in [1024, 7, 64, 11, 2624, 1] {
            for m in [512, 31, 6, 1, 112, 5, 0] {
                let x: Vec<_> = (0..m * k)
                    .map(|i| {
                        if i % 31 == 0 {
                            -0.
                        } else {
                            (i as f32 * 0.05).sin()
                        }
                    })
                    .collect();
                let prior = got.len();
                let len = prepare(&x, k, &mut got, None);
                assert_eq!(len, m.div_ceil(6) * 6 * k);
                assert!(got.len() >= prior);
                for r in 0..m {
                    for i in 0..k {
                        assert_eq!(
                            got[r / 6 * 6 * k + i * 6 + r % 6].to_bits(),
                            x[r * k + i].to_bits()
                        );
                    }
                }
            }
        }
    }
}
