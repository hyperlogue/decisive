use super::int8::{GROUP, group_scale_bits, pack_bits};

pub(super) const NF4: [f32; 16] = [
    -1.,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.28444138,
    -0.18477343,
    -0.091050036,
    0.,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.,
];




pub(super) fn mse(x: &[f32], out: &mut [i8], avx2: bool) -> f32 {
    assert!(!x.is_empty() && x.len() <= GROUP && out.len() == x.len());
    let maximum = x.iter().fold(0_f32, |a, v| a.max(v.abs()));
    let baseline = group_scale_bits(x, avx2, 4);
    pack_bits(x, out, baseline, avx2, 4);
    if maximum == 0. {
        return baseline;
    }
    let mut normalized = [0.; GROUP];
    for (v, &x) in normalized.iter_mut().zip(x) {
        *v = x / maximum;
    }
    let values = &normalized[..x.len()];
    let mut best_error: f32 = values
        .iter()
        .zip(out.iter())
        .map(|(x, q)| {
            let error = x - *q as f32 * (baseline / maximum);
            error * error
        })
        .sum();
    let mut best = baseline;
    for sign in [-1., 1.] {
        for ratio in [0.8, 0.95, 1.1] {
            let mut scale = sign * ratio / 7.;
            for _ in 0..4 {
                let (numerator, denominator, error) = stats(values, scale, avx2);
                if error < best_error {
                    best_error = error;
                    best = scale * maximum;
                    for (code, &value) in out.iter_mut().zip(values) {
                        *code = (value * scale.recip()).round_ties_even().clamp(-8., 7.) as i8;
                    }
                }
                if denominator == 0. {
                    break;
                }
                let next = numerator / denominator;
                if !next.is_finite() || next.abs() < f32::MIN_POSITIVE || next == scale {
                    break;
                }
                scale = next;
            }
        }
    }
    best
}

fn stats(x: &[f32], scale: f32, avx2: bool) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return unsafe { stats_avx2(x, scale) };
    }
    let _ = avx2;
    let mut result = (0., 0., 0.);
    for &value in x {
        let q = (value * scale.recip()).round_ties_even().clamp(-8., 7.);
        let error = value - q * scale;
        result.0 += value * q;
        result.1 += q * q;
        result.2 += error * error;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stats_avx2(x: &[f32], scale: f32) -> (f32, f32, f32) {
    use std::arch::x86_64::*;
    unsafe {
        let inverse = _mm256_set1_ps(scale.recip());
        let scale_vector = _mm256_set1_ps(scale);
        let mut numerator = _mm256_setzero_ps();
        let mut denominator = _mm256_setzero_ps();
        let mut error_sum = _mm256_setzero_ps();
        let end = x.len() / 8 * 8;
        for i in (0..end).step_by(8) {
            let value = _mm256_loadu_ps(x.as_ptr().add(i));
            let q = _mm256_min_ps(
                _mm256_set1_ps(7.),
                _mm256_max_ps(
                    _mm256_set1_ps(-8.),
                    _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                        _mm256_mul_ps(value, inverse),
                    ),
                ),
            );
            let error = _mm256_sub_ps(value, _mm256_mul_ps(q, scale_vector));
            numerator = _mm256_add_ps(numerator, _mm256_mul_ps(value, q));
            denominator = _mm256_add_ps(denominator, _mm256_mul_ps(q, q));
            error_sum = _mm256_add_ps(error_sum, _mm256_mul_ps(error, error));
        }
        let sum = |v| {
            let mut lanes = [0.; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), v);
            lanes.iter().sum::<f32>()
        };
        let tail = stats(&x[end..], scale, false);
        (
            sum(numerator) + tail.0,
            sum(denominator) + tail.1,
            sum(error_sum) + tail.2,
        )
    }
}

fn nf4_code(value: f32) -> usize {
    let mut low = 0;
    let mut high = 15;
    while low < high {
        let middle = (low + high) / 2;
        if value > (NF4[middle] + NF4[middle + 1]) * 0.5 {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}



pub(super) fn affine(x: &[f32], out: &mut [i8], avx2: bool) -> (f32, f32) {
    assert!(!x.is_empty() && x.len() <= GROUP && out.len() == x.len());
    let maximum = x.iter().fold(0_f32, |a, v| a.max(v.abs()));
    if maximum == 0. {
        out.fill(0);
        return (f32::MIN_POSITIVE, 0.);
    }
    let mut normalized = [0.; GROUP];
    for (v, &x) in normalized.iter_mut().zip(x) {
        *v = x / maximum;
    }
    let values = &normalized[..x.len()];
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if low == high {
        out.fill(0);
        return (f32::MIN_POSITIVE, x[0]);
    }
    let mean = values.iter().sum::<f32>() / x.len() as f32;
    let mut best_error = f32::INFINITY;
    let mut best = (0., 0.);
    for ratio in [0.8, 1., 1.2] {
        let mut scale = (high - low) / 15. * ratio;
        let mut offset = (high + low + scale) * 0.5;
        for _ in 0..4 {
            let (vq, qq, error, qsum) = affine_stats(values, scale, offset, avx2);
            if error < best_error {
                best_error = error;
                best = (scale * maximum, offset * maximum);
                for (code, &value) in out.iter_mut().zip(values) {
                    *code = ((value - offset) * scale.recip())
                        .round_ties_even()
                        .clamp(-8., 7.) as i8;
                }
            }
            let mean_q = qsum / x.len() as f32;
            let denominator = qq - qsum * mean_q;
            if denominator <= 0. {
                break;
            }
            let next = (vq - qsum * mean) / denominator;
            if !next.is_finite() || next <= 0. {
                break;
            }
            let next_offset = mean - next * mean_q;
            if next == scale && next_offset == offset {
                break;
            }
            scale = next;
            offset = next_offset;
        }
    }
    best
}

fn affine_stats(x: &[f32], scale: f32, offset: f32, avx2: bool) -> (f32, f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return unsafe { affine_stats_avx2(x, scale, offset) };
    }
    let _ = avx2;
    let mut sums = (0., 0., 0., 0.);
    for &v in x {
        let q = ((v - offset) * scale.recip())
            .round_ties_even()
            .clamp(-8., 7.);
        let error = v - (q * scale + offset);
        sums.0 += v * q;
        sums.1 += q * q;
        sums.2 += error * error;
        sums.3 += q;
    }
    sums
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn affine_stats_avx2(x: &[f32], scale: f32, offset: f32) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::*;
    unsafe {
        let inverse = _mm256_set1_ps(scale.recip());
        let sv = _mm256_set1_ps(scale);
        let ov = _mm256_set1_ps(offset);
        let mut sums = [_mm256_setzero_ps(); 4];
        let end = x.len() / 8 * 8;
        for i in (0..end).step_by(8) {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            let q = _mm256_min_ps(
                _mm256_set1_ps(7.),
                _mm256_max_ps(
                    _mm256_set1_ps(-8.),
                    _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                        _mm256_mul_ps(_mm256_sub_ps(v, ov), inverse),
                    ),
                ),
            );
            let e = _mm256_sub_ps(v, _mm256_add_ps(_mm256_mul_ps(q, sv), ov));
            sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(v, q));
            sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(q, q));
            sums[2] = _mm256_add_ps(sums[2], _mm256_mul_ps(e, e));
            sums[3] = _mm256_add_ps(sums[3], q);
        }
        let tail = affine_stats(&x[end..], scale, offset, false);
        let mut result = [tail.0, tail.1, tail.2, tail.3];
        for (out, v) in result.iter_mut().zip(sums) {
            let mut lanes = [0.; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), v);
            *out += lanes.iter().sum::<f32>();
        }
        (result[0], result[1], result[2], result[3])
    }
}

pub(super) fn nf4(x: &[f32], out: &mut [i8]) -> f32 {
    assert!(!x.is_empty() && x.len() <= GROUP && out.len() == x.len());
    let maximum = x.iter().fold(0_f32, |a, v| a.max(v.abs()));
    if maximum == 0. {
        out.fill(-1);
        return f32::MIN_POSITIVE;
    }
    let mut values = [0.; GROUP];
    for (v, &x) in values.iter_mut().zip(x) {
        *v = x / maximum;
    }
    let mut best_error = f32::INFINITY;
    let mut best_scale = maximum;
    for initial in [0.85, 1., 1.15] {
        let mut scale: f32 = initial;
        for _ in 0..4 {
            let mut codes = [0; GROUP];
            let (mut numerator, mut denominator, mut error) = (0., 0., 0.);
            for (i, &v) in values[..x.len()].iter().enumerate() {
                let index = nf4_code(v / scale);
                codes[i] = index as i8 - 8;
                let q = NF4[index];
                numerator += v * q;
                denominator += q * q;
                error += (v - scale * q).powi(2);
            }
            if error < best_error {
                best_error = error;
                best_scale = scale * maximum;
                out.copy_from_slice(&codes[..x.len()]);
            }
            if denominator == 0. {
                break;
            }
            let next = numerator / denominator;
            if next == scale || next <= 0. {
                break;
            }
            scale = next;
        }
    }
    best_scale
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_preserves_constants_and_fits_offset_grids() {
        for avx2 in [false, super::super::cpu::avx2_available()] {
            for n in [1, 7, 16, 63, 64] {
                let mut codes = vec![0; n];
                for v in [0., -3.25, 12.] {
                    let (s, b) = affine(&vec![v; n], &mut codes, avx2);
                    assert!(codes.iter().all(|&q| q as f32 * s + b == v));
                }
                let values: Vec<_> = (0..n)
                    .map(|i| 4. + (i * 15 / (n - 1).max(1)) as f32 * 0.25)
                    .collect();
                let (s, b) = affine(&values, &mut codes, avx2);
                for (&v, &q) in values.iter().zip(&codes) {
                    assert!((v - (q as f32 * s + b)).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn fitting_reduces_reconstruction_error_and_handles_tails() {
        for avx2 in [false, super::super::cpu::avx2_available()] {
            for n in [1, 7, 8, 31, 63, 64] {
                for seed in 0..32 {
                    let values: Vec<_> = (0..n)
                        .map(|i| {
                            let v = ((i * 137 + seed * 19) % 101) as f32 - 50.;
                            if i == 0 { v * 3.7 } else { v }
                        })
                        .collect();
                    let mut base = vec![0; n];
                    let s = group_scale_bits(&values, avx2, 4);
                    pack_bits(&values, &mut base, s, avx2, 4);
                    let mut codes = vec![0; n];
                    let fit = mse(&values, &mut codes, avx2);
                    let error = |q: &[i8], scale: f32| {
                        values
                            .iter()
                            .zip(q)
                            .map(|(&v, &q)| (v as f64 - q as f64 * scale as f64).powi(2))
                            .sum::<f64>()
                    };
                    assert!(error(&codes, fit) <= error(&base, s) + 1e-5);
                    assert!(codes.iter().all(|&q| (-8..=7).contains(&q)));
                }
            }
            let mut codes = [42; 64];
            assert!(mse(&[0.; 64], &mut codes, avx2).is_finite());
            assert_eq!(codes, [0; 64]);
        }
    }

    #[test]
    fn nf4_roundtrips_codebook_and_zero() {
        let mut codes = [0; 16];
        let scale = nf4(&NF4, &mut codes);
        for (i, code) in codes.iter().enumerate() {
            assert!((NF4[(*code + 8) as usize] * scale - NF4[i]).abs() < 1e-6);
        }
        let scale = nf4(&[0.; 16], &mut codes);
        assert!(scale.is_finite());
        assert!(codes.iter().all(|&q| NF4[(q + 8) as usize] == 0.));
    }
}
