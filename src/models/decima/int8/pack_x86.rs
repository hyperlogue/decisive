use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(super) unsafe fn row(input: &[f32], block: usize, output: &mut [i8], scales: &mut [f32]) {
    unsafe {
        output.fill(0);
        for (g, values) in input.chunks(block).enumerate() {
            let end = values.len() / 8 * 8;
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
            let mut maximum = _mm256_setzero_ps();
            for k in (0..end).step_by(8) {
                maximum = _mm256_max_ps(
                    maximum,
                    _mm256_and_ps(_mm256_loadu_ps(values.as_ptr().add(k)), mask),
                );
            }
            let halves = _mm_max_ps(
                _mm256_castps256_ps128(maximum),
                _mm256_extractf128_ps::<1>(maximum),
            );
            let pairs = _mm_max_ps(halves, _mm_movehl_ps(halves, halves));
            let scalar = _mm_max_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs));
            let mut magnitude = _mm_cvtss_f32(scalar);
            for &v in &values[end..] {
                magnitude = magnitude.max(v.abs());
            }
            let scale = (magnitude / 127.).max(f32::MIN_POSITIVE);
            scales[g] = scale;
            let destination = output.as_mut_ptr().add(g * block);
            for k in (0..end).step_by(8) {
                let value = _mm256_div_ps(
                    _mm256_loadu_ps(values.as_ptr().add(k)),
                    _mm256_set1_ps(scale),
                );
                let truncated = _mm256_cvttps_epi32(value);
                let fraction =
                    _mm256_and_ps(_mm256_sub_ps(value, _mm256_cvtepi32_ps(truncated)), mask);
                let increment = _mm256_or_si256(
                    _mm256_srai_epi32::<31>(_mm256_castps_si256(value)),
                    _mm256_set1_epi32(1),
                );
                let correction = _mm256_and_si256(
                    increment,
                    _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GE_OQ>(fraction, _mm256_set1_ps(0.5))),
                );
                let rounded = _mm256_add_epi32(truncated, correction);
                let shorts = _mm_packs_epi32(
                    _mm256_castsi256_si128(rounded),
                    _mm256_extracti128_si256::<1>(rounded),
                );
                let bytes = _mm_packs_epi16(shorts, _mm_setzero_si128());
                _mm_storel_epi64(destination.add(k).cast(), bytes);
            }
            for k in end..values.len() {
                *destination.add(k) = super::quantize(values[k], scale);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vector_rounding_and_scales_match_scalar_bitwise() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut values = vec![-127., 127., 0., -0., f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
        for i in -127..127 {
            let half = i as f32 + 0.5;
            values.extend([half.next_down(), half, half.next_up()]);
        }
        for i in 0..4096_u32 {
            let bits = (i.wrapping_mul(747796405).wrapping_add(2891336453)) & 0xff7f_ffff;
            values.push(f32::from_bits(bits));
        }
        for block in [16, 32, 64, 128] {
            for length in [1, 7, 8, 9, 15, 17, 31, 33, values.len()] {
                let input = &values[..length];
                let mut out = vec![i8::MIN; length.div_ceil(block) * block];
                let mut scales = vec![f32::NAN; length.div_ceil(block)];
                unsafe {
                    row(input, block, &mut out, &mut scales);
                }
                for (g, group) in input.chunks(block).enumerate() {
                    let s = super::super::scale(group);
                    assert_eq!(s.to_bits(), scales[g].to_bits());
                    for (k, &v) in group.iter().enumerate() {
                        assert_eq!(out[g * block + k], super::super::quantize(v, s), "{v}, {s}");
                    }
                }
                assert!(out[length..].iter().all(|&q| q == 0));
            }
        }
    }
}
