use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe {
        let mut sum = _mm256_setzero_ps();
        let end = a.len() / 8 * 8;
        for i in (0..end).step_by(8) {
            sum = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                sum,
            );
        }
        let mut lanes = [0.0; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        lanes.iter().sum::<f32>()
            + a[end..]
                .iter()
                .zip(&b[end..])
                .map(|(x, y)| x * y)
                .sum::<f32>()
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn add_scaled(dst: &mut [f32], src: &[f32], scale: f32) {
    assert_eq!(dst.len(), src.len());
    unsafe {
        let p = _mm256_set1_ps(scale);
        let end = dst.len() / 8 * 8;
        for i in (0..end).step_by(8) {
            let y = _mm256_fmadd_ps(
                p,
                _mm256_loadu_ps(src.as_ptr().add(i)),
                _mm256_loadu_ps(dst.as_ptr().add(i)),
            );
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), y);
        }
        for i in end..dst.len() {
            dst[i] += scale * src[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vector_reductions_and_tails() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for n in 0..80 {
            let a: Vec<_> = (0..n).map(|i| (i as f32).sin()).collect();
            let b: Vec<_> = (0..n).map(|i| (i as f32).cos()).collect();
            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((unsafe { dot(&a, &b) } - expected).abs() < 1e-5);
            let mut out = a.clone();
            unsafe { add_scaled(&mut out, &b, 0.25) };
            for i in 0..n {
                assert!((out[i] - (a[i] + 0.25 * b[i])).abs() < 1e-6);
            }
        }
    }
}
