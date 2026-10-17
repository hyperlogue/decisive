use std::arch::x86_64::*;

pub(super) fn exact(x: &mut [f32]) {
    assert!(crate::laya::cpu::packed_supported());
    let max = unsafe { maximum(x) };
    let mut sum = 0.;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    unsafe {
        divide(x, sum);
    }
}

pub(super) fn table_exp(x: &mut [f32]) {
    let sum = table_scale(x, false);
    unsafe {
        divide(x, sum);
    }
}
pub(super) fn table_exp_sum(x: &mut [f32]) {
    let sum = table_scale(x, true);
    unsafe {
        divide(x, sum);
    }
}
pub(super) fn polynomial_exp_sum(x: &mut [f32]) {
    let sum = polynomial_scale(x);
    unsafe {
        divide(x, sum);
    }
}
pub(super) fn polynomial_scale(x: &mut [f32]) -> f32 {
    scale(x, true, true).1
}
pub(super) fn table_scale(x: &mut [f32], vector_sum: bool) -> f32 {
    scale(x, vector_sum, false).1
}
pub(super) fn raw_stats(x: &mut [f32], math: crate::laya::cpu::AttentionMath) -> (f32, f32) {
    use crate::laya::cpu::AttentionMath::*;
    match math {
        TableExp | TableExpRows => scale(x, false, false),
        TableExpSum => scale(x, true, false),
        PolynomialExpSum => scale(x, true, true),
        Scalar | ExactSimd => {
            let max = unsafe { maximum(x) };
            let mut sum = 0.;
            for v in x {
                *v = (*v - max).exp();
                sum += *v;
            }
            (max, sum)
        }
    }
}
fn scale(x: &mut [f32], vector_sum: bool, polynomial: bool) -> (f32, f32) {
    assert!(crate::laya::cpu::packed_supported());
    let max = unsafe { maximum(x) };
    if !max.is_finite() {
        crate::laya::cpu::softmax(x);
        return (max, 1.);
    }
    unsafe {
        if polynomial {
            polynomial_exponentials(x, max);
        } else {
            exponentials(x, max);
        }
    }
    if vector_sum {
        return (max, unsafe { sum(x) });
    }
    let mut total = 0.;
    for &v in x.iter() {
        total += v;
    }
    (max, total)
}
#[target_feature(enable = "avx2,fma")]
unsafe fn polynomial_exponentials(x: &mut [f32], max: f32) {
    unsafe {
        let end = x.len() / 8 * 8;
        let maximum = _mm256_set1_ps(max);
        for i in (0..end).step_by(8) {
            let difference = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), maximum);
            _mm256_storeu_ps(x.as_mut_ptr().add(i), super::exp::negative8(difference));
        }
        for v in &mut x[end..] {
            *v = (*v - max).exp();
        }
    }
}
#[target_feature(enable = "avx2")]
unsafe fn sum(x: &[f32]) -> f32 {
    unsafe {
        let mut a = _mm256_setzero_ps();
        let mut b = a;
        let mut c = a;
        let mut d = a;
        let mut i = 0;
        while i + 32 <= x.len() {
            a = _mm256_add_ps(a, _mm256_loadu_ps(x.as_ptr().add(i)));
            b = _mm256_add_ps(b, _mm256_loadu_ps(x.as_ptr().add(i + 8)));
            c = _mm256_add_ps(c, _mm256_loadu_ps(x.as_ptr().add(i + 16)));
            d = _mm256_add_ps(d, _mm256_loadu_ps(x.as_ptr().add(i + 24)));
            i += 32;
        }
        a = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
        let mut lanes = [0.; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), a);
        lanes.into_iter().chain(x[i..].iter().copied()).sum()
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn exponentials(x: &mut [f32], max: f32) {
    unsafe {
        let end = x.len() / 4 * 4;
        let maximum = _mm_set1_ps(max);
        for i in (0..end).step_by(4) {
            let difference = _mm_sub_ps(_mm_loadu_ps(x.as_ptr().add(i)), maximum);
            _mm_storeu_ps(x.as_mut_ptr().add(i), super::exp::negative4(difference));
        }
        for v in &mut x[end..] {
            *v = (*v - max).exp();
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn maximum(x: &[f32]) -> f32 {
    unsafe {
        let mut a = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut b = a;
        let mut c = a;
        let mut d = a;
        let mut i = 0;
        while i + 32 <= x.len() {
            a = _mm256_max_ps(a, _mm256_loadu_ps(x.as_ptr().add(i)));
            b = _mm256_max_ps(b, _mm256_loadu_ps(x.as_ptr().add(i + 8)));
            c = _mm256_max_ps(c, _mm256_loadu_ps(x.as_ptr().add(i + 16)));
            d = _mm256_max_ps(d, _mm256_loadu_ps(x.as_ptr().add(i + 24)));
            i += 32;
        }
        a = _mm256_max_ps(_mm256_max_ps(a, b), _mm256_max_ps(c, d));
        while i + 8 <= x.len() {
            a = _mm256_max_ps(a, _mm256_loadu_ps(x.as_ptr().add(i)));
            i += 8;
        }
        let mut lanes = [0.; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), a);
        lanes
            .iter()
            .chain(&x[i..])
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }
}
#[target_feature(enable = "avx2")]
unsafe fn divide(x: &mut [f32], sum: f32) {
    unsafe {
        let scale = _mm256_set1_ps(sum);
        let end = x.len() / 8 * 8;
        for i in (0..end).step_by(8) {
            _mm256_storeu_ps(
                x.as_mut_ptr().add(i),
                _mm256_div_ps(_mm256_loadu_ps(x.as_ptr().add(i)), scale),
            );
        }
        for v in &mut x[end..] {
            *v /= sum;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn table_softmax_bounds_probabilities_and_preserves_special_values() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        for n in [0, 1, 3, 4, 5, 31, 129, 512, 2049] {
            for scale in [0., 0.01, 1., 17., 1000.] {
                let mut reference: Vec<_> =
                    (0..n).map(|i| (i as f32 * 0.17).sin() * scale).collect();
                let mut actual = reference.clone();
                crate::laya::cpu::softmax(&mut reference);
                table_exp(&mut actual);
                assert!(
                    actual
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| (a - b).abs() <= 1e-6),
                    "n={n} scale={scale}"
                );
            }
        }
        for special in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.,
            f32::MAX,
            -f32::MAX,
        ] {
            for position in 0..35 {
                let mut reference = vec![0.; 35];
                reference[position] = special;
                let mut actual = reference.clone();
                crate::laya::cpu::softmax(&mut reference);
                table_exp(&mut actual);
                assert!(
                    actual
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()))
                );
            }
        }
    }
    #[test]
    fn exact_softmax_keeps_scalar_exp_and_sum_bits() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        for n in [0, 1, 7, 8, 9, 31, 32, 33, 64, 129, 512, 1024, 2048, 2049] {
            for scale in [0., 1., 17., 1000.] {
                let data: Vec<_> = (0..n).map(|i| (i as f32 * 0.17).sin() * scale).collect();
                let mut reference = data.clone();
                crate::laya::cpu::softmax(&mut reference);
                let mut actual = data;
                exact(&mut actual);
                assert!(
                    actual
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "n={n} scale={scale}"
                );
            }
        }
        for special in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.] {
            for position in 0..35 {
                let mut reference = vec![0.; 35];
                reference[position] = special;
                let mut actual = reference.clone();
                crate::laya::cpu::softmax(&mut reference);
                exact(&mut actual);
                assert!(
                    actual
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()))
                );
            }
        }
    }
}
