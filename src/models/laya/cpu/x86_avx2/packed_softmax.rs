use std::arch::x86_64::*;

pub(super) fn normalize(values: &mut [f32], rows: usize, keys: usize) {
    assert!(crate::laya::cpu::packed_supported() && rows > 0 && keys > 0);
    let blocks = rows.div_ceil(6);
    assert_eq!(values.len(), blocks * 6 * keys);
    unsafe {
        let p = values.as_mut_ptr();
        for block in (0..blocks).step_by(2) {
            let a = p.add(block * 6 * keys);
            group::<0>(a, a, keys);
            if block + 1 < blocks {
                let b = a.add(6 * keys);
                group::<1>(a.add(4), b, keys);
                group::<0>(b.add(2), b, keys);
            } else {
                group::<2>(a.add(4), a, keys);
            }
        }
    }
}
#[inline]
unsafe fn load<const MODE: usize>(a: *const f32, b: *const f32) -> __m128 {
    unsafe {
        if MODE == 0 {
            _mm_loadu_ps(a)
        } else {
            let low = _mm_castsi128_ps(_mm_loadl_epi64(a.cast()));
            if MODE == 1 {
                _mm_movelh_ps(low, _mm_castsi128_ps(_mm_loadl_epi64(b.cast())))
            } else {
                low
            }
        }
    }
}
#[inline]
unsafe fn store<const MODE: usize>(a: *mut f32, b: *mut f32, value: __m128) {
    unsafe {
        if MODE == 0 {
            _mm_storeu_ps(a, value);
        } else {
            _mm_storel_epi64(a.cast(), _mm_castps_si128(value));
            if MODE == 1 {
                _mm_storel_epi64(b.cast(), _mm_castps_si128(_mm_movehl_ps(value, value)));
            }
        }
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn group<const MODE: usize>(a: *mut f32, b: *mut f32, keys: usize) {
    unsafe {
        let mut max = _mm_set1_ps(f32::NEG_INFINITY);
        for key in 0..keys {
            max = _mm_max_ps(max, load::<MODE>(a.add(key * 6), b.add(key * 6)));
        }
        let mut sum = _mm_setzero_ps();
        let vector_keys = keys / 4 * 4;
        for key in 0..keys {
            let x = _mm_sub_ps(load::<MODE>(a.add(key * 6), b.add(key * 6)), max);
            let exp = if key < vector_keys {
                super::exp::negative4(x)
            } else {
                let mut lanes = [0.; 4];
                _mm_storeu_ps(lanes.as_mut_ptr(), x);
                for v in &mut lanes {
                    *v = v.exp();
                }
                _mm_loadu_ps(lanes.as_ptr())
            };
            sum = _mm_add_ps(sum, exp);
            store::<MODE>(a.add(key * 6), b.add(key * 6), exp);
        }
        for key in 0..keys {
            store::<MODE>(
                a.add(key * 6),
                b.add(key * 6),
                _mm_div_ps(load::<MODE>(a.add(key * 6), b.add(key * 6)), sum),
            );
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_query_sums_preserve_each_row_including_tails_and_special_values() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        for rows in [1, 5, 6, 7, 11, 12, 13, 64, 65] {
            for keys in [1, 3, 4, 17, 129, 512, 2049] {
                let original: Vec<_> = (0..rows * keys)
                    .map(|i| (i as f32 * 0.071).sin() * 17.)
                    .collect();
                let mut expected = original.clone();
                for row in expected.chunks_exact_mut(keys) {
                    super::super::softmax::table_exp(row);
                }
                let size = rows.div_ceil(6) * 6 * keys;
                let mut packed = vec![-17.; size + 2];
                for r in 0..rows {
                    for k in 0..keys {
                        packed[1 + r / 6 * 6 * keys + k * 6 + r % 6] = original[r * keys + k];
                    }
                }
                normalize(&mut packed[1..size + 1], rows, keys);
                for r in 0..rows {
                    for k in 0..keys {
                        assert_eq!(
                            packed[1 + r / 6 * 6 * keys + k * 6 + r % 6].to_bits(),
                            expected[r * keys + k].to_bits(),
                            "rows={rows} keys={keys} r={r} k={k}"
                        );
                    }
                }
                assert_eq!(packed[0], -17.);
                assert_eq!(packed[size + 1], -17.);
            }
        }
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.,
            f32::MAX,
            -f32::MAX,
        ] {
            let rows = 12;
            let keys = 17;
            let mut reference = vec![0.; rows * keys];
            reference[5 * keys + 3] = value;
            reference[6 * keys + 11] = value;
            let mut packed = vec![0.; rows * keys];
            for r in 0..rows {
                for k in 0..keys {
                    packed[r / 6 * 6 * keys + k * 6 + r % 6] = reference[r * keys + k];
                }
            }
            for row in reference.chunks_exact_mut(keys) {
                super::super::softmax::table_exp(row);
            }
            normalize(&mut packed, rows, keys);
            for r in 0..rows {
                for k in 0..keys {
                    let a = packed[r / 6 * 6 * keys + k * 6 + r % 6];
                    let b = reference[r * keys + k];
                    assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
                }
            }
        }
    }
}
