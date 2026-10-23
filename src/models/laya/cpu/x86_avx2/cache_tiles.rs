use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn multiply<const HALF: bool>(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    row_block: usize,
    reduction: usize,
) {
    assert!(crate::laya::cpu::packed_supported() && k > 0 && n > 0 && n.is_multiple_of(16));
    assert!(row_block > 0 && row_block.is_multiple_of(6));
    let m = c.len() / n;
    let bytes = if HALF { 2 } else { 4 };
    assert_eq!(c.len(), m * n);
    assert_eq!(a.len(), m.div_ceil(6) * 6 * k);
    assert_eq!(b.len(), n * k * bytes);
    assert!(bias.is_none_or(|b| b.len() == n));
    let panels = n / 16;
    let panel_block = 8;
    let column_blocks = panels.div_ceil(panel_block);
    let output = Output(c.as_mut_ptr());
    (0..m.div_ceil(row_block) * column_blocks)
        .into_par_iter()
        .for_each(|job| {
            let row = job / column_blocks * row_block;
            let panel = job % column_blocks * panel_block;
            unsafe {
                block::<HALF>(
                    a.as_ptr().add(row * k),
                    b.as_ptr().add(panel * k * 16 * bytes),
                    output.ptr().add(row * n + panel * 16),
                    (m - row).min(row_block),
                    k,
                    n,
                    (panels - panel).min(panel_block),
                    bias.map(|b| b.as_ptr().add(panel * 16)),
                    reduction.max(1).min(k),
                );
            }
        });
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn block<const HALF: bool>(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: usize,
    k: usize,
    stride: usize,
    panels: usize,
    bias: Option<*const f32>,
    reduction: usize,
) {
    let bytes = if HALF { 2 } else { 4 };
    for p in 0..panels {
        for begin in (0..k).step_by(reduction) {
            let count = (k - begin).min(reduction);
            let last = begin + count == k;
            for r in (0..m).step_by(6) {
                unsafe {
                    let a = a.add(r * k + begin * 6);
                    let b = b.add((p * k + begin) * 16 * bytes);
                    let c = c.add(r * stride + p * 16);
                    let bias = if last {
                        bias.map(|b| b.add(p * 16))
                    } else {
                        None
                    };
                    macro_rules! run {
                        ($r:literal) => {
                            tile::<HALF, $r>(a, b, c, count, stride, bias, begin != 0)
                        };
                    }
                    match (m - r).min(6) {
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
unsafe fn tile<const HALF: bool, const ROWS: usize>(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    k: usize,
    stride: usize,
    bias: Option<*const f32>,
    accumulate: bool,
) {
    unsafe {
        let mut lo = [_mm256_setzero_ps(); ROWS];
        let mut hi = lo;
        if accumulate {
            for row in 0..ROWS {
                lo[row] = _mm256_loadu_ps(c.add(row * stride));
                hi[row] = _mm256_loadu_ps(c.add(row * stride + 8));
            }
        }
        for t in 0..k {
            let (w0, w1) = load_weights::<HALF>(b, t);
            for row in 0..ROWS {
                let x = _mm256_broadcast_ss(&*a.add(t * 6 + row));
                lo[row] = _mm256_fmadd_ps(x, w0, lo[row]);
                hi[row] = _mm256_fmadd_ps(x, w1, hi[row]);
            }
        }
        if let Some(bias) = bias {
            let b0 = _mm256_loadu_ps(bias);
            let b1 = _mm256_loadu_ps(bias.add(8));
            for row in 0..ROWS {
                lo[row] = _mm256_add_ps(lo[row], b0);
                hi[row] = _mm256_add_ps(hi[row], b1);
            }
        }
        for row in 0..ROWS {
            _mm256_storeu_ps(c.add(row * stride), lo[row]);
            _mm256_storeu_ps(c.add(row * stride + 8), hi[row]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_tiles_preserve_full_reduction_and_tails() {
        if !crate::laya::cpu::packed_supported() {
            return;
        }
        for m in [0, 1, 5, 6, 7, 47, 48, 49, 97, 513] {
            for k in [1, 63, 128, 129, 1024] {
                let n = 144;
                let a: Vec<_> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
                let b: Vec<_> = (0..n * k).map(|i| (i as f32 * 0.031).cos()).collect();
                let half: Vec<_> = b.iter().map(|&v| half::f16::from_f32(v)).collect();
                let bias: Vec<_> = (0..n).map(|i| i as f32 * 0.01).collect();
                let mut packed = Vec::new();
                let len = input::prepare(&a, k, &mut packed, None);
                for half_mode in [false, true] {
                    let mut expected = vec![0.; m * n];
                    if half_mode {
                        inline_tiles::<true, true>(
                            &packed[..len],
                            bytemuck::cast_slice(&half),
                            &mut expected,
                            k,
                            n,
                            Some(&bias),
                            1,
                        );
                    } else {
                        inline_tiles::<false, true>(
                            &packed[..len],
                            bytemuck::cast_slice(&b),
                            &mut expected,
                            k,
                            n,
                            Some(&bias),
                            1,
                        );
                    }
                    for reduction in [64, 128, k] {
                        let mut got = vec![f32::NAN; m * n + 2];
                        got[0] = -17.;
                        got[m * n + 1] = 31.;
                        if half_mode {
                            multiply::<true>(
                                &packed[..len],
                                bytemuck::cast_slice(&half),
                                &mut got[1..m * n + 1],
                                k,
                                n,
                                Some(&bias),
                                48,
                                reduction,
                            );
                        } else {
                            multiply::<false>(
                                &packed[..len],
                                bytemuck::cast_slice(&b),
                                &mut got[1..m * n + 1],
                                k,
                                n,
                                Some(&bias),
                                48,
                                reduction,
                            );
                        }
                        assert!(
                            got[1..m * n + 1]
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "m={m} k={k} half={half_mode} reduction={reduction}"
                        );
                        assert_eq!(got[0], -17.);
                        assert_eq!(got[m * n + 1], 31.);
                    }
                }
            }
        }
    }
}
