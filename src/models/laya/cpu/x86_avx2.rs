use rayon::prelude::*;
mod activation;
mod input;
pub(super) use input::prepare as prepare_input;
mod block_attention;
mod cache_tiles;
mod online_attention;
mod packed_attention;
pub(super) use block_attention::head as block_attention_head;
pub(super) use online_attention::apply as online_attention;
mod row_attention;
pub(super) use cache_tiles::multiply as cache_tiles;
pub(super) use row_attention::apply as row_attention;
mod exp;
mod packed_softmax;
mod softmax;
pub(super) use activation::gated as gated_activation;
pub(super) use packed_attention::apply as packed_attention;
pub(super) use packed_attention::apply_selected as packed_attention_selected;
use std::arch::x86_64::*;

pub(super) fn activate_pack(
    x: &[f32],
    temporary: &mut [f32],
    packed: &mut Vec<f32>,
    width: usize,
    gated: bool,
) -> usize {
    assert!(super::packed_supported() && width > 0 && temporary.len() % width == 0);
    let rows = temporary.len() / width;
    let inputs = if gated { 2 } else { 1 };
    assert_eq!(x.len(), rows * width * inputs);
    let len = rows.div_ceil(6) * 6 * width;
    if packed.len() < len {
        packed.resize(len, 0.);
    }
    let run = |((src, tmp), dst): ((&[f32], &mut [f32]), &mut [f32])| {
        if gated {
            for (src, row) in src.chunks_exact(width * 2).zip(tmp.chunks_exact_mut(width)) {
                unsafe {
                    gated_activation(&src[..width], &src[width..], row);
                }
            }
        } else {
            for (out, &value) in tmp.iter_mut().zip(src) {
                *out = value.max(0.);
            }
        }
        input::plain_rows(tmp, width, dst);
    };
    if rows >= 96 {
        x.par_chunks(6 * width * inputs)
            .zip(temporary.par_chunks_mut(6 * width))
            .zip(packed[..len].par_chunks_mut(6 * width))
            .with_min_len(2)
            .for_each(run);
    } else {
        x.chunks(6 * width * inputs)
            .zip(temporary.chunks_mut(6 * width))
            .zip(packed[..len].chunks_mut(6 * width))
            .for_each(run);
    }
    len
}

#[derive(Clone, Copy)]
struct Output(*mut f32);
unsafe impl Send for Output {}
unsafe impl Sync for Output {}
impl Output {
    fn ptr(self) -> *mut f32 {
        self.0
    }
}

pub(super) fn multiply(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    group: usize,
) {
    multiply_bytes::<false>(a, bytemuck::cast_slice(b), c, m, k, n, bias, group);
}
#[cfg(any(test, feature = "fp16"))]
pub(super) fn multiply_half(
    a: &[f32],
    b: &[half::f16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    group: usize,
) {
    multiply_bytes::<true>(a, bytemuck::cast_slice(b), c, m, k, n, bias, group);
}
fn multiply_bytes<const HALF: bool>(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    group: usize,
) {
    assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
    assert!(is_x86_feature_detected!("f16c"));
    let bytes = if HALF { 2 } else { 4 };
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n.div_ceil(16) * k * 16 * bytes);
    assert_eq!(c.len(), m * n);
    let out = Output(c.as_mut_ptr());
    b.par_chunks_exact(k * 16 * bytes)
        .enumerate()
        .with_min_len(group)
        .for_each(|(panel, weights)| {
            for row in (0..m).step_by(6) {
                unsafe {
                    kernel::<HALF>(
                        a.as_ptr().add(row * k),
                        weights.as_ptr(),
                        out.ptr().add(row * n + panel * 16),
                        k,
                        n,
                        (m - row).min(6),
                        (n - panel * 16).min(16),
                        bias.map(|v| v.as_ptr().add(panel * 16)),
                    );
                }
            }
        });
}

pub(super) fn tiled_tails<const HALF: bool>(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    group: usize,
) {
    assert!(super::packed_supported() && k > 0 && n > 0 && group > 0);
    let bytes = if HALF { 2 } else { 4 };
    let m = a.len() / k;
    assert_eq!(a.len(), m * k);
    assert_eq!(c.len(), m * n);
    assert_eq!(b.len(), n.div_ceil(16) * k * 16 * bytes);
    assert!(bias.is_none_or(|b| b.len() == n));
    let out = Output(c.as_mut_ptr());
    b.par_chunks_exact(k * 16 * bytes)
        .enumerate()
        .with_min_len(group)
        .for_each(|(panel, weights)| {
            for row in (0..m).step_by(6) {
                unsafe {
                    let a = a.as_ptr().add(row * k);
                    let b = weights.as_ptr();
                    let c = out.ptr().add(row * n + panel * 16);
                    let width = (n - panel * 16).min(16);
                    let bias = bias.map(|b| b.as_ptr().add(panel * 16));
                    match (m - row).min(6) {
                        2 => tail::<HALF, 2>(a, b, c, k, n, width, bias),
                        3 => tail::<HALF, 3>(a, b, c, k, n, width, bias),
                        4 => tail::<HALF, 4>(a, b, c, k, n, width, bias),
                        5 => tail::<HALF, 5>(a, b, c, k, n, width, bias),
                        rows => kernel::<HALF>(a, b, c, k, n, rows, width, bias),
                    }
                }
            }
        });
}
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tail<const HALF: bool, const ROWS: usize>(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    k: usize,
    stride: usize,
    n: usize,
    bias: Option<*const f32>,
) {
    unsafe {
        let mut lo = [_mm256_setzero_ps(); ROWS];
        let mut hi = lo;
        for t in 0..k {
            let (w0, w1) = load_weights::<HALF>(b, t);
            for row in 0..ROWS {
                let x = _mm256_broadcast_ss(&*a.add(row * k + t));
                lo[row] = _mm256_fmadd_ps(x, w0, lo[row]);
                hi[row] = _mm256_fmadd_ps(x, w1, hi[row]);
            }
        }
        for row in 0..ROWS {
            store(c.add(row * stride), lo[row], hi[row], n, bias);
        }
    }
}

pub(super) fn inline_tiles<const HALF: bool, const PACKED: bool>(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    group: usize,
) {
    assert!(super::packed_supported() && k > 0 && n > 0 && group > 0);
    let bytes = if HALF { 2 } else { 4 };
    let m = c.len() / n;
    assert_eq!(c.len(), m * n);
    assert_eq!(a.len(), if PACKED { m.div_ceil(6) * 6 * k } else { m * k });
    assert!(!PACKED || n.is_multiple_of(16));
    assert_eq!(b.len(), n.div_ceil(16) * k * 16 * bytes);
    assert!(bias.is_none_or(|v| v.len() == n));
    if m == 0 {
        return;
    }
    let output = Output(c.as_mut_ptr());
    b.par_chunks_exact(k * 16 * bytes)
        .enumerate()
        .with_min_len(group)
        .for_each(|(panel, weights)| unsafe {
            inline_panel::<HALF, PACKED>(
                a.as_ptr(),
                weights.as_ptr(),
                output.ptr().add(panel * 16),
                m,
                k,
                n,
                (n - panel * 16).min(16),
                bias.map(|b| b.as_ptr().add(panel * 16)),
            );
        });
}
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn inline_panel<const HALF: bool, const PACKED: bool>(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: usize,
    k: usize,
    stride: usize,
    width: usize,
    bias: Option<*const f32>,
) {
    unsafe {
        if width != 16 {
            debug_assert!(!PACKED);
            for row in (0..m).step_by(6) {
                kernel::<HALF>(
                    a.add(row * k),
                    b,
                    c.add(row * stride),
                    k,
                    stride,
                    (m - row).min(6),
                    width,
                    bias,
                );
            }
            return;
        }
        let full = m / 6 * 6;
        for row in (0..full).step_by(6) {
            inline_tile::<HALF, 6, PACKED>(a.add(row * k), b, c.add(row * stride), k, stride, bias);
        }
        if full == m {
            return;
        }
        let a = a.add(full * k);
        let c = c.add(full * stride);
        macro_rules! run {
            ($r:literal) => {
                inline_tile::<HALF, $r, PACKED>(a, b, c, k, stride, bias)
            };
        }
        match m - full {
            0 => {}
            1 => run!(1),
            2 => run!(2),
            3 => run!(3),
            4 => run!(4),
            5 => run!(5),
            _ => unreachable!(),
        }
    }
}
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn inline_tile<const HALF: bool, const ROWS: usize, const PACKED: bool>(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    k: usize,
    stride: usize,
    bias: Option<*const f32>,
) {
    unsafe {
        let mut lo = [_mm256_setzero_ps(); ROWS];
        let mut hi = lo;
        for t in 0..k {
            let (w0, w1) = load_weights::<HALF>(b, t);
            for row in 0..ROWS {
                let x =
                    _mm256_broadcast_ss(&*a.add(if PACKED { t * 6 + row } else { row * k + t }));
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

pub(super) fn blocked<const HALF: bool>(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    group: usize,
    block: usize,
) {
    assert!(super::packed_supported());
    assert!(k > 0 && n > 0 && block > 0 && group > 0);
    let bytes = if HALF { 2 } else { 4 };
    let m = a.len() / k;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n.div_ceil(16) * k * 16 * bytes);
    assert_eq!(c.len(), m * n);
    assert!(bias.is_none_or(|b| b.len() == n));
    let out = Output(c.as_mut_ptr());
    b.par_chunks_exact(k * 16 * bytes)
        .enumerate()
        .with_min_len(group)
        .for_each(|(panel, weights)| {
            for begin in (0..k).step_by(block) {
                let len = (k - begin).min(block);
                let last = begin + len == k;
                for row in (0..m).step_by(6) {
                    unsafe {
                        kernel_blocked::<HALF>(
                            a.as_ptr().add(row * k + begin),
                            weights.as_ptr().add(begin * 16 * bytes),
                            out.ptr().add(row * n + panel * 16),
                            len,
                            k,
                            n,
                            (m - row).min(6),
                            (n - panel * 16).min(16),
                            if last {
                                bias.map(|b| b.as_ptr().add(panel * 16))
                            } else {
                                None
                            },
                            begin != 0,
                        );
                    }
                }
            }
        });
}

pub(super) fn gated<const HALF: bool>(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    k: usize,
    width: usize,
    bias: Option<&[f32]>,
    group: usize,
) {
    assert!(super::packed_supported());
    assert!(width.is_multiple_of(16) && width > 0 && k > 0 && group > 0);
    assert_eq!(a.len() % k, 0);
    let m = a.len() / k;
    let bytes = if HALF { 2 } else { 4 };
    let panel_bytes = k * 16 * bytes;
    let gate_bytes = width * k * bytes;
    assert_eq!(b.len(), 2 * gate_bytes);
    assert_eq!(c.len(), m * width);
    assert!(bias.is_none_or(|v| v.len() == width * 2));
    let output = Output(c.as_mut_ptr());
    b[..gate_bytes]
        .par_chunks_exact(panel_bytes)
        .enumerate()
        .with_min_len(group)
        .for_each(|(panel, gate)| {
            let up = &b[gate_bytes + panel * panel_bytes..][..panel_bytes];
            let mut g = [0.0_f32; 96];
            let mut u = [0.0_f32; 96];
            for row in (0..m).step_by(6) {
                let rows = (m - row).min(6);
                unsafe {
                    kernel::<HALF>(
                        a.as_ptr().add(row * k),
                        gate.as_ptr(),
                        g.as_mut_ptr(),
                        k,
                        16,
                        rows,
                        16,
                        bias.map(|v| v.as_ptr().add(panel * 16)),
                    );
                    kernel::<HALF>(
                        a.as_ptr().add(row * k),
                        up.as_ptr(),
                        u.as_mut_ptr(),
                        k,
                        16,
                        rows,
                        16,
                        bias.map(|v| v.as_ptr().add(width + panel * 16)),
                    );
                    for r in 0..rows {
                        for col in 0..16 {
                            *output.ptr().add((row + r) * width + panel * 16 + col) =
                                super::gelu(g[r * 16 + col]) * u[r * 16 + col];
                        }
                    }
                }
            }
        });
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn kernel<const HALF: bool>(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    k: usize,
    stride: usize,
    m: usize,
    n: usize,
    bias: Option<*const f32>,
) {
    unsafe {
        let mut z0 = _mm256_setzero_ps();
        let mut z1 = z0;
        let mut z2 = z0;
        let mut z3 = z0;
        let mut z4 = z0;
        let mut z5 = z0;
        let mut z6 = z0;
        let mut z7 = z0;
        let mut z8 = z0;
        let mut z9 = z0;
        let mut z10 = z0;
        let mut z11 = z0;
        if m == 6 {
            for t in 0..k {
                let (w0, w1) = load_weights::<HALF>(b, t);
                let x0 = _mm256_broadcast_ss(&*a.add(t));
                z0 = _mm256_fmadd_ps(x0, w0, z0);
                z1 = _mm256_fmadd_ps(x0, w1, z1);
                let x1 = _mm256_broadcast_ss(&*a.add(k + t));
                z2 = _mm256_fmadd_ps(x1, w0, z2);
                z3 = _mm256_fmadd_ps(x1, w1, z3);
                let x2 = _mm256_broadcast_ss(&*a.add(2 * k + t));
                z4 = _mm256_fmadd_ps(x2, w0, z4);
                z5 = _mm256_fmadd_ps(x2, w1, z5);
                let x3 = _mm256_broadcast_ss(&*a.add(3 * k + t));
                z6 = _mm256_fmadd_ps(x3, w0, z6);
                z7 = _mm256_fmadd_ps(x3, w1, z7);
                let x4 = _mm256_broadcast_ss(&*a.add(4 * k + t));
                z8 = _mm256_fmadd_ps(x4, w0, z8);
                z9 = _mm256_fmadd_ps(x4, w1, z9);
                let x5 = _mm256_broadcast_ss(&*a.add(5 * k + t));
                z10 = _mm256_fmadd_ps(x5, w0, z10);
                z11 = _mm256_fmadd_ps(x5, w1, z11);
            }
            let all = [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11];
            for row in 0..6 {
                store(c.add(row * stride), all[row * 2], all[row * 2 + 1], n, bias);
            }
        } else {
            for row in 0..m {
                let mut lo = _mm256_setzero_ps();
                let mut hi = lo;
                for t in 0..k {
                    let x = _mm256_broadcast_ss(&*a.add(row * k + t));
                    let (w0, w1) = load_weights::<HALF>(b, t);
                    lo = _mm256_fmadd_ps(x, w0, lo);
                    hi = _mm256_fmadd_ps(x, w1, hi);
                }
                store(c.add(row * stride), lo, hi, n, bias);
            }
        }
    }
}
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn kernel_blocked<const HALF: bool>(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    k: usize,
    a_stride: usize,
    stride: usize,
    m: usize,
    n: usize,
    bias: Option<*const f32>,
    accumulate: bool,
) {
    unsafe {
        let mut z0 = _mm256_setzero_ps();
        let mut z1 = z0;
        let mut z2 = z0;
        let mut z3 = z0;
        let mut z4 = z0;
        let mut z5 = z0;
        let mut z6 = z0;
        let mut z7 = z0;
        let mut z8 = z0;
        let mut z9 = z0;
        let mut z10 = z0;
        let mut z11 = z0;
        if m == 6 {
            if accumulate {
                (z0, z1) = load_output(c.add(0 * stride), n);
                (z2, z3) = load_output(c.add(1 * stride), n);
                (z4, z5) = load_output(c.add(2 * stride), n);
                (z6, z7) = load_output(c.add(3 * stride), n);
                (z8, z9) = load_output(c.add(4 * stride), n);
                (z10, z11) = load_output(c.add(5 * stride), n);
            }
            for t in 0..k {
                let (w0, w1) = load_weights::<HALF>(b, t);
                let x0 = _mm256_broadcast_ss(&*a.add(t));
                z0 = _mm256_fmadd_ps(x0, w0, z0);
                z1 = _mm256_fmadd_ps(x0, w1, z1);
                let x1 = _mm256_broadcast_ss(&*a.add(a_stride + t));
                z2 = _mm256_fmadd_ps(x1, w0, z2);
                z3 = _mm256_fmadd_ps(x1, w1, z3);
                let x2 = _mm256_broadcast_ss(&*a.add(2 * a_stride + t));
                z4 = _mm256_fmadd_ps(x2, w0, z4);
                z5 = _mm256_fmadd_ps(x2, w1, z5);
                let x3 = _mm256_broadcast_ss(&*a.add(3 * a_stride + t));
                z6 = _mm256_fmadd_ps(x3, w0, z6);
                z7 = _mm256_fmadd_ps(x3, w1, z7);
                let x4 = _mm256_broadcast_ss(&*a.add(4 * a_stride + t));
                z8 = _mm256_fmadd_ps(x4, w0, z8);
                z9 = _mm256_fmadd_ps(x4, w1, z9);
                let x5 = _mm256_broadcast_ss(&*a.add(5 * a_stride + t));
                z10 = _mm256_fmadd_ps(x5, w0, z10);
                z11 = _mm256_fmadd_ps(x5, w1, z11);
            }
            let all = [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11];
            for row in 0..6 {
                store(c.add(row * stride), all[row * 2], all[row * 2 + 1], n, bias);
            }
        } else {
            for row in 0..m {
                let mut lo = _mm256_setzero_ps();
                let mut hi = lo;
                if accumulate {
                    (lo, hi) = load_output(c.add(row * stride), n);
                }
                for t in 0..k {
                    let x = _mm256_broadcast_ss(&*a.add(row * a_stride + t));
                    let (w0, w1) = load_weights::<HALF>(b, t);
                    lo = _mm256_fmadd_ps(x, w0, lo);
                    hi = _mm256_fmadd_ps(x, w1, hi);
                }
                store(c.add(row * stride), lo, hi, n, bias);
            }
        }
    }
}
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_output(c: *const f32, n: usize) -> (__m256, __m256) {
    unsafe {
        if n == 16 {
            return (_mm256_loadu_ps(c), _mm256_loadu_ps(c.add(8)));
        }
        let mut values = [0.0_f32; 16];
        std::ptr::copy_nonoverlapping(c, values.as_mut_ptr(), n);
        (
            _mm256_loadu_ps(values.as_ptr()),
            _mm256_loadu_ps(values.as_ptr().add(8)),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn load_weights<const HALF: bool>(b: *const u8, t: usize) -> (__m256, __m256) {
    unsafe {
        if HALF {
            (
                _mm256_cvtph_ps(_mm_loadu_si128(b.add(t * 32).cast())),
                _mm256_cvtph_ps(_mm_loadu_si128(b.add(t * 32 + 16).cast())),
            )
        } else {
            (
                _mm256_loadu_ps(b.add(t * 64).cast()),
                _mm256_loadu_ps(b.add(t * 64 + 32).cast()),
            )
        }
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn store(c: *mut f32, mut lo: __m256, mut hi: __m256, n: usize, bias: Option<*const f32>) {
    unsafe {
        if n == 16 {
            if let Some(b) = bias {
                lo = _mm256_add_ps(lo, _mm256_loadu_ps(b));
                hi = _mm256_add_ps(hi, _mm256_loadu_ps(b.add(8)));
            }
            _mm256_storeu_ps(c, lo);
            _mm256_storeu_ps(c.add(8), hi);
        } else {
            let mut values = [0.0; 16];
            _mm256_storeu_ps(values.as_mut_ptr(), lo);
            _mm256_storeu_ps(values.as_mut_ptr().add(8), hi);
            for i in 0..n {
                *c.add(i) = values[i] + bias.map_or(0.0, |b| *b.add(i));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reduction_blocks_are_bit_exact_including_tails_and_bias() {
        if !super::super::packed_supported() {
            return;
        }
        for m in [0, 1, 2, 3, 4, 5, 6, 7, 11, 13, 95, 96, 97] {
            for k in [37, 259, 521] {
                for n in [1_usize, 16, 23] {
                    let a: Vec<_> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
                    let b: Vec<_> = (0..n.div_ceil(16) * k * 16)
                        .map(|i| (i as f32 * 0.07).cos())
                        .collect();
                    let half: Vec<_> = b.iter().map(|&v| half::f16::from_f32(v)).collect();
                    for bias in [
                        None,
                        Some((0..n).map(|i| i as f32 * 0.01 - 0.1).collect::<Vec<_>>()),
                    ] {
                        for half_mode in [false, true] {
                            let mut expected = vec![0.; m * n];
                            if half_mode {
                                multiply_half(
                                    &a,
                                    &half,
                                    &mut expected,
                                    m,
                                    k,
                                    n,
                                    bias.as_deref(),
                                    1,
                                );
                            } else {
                                multiply(&a, &b, &mut expected, m, k, n, bias.as_deref(), 1);
                            }
                            let mut tiled = vec![f32::NAN; m * n];
                            if half_mode {
                                tiled_tails::<true>(
                                    &a,
                                    bytemuck::cast_slice(&half),
                                    &mut tiled,
                                    k,
                                    n,
                                    bias.as_deref(),
                                    1,
                                );
                            } else {
                                tiled_tails::<false>(
                                    &a,
                                    bytemuck::cast_slice(&b),
                                    &mut tiled,
                                    k,
                                    n,
                                    bias.as_deref(),
                                    1,
                                );
                            }
                            assert!(
                                tiled
                                    .iter()
                                    .zip(&expected)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "tiled tails m={m} k={k} n={n}"
                            );
                            let mut got = vec![f32::NAN; m * n + 2];
                            got[0] = -17.;
                            got[m * n + 1] = 31.;
                            if half_mode {
                                inline_tiles::<true, false>(
                                    &a,
                                    bytemuck::cast_slice(&half),
                                    &mut got[1..m * n + 1],
                                    k,
                                    n,
                                    bias.as_deref(),
                                    1,
                                );
                            } else {
                                inline_tiles::<false, false>(
                                    &a,
                                    bytemuck::cast_slice(&b),
                                    &mut got[1..m * n + 1],
                                    k,
                                    n,
                                    bias.as_deref(),
                                    1,
                                );
                            }
                            assert!(
                                got[1..m * n + 1]
                                    .iter()
                                    .zip(&expected)
                                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                                "inline m={m} k={k} n={n}"
                            );
                            assert_eq!(got[0], -17.);
                            assert_eq!(got[m * n + 1], 31.);
                            if n == 16 {
                                let mut packed = vec![f32::NAN; m.div_ceil(6) * 6 * k];
                                for row in 0..m {
                                    for t in 0..k {
                                        packed[row / 6 * 6 * k + t * 6 + row % 6] = a[row * k + t];
                                    }
                                }
                                let mut actual = vec![f32::NAN; m * n];
                                if half_mode {
                                    inline_tiles::<true, true>(
                                        &packed,
                                        bytemuck::cast_slice(&half),
                                        &mut actual,
                                        k,
                                        n,
                                        bias.as_deref(),
                                        1,
                                    );
                                } else {
                                    inline_tiles::<false, true>(
                                        &packed,
                                        bytemuck::cast_slice(&b),
                                        &mut actual,
                                        k,
                                        n,
                                        bias.as_deref(),
                                        1,
                                    );
                                }
                                assert!(
                                    actual
                                        .iter()
                                        .zip(&expected)
                                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                                    "packed input m={m} k={k}"
                                );
                            }
                            for block in [32, 128, 256] {
                                let mut actual = vec![f32::NAN; m * n + 2];
                                actual[0] = 123.;
                                actual[m * n + 1] = 456.;
                                if half_mode {
                                    blocked::<true>(
                                        &a,
                                        bytemuck::cast_slice(&half),
                                        &mut actual[1..m * n + 1],
                                        k,
                                        n,
                                        bias.as_deref(),
                                        1,
                                        block,
                                    );
                                } else {
                                    blocked::<false>(
                                        &a,
                                        bytemuck::cast_slice(&b),
                                        &mut actual[1..m * n + 1],
                                        k,
                                        n,
                                        bias.as_deref(),
                                        1,
                                        block,
                                    );
                                }
                                assert_eq!(actual[0], 123.);
                                assert_eq!(actual[m * n + 1], 456.);
                                assert!(
                                    actual[1..m * n + 1]
                                        .iter()
                                        .zip(&expected)
                                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                                    "m={m} k={k} n={n} block={block} half={half_mode}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn gated_tiles_preserve_accumulation_bias_and_row_tails() {
        if !super::super::packed_supported() {
            return;
        }
        for m in [1, 5, 6, 7, 11, 12, 13] {
            for width in [16, 32, 80] {
                let k = 37;
                let a: Vec<_> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
                let weights: Vec<_> = (0..2 * width * k)
                    .map(|i| (i as f32 * 0.07).cos())
                    .collect();
                let half: Vec<_> = weights.iter().map(|&v| half::f16::from_f32(v)).collect();
                for bias in [None, Some(vec![0.03125; 2 * width])] {
                    for group in [1, 4, 16] {
                        let mut full = vec![0.; m * 2 * width];
                        let mut expected = vec![0.; m * width];
                        let mut actual = expected.clone();
                        multiply(
                            &a,
                            &weights,
                            &mut full,
                            m,
                            k,
                            width * 2,
                            bias.as_deref(),
                            group,
                        );
                        super::super::gated_gelu(&full, &mut expected, width);
                        gated::<false>(
                            &a,
                            bytemuck::cast_slice(&weights),
                            &mut actual,
                            k,
                            width,
                            bias.as_deref(),
                            group,
                        );
                        assert_eq!(expected, actual);
                        multiply_half(
                            &a,
                            &half,
                            &mut full,
                            m,
                            k,
                            width * 2,
                            bias.as_deref(),
                            group,
                        );
                        super::super::gated_gelu(&full, &mut expected, width);
                        gated::<true>(
                            &a,
                            bytemuck::cast_slice(&half),
                            &mut actual,
                            k,
                            width,
                            bias.as_deref(),
                            group,
                        );
                        assert_eq!(expected, actual);
                    }
                }
            }
        }
    }
    #[test]
    fn packed_kernel_handles_all_tails() {
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("f16c")
        {
            return;
        }
        for m in 1..14 {
            for n in [1_usize, 7, 16, 23, 32] {
                let k = 521;
                let a: Vec<_> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
                let w: Vec<_> = (0..n * k).map(|i| (i as f32 * 0.17).cos()).collect();
                let mut packed = vec![0.0; n.div_ceil(16) * k * 16];
                for j in 0..n {
                    for t in 0..k {
                        packed[(j / 16) * k * 16 + t * 16 + j % 16] = w[j * k + t];
                    }
                }
                let mut out = vec![0.0; m * n];
                let bias = vec![0.125; n];
                multiply(&a, &packed, &mut out, m, k, n, Some(&bias), 1);
                for i in 0..m {
                    for j in 0..n {
                        let expected =
                            (0..k).map(|t| a[i * k + t] * w[j * k + t]).sum::<f32>() + 0.125;
                        assert!((out[i * n + j] - expected).abs() < 2e-4, "m={m} n={n}");
                    }
                }
                let half: Vec<_> = packed.iter().map(|&v| half::f16::from_f32(v)).collect();
                multiply_half(&a, &half, &mut out, m, k, n, Some(&bias), 1);
                for i in 0..m {
                    for j in 0..n {
                        let expected = (0..k)
                            .map(|t| a[i * k + t] * half::f16::from_f32(w[j * k + t]).to_f32())
                            .sum::<f32>()
                            + 0.125;
                        assert!((out[i * n + j] - expected).abs() < 2e-4, "half m={m} n={n}");
                    }
                }
            }
        }
    }
}
