use super::Matrix;
use core::arch::wasm32::*;
#[cfg(not(feature = "decima-wasm-fma"))]
const PANEL: usize = 12;
#[cfg(feature = "decima-wasm-fma")]
const PANEL: usize = 8;
const PARTS: usize = PANEL / 4;

#[cfg(feature = "decima-wasm-threads")]
pub(super) fn apply_parallel(
    w: &Matrix,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
    tile: usize,
) -> bool {
    use rayon::prelude::*;
    use std::cell::RefCell;
    thread_local! {
        static PANEL_WORK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }
    let count = x.len() / w.cols;
    let tasks = super::super::wasm_threads::count().min(count.div_ceil(24));
    if tasks <= 1 || count * w.cols * w.rows < 2_000_000 {
        return false;
    }
    let rows = count.div_ceil(tasks).div_ceil(tile) * tile;
    out.par_chunks_mut(rows * w.rows)
        .enumerate()
        .for_each(|(chunk, out)| {
            let first = chunk * rows;
            let count = out.len() / w.rows;
            PANEL_WORK.with(|work| {
                // Never submit nested Rayon work while this TLS panel is borrowed.
                apply(
                    w,
                    &x[first * w.cols..(first + count) * w.cols],
                    out,
                    bias,
                    tile,
                    &mut work.borrow_mut(),
                );
            });
        });
    true
}

#[inline]
#[cfg_attr(feature = "decima-wasm-fma", target_feature(enable = "relaxed-simd"))]
unsafe fn madd(a: v128, b: v128, c: v128) -> v128 {
    #[cfg(feature = "decima-wasm-fma")]
    {
        f32x4_relaxed_madd(a, b, c)
    }
    #[cfg(not(feature = "decima-wasm-fma"))]
    {
        f32x4_add(c, f32x4_mul(a, b))
    }
}

pub(super) fn apply(
    w: &Matrix,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
    rows: usize,
    work: &mut Vec<f32>,
) {
    let count = x.len() / w.cols;
    #[cfg(not(feature = "decima-wasm-fma"))]
    let rows = rows.min(if count <= 32 { 2 } else { 3 });
    let block_rows = if (256..=512).contains(&count) {
        count.max(1)
    } else {
        (131072 / w.cols / 4).clamp(rows, 32) / rows * rows
    };
    work.resize(w.cols * PANEL, 0.);
    for first in (0..count).step_by(block_rows) {
        let end = (first + block_rows).min(count);
        for panel in 0..w.rows.div_ceil(PANEL) {
            unsafe {
                decode(w, panel, work);
            }
            for base in (first..end).step_by(rows) {
                let last = (base + rows).min(end);
                unsafe {
                    match last - base {
                        1 => kernel::<1>(x, work, out, w.cols, w.rows, base, panel, bias),
                        2 => kernel::<2>(x, work, out, w.cols, w.rows, base, panel, bias),
                        3 => kernel::<3>(x, work, out, w.cols, w.rows, base, panel, bias),
                        4 => kernel::<4>(x, work, out, w.cols, w.rows, base, panel, bias),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
}
#[cfg(not(feature = "decima-wasm-fma"))]
unsafe fn decode(w: &Matrix, panel: usize, out: &mut [f32]) {
    unsafe {
        let groups = w.groups();
        for part in 0..PARTS {
            let column = panel * PANEL + part * 4;
            if column >= w.rows {
                for j in 0..w.cols {
                    v128_store(
                        out.as_mut_ptr().add(j * PANEL + part * 4).cast(),
                        f32x4_splat(0.),
                    );
                }
                continue;
            }
            for g in 0..groups {
                let source = w
                    .data
                    .as_ptr()
                    .add(((column / 8) * groups + g) * w.block * 8 + column % 8);
                let scale = v128_load(
                    w.scales
                        .as_ptr()
                        .add(((column / 8) * groups + g) * 8 + column % 8)
                        .cast(),
                );
                for j in 0..w.block.min(w.cols - g * w.block) {
                    let bytes = v128_load32_zero(source.add(j * 8).cast());
                    let value =
                        f32x4_convert_i32x4(i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(bytes)));
                    v128_store(
                        out.as_mut_ptr()
                            .add((g * w.block + j) * PANEL + part * 4)
                            .cast(),
                        f32x4_mul(value, scale),
                    );
                }
            }
        }
    }
}
#[cfg(feature = "decima-wasm-fma")]
unsafe fn decode(w: &Matrix, panel: usize, out: &mut [f32]) {
    unsafe {
        let groups = w.groups();
        for g in 0..groups {
            let source = w.data.as_ptr().add((panel * groups + g) * w.block * 8);
            let scale = w.scales.as_ptr().add((panel * groups + g) * 8);
            let s0 = v128_load(scale.cast());
            let s1 = v128_load(scale.add(4).cast());
            for j in 0..w.block.min(w.cols - g * w.block) {
                let q = i16x8_extend_low_i8x16(v128_load64_zero(source.add(j * 8).cast()));
                let low = f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(q)), s0);
                let high = f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(q)), s1);
                let dst = out.as_mut_ptr().add((g * w.block + j) * 8);
                v128_store(dst.cast(), low);
                v128_store(dst.add(4).cast(), high);
            }
        }
    }
}
#[cfg_attr(feature = "decima-wasm-fma", target_feature(enable = "relaxed-simd"))]
unsafe fn kernel<const R: usize>(
    x: &[f32],
    weights: &[f32],
    out: &mut [f32],
    k: usize,
    n: usize,
    first: usize,
    panel: usize,
    bias: Option<&[f32]>,
) {
    unsafe {
        let mut sum = [[f32x4_splat(0.); PARTS]; R];
        for j in 0..k {
            let b = weights.as_ptr().add(j * PANEL);
            let weights: [v128; PARTS] = std::array::from_fn(|p| v128_load(b.add(p * 4).cast()));
            for r in 0..R {
                let a = f32x4_splat(*x.get_unchecked((first + r) * k + j));
                for p in 0..PARTS {
                    sum[r][p] = madd(a, weights[p], sum[r][p]);
                }
            }
        }
        for r in 0..R {
            let column = panel * PANEL;
            if column + PANEL <= n {
                for p in 0..PARTS {
                    let b = bias.map_or(f32x4_splat(0.), |b| {
                        v128_load(b.as_ptr().add(column + p * 4).cast())
                    });
                    v128_store(
                        out.as_mut_ptr()
                            .add((first + r) * n + column + p * 4)
                            .cast(),
                        f32x4_add(sum[r][p], b),
                    );
                }
            } else {
                let mut values = [0.; PANEL];
                for p in 0..PARTS {
                    v128_store(values.as_mut_ptr().add(p * 4).cast(), sum[r][p]);
                }
                for c in column..n {
                    out[(first + r) * n + c] = values[c - column] + bias.map_or(0., |b| b[c]);
                }
            }
        }
    }
}
