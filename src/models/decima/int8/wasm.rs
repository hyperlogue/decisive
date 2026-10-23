use super::Matrix;
#[cfg(feature = "kernel-checks")]
use super::Mode;
use core::arch::wasm32::*;

pub(super) fn pack_for_rows(
    x: &[f32],
    cols: usize,
    block: usize,
    packed: &mut [i16],
    scales: &mut [f32],
    tile: usize,
) -> bool {
    if tile != 4 || x.len() / cols < 4 {
        return pack_activations(x, cols, block, packed, scales);
    }
    let count = x.len() / cols;
    let full = count / 4 * 4;
    let groups = cols.div_ceil(block);
    let padded = groups * block;
    packed.fill(0);
    for first in (0..full).step_by(4) {
        for g in 0..groups {
            let start = g * block;
            let len = block.min(cols - start);
            let mut scale = [0.; 4];
            for r in 0..4 {
                let values = &x[(first + r) * cols + start..][..len];
                let Some(s) = group_scale(values) else {
                    return false;
                };
                scale[r] = s;
                scales[(first + r) * groups + g] = s;
            }
            let base = first * padded + g * block * 4;
            let end = len / 8 * 8;
            unsafe {
                for k in (0..end).step_by(8) {
                    let input = x.as_ptr().add(first * cols + start + k);
                    let r0 = quantize_eight(input, scale[0]);
                    let r1 = quantize_eight(input.add(cols), scale[1]);
                    let r2 = quantize_eight(input.add(2 * cols), scale[2]);
                    let r3 = quantize_eight(input.add(3 * cols), scale[3]);
                    let ab0 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1);
                    let ab1 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1);
                    let cd0 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3);
                    let cd1 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3);
                    let out = packed.as_mut_ptr().add(base + k * 4);
                    v128_store(out.cast(), i64x2_shuffle::<0, 2>(ab0, cd0));
                    v128_store(out.add(8).cast(), i64x2_shuffle::<1, 3>(ab0, cd0));
                    v128_store(out.add(16).cast(), i64x2_shuffle::<0, 2>(ab1, cd1));
                    v128_store(out.add(24).cast(), i64x2_shuffle::<1, 3>(ab1, cd1));
                }
            }
            for k in end..len {
                for r in 0..4 {
                    packed[base + (k / 2) * 8 + r * 2 + k % 2] =
                        super::quantize(x[(first + r) * cols + start + k], scale[r]) as i16;
                }
            }
        }
    }
    pack_activations(
        &x[full * cols..],
        cols,
        block,
        &mut packed[full * padded..],
        &mut scales[full * groups..],
    )
}

#[inline]
unsafe fn quantize_eight(input: *const f32, scale: f32) -> v128 {
    unsafe {
        let s = f32x4_splat(scale);
        let low = rounded(f32x4_div(v128_load(input.cast()), s));
        let high = rounded(f32x4_div(v128_load(input.add(4).cast()), s));
        i16x8_narrow_i32x4(low, high)
    }
}

#[inline]
fn group_scale(values: &[f32]) -> Option<f32> {
    unsafe {
        let end = values.len() / 4 * 4;
        let mut maximum = f32x4_splat(0.);
        let mut finite = i32x4_splat(-1);
        for k in (0..end).step_by(4) {
            let v = f32x4_abs(v128_load(values.as_ptr().add(k).cast()));
            finite = v128_and(finite, f32x4_le(v, f32x4_splat(f32::MAX)));
            maximum = f32x4_pmax(maximum, v);
        }
        if i32x4_bitmask(finite) != 15 {
            return None;
        }
        maximum = f32x4_pmax(maximum, i32x4_shuffle::<2, 3, 0, 1>(maximum, maximum));
        maximum = f32x4_pmax(maximum, i32x4_shuffle::<1, 0, 3, 2>(maximum, maximum));
        let mut magnitude = f32x4_extract_lane::<0>(maximum);
        for &v in &values[end..] {
            if !v.is_finite() {
                return None;
            }
            if v.abs() > magnitude {
                magnitude = v.abs();
            }
        }
        Some((magnitude / 127.).max(f32::MIN_POSITIVE))
    }
}

pub(super) fn pack_activations(
    x: &[f32],
    cols: usize,
    block: usize,
    packed: &mut [i16],
    scales: &mut [f32],
) -> bool {
    let groups = cols.div_ceil(block);
    packed.fill(0);
    for (row, input) in x.chunks_exact(cols).enumerate() {
        for g in 0..groups {
            let start = g * block;
            let values = &input[start..(start + block).min(cols)];
            let end = values.len() / 4 * 4;
            unsafe {
                let mut maximum = f32x4_splat(0.);
                let mut finite = i32x4_splat(-1);
                for k in (0..end).step_by(4) {
                    let v = f32x4_abs(v128_load(values.as_ptr().add(k).cast()));
                    finite = v128_and(finite, f32x4_le(v, f32x4_splat(f32::MAX)));
                    maximum = f32x4_pmax(maximum, v);
                }
                if i32x4_bitmask(finite) != 15 {
                    return false;
                }
                maximum = f32x4_pmax(maximum, i32x4_shuffle::<2, 3, 0, 1>(maximum, maximum));
                maximum = f32x4_pmax(maximum, i32x4_shuffle::<1, 0, 3, 2>(maximum, maximum));
                let mut magnitude = f32x4_extract_lane::<0>(maximum);
                for &v in &values[end..] {
                    if !v.is_finite() {
                        return false;
                    }
                    if v.abs() > magnitude {
                        magnitude = v.abs();
                    }
                }
                let s = magnitude / 127.;
                let s = if s < f32::MIN_POSITIVE {
                    f32::MIN_POSITIVE
                } else {
                    s
                };
                scales[row * groups + g] = s;
                let out = &mut packed[(row * groups + g) * block..][..block];
                let end16 = values.len() / 16 * 16;
                for k in (0..end16).step_by(16) {
                    let q0 = rounded(f32x4_div(
                        v128_load(values.as_ptr().add(k).cast()),
                        f32x4_splat(s),
                    ));
                    let q1 = rounded(f32x4_div(
                        v128_load(values.as_ptr().add(k + 4).cast()),
                        f32x4_splat(s),
                    ));
                    let q2 = rounded(f32x4_div(
                        v128_load(values.as_ptr().add(k + 8).cast()),
                        f32x4_splat(s),
                    ));
                    let q3 = rounded(f32x4_div(
                        v128_load(values.as_ptr().add(k + 12).cast()),
                        f32x4_splat(s),
                    ));
                    let low = i16x8_narrow_i32x4(q0, q1);
                    let high = i16x8_narrow_i32x4(q2, q3);
                    v128_store(out.as_mut_ptr().add(k).cast(), low);
                    v128_store(out.as_mut_ptr().add(k + 8).cast(), high);
                }
                for k in (end16..end).step_by(4) {
                    let q = rounded(f32x4_div(
                        v128_load(values.as_ptr().add(k).cast()),
                        f32x4_splat(s),
                    ));
                    let shorts = i16x8_narrow_i32x4(q, i32x4_splat(0));
                    out.as_mut_ptr()
                        .add(k)
                        .cast::<i64>()
                        .write_unaligned(i64x2_extract_lane::<0>(shorts));
                }
                for k in end..values.len() {
                    out[k] = super::quantize(values[k], s) as i16;
                }
            }
        }
    }
    true
}

#[inline(always)]
fn rounded(v: v128) -> v128 {
    let truncated = i32x4_trunc_sat_f32x4(v);
    let fraction = f32x4_abs(f32x4_sub(v, f32x4_convert_i32x4(truncated)));
    let increment = v128_or(i32x4_shr(v, 31), i32x4_splat(1));
    i32x4_add(
        truncated,
        v128_and(increment, f32x4_ge(fraction, f32x4_splat(0.5))),
    )
}

#[cfg(feature = "decima-wasm-threads")]
pub(super) fn apply_parallel(
    w: &Matrix,
    a: &[i16],
    sa: &[f32],
    dst: &mut [f32],
    bias: Option<&[f32]>,
    tile: usize,
) -> bool {
    use rayon::prelude::*;
    let count = dst.len() / w.rows;
    let tasks = super::super::wasm_threads::count().min(count.div_ceil(16));
    if tasks <= 1 {
        return false;
    }
    let rows = count.div_ceil(tasks).div_ceil(tile) * tile;
    let groups = w.groups();
    let padded = groups * w.block;
    dst.par_chunks_mut(rows * w.rows)
        .enumerate()
        .for_each(|(chunk, out)| {
            let first = chunk * rows;
            let count = out.len() / w.rows;
            apply_in_worker(
                w,
                &a[first * padded..(first + count) * padded],
                &sa[first * groups..(first + count) * groups],
                out,
                bias,
                tile,
            );
        });
    true
}

#[cfg(feature = "decima-wasm-threads")]
fn apply_in_worker(
    w: &Matrix,
    a: &[i16],
    sa: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
    tile: usize,
) {
    use std::cell::RefCell;
    thread_local! {
        static PANEL: RefCell<Vec<i16>> = const { RefCell::new(Vec::new()) };
    }
    PANEL.with(|panel| {
        let mut panel = panel.borrow_mut();
        let required = w.groups() * w.block * 8;
        if panel.len() < required {
            panel.resize(required, 0);
        }
        apply(w, a, sa, out, bias, tile, &mut panel[..required]);
    });
}

#[cfg(feature = "decima-wasm-threads")]
pub(super) fn apply_three(
    matrices: [&Matrix; 3],
    a: &[i16],
    sa: &[f32],
    outputs: [&mut [f32]; 3],
    biases: [&[f32]; 3],
    tile: usize,
) {
    let [q, k, v] = outputs;
    let run = |index: usize, out: &mut [f32]| {
        let w = matrices[index];
        if !apply_parallel(w, a, sa, out, Some(biases[index]), tile) {
            apply_in_worker(w, a, sa, out, Some(biases[index]), tile);
        }
    };
    rayon::join(|| run(0, q), || rayon::join(|| run(1, k), || run(2, v)));
}

#[inline(never)]
pub(super) fn apply(
    w: &Matrix,
    a: &[i16],
    sa: &[f32],
    dst: &mut [f32],
    bias: Option<&[f32]>,
    tile: usize,
    workspace: &mut [i16],
) {
    let input_bytes = 2;
    let cache_rows = (262144 / w.cols / input_bytes).clamp(tile, 64) / tile * tile;
    let wide = workspace;
    let count = dst.len() / w.rows;
    let panel_size = w.groups() * w.block * 8;

    for base in (0..count).step_by(cache_rows) {
        let end = (base + cache_rows).min(count);
        for panel in 0..w.rows.div_ceil(8) {
            unsafe {
                for k in (0..panel_size).step_by(16) {
                    let b = v128_load(w.data.as_ptr().add(panel * panel_size + k).cast());
                    v128_store(wide.as_mut_ptr().add(k).cast(), i16x8_extend_low_i8x16(b));
                    v128_store(
                        wide.as_mut_ptr().add(k + 8).cast(),
                        i16x8_extend_high_i8x16(b),
                    );
                }
            }

            for first in (base..end).step_by(tile) {
                let last = (first + tile).min(end);
                run(
                    w,
                    a,
                    sa,
                    first,
                    &mut dst[first * w.rows..last * w.rows],
                    bias,
                    panel,
                    wide,
                );
            }
        }
    }
}
fn run(
    w: &Matrix,
    a: &[i16],
    sa: &[f32],
    first: usize,
    dst: &mut [f32],
    bias: Option<&[f32]>,
    panel: usize,
    wide: &[i16],
) {
    match dst.len() / w.rows {
        1 => rows::<1>(w, a, sa, first, dst, bias, panel, wide),
        2 => rows::<2>(w, a, sa, first, dst, bias, panel, wide),
        3 => rows::<3>(w, a, sa, first, dst, bias, panel, wide),
        4 => rows4(w, a, sa, first, dst, bias, panel, wide),
        _ => unreachable!(),
    }
}
fn rows<const R: usize>(
    w: &Matrix,
    a: &[i16],
    sa: &[f32],
    first: usize,
    dst: &mut [f32],
    bias: Option<&[f32]>,
    panel: usize,
    wide: &[i16],
) {
    let groups = w.groups();
    unsafe {
        let mut sums = [[f32x4_splat(0.); 2]; R];
        for g in 0..groups {
            let scales = w.scales.as_ptr().add((panel * groups + g) * 8);
            let sw = [v128_load(scales.cast()), v128_load(scales.add(4).cast())];

            let mut acc = [[i32x4_splat(0); 2]; R];
            for k in (0..w.block).step_by(2) {
                let local = (g * w.block + k) * 8;
                let low = v128_load(wide.as_ptr().add(local).cast());
                let high = v128_load(wide.as_ptr().add(local + 8).cast());
                for r in 0..R {
                    let q = a
                        .as_ptr()
                        .add(((first + r) * groups + g) * w.block + k)
                        .cast::<i32>()
                        .read_unaligned();
                    let input = i32x4_splat(q);
                    acc[r][0] = i32x4_add(acc[r][0], i32x4_dot_i16x8(input, low));
                    acc[r][1] = i32x4_add(acc[r][1], i32x4_dot_i16x8(input, high));
                }
            }
            for r in 0..R {
                let scale = f32x4_splat(sa[(first + r) * groups + g]);
                for part in 0..2 {
                    sums[r][part] = f32x4_add(
                        sums[r][part],
                        f32x4_mul(
                            f32x4_convert_i32x4(acc[r][part]),
                            f32x4_mul(sw[part], scale),
                        ),
                    );
                }
            }
        }
        for r in 0..R {
            let mut values = [0.; 8];
            v128_store(values.as_mut_ptr().cast(), sums[r][0]);
            v128_store(values.as_mut_ptr().add(4).cast(), sums[r][1]);
            for n in 0..8.min(w.rows - panel * 8) {
                let col = panel * 8 + n;
                dst[r * w.rows + col] = values[n] + bias.map_or(0., |b| b[col]);
            }
        }
    }
}

fn rows4(
    w: &Matrix,
    a: &[i16],
    sa: &[f32],
    first: usize,
    dst: &mut [f32],
    bias: Option<&[f32]>,
    panel: usize,
    wide: &[i16],
) {
    let groups = w.groups();
    let padded = groups * w.block;
    unsafe {
        let mut sum00 = f32x4_splat(0.);
        let mut sum01 = f32x4_splat(0.);
        let mut sum10 = f32x4_splat(0.);
        let mut sum11 = f32x4_splat(0.);
        let mut sum20 = f32x4_splat(0.);
        let mut sum21 = f32x4_splat(0.);
        let mut sum30 = f32x4_splat(0.);
        let mut sum31 = f32x4_splat(0.);
        let mut input_ptr = a.as_ptr().add(first * padded);
        let mut weight = wide.as_ptr();
        for g in 0..groups {
            let mut acc00 = i32x4_splat(0);
            let mut acc01 = i32x4_splat(0);
            let mut acc10 = i32x4_splat(0);
            let mut acc11 = i32x4_splat(0);
            let mut acc20 = i32x4_splat(0);
            let mut acc21 = i32x4_splat(0);
            let mut acc30 = i32x4_splat(0);
            let mut acc31 = i32x4_splat(0);
            let end = weight.add(w.block * 8);
            while weight != end {
                let low = v128_load(weight.cast());
                let high = v128_load(weight.add(8).cast());
                let lanes = v128_load(input_ptr.cast());
                let input = i32x4_shuffle::<0, 0, 0, 0>(lanes, lanes);
                acc00 = i32x4_add(acc00, i32x4_dot_i16x8(input, low));
                acc01 = i32x4_add(acc01, i32x4_dot_i16x8(input, high));
                let input = i32x4_shuffle::<1, 1, 1, 1>(lanes, lanes);
                acc10 = i32x4_add(acc10, i32x4_dot_i16x8(input, low));
                acc11 = i32x4_add(acc11, i32x4_dot_i16x8(input, high));
                let input = i32x4_shuffle::<2, 2, 2, 2>(lanes, lanes);
                acc20 = i32x4_add(acc20, i32x4_dot_i16x8(input, low));
                acc21 = i32x4_add(acc21, i32x4_dot_i16x8(input, high));
                let input = i32x4_shuffle::<3, 3, 3, 3>(lanes, lanes);
                acc30 = i32x4_add(acc30, i32x4_dot_i16x8(input, low));
                acc31 = i32x4_add(acc31, i32x4_dot_i16x8(input, high));
                weight = weight.add(16);
                input_ptr = input_ptr.add(8);
            }
            let scale = w.scales.as_ptr().add((panel * groups + g) * 8);
            let low = v128_load(scale.cast());
            let high = v128_load(scale.add(4).cast());
            let scale = f32x4_splat(*sa.get_unchecked((first + 0) * groups + g));
            sum00 = f32x4_add(
                sum00,
                f32x4_mul(f32x4_convert_i32x4(acc00), f32x4_mul(low, scale)),
            );
            sum01 = f32x4_add(
                sum01,
                f32x4_mul(f32x4_convert_i32x4(acc01), f32x4_mul(high, scale)),
            );
            let scale = f32x4_splat(*sa.get_unchecked((first + 1) * groups + g));
            sum10 = f32x4_add(
                sum10,
                f32x4_mul(f32x4_convert_i32x4(acc10), f32x4_mul(low, scale)),
            );
            sum11 = f32x4_add(
                sum11,
                f32x4_mul(f32x4_convert_i32x4(acc11), f32x4_mul(high, scale)),
            );
            let scale = f32x4_splat(*sa.get_unchecked((first + 2) * groups + g));
            sum20 = f32x4_add(
                sum20,
                f32x4_mul(f32x4_convert_i32x4(acc20), f32x4_mul(low, scale)),
            );
            sum21 = f32x4_add(
                sum21,
                f32x4_mul(f32x4_convert_i32x4(acc21), f32x4_mul(high, scale)),
            );
            let scale = f32x4_splat(*sa.get_unchecked((first + 3) * groups + g));
            sum30 = f32x4_add(
                sum30,
                f32x4_mul(f32x4_convert_i32x4(acc30), f32x4_mul(low, scale)),
            );
            sum31 = f32x4_add(
                sum31,
                f32x4_mul(f32x4_convert_i32x4(acc31), f32x4_mul(high, scale)),
            );
        }
        let column = panel * 8;
        if column + 8 <= w.rows {
            let b0 = bias.map_or(f32x4_splat(0.), |b| {
                v128_load(b.as_ptr().add(column).cast())
            });
            let b1 = bias.map_or(f32x4_splat(0.), |b| {
                v128_load(b.as_ptr().add(column + 4).cast())
            });
            v128_store(
                dst.as_mut_ptr().add(0 * w.rows + column + 0).cast(),
                f32x4_add(sum00, b0),
            );
            v128_store(
                dst.as_mut_ptr().add(0 * w.rows + column + 4).cast(),
                f32x4_add(sum01, b1),
            );
            v128_store(
                dst.as_mut_ptr().add(1 * w.rows + column + 0).cast(),
                f32x4_add(sum10, b0),
            );
            v128_store(
                dst.as_mut_ptr().add(1 * w.rows + column + 4).cast(),
                f32x4_add(sum11, b1),
            );
            v128_store(
                dst.as_mut_ptr().add(2 * w.rows + column + 0).cast(),
                f32x4_add(sum20, b0),
            );
            v128_store(
                dst.as_mut_ptr().add(2 * w.rows + column + 4).cast(),
                f32x4_add(sum21, b1),
            );
            v128_store(
                dst.as_mut_ptr().add(3 * w.rows + column + 0).cast(),
                f32x4_add(sum30, b0),
            );
            v128_store(
                dst.as_mut_ptr().add(3 * w.rows + column + 4).cast(),
                f32x4_add(sum31, b1),
            );
        } else {
            let mut values = [0.; 32];
            v128_store(values.as_mut_ptr().add(0).cast(), sum00);
            v128_store(values.as_mut_ptr().add(4).cast(), sum01);
            v128_store(values.as_mut_ptr().add(8).cast(), sum10);
            v128_store(values.as_mut_ptr().add(12).cast(), sum11);
            v128_store(values.as_mut_ptr().add(16).cast(), sum20);
            v128_store(values.as_mut_ptr().add(20).cast(), sum21);
            v128_store(values.as_mut_ptr().add(24).cast(), sum30);
            v128_store(values.as_mut_ptr().add(28).cast(), sum31);
            for r in 0..4 {
                for n in 0..w.rows - column {
                    dst[r * w.rows + column + n] =
                        values[r * 8 + n] + bias.map_or(0., |b| b[column + n]);
                }
            }
        }
    }
}

#[cfg(feature = "kernel-checks")]
pub(super) fn check() -> anyhow::Result<usize> {
    let mut checked =
        check_quantization_boundaries()? + check_shared_input()? + check_interleaved_pack()?;
    for (m, n, k) in [
        (1, 1, 1),
        (3, 11, 33),
        (7, 17, 127),
        (8, 16, 64),
        (13, 65, 384),
        (65, 17, 127),
        (129, 11, 33),
        (255, 9, 33),
        (256, 9, 33),
        (257, 9, 33),
        (511, 9, 33),
        (512, 9, 33),
        (513, 9, 33),
    ] {
        for mode in [Mode::Weights, Mode::WeightsActivations] {
            for block in [16, 32, 64, 128] {
                for extreme in [false, true] {
                    let x: Vec<_> = (0..m * k)
                        .map(|i| {
                            if extreme {
                                if i % 3 == 0 { -127. } else { 127. }
                            } else {
                                ((i * 17 % 129) as f32 - 64.) / 65.
                            }
                        })
                        .collect();
                    let v: Vec<_> = (0..n * k)
                        .map(|i| {
                            if extreme {
                                if i % 7 == 0 {
                                    0.
                                } else if i % 3 == 0 {
                                    127.
                                } else {
                                    -127.
                                }
                            } else {
                                ((i * 31 % 127) as f32 - 63.) / 71.
                            }
                        })
                        .collect();
                    let bias: Vec<_> = (0..n).map(|i| i as f32 / 71.).collect();
                    let w = Matrix::new(&v, n, k, block, mode, false)?;
                    let mut actual = vec![];
                    let mut workspace = vec![];
                    w.apply(&x, &mut actual, Some(&bias), &mut workspace);
                    let bytes = if mode == Mode::WeightsActivations {
                        m * w.groups() * block
                    } else {
                        0
                    };
                    let mut reference_scales = vec![0.; if bytes > 0 { m * w.groups() } else { 0 }];
                    let mut reference_wide = vec![0_i16; bytes];
                    let (packed, scales): (Vec<i8>, &[f32]) = if mode == Mode::WeightsActivations {
                        anyhow::ensure!(pack_activations(
                            &x,
                            k,
                            block,
                            &mut reference_wide,
                            &mut reference_scales
                        ));
                        (
                            reference_wide.iter().map(|&v| v as i8).collect(),
                            &reference_scales,
                        )
                    } else {
                        (vec![], &[])
                    };
                    if mode == Mode::WeightsActivations {
                        for row in 0..m {
                            for g in 0..w.groups() {
                                let start = g * block;
                                let values = &x[row * k + start..row * k + (start + block).min(k)];
                                let s = super::scale(values);
                                anyhow::ensure!(
                                    scales[row * w.groups() + g].to_bits() == s.to_bits(),
                                    "INT8 scale changed"
                                );
                                for j in 0..block {
                                    let q = if j < values.len() {
                                        super::quantize(values[j], s)
                                    } else {
                                        0
                                    };
                                    anyhow::ensure!(
                                        packed[(row * w.groups() + g) * block + j] == q,
                                        "INT8 activation rounding changed"
                                    );
                                }
                            }
                        }
                    }
                    let mut reference = vec![0.; m * n];
                    w.portable(&x, &packed, scales, 0, &mut reference, Some(&bias));
                    for tile in 1..=4 {
                        w.apply_with_rows(&x, &mut actual, Some(&bias), &mut workspace, tile);
                        for (a, b) in actual.iter().zip(&reference) {
                            anyhow::ensure!(
                                if mode == Mode::WeightsActivations {
                                    a.to_bits() == b.to_bits()
                                } else {
                                    (a - b).abs() <= 1e-4 + b.abs() * 1e-5
                                },
                                "INT8 SIMD differs: {m}x{k}x{n}, {block}, {mode:?}: {a} vs {b}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    let edge = [
        -127.,
        -3.5,
        -2.5,
        -0.50000006,
        -0.5,
        -0.49999997,
        -0.,
        0.,
        0.49999997,
        0.5,
        0.50000006,
        2.5,
        3.5,
        127.,
    ];
    let mut packed = vec![0; 32];
    let mut scales = vec![0.; 1];
    anyhow::ensure!(
        pack_activations(&edge, edge.len(), 32, &mut packed, &mut scales),
        "finite quantization rejected"
    );
    anyhow::ensure!(scales[0] == 1., "rounding fixture scale");
    for (&a, &x) in packed.iter().zip(&edge) {
        anyhow::ensure!(a == super::quantize(x, 1.) as i16, "rounding edge {x}");
    }
    let w = Matrix::new(
        &vec![1.; 33 * 5],
        5,
        33,
        32,
        Mode::WeightsActivations,
        false,
    )?;
    for exceptional in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in [0, 31, 32, 65] {
            let mut x = vec![1.; 66];
            x[index] = exceptional;
            let mut y = vec![];
            w.apply(&x, &mut y, None, &mut Vec::new());
            anyhow::ensure!(
                y.iter().all(|v| v.is_nan()),
                "nonfinite quantization must propagate"
            );
        }
    }
    Ok(checked)
}

#[cfg(feature = "kernel-checks")]
fn check_interleaved_pack() -> anyhow::Result<usize> {
    let mut checked = 0;
    let mut seed = 0x6d2b79f5_u32;
    for (m, k) in [
        (1_usize, 1_usize),
        (3, 15),
        (4, 16),
        (5, 17),
        (7, 31),
        (8, 32),
        (9, 33),
        (32, 127),
        (129, 384),
        (128, 1536),
    ] {
        for block in [16, 32, 64, 128] {
            for trial in 0..4 {
                let values: Vec<_> = (0..m * k)
                    .map(|i| {
                        if trial == 0 {
                            return if i % block == 0 {
                                127.
                            } else {
                                (i % 127) as f32 - 63.5
                            };
                        }
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        f32::from_bits(if seed & 0x7f800000 == 0x7f800000 {
                            seed ^ 0x00800000
                        } else {
                            seed
                        })
                    })
                    .collect();
                let groups = k.div_ceil(block);
                let padded = groups * block;
                let mut reference = vec![0_i16; m * padded];
                let mut actual = reference.clone();
                let mut expected_scales = vec![0.; m * groups];
                let mut actual_scales = expected_scales.clone();
                anyhow::ensure!(pack_activations(
                    &values,
                    k,
                    block,
                    &mut reference,
                    &mut expected_scales
                ));
                anyhow::ensure!(pack_for_rows(
                    &values,
                    k,
                    block,
                    &mut actual,
                    &mut actual_scales,
                    4
                ));
                anyhow::ensure!(
                    actual_scales
                        .iter()
                        .zip(&expected_scales)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "interleaved scales differ"
                );
                for row in 0..m {
                    for c in 0..padded {
                        let index = if row < m / 4 * 4 {
                            (row / 4 * 4) * padded + (c / 2) * 8 + (row % 4) * 2 + c % 2
                        } else {
                            row * padded + c
                        };
                        anyhow::ensure!(
                            reference[row * padded + c] == actual[index],
                            "interleaved pack differs: {m}x{k}, block={block}, row={row}, col={c}"
                        );
                        checked += 1;
                    }
                }
                for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut bad = values.clone();
                    bad[m * k - 1] = invalid;
                    anyhow::ensure!(
                        !pack_for_rows(&bad, k, block, &mut actual, &mut actual_scales, 4),
                        "interleaved packing hid nonfinite input"
                    );
                }
            }
        }
    }
    Ok(checked)
}

#[cfg(feature = "kernel-checks")]
fn check_quantization_boundaries() -> anyhow::Result<usize> {
    let mut values = [0_f32; 32];
    let mut packed = [0_i16; 32];
    let mut scales = [0_f32; 1];
    let mut checked = 0;
    let mut compare = |values: &[f32; 32]| -> anyhow::Result<()> {
        anyhow::ensure!(pack_activations(values, 32, 32, &mut packed, &mut scales));
        let scale = super::scale(values);
        anyhow::ensure!(scales[0].to_bits() == scale.to_bits(), "scale drift");
        for (i, (&x, &q)) in values.iter().zip(&packed).enumerate() {
            let expected = super::quantize(x, scale) as i16;
            anyhow::ensure!(
                q == expected,
                "quantization boundary {i}: {x:e}/{scale:e} -> {q}, expected {expected}"
            );
            checked += 1;
        }
        Ok(())
    };
    for exponent in (-126_i32..=120).step_by(5) {
        let scale = f32::from_bits(((exponent + 127) as u32) << 23);
        for integer in -127..127 {
            values[0] = 127. * scale;
            values[1] = -127. * scale;
            let center = (integer as f32 + 0.5) * scale;
            for (i, v) in values[2..].iter_mut().enumerate() {
                *v = f32::from_bits(center.to_bits().wrapping_add_signed((i % 5) as i32 - 2));
            }
            compare(&values)?;
        }
    }
    let mut seed = 0x6d2b79f5_u32;
    for _ in 0..4096 {
        for v in &mut values {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = f32::from_bits(if seed & 0x7f800000 == 0x7f800000 {
                seed ^ 0x00800000
            } else {
                seed
            });
        }
        compare(&values)?;
    }
    Ok(checked)
}

#[cfg(feature = "kernel-checks")]
fn check_shared_input() -> anyhow::Result<usize> {
    let mut checked = 0;
    for (m, k) in [(1, 33), (7, 127), (128, 384)] {
        let input: Vec<_> = (0..m * k)
            .map(|i| ((i * 37 % 131) as f32 - 65.) / 67.)
            .collect();
        let mut matrices = Vec::new();
        let mut biases = Vec::new();
        for (seed, n) in [9, 13, 17].into_iter().enumerate() {
            let weights: Vec<_> = (0..n * k)
                .map(|i| ((i * 53 + seed * 31) % 127) as f32 / 59. - 1.)
                .collect();
            matrices.push(Matrix::new(
                &weights,
                n,
                k,
                32,
                Mode::WeightsActivations,
                false,
            )?);
            biases.push((0..n).map(|i| i as f32 / 17.).collect::<Vec<_>>());
        }
        let mut work = Vec::new();
        for tile in 1..=4 {
            let mut reference = [Vec::new(), Vec::new(), Vec::new()];
            for i in 0..3 {
                matrices[i].apply_with_rows(
                    &input,
                    &mut reference[i],
                    Some(&biases[i]),
                    &mut work,
                    tile,
                );
            }
            let [mut q, mut key, mut value] = [Vec::new(), Vec::new(), Vec::new()];
            anyhow::ensure!(Matrix::apply_shared(
                [&matrices[0], &matrices[1], &matrices[2]],
                &input,
                [&mut q, &mut key, &mut value],
                [&biases[0], &biases[1], &biases[2]],
                &mut work,
                tile,
            ));
            for (actual, expected) in [q, key, value].iter().zip(&reference) {
                anyhow::ensure!(
                    actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "shared Q/K/V quantization changed outputs"
                );
                checked += actual.len();
            }
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = input.clone();
            bad[m * k - 1] = invalid;
            for (matrix, bias) in matrices.iter().zip(&biases) {
                let mut output = Vec::new();
                matrix.apply_with_rows(&bad, &mut output, Some(bias), &mut work, 4);
                anyhow::ensure!(
                    output.iter().all(|v| v.is_nan()),
                    "worker-local packing must invalidate every output row"
                );
            }
            let [mut q, mut key, mut value] = [Vec::new(), Vec::new(), Vec::new()];
            anyhow::ensure!(Matrix::apply_shared(
                [&matrices[0], &matrices[1], &matrices[2]],
                &bad,
                [&mut q, &mut key, &mut value],
                [&biases[0], &biases[1], &biases[2]],
                &mut work,
                4,
            ));
            anyhow::ensure!(
                [q, key, value].iter().all(|v| v.iter().all(|x| x.is_nan())),
                "shared quantization hid a nonfinite activation"
            );
        }
    }
    Ok(checked)
}
