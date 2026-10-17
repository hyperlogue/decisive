use super::{Matrix, Mode};
use core::arch::wasm32::*;

pub(super) fn pack_activations(
    x: &[f32],
    cols: usize,
    block: usize,
    packed: &mut [i8],
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
                for k in (0..end).step_by(4) {
                    let v = f32x4_div(v128_load(values.as_ptr().add(k).cast()), f32x4_splat(s));
                    let truncated = i32x4_trunc_sat_f32x4(v);
                    let fraction = f32x4_abs(f32x4_sub(v, f32x4_convert_i32x4(truncated)));
                    let increment = v128_or(i32x4_shr(v, 31), i32x4_splat(1));
                    let rounded = i32x4_add(
                        truncated,
                        v128_and(increment, f32x4_ge(fraction, f32x4_splat(0.5))),
                    );
                    let shorts = i16x8_narrow_i32x4(rounded, i32x4_splat(0));
                    let bytes = i8x16_narrow_i16x8(shorts, i16x8_splat(0));
                    out.as_mut_ptr()
                        .add(k)
                        .cast::<i32>()
                        .write_unaligned(i32x4_extract_lane::<0>(bytes));
                }
                for k in end..values.len() {
                    out[k] = super::quantize(values[k], s);
                }
            }
        }
    }
    true
}

pub(super) fn apply(
    w: &Matrix,
    x: &[f32],
    a: &[i8],
    sa: &[f32],
    dst: &mut [f32],
    bias: Option<&[f32]>,
    tile: usize,
) {
    let input_bytes = if w.mode == Mode::WeightsActivations {
        1
    } else {
        4
    };
    let cache_rows = (262144 / w.cols / input_bytes).clamp(tile, 64) / tile * tile;
    let count = dst.len() / w.rows;
    for base in (0..count).step_by(cache_rows) {
        let end = (base + cache_rows).min(count);
        for panel in 0..w.rows.div_ceil(8) {
            for first in (base..end).step_by(tile) {
                let last = (first + tile).min(end);
                run(
                    w,
                    x,
                    a,
                    sa,
                    first,
                    &mut dst[first * w.rows..last * w.rows],
                    bias,
                    panel,
                );
            }
        }
    }
}
fn run(
    w: &Matrix,
    x: &[f32],
    a: &[i8],
    sa: &[f32],
    first: usize,
    dst: &mut [f32],
    bias: Option<&[f32]>,
    panel: usize,
) {
    match dst.len() / w.rows {
        1 => rows::<1>(w, x, a, sa, first, dst, bias, panel),
        2 => rows::<2>(w, x, a, sa, first, dst, bias, panel),
        3 => rows::<3>(w, x, a, sa, first, dst, bias, panel),
        4 => rows::<4>(w, x, a, sa, first, dst, bias, panel),
        _ => unreachable!(),
    }
}
fn rows<const R: usize>(
    w: &Matrix,
    x: &[f32],
    a: &[i8],
    sa: &[f32],
    first: usize,
    dst: &mut [f32],
    bias: Option<&[f32]>,
    panel: usize,
) {
    let groups = w.groups();
    unsafe {
        let mut sums = [[f32x4_splat(0.); 2]; R];
        for g in 0..groups {
            let base = (panel * groups + g) * w.block * 8;
            let scales = w.scales.as_ptr().add((panel * groups + g) * 8);
            let sw = [v128_load(scales.cast()), v128_load(scales.add(4).cast())];
            if w.mode == Mode::WeightsActivations {
                let mut acc = [[i32x4_splat(0); 2]; R];
                for k in (0..w.block).step_by(2) {
                    let b = v128_load(w.data.as_ptr().add(base + k * 8).cast());
                    let low = i16x8_extend_low_i8x16(b);
                    let high = i16x8_extend_high_i8x16(b);
                    for r in 0..R {
                        let q = a
                            .as_ptr()
                            .add(((first + r) * groups + g) * w.block + k)
                            .cast::<i16>()
                            .read_unaligned();
                        let input = i16x8_extend_low_i8x16(i16x8_splat(q));
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
            } else {
                let mut acc = [[f32x4_splat(0.); 2]; R];
                for k in 0..w.block.min(w.cols - g * w.block) {
                    let b = v128_load64_zero(w.data.as_ptr().add(base + k * 8).cast());
                    let wide = i16x8_extend_low_i8x16(b);
                    let b0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(wide));
                    let b1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(wide));
                    for r in 0..R {
                        let av = f32x4_splat(x[(first + r) * w.cols + g * w.block + k]);
                        acc[r][0] = f32x4_add(acc[r][0], f32x4_mul(av, b0));
                        acc[r][1] = f32x4_add(acc[r][1], f32x4_mul(av, b1));
                    }
                }
                for r in 0..R {
                    for part in 0..2 {
                        sums[r][part] = f32x4_add(sums[r][part], f32x4_mul(acc[r][part], sw[part]));
                    }
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

#[cfg(feature = "kernel-checks")]
pub(super) fn check() -> anyhow::Result<usize> {
    let mut checked = 0;
    for (m, n, k) in [
        (1, 1, 1),
        (3, 11, 33),
        (7, 17, 127),
        (8, 16, 64),
        (13, 65, 384),
        (65, 17, 127),
        (129, 11, 33),
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
                    let (packed, scales) = workspace.split_at(bytes / 4);
                    let packed: &[i8] = bytemuck::cast_slice(packed);
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
                    w.portable(&x, packed, scales, 0, &mut reference, Some(&bias));
                    for tile in 1..=4 {
                        w.apply_with_rows(&x, &mut actual, Some(&bias), &mut workspace, tile);
                        for (a, b) in actual.iter().zip(&reference) {
                            anyhow::ensure!(
                                a.to_bits() == b.to_bits(),
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
        anyhow::ensure!(a == super::quantize(x, 1.), "rounding edge {x}");
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
