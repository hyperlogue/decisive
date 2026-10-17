use super::weights::Weights;
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
use crate::avx2_exp as exp;
use anyhow::{Result, ensure};
#[cfg(feature = "cpu")]
use rayon::prelude::*;

pub(super) fn avx2_available() -> bool {
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    {
        return std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    }
    #[allow(unreachable_code)]
    false
}
pub(super) struct Linear {
    pub input: usize,
    pub output: usize,
    weight: Vec<f32>,
    weight_bf16: Vec<u16>,
    bias: Vec<f32>,
    packed: bool,
}
impl Linear {
    pub fn load(
        w: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        packed: bool,
    ) -> Result<Self> {
        Self::new(
            w.get(&format!("{name}.weight"), &[output, input])?,
            vec![0.; output],
            input,
            output,
            packed,
        )
    }
    pub fn new(
        data: Vec<f32>,
        bias: Vec<f32>,
        input: usize,
        output: usize,
        packed: bool,
    ) -> Result<Self> {
        ensure!(
            data.len() == input * output && bias.len() == output && (!packed || avx2_available()),
            "invalid linear layout/kernel"
        );
        let mut weight_bf16 = Vec::new();
        let weight = if packed && data.iter().all(|v| v.to_bits() & 0xffff == 0) {
            weight_bf16.resize(output.div_ceil(16) * 16 * input, 0);
            for n in 0..output {
                for k in 0..input {
                    weight_bf16[(n / 16 * input + k) * 16 + n % 16] =
                        (data[n * input + k].to_bits() >> 16) as u16;
                }
            }
            Vec::new()
        } else if packed {
            let mut p = vec![0.; output.div_ceil(16) * 16 * input];
            for n in 0..output {
                for k in 0..input {
                    p[(n / 16 * input + k) * 16 + n % 16] = data[n * input + k];
                }
            }
            p
        } else {
            data
        };
        Ok(Self {
            input,
            output,
            weight,
            weight_bf16,
            bias,
            packed,
        })
    }
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn project(
        &self,
        x: &[f32],
        y: *mut f32,
        count: usize,
        first: usize,
        rows: usize,
        panel_start: usize,
        panel_end: usize,
    ) {
        unsafe {
            if self.weight_bf16.is_empty() {
                multiply::<false>(
                    x,
                    self.weight.as_ptr().cast(),
                    y,
                    count,
                    first,
                    self.input,
                    self.output,
                    &self.bias,
                    rows,
                    panel_start,
                    panel_end,
                );
            } else {
                multiply::<true>(
                    x,
                    self.weight_bf16.as_ptr().cast(),
                    y,
                    count,
                    first,
                    self.input,
                    self.output,
                    &self.bias,
                    rows,
                    panel_start,
                    panel_end,
                );
            }
        }
    }
    pub fn apply(&self, x: &[f32], y: &mut Vec<f32>, rows: usize) {
        #[cfg(not(all(feature = "cpu", target_arch = "x86_64")))]
        debug_assert!(!self.packed);
        assert_eq!(x.len() % self.input, 0);
        assert!((1..=6).contains(&rows));
        y.resize(x.len() / self.input * self.output, 0.);
        #[cfg(feature = "cpu")]
        let threads = rayon::current_num_threads();
        #[cfg(not(feature = "cpu"))]
        let threads = 1;
        let tokens = x.len() / self.input;
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if self.packed
            && self.output >= 512
            && (threads > 1 || (tokens > rows && (self.weight_bf16.is_empty() || tokens <= 128)))
        {
            let panels = if threads > 1 && tokens >= 128 && self.output.div_ceil(64) % threads != 0
            {
                1
            } else {
                4
            };
            let dst = Columns(y.as_mut_ptr());
            (0..self.output.div_ceil(panels * 16))
                .into_par_iter()
                .for_each(|tile| {
                    let start = tile * panels;
                    unsafe {
                        self.project(
                            x,
                            dst.pointer(),
                            tokens,
                            0,
                            rows,
                            start,
                            (start + panels).min(self.output.div_ceil(16)),
                        );
                    }
                });
            return;
        }
        let tile_rows = if self.packed {
            rows * tokens.div_ceil(rows * threads * 2).clamp(1, 16)
        } else {
            rows
        };
        let run = |(tile, dst): (usize, &mut [f32])| {
            let first = tile * tile_rows;
            #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
            if self.packed {
                unsafe {
                    self.project(
                        x,
                        dst.as_mut_ptr(),
                        dst.len() / self.output,
                        first,
                        rows,
                        0,
                        self.output.div_ceil(16),
                    );
                }
                return;
            }
            unsafe {
                matrixmultiply::sgemm(
                    dst.len() / self.output,
                    self.input,
                    self.output,
                    1.,
                    x.as_ptr().add(first * self.input),
                    self.input as isize,
                    1,
                    self.weight.as_ptr(),
                    1,
                    self.input as isize,
                    0.,
                    dst.as_mut_ptr(),
                    self.output as isize,
                    1,
                );
            }
            for row in dst.chunks_mut(self.output) {
                for (v, b) in row.iter_mut().zip(&self.bias) {
                    *v += b;
                }
            }
        };
        #[cfg(feature = "cpu")]
        y.par_chunks_mut(tile_rows * self.output)
            .enumerate()
            .for_each(run);
        #[cfg(not(feature = "cpu"))]
        y.chunks_mut(tile_rows * self.output)
            .enumerate()
            .for_each(run);
    }
}
pub(super) fn rms(x: &mut [f32], weight: &[f32], eps: f32, zero_centered: bool) {
    let run = |x: &mut [f32]| {
        for row in x.chunks_exact_mut(weight.len()) {
            let scale =
                1. / (row.iter().map(|v| v * v).sum::<f32>() / weight.len() as f32 + eps).sqrt();
            for (v, w) in row.iter_mut().zip(weight) {
                *v = (*v * scale) * if zero_centered { 1. + w } else { *w };
            }
        }
    };
    #[cfg(feature = "cpu")]
    if x.len() >= 16384 && rayon::current_num_threads() > 1 {
        x.par_chunks_mut(weight.len() * 16).for_each(run);
        return;
    }
    run(x);
}
pub(super) fn rms_from(x: &[f32], y: &mut Vec<f32>, weight: &[f32], eps: f32, zero_centered: bool) {
    assert_eq!(x.len() % weight.len(), 0);
    y.resize(x.len(), 0.);
    let tile = weight.len() * 16;
    let run = |(index, dst): (usize, &mut [f32])| {
        let src = &x[index * tile..index * tile + dst.len()];
        for (src, dst) in src
            .chunks_exact(weight.len())
            .zip(dst.chunks_exact_mut(weight.len()))
        {
            let scale =
                1. / (src.iter().map(|v| v * v).sum::<f32>() / weight.len() as f32 + eps).sqrt();
            for ((v, w), out) in src.iter().zip(weight).zip(dst) {
                *out = (*v * scale) * if zero_centered { 1. + w } else { *w };
            }
        }
    };
    #[cfg(feature = "cpu")]
    if x.len() >= 16384 && rayon::current_num_threads() > 1 {
        y.par_chunks_mut(tile).enumerate().for_each(run);
        return;
    }
    y.chunks_mut(tile).enumerate().for_each(run);
}
pub(super) fn add(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    let run = |(x, y): (&mut [f32], &[f32])| {
        for (a, b) in x.iter_mut().zip(y) {
            *a += b;
        }
    };
    #[cfg(feature = "cpu")]
    if x.len() >= 16384 && rayon::current_num_threads() > 1 {
        x.par_chunks_mut(4096).zip(y.par_chunks(4096)).for_each(run);
        return;
    }
    run((x, y));
}
pub(super) fn silu_mul(gate: &mut [f32], up: &[f32], avx2: bool) {
    assert_eq!(gate.len(), up.len());
    let run = |(g, u): (&mut [f32], &[f32])| {
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if avx2 {
            unsafe {
                silu_mul_avx2(g, u);
            }
            return;
        }
        let _ = avx2;
        for (g, u) in g.iter_mut().zip(u) {
            *g = silu(*g) * u;
        }
    };
    #[cfg(feature = "cpu")]
    if gate.len() >= 16384 && rayon::current_num_threads() > 1 {
        gate.par_chunks_mut(4096)
            .zip(up.par_chunks(4096))
            .for_each(run);
        return;
    }
    run((gate, up));
}
pub(super) fn conv_silu(rows: [&[f32]; 4], weight: &[f32], out: &mut [f32], avx2: bool) {
    let width = out.len();
    assert!(rows.iter().all(|r| r.len() == width));
    assert_eq!(weight.len(), 4 * width);
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    if avx2 {
        unsafe {
            conv_silu_avx2(rows, weight, out);
        }
        return;
    }
    let _ = avx2;
    for (c, out) in out.iter_mut().enumerate() {
        let mut sum = 0.;
        for tap in 0..4 {
            sum += rows[tap][c] * weight[tap * width + c];
        }
        *out = silu(sum);
    }
}
pub(super) fn delta_gate(
    src: &[f32],
    weight: &[f32],
    scale: f32,
    z: &[f32],
    out: &mut [f32],
    avx2: bool,
) {
    assert_eq!(src.len(), out.len());
    assert_eq!(weight.len(), out.len());
    assert_eq!(z.len(), out.len());
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    if avx2 {
        unsafe {
            delta_gate_avx2(src, weight, scale, z, out);
        }
        return;
    }
    let _ = avx2;
    for i in 0..out.len() {
        out[i] = ((src[i] * scale) * weight[i]) * silu(z[i]);
    }
}
pub(super) fn attention_gate(src: &[f32], z: &[f32], out: &mut [f32], avx2: bool) {
    assert_eq!(src.len(), out.len());
    assert_eq!(z.len(), out.len());
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    if avx2 {
        unsafe {
            attention_gate_avx2(src, z, out);
        }
        return;
    }
    let _ = avx2;
    for i in 0..out.len() {
        out[i] = src[i] * sigmoid(z[i]);
    }
}
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul_avx2(gate: &mut [f32], up: &[f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let end = gate.len() / 8 * 8;
        for i in (0..end).step_by(8) {
            let g = _mm256_loadu_ps(gate.as_ptr().add(i));
            let activated = _mm256_mul_ps(g, exp::sigmoid8(g));
            _mm256_storeu_ps(
                gate.as_mut_ptr().add(i),
                _mm256_mul_ps(activated, _mm256_loadu_ps(up.as_ptr().add(i))),
            );
        }
        for i in end..gate.len() {
            gate[i] = silu(gate[i]) * up[i];
        }
    }
}
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_silu_avx2(rows: [&[f32]; 4], weight: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let width = out.len();
        let end = width / 8 * 8;
        for c in (0..end).step_by(8) {
            let mut sum = _mm256_setzero_ps();
            for tap in 0..4 {
                let x = _mm256_loadu_ps(rows[tap].as_ptr().add(c));
                let w = _mm256_loadu_ps(weight.as_ptr().add(tap * width + c));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(x, w));
            }
            _mm256_storeu_ps(
                out.as_mut_ptr().add(c),
                _mm256_mul_ps(sum, exp::sigmoid8(sum)),
            );
        }
        for c in end..width {
            let mut sum = 0.;
            for tap in 0..4 {
                sum += rows[tap][c] * weight[tap * width + c];
            }
            out[c] = silu(sum);
        }
    }
}
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn delta_gate_avx2(src: &[f32], weight: &[f32], scale: f32, z: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let end = out.len() / 8 * 8;
        let sc = _mm256_set1_ps(scale);
        for i in (0..end).step_by(8) {
            let x = _mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i)), sc);
            let x = _mm256_mul_ps(x, _mm256_loadu_ps(weight.as_ptr().add(i)));
            let gate = _mm256_loadu_ps(z.as_ptr().add(i));
            let activated = _mm256_mul_ps(gate, exp::sigmoid8(gate));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(x, activated));
        }
        for i in end..out.len() {
            out[i] = ((src[i] * scale) * weight[i]) * silu(z[i]);
        }
    }
}
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn attention_gate_avx2(src: &[f32], z: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let end = out.len() / 8 * 8;
        for i in (0..end).step_by(8) {
            let gate = _mm256_loadu_ps(z.as_ptr().add(i));
            let value = _mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i)), exp::sigmoid8(gate));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), value);
        }
        for i in end..out.len() {
            out[i] = src[i] * sigmoid(z[i]);
        }
    }
}

pub(super) fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
pub(super) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
pub(super) fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.;
    for v in &mut *x {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x {
        *v /= sum;
    }
}
// Jobs write disjoint columns and join before Linear::apply releases the output borrow.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
struct Columns(*mut f32);
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
unsafe impl Sync for Columns {}
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
impl Columns {
    fn pointer(&self) -> *mut f32 {
        self.0
    }
}
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn multiply<const BF16: bool>(
    x: &[f32],
    w: *const u8,
    y: *mut f32,
    count: usize,
    first: usize,
    k: usize,
    n: usize,
    bias: &[f32],
    rows: usize,
    panel_start: usize,
    panel_end: usize,
) {
    let full = count / rows * rows;
    let tail = count - full;
    for (offset, count, tile) in [(0, full, rows), (full, tail, tail)] {
        unsafe {
            let dst = y.add(offset * n);
            let start = first + offset;
            match tile {
                0 => {}
                1 => multiply_rows::<1, BF16>(
                    x,
                    w,
                    dst,
                    count,
                    start,
                    k,
                    n,
                    bias,
                    panel_start,
                    panel_end,
                ),
                2 => multiply_rows::<2, BF16>(
                    x,
                    w,
                    dst,
                    count,
                    start,
                    k,
                    n,
                    bias,
                    panel_start,
                    panel_end,
                ),
                3 => multiply_rows::<3, BF16>(
                    x,
                    w,
                    dst,
                    count,
                    start,
                    k,
                    n,
                    bias,
                    panel_start,
                    panel_end,
                ),
                4 => multiply_rows::<4, BF16>(
                    x,
                    w,
                    dst,
                    count,
                    start,
                    k,
                    n,
                    bias,
                    panel_start,
                    panel_end,
                ),
                5 => multiply_rows::<5, BF16>(
                    x,
                    w,
                    dst,
                    count,
                    start,
                    k,
                    n,
                    bias,
                    panel_start,
                    panel_end,
                ),
                6 => multiply_rows::<6, BF16>(
                    x,
                    w,
                    dst,
                    count,
                    start,
                    k,
                    n,
                    bias,
                    panel_start,
                    panel_end,
                ),
                _ => unreachable!("six-row projection tile"),
            }
        }
    }
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn multiply_rows<const R: usize, const BF16: bool>(
    x: &[f32],
    w: *const u8,
    y: *mut f32,
    count: usize,
    first: usize,
    k: usize,
    n: usize,
    bias: &[f32],
    panel_start: usize,
    panel_end: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        for panel in panel_start..panel_end {
            for row in (0..count).step_by(R) {
                let first = first + row;
                let y = y.add(row * n);
                let mut acc = [[_mm256_setzero_ps(); 2]; R];
                for j in 0..k {
                    let b = w.add((panel * k + j) * 16 * if BF16 { 2 } else { 4 });
                    let (b0, b1) = if BF16 {
                        (
                            _mm256_castsi256_ps(_mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(_mm_loadu_si128(b.cast())),
                                16,
                            )),
                            _mm256_castsi256_ps(_mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(_mm_loadu_si128(b.add(16).cast())),
                                16,
                            )),
                        )
                    } else {
                        (_mm256_loadu_ps(b.cast()), _mm256_loadu_ps(b.add(32).cast()))
                    };
                    for i in 0..R {
                        let a = _mm256_set1_ps(*x.get_unchecked((first + i) * k + j));
                        acc[i][0] = _mm256_fmadd_ps(a, b0, acc[i][0]);
                        acc[i][1] = _mm256_fmadd_ps(a, b1, acc[i][1]);
                    }
                }
                for i in 0..R {
                    if panel * 16 + 16 <= n {
                        let b = bias.as_ptr().add(panel * 16);
                        let dst = y.add(i * n + panel * 16);
                        _mm256_storeu_ps(dst, _mm256_add_ps(acc[i][0], _mm256_loadu_ps(b)));
                        _mm256_storeu_ps(
                            dst.add(8),
                            _mm256_add_ps(acc[i][1], _mm256_loadu_ps(b.add(8))),
                        );
                    } else {
                        let mut tmp = [0.; 16];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), acc[i][0]);
                        _mm256_storeu_ps(tmp.as_mut_ptr().add(8), acc[i][1]);
                        for c in 0..n - panel * 16 {
                            *y.add(i * n + panel * 16 + c) = tmp[c] + bias[panel * 16 + c];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn value(i: usize) -> f32 {
        (((i as u64 * 48271 + 731) % 65521) as f32 - 32760.) / 32768.
    }
    #[test]
    fn projection_layout_tails_and_bias_match_f64() -> Result<()> {
        let packed = avx2_available();
        for (m, k, n) in [
            (1, 7, 1),
            (2, 31, 15),
            (5, 259, 17),
            (7, 31, 35),
            (23, 7, 16),
            (24, 31, 32),
            (25, 259, 35),
            (95, 31, 17),
            (96, 7, 16),
            (97, 31, 17),
            (13, 3584, 19),
            (17, 31, 515),
            (13, 1024, 1031),
            (257, 31, 1031),
        ] {
            for bf16 in [false, true] {
                let input: Vec<_> = (0..m * k).map(value).collect();
                let weight: Vec<_> = (0..k * n)
                    .map(|i| {
                        let v = value(i + 13) * 0.02;
                        if bf16 {
                            f32::from_bits(v.to_bits() & 0xffff0000)
                        } else {
                            v
                        }
                    })
                    .collect();
                let bias: Vec<_> = (0..n).map(|i| value(i + 73) * 0.03).collect();
                let linear = Linear::new(weight.clone(), bias.clone(), k, n, packed)?;
                for rows in 1..=6 {
                    let mut output = vec![f32::NAN; m * n];
                    linear.apply(&input, &mut output, rows);
                    for r in 0..m {
                        for c in 0..n {
                            let expected = (0..k)
                                .map(|i| input[r * k + i] as f64 * weight[c * k + i] as f64)
                                .sum::<f64>()
                                + bias[c] as f64;
                            if packed {
                                let mut exact = 0f32;
                                for i in 0..k {
                                    exact = input[r * k + i].mul_add(weight[c * k + i], exact);
                                }
                                exact += bias[c];
                                assert_eq!(
                                    output[r * n + c].to_bits(),
                                    exact.to_bits(),
                                    "FP32 accumulation order changed"
                                );
                            }
                            assert!(
                                (output[r * n + c] as f64 - expected).abs() < 1e-4,
                                "m={m} k={k} n={n} tile={rows} r={r} c={c}: {} != {expected}",
                                output[r * n + c]
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn convolution_layout_and_gates_preserve_scalar_results() {
        for width in [1, 7, 8, 17, 33, 128, 6144] {
            let rows: [Vec<f32>; 4] =
                std::array::from_fn(|t| (0..width).map(|i| value(i + t * width)).collect());
            let original: Vec<_> = (0..4 * width).map(|i| value(i + 37) * 0.1).collect();
            let packed: Vec<_> = (0..4)
                .flat_map(|tap| {
                    (0..width).map({
                        let original = &original;
                        move |c| original[c * 4 + tap]
                    })
                })
                .collect();
            let expected: Vec<_> = (0..width)
                .map(|c| {
                    let mut sum = 0.;
                    for tap in 0..4 {
                        sum += rows[tap][c] * original[c * 4 + tap];
                    }
                    silu(sum)
                })
                .collect();
            for avx2 in [false, avx2_available()] {
                let mut actual = vec![f32::NAN; width];
                conv_silu(
                    std::array::from_fn(|i| rows[i].as_slice()),
                    &packed,
                    &mut actual,
                    avx2,
                );
                assert_eq!(actual, expected, "convolution width={width}");
                let weight: Vec<_> = (0..width).map(|i| value(i + 87)).collect();
                let expected: Vec<_> = (0..width)
                    .map(|i| ((rows[0][i] * 0.43) * weight[i]) * silu(rows[1][i]))
                    .collect();
                delta_gate(&rows[0], &weight, 0.43, &rows[1], &mut actual, avx2);
                assert_eq!(actual, expected, "DeltaNet output gate width={width}");
                let expected: Vec<_> = (0..width)
                    .map(|i| rows[0][i] * sigmoid(rows[1][i]))
                    .collect();
                attention_gate(&rows[0], &rows[1], &mut actual, avx2);
                assert_eq!(actual, expected, "attention output gate width={width}");
            }
        }
    }
    #[test]
    fn independent_pointwise_tiles_preserve_scalar_results() {
        for (rows, width) in [(1, 128), (137, 128), (71, 256), (17, 1024)] {
            let input: Vec<_> = (0..rows * width).map(value).collect();
            let weight: Vec<_> = (0..width).map(|i| value(i + 41)).collect();
            for centered in [false, true] {
                let mut expected = input.clone();
                for row in expected.chunks_exact_mut(width) {
                    let scale =
                        1. / (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-6).sqrt();
                    for (v, w) in row.iter_mut().zip(&weight) {
                        *v = (*v * scale) * if centered { 1. + w } else { *w };
                    }
                }
                let mut actual = input.clone();
                rms(&mut actual, &weight, 1e-6, centered);
                assert_eq!(actual, expected);
                actual.fill(f32::NAN);
                rms_from(&input, &mut actual, &weight, 1e-6, centered);
                assert_eq!(actual, expected);
            }
            let rhs: Vec<_> = (0..input.len()).map(|i| value(i + 29)).collect();
            let expected: Vec<_> = input.iter().zip(&rhs).map(|(a, b)| a + b).collect();
            let mut actual = input.clone();
            add(&mut actual, &rhs);
            assert_eq!(actual, expected);
            let expected: Vec<_> = input.iter().zip(&rhs).map(|(a, b)| silu(*a) * b).collect();
            let mut actual = input.clone();
            silu_mul(&mut actual, &rhs, avx2_available());
            assert_eq!(actual, expected);
        }
    }
}
