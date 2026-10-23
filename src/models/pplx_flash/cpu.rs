use super::weights::Weights;
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
use crate::avx2_exp as exp;
use anyhow::Result;
#[cfg(feature = "cpu")]
use rayon::prelude::*;

#[derive(Default)]
pub(super) struct ProjectionScratch {
    #[cfg(feature = "cpu")]
    pub profile: bool,
    #[cfg(feature = "cpu")]
    pub projection_ms: f64,
    #[cfg(all(feature = "cpu", feature = "int8"))]
    int8: super::int8::Scratch,
}

#[cfg(any(feature = "cpu", test))]
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
    weight_bf16: Vec<u16>,
    #[cfg(all(feature = "cpu", feature = "int8"))]
    int8: Option<super::int8::Matrix>,
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    avx2: bool,
}
impl Linear {
    pub fn load(
        w: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        avx2: bool,
    ) -> Result<Self> {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let _ = avx2;
        let data = w.bytes(&format!("{name}.weight"), &[output, input])?;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let weight_bf16 = super::wasm_simd::pack(data, input, output);
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        let weight_bf16 = {
            let mut weight_bf16 = vec![0; output.div_ceil(16) * 16 * input];
            for panel in 0..output.div_ceil(16) {
                let width = 16.min(output - panel * 16);
                for k in 0..input {
                    for c in 0..width {
                        let at = ((panel * 16 + c) * input + k) * 2;
                        unsafe {
                            *weight_bf16.get_unchecked_mut((panel * input + k) * 16 + c) =
                                u16::from_le(std::ptr::read_unaligned(
                                    data.as_ptr().add(at).cast(),
                                ));
                        }
                    }
                }
            }
            weight_bf16
        };
        Ok(Self {
            input,
            output,
            weight_bf16,
            #[cfg(all(feature = "cpu", feature = "int8"))]
            int8: None,
            #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
            avx2,
        })
    }
    #[cfg(all(feature = "cpu", feature = "int8"))]
    pub fn load_quantized(
        w: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        avx2: bool,
        weight_bits: u8,
        activation_bits: u8,
        optimized: bool,
        affine: bool,
    ) -> Result<Self> {
        Ok(Self {
            input,
            output,
            weight_bf16: Vec::new(),
            int8: Some(super::int8::Matrix::from_bf16_optimized(
                w.bytes(&format!("{name}.weight"), &[output, input])?,
                input,
                output,
                avx2,
                weight_bits,
                activation_bits,
                optimized,
                affine,
            )?),
            #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
            avx2,
        })
    }
    pub fn apply_pair(
        linears: [&Self; 2],
        x: &[f32],
        outputs: [&mut Vec<f32>; 2],
        rows: usize,
        scratch: &mut ProjectionScratch,
    ) {
        #[cfg(all(feature = "cpu", feature = "int8"))]
        if let (Some(a), Some(b)) = (&linears[0].int8, &linears[1].int8) {
            let start = scratch.profile.then(std::time::Instant::now);
            super::int8::Matrix::apply_pair([a, b], x, outputs, &mut scratch.int8);
            if let Some(start) = start {
                scratch.projection_ms += start.elapsed().as_secs_f64() * 1000.;
            }
            return;
        }
        for (linear, out) in linears.into_iter().zip(outputs) {
            linear.apply(x, out, rows, scratch);
        }
    }
    #[cfg(feature = "cpu")]
    pub fn bytes(&self) -> usize {
        #[cfg(feature = "int8")]
        if let Some(w) = &self.int8 {
            return w.bytes();
        }
        self.weight_bf16.len() * 2
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
            multiply(
                x,
                self.weight_bf16.as_ptr(),
                y,
                count,
                first,
                self.input,
                self.output,
                rows,
                panel_start,
                panel_end,
            );
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub fn apply(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        _rows: usize,
        _scratch: &mut ProjectionScratch,
    ) {
        assert_eq!(x.len() % self.input, 0);
        y.resize(x.len() / self.input * self.output, 0.);
        super::wasm_simd::project(x, &self.weight_bf16, y, self.input, self.output);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]

    pub fn apply(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        rows: usize,
        _scratch: &mut ProjectionScratch,
    ) {
        #[cfg(feature = "cpu")]
        let start = _scratch.profile.then(std::time::Instant::now);
        self.apply_native(x, y, rows, _scratch);
        #[cfg(feature = "cpu")]
        if let Some(start) = start {
            _scratch.projection_ms += start.elapsed().as_secs_f64() * 1000.;
        }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn apply_native(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        rows: usize,
        _scratch: &mut ProjectionScratch,
    ) {
        #[cfg(all(feature = "cpu", feature = "int8"))]
        if let Some(w) = &self.int8 {
            w.apply(x, y, &mut _scratch.int8);
            return;
        }
        #[cfg(not(all(feature = "cpu", target_arch = "x86_64")))]
        debug_assert!(!self.avx2);
        assert_eq!(x.len() % self.input, 0);
        assert!((1..=6).contains(&rows));
        y.resize(x.len() / self.input * self.output, 0.);
        #[cfg(feature = "cpu")]
        let threads = rayon::current_num_threads();
        #[cfg(not(feature = "cpu"))]
        let threads = 1;
        let tokens = x.len() / self.input;
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if self.avx2 && self.output >= 512 && (threads > 1 || tokens > rows) {
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
        let tile_rows = if self.avx2 {
            rows * tokens.div_ceil(rows * threads * 2).clamp(1, 16)
        } else {
            rows
        };
        let run = |(tile, dst): (usize, &mut [f32])| {
            let first = tile * tile_rows;
            #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
            if self.avx2 {
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
            for (r, row) in dst.chunks_mut(self.output).enumerate() {
                row.fill(0.);
                for panel in 0..self.output.div_ceil(16) {
                    let mut acc = [0.; 16];
                    for k in 0..self.input {
                        let a = x[(first + r) * self.input + k];
                        for j in 0..16 {
                            let b = f32::from_bits(
                                (self.weight_bf16[(panel * self.input + k) * 16 + j] as u32) << 16,
                            );
                            acc[j] += a * b;
                        }
                    }
                    for j in 0..16.min(self.output - panel * 16) {
                        row[panel * 16 + j] = acc[j];
                    }
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
unsafe fn multiply(
    x: &[f32],
    w: *const u16,
    y: *mut f32,
    count: usize,
    first: usize,
    k: usize,
    n: usize,
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
                1 => multiply_rows::<1>(x, w, dst, count, start, k, n, panel_start, panel_end),
                2 => multiply_rows::<2>(x, w, dst, count, start, k, n, panel_start, panel_end),
                3 => multiply_rows::<3>(x, w, dst, count, start, k, n, panel_start, panel_end),
                4 => multiply_rows::<4>(x, w, dst, count, start, k, n, panel_start, panel_end),
                5 => multiply_rows::<5>(x, w, dst, count, start, k, n, panel_start, panel_end),
                6 => multiply_rows::<6>(x, w, dst, count, start, k, n, panel_start, panel_end),
                _ => unreachable!("six-row projection tile"),
            }
        }
    }
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn multiply_rows<const R: usize>(
    x: &[f32],
    w: *const u16,
    y: *mut f32,
    count: usize,
    first: usize,
    k: usize,
    n: usize,
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
                    let b = w.add((panel * k + j) * 16);
                    let b0 = _mm256_castsi256_ps(_mm256_slli_epi32(
                        _mm256_cvtepu16_epi32(_mm_loadu_si128(b.cast())),
                        16,
                    ));
                    let b1 = _mm256_castsi256_ps(_mm256_slli_epi32(
                        _mm256_cvtepu16_epi32(_mm_loadu_si128(b.add(8).cast())),
                        16,
                    ));
                    for i in 0..R {
                        let a = _mm256_set1_ps(*x.get_unchecked((first + i) * k + j));
                        acc[i][0] = _mm256_fmadd_ps(a, b0, acc[i][0]);
                        acc[i][1] = _mm256_fmadd_ps(a, b1, acc[i][1]);
                    }
                }
                for i in 0..R {
                    if panel * 16 + 16 <= n {
                        let dst = y.add(i * n + panel * 16);
                        _mm256_storeu_ps(dst, acc[i][0]);
                        _mm256_storeu_ps(dst.add(8), acc[i][1]);
                    } else {
                        let mut tmp = [0.; 16];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), acc[i][0]);
                        _mm256_storeu_ps(tmp.as_mut_ptr().add(8), acc[i][1]);
                        for c in 0..n - panel * 16 {
                            *y.add(i * n + panel * 16 + c) = tmp[c];
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
    #[test]
    fn bf16_projection_tails_match_f64() -> Result<()> {
        use safetensors::{Dtype, tensor::TensorView};
        for (m, k, n) in [(1, 7, 1), (3, 17, 17), (7, 128, 32), (13, 256, 65)] {
            let data: Vec<u8> = (0..k * n)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 17 % 113) as f32 - 56.) / 64.)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect();
            let weights = Weights(std::collections::HashMap::from([(
                "p.weight".into(),
                TensorView::new(Dtype::BF16, vec![n, k], &data)?,
            )]));
            let x: Vec<_> = (0..m * k)
                .map(|i| ((i * 23 % 127) as f32 - 63.) / 128.)
                .collect();
            let mut expected = vec![0.; m * n];
            for r in 0..m {
                for c in 0..n {
                    expected[r * n + c] = (0..k)
                        .map(|j| {
                            let at = (c * k + j) * 2;
                            let b = f32::from_bits(
                                (u16::from_le_bytes([data[at], data[at + 1]]) as u32) << 16,
                            );
                            x[r * k + j] as f64 * b as f64
                        })
                        .sum::<f64>() as f32;
                }
            }
            for avx2 in [false, true] {
                if avx2 && !avx2_available() {
                    continue;
                }
                let linear = Linear::load(&weights, "p", k, n, avx2)?;
                let mut out = Vec::new();
                linear.apply(&x, &mut out, 6, &mut ProjectionScratch::default());
                for (a, b) in out.iter().zip(&expected) {
                    assert!((a - b).abs() < 1e-5, "m={m},k={k},n={n}: {a} != {b}");
                }
            }
        }
        Ok(())
    }
}
