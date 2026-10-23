use anyhow::{Result, ensure};
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
mod pack_x86;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod weights_wasm;
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
mod weights_x86;
#[cfg(all(
    feature = "kernel-checks",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
pub(crate) fn wasm_check() -> Result<usize> {
    wasm::check()
}
#[cfg(feature = "cpu")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Weights,
    WeightsActivations,
}

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub block: usize,
    pub mode: Mode,
    data: Vec<i8>,
    scales: Vec<f32>,
    #[cfg_attr(not(all(feature = "cpu", target_arch = "x86_64")), allow(dead_code))]
    avx2: bool,
}

impl Matrix {
    pub(super) fn from_quantized(
        values: &[i8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        block: usize,
        mode: Mode,
        avx2: bool,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && cols > 0 && rows.checked_mul(cols) == Some(values.len()),
            "invalid INT8 shape"
        );
        ensure!(
            matches!(block, 16 | 32 | 64 | 128) && cols % block == 0,
            "invalid compact INT8 groups"
        );
        let groups = cols / block;
        ensure!(
            scales.len() == rows * groups
                && scales
                    .iter()
                    .all(|v| v.is_finite() && *v >= f32::MIN_POSITIVE),
            "invalid compact INT8 scales"
        );
        ensure!(
            values.iter().all(|&v| v != i8::MIN),
            "invalid INT8 symmetric range"
        );
        ensure!(
            !avx2 || super::cpu::packed_supported(),
            "INT8 AVX2 kernel unavailable"
        );
        let mut result = Self {
            rows,
            cols,
            block,
            mode,
            avx2,
            data: vec![0; rows.div_ceil(8) * 8 * groups * block],
            scales: vec![0.; rows.div_ceil(8) * 8 * groups],
        };
        for n in 0..rows {
            for g in 0..groups {
                result.scales[(n / 8 * groups + g) * 8 + n % 8] = scales[n * groups + g];
                for k in g * block..(g + 1) * block {
                    let index = result.index(n, k);
                    result.data[index] = values[n * cols + k];
                }
            }
        }
        Ok(result)
    }
    pub fn new(
        values: &[f32],
        rows: usize,
        cols: usize,
        block: usize,
        mode: Mode,
        avx2: bool,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && cols > 0 && rows.checked_mul(cols) == Some(values.len()),
            "invalid INT8 matrix shape"
        );
        ensure!(
            matches!(block, 16 | 32 | 64 | 128),
            "INT8 block must be 16, 32, 64 or 128"
        );
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite INT8 weights"
        );
        ensure!(
            !avx2 || super::cpu::packed_supported(),
            "INT8 AVX2 kernel unavailable"
        );
        let groups = cols.div_ceil(block);
        let mut result = Self {
            rows,
            cols,
            block,
            mode,
            data: vec![0; rows.div_ceil(8) * 8 * groups * block],
            scales: vec![0.; rows.div_ceil(8) * 8 * groups],
            avx2,
        };
        for n in 0..rows {
            for g in 0..groups {
                let start = g * block;
                let slice = &values[n * cols + start..n * cols + (start + block).min(cols)];
                let scale = scale(slice);
                result.scales[(n / 8 * groups + g) * 8 + n % 8] = scale;
                for (k, &v) in slice.iter().enumerate() {
                    let index = result.index(n, start + k);
                    result.data[index] = quantize(v, scale);
                }
            }
        }
        Ok(result)
    }
    fn groups(&self) -> usize {
        self.cols.div_ceil(self.block)
    }
    fn index(&self, n: usize, k: usize) -> usize {
        let base = (n / 8 * self.groups() + k / self.block) * self.block * 8;
        let k = k % self.block;
        base + match self.mode {
            Mode::Weights => k * 8 + n % 8,
            Mode::WeightsActivations => {
                if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
                    k / 2 * 16 + n % 8 * 2 + k % 2
                } else {
                    k / 4 * 32 + n % 8 * 4 + k % 4
                }
            }
        }
    }
    #[cfg(any(test, feature = "kernel-checks"))]
    pub fn apply(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        bias: Option<&[f32]>,
        workspace: &mut Vec<f32>,
    ) {
        self.apply_with_rows(x, y, bias, workspace, 4);
    }
    pub fn apply_with_rows(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        bias: Option<&[f32]>,
        workspace: &mut Vec<f32>,
        rows: usize,
    ) {
        assert!((1..=4).contains(&rows));
        assert_eq!(x.len() % self.cols, 0);
        assert!(bias.is_none_or(|b| b.len() == self.rows));
        let m = x.len() / self.cols;
        y.resize(m * self.rows, 0.);
        let needs_scan = !cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
            || self.mode == Mode::Weights;
        if needs_scan && !x.iter().all(|v| v.is_finite()) {
            y.fill(f32::NAN);
            return;
        }
        if self.mode == Mode::Weights {
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                #[cfg(feature = "decima-wasm-threads")]
                if weights_wasm::apply_parallel(self, x, y, bias, rows) {
                    return;
                }
                weights_wasm::apply(self, x, y, bias, rows, workspace);
                return;
            }
            #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
            if self.avx2 && (m >= 64 || rayon::current_num_threads() < 4) {
                weights_x86::apply(self, x, y, bias, rows, workspace);
                return;
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let groups = self.groups();
            let activation_len = m * groups * self.block;
            let panel_len = groups * self.block * 8;
            workspace.resize(m * groups + (activation_len + panel_len).div_ceil(2), 0.);
            let (scales, wide) = workspace.split_at_mut(m * groups);
            let (activations, panel) =
                bytemuck::cast_slice_mut::<f32, i16>(wide).split_at_mut(activation_len);
            if !wasm::pack_for_rows(x, self.cols, self.block, activations, scales, rows) {
                y.fill(f32::NAN);
                return;
            }
            #[cfg(feature = "decima-wasm-threads")]
            if wasm::apply_parallel(self, activations, scales, y, bias, rows) {
                return;
            }
            wasm::apply(self, activations, scales, y, bias, rows, panel);
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            let groups = self.groups();
            let padded = groups * self.block;
            let bytes = if self.mode == Mode::WeightsActivations {
                m * padded
            } else {
                0
            };
            let scale_count = if bytes > 0 { m * groups } else { 0 };
            workspace.resize(bytes / 4 + scale_count, 0.);
            let (packed, scales) = workspace.split_at_mut(bytes / 4);
            let packed: &mut [i8] = bytemuck::cast_slice_mut(packed);
            if bytes > 0 {
                {
                    let fill = |(i, (row, scales)): (usize, (&mut [i8], &mut [f32]))| {
                        let input = &x[i * self.cols..(i + 1) * self.cols];
                        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
                        if self.avx2 {
                            unsafe {
                                pack_x86::row(input, self.block, row, scales);
                            }
                            return;
                        }
                        row.fill(0);
                        for g in 0..groups {
                            let s = g * self.block;
                            let values = &input[s..(s + self.block).min(self.cols)];
                            scales[g] = scale(values);
                            for (q, &v) in row[s..].iter_mut().zip(values) {
                                *q = quantize(v, scales[g]);
                            }
                        }
                    };
                    #[cfg(feature = "cpu")]
                    packed
                        .par_chunks_mut(padded)
                        .zip(scales.par_chunks_mut(groups))
                        .enumerate()
                        .for_each(fill);
                    #[cfg(not(feature = "cpu"))]
                    packed
                        .chunks_mut(padded)
                        .zip(scales.chunks_mut(groups))
                        .enumerate()
                        .for_each(fill);
                }
            }
            {
                let run = |(tile, dst): (usize, &mut [f32])| {
                    let first = tile * rows;
                    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
                    if self.avx2 {
                        unsafe {
                            x86(self, x, packed, scales, first, dst, bias);
                        }
                        return;
                    }
                    self.portable(x, packed, scales, first, dst, bias);
                };
                #[cfg(feature = "cpu")]
                y.par_chunks_mut(rows * self.rows).enumerate().for_each(run);
                #[cfg(not(feature = "cpu"))]
                y.chunks_mut(rows * self.rows).enumerate().for_each(run);
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub(super) fn apply_shared(
        matrices: [&Self; 3],
        x: &[f32],
        outputs: [&mut Vec<f32>; 3],
        biases: [&[f32]; 3],
        workspace: &mut Vec<f32>,
        rows: usize,
    ) -> bool {
        let first = matrices[0];
        if matrices.iter().any(|w| {
            w.mode != Mode::WeightsActivations || w.cols != first.cols || w.block != first.block
        }) {
            return false;
        }
        assert!((1..=4).contains(&rows));
        assert_eq!(x.len() % first.cols, 0);
        let m = x.len() / first.cols;
        let groups = first.groups();
        let activation_len = m * groups * first.block;
        let panel_len = groups * first.block * 8;
        workspace.resize(m * groups + (activation_len + panel_len).div_ceil(2), 0.);
        let (scales, wide) = workspace.split_at_mut(m * groups);
        let (activations, panel) =
            bytemuck::cast_slice_mut::<f32, i16>(wide).split_at_mut(activation_len);
        let finite = wasm::pack_for_rows(x, first.cols, first.block, activations, scales, rows);
        #[cfg(feature = "decima-wasm-threads")]
        if finite && m >= 16 && super::wasm_threads::count() > 1 {
            let [q, k, v] = outputs;
            for (index, out) in [&mut *q, &mut *k, &mut *v].into_iter().enumerate() {
                assert_eq!(biases[index].len(), matrices[index].rows);
                out.resize(m * matrices[index].rows, 0.);
            }
            wasm::apply_three(matrices, activations, scales, [q, k, v], biases, rows);
            return true;
        }
        for ((w, out), bias) in matrices.into_iter().zip(outputs).zip(biases) {
            assert_eq!(bias.len(), w.rows);
            out.resize(m * w.rows, 0.);
            if finite {
                #[cfg(feature = "decima-wasm-threads")]
                if wasm::apply_parallel(w, activations, scales, out, Some(bias), rows) {
                    continue;
                }
                wasm::apply(w, activations, scales, out, Some(bias), rows, panel);
            } else {
                out.fill(f32::NAN);
            }
        }
        true
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub(super) fn apply_serial_wasm(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        bias: &[f32],
        workspace: &mut Vec<f32>,
        rows: usize,
    ) {
        assert_eq!(self.mode, Mode::WeightsActivations);
        assert!((1..=4).contains(&rows));
        assert_eq!(x.len() % self.cols, 0);
        assert_eq!(bias.len(), self.rows);
        let m = x.len() / self.cols;
        y.resize(m * self.rows, 0.);
        let groups = self.groups();
        let activation_len = m * groups * self.block;
        let panel_len = groups * self.block * 8;
        workspace.resize(m * groups + (activation_len + panel_len).div_ceil(2), 0.);
        let (scales, wide) = workspace.split_at_mut(m * groups);
        let (activations, panel) =
            bytemuck::cast_slice_mut::<f32, i16>(wide).split_at_mut(activation_len);
        if !wasm::pack_for_rows(x, self.cols, self.block, activations, scales, rows) {
            y.fill(f32::NAN);
            return;
        }
        wasm::apply(self, activations, scales, y, Some(bias), rows, panel);
    }

    fn portable(
        &self,
        x: &[f32],
        a: &[i8],
        sa: &[f32],
        first: usize,
        dst: &mut [f32],
        bias: Option<&[f32]>,
    ) {
        let groups = self.groups();
        for (i, out) in dst.chunks_mut(self.rows).enumerate() {
            let i = first + i;
            for (n, out) in out.iter_mut().enumerate() {
                let mut sum = 0.;
                for g in 0..groups {
                    let sw = self.scales[(n / 8 * groups + g) * 8 + n % 8];
                    let base = (n / 8 * groups + g) * self.block * 8;
                    match self.mode {
                        Mode::WeightsActivations => {
                            let mut dot = 0_i32;
                            for k in 0..self.block.min(self.cols - g * self.block) {
                                dot += a[(i * groups + g) * self.block + k] as i32
                                    * self.data[self.index(n, g * self.block + k)] as i32;
                            }
                            sum += dot as f32 * (sw * sa[i * groups + g]);
                        }
                        Mode::Weights => {
                            let mut dot = 0.;
                            for k in 0..self.block.min(self.cols - g * self.block) {
                                dot += x[i * self.cols + g * self.block + k]
                                    * self.data[base + k * 8 + n % 8] as f32;
                            }
                            sum += dot * sw;
                        }
                    }
                }
                *out = sum + bias.map_or(0., |b| b[n]);
            }
        }
    }
}

fn scale(x: &[f32]) -> f32 {
    (x.iter().fold(0_f32, |a, v| a.max(v.abs())) / 127.).max(f32::MIN_POSITIVE)
}
fn quantize(x: f32, s: f32) -> i8 {
    (x / s).round().clamp(-127., 127.) as i8
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn x86(
    w: &Matrix,
    x: &[f32],
    a: &[i8],
    sa: &[f32],
    first: usize,
    dst: &mut [f32],
    bias: Option<&[f32]>,
) {
    unsafe {
        match dst.len() / w.rows {
            1 => x86_rows::<1>(w, x, a, sa, first, dst, bias),
            2 => x86_rows::<2>(w, x, a, sa, first, dst, bias),
            3 => x86_rows::<3>(w, x, a, sa, first, dst, bias),
            4 => x86_rows::<4>(w, x, a, sa, first, dst, bias),
            _ => unreachable!("INT8 row tile"),
        }
    }
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn x86_rows<const R: usize>(
    w: &Matrix,
    x: &[f32],
    a: &[i8],
    sa: &[f32],
    first: usize,
    dst: &mut [f32],
    bias: Option<&[f32]>,
) {
    use std::arch::x86_64::*;
    unsafe {
        let groups = w.groups();
        let one = _mm256_set1_epi16(1);
        for panel in 0..w.rows.div_ceil(8) {
            let mut sums = [_mm256_setzero_ps(); R];
            for g in 0..groups {
                let base = (panel * groups + g) * w.block * 8;
                let sw = _mm256_loadu_ps(w.scales.as_ptr().add((panel * groups + g) * 8));
                if w.mode == Mode::WeightsActivations {
                    let mut acc = [_mm256_setzero_si256(); R];
                    for k in (0..w.block).step_by(4) {
                        let b = _mm256_loadu_si256(w.data.as_ptr().add(base + k * 8).cast());
                        for r in 0..R {
                            let q = a
                                .as_ptr()
                                .add(((first + r) * groups + g) * w.block + k)
                                .cast::<i32>()
                                .read_unaligned();
                            let av = _mm256_set1_epi32(q);
                            let pairs =
                                _mm256_maddubs_epi16(_mm256_abs_epi8(av), _mm256_sign_epi8(b, av));
                            acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(pairs, one));
                        }
                    }
                    for r in 0..R {
                        let s = _mm256_mul_ps(sw, _mm256_set1_ps(sa[(first + r) * groups + g]));
                        sums[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[r]), s, sums[r]);
                    }
                } else {
                    let mut acc = [_mm256_setzero_ps(); R];
                    for k in 0..w.block.min(w.cols - g * w.block) {
                        let b = _mm_loadl_epi64(w.data.as_ptr().add(base + k * 8).cast());
                        let b = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
                        for r in 0..R {
                            let av = _mm256_set1_ps(x[(first + r) * w.cols + g * w.block + k]);
                            acc[r] = _mm256_fmadd_ps(av, b, acc[r]);
                        }
                    }
                    for r in 0..R {
                        sums[r] = _mm256_fmadd_ps(acc[r], sw, sums[r]);
                    }
                }
            }
            for r in 0..R {
                let mut values = [0_f32; 8];
                _mm256_storeu_ps(values.as_mut_ptr(), sums[r]);
                for n in 0..8.min(w.rows - panel * 8) {
                    let col = panel * 8 + n;
                    dst[r * w.rows + col] = values[n] + bias.map_or(0., |b| b[col]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn imported_quantization_preserves_packed_values_scales_and_outputs() {
        let (n, k) = (11, 64);
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i * 17 % 129) as f32 - 64.) / 65.)
            .collect();
        let mut quantized = vec![];
        let mut scales = vec![];
        for group in values.chunks_exact(32) {
            let s = scale(group);
            scales.push(s);
            quantized.extend(group.iter().map(|&v| quantize(v, s)));
        }
        let x: Vec<f32> = (0..7 * k).map(|i| (i as f32 * 0.3).sin()).collect();
        for mode in [Mode::Weights, Mode::WeightsActivations] {
            for avx2 in [false, super::super::cpu::packed_supported()] {
                let original = Matrix::new(&values, n, k, 32, mode, avx2).unwrap();
                let loaded =
                    Matrix::from_quantized(&quantized, &scales, n, k, 32, mode, avx2).unwrap();
                assert_eq!(original.data, loaded.data);
                assert_eq!(original.scales, loaded.scales);
                for rows in 1..=4 {
                    let (mut a, mut b) = (vec![], vec![]);
                    original.apply_with_rows(&x, &mut a, None, &mut vec![], rows);
                    loaded.apply_with_rows(&x, &mut b, None, &mut vec![], rows);
                    assert_eq!(a, b);
                }
            }
        }
        assert!(
            Matrix::from_quantized(&[-128; 32], &[1.], 1, 32, 32, Mode::Weights, false).is_err()
        );
        assert!(
            Matrix::from_quantized(&[0; 32], &[f32::INFINITY], 1, 32, 32, Mode::Weights, false)
                .is_err()
        );
    }
    #[test]
    fn tails_extrema_zero_and_outliers_match_portable() {
        for (m, n, k) in [
            (1, 1, 1),
            (3, 11, 33),
            (7, 17, 127),
            (8, 16, 64),
            (127, 9, 33),
            (128, 9, 33),
            (129, 9, 33),
            (511, 9, 33),
            (512, 9, 33),
            (513, 9, 33),
        ] {
            let x: Vec<_> = (0..m * k)
                .map(|i| {
                    if i % 11 == 0 {
                        0.
                    } else if i % 7 == 0 {
                        70.
                    } else {
                        (i as f32 * 0.31).sin()
                    }
                })
                .collect();
            let v: Vec<_> = (0..n * k)
                .map(|i| if i % 3 == 0 { -127. } else { 127. })
                .collect();
            for mode in [Mode::Weights, Mode::WeightsActivations] {
                for block in [16, 32, 64, 128] {
                    let p = Matrix::new(&v, n, k, block, mode, false).unwrap();
                    let mut y = vec![];
                    p.apply(&x, &mut y, None, &mut vec![]);
                    assert!(y.iter().all(|v| v.is_finite()));
                    if super::super::cpu::packed_supported() {
                        let q = Matrix::new(&v, n, k, block, mode, true).unwrap();
                        let mut z = vec![];
                        q.apply(&x, &mut z, None, &mut vec![]);
                        for (&a, &b) in y.iter().zip(&z) {
                            assert!((a - b).abs() <= 0.02 + a.abs() * 1e-5, "{a} {b}");
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn invalid_and_nonfinite_values_are_not_hidden() {
        assert!(Matrix::new(&[], usize::MAX, 2, 32, Mode::Weights, false).is_err());
        assert!(Matrix::new(&[f32::NAN], 1, 1, 32, Mode::Weights, false).is_err());
        assert!(Matrix::new(&[1.], 1, 1, 3, Mode::Weights, false).is_err());
        let q = Matrix::new(&[0.], 1, 1, 32, Mode::WeightsActivations, false).unwrap();
        let mut y = vec![];
        q.apply(&[f32::INFINITY], &mut y, None, &mut vec![]);
        assert!(y[0].is_nan());
        q.apply(&[0.], &mut y, Some(&[2.]), &mut vec![]);
        assert_eq!(y, vec![2.]);
    }
}
