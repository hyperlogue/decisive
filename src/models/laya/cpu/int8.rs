use anyhow::{Result, ensure};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
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
            !avx2 || super::packed_supported(),
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
    pub fn value(&self, n: usize, k: usize) -> f32 {
        self.data[self.index(n, k)] as f32
            * self.scales[(n / 8 * self.groups() + k / self.block) * 8 + n % 8]
    }
    pub fn bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

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
        let groups = self.groups();
        let padded = groups * self.block;
        let bytes = if self.mode == Mode::WeightsActivations {
            m * padded
        } else {
            0
        };
        workspace.resize(bytes / 4 + if bytes > 0 { m * groups } else { 0 }, 0.);
        let (packed, scales) = workspace.split_at_mut(bytes / 4);
        let packed: &mut [i8] = bytemuck::cast_slice_mut(packed);
        if bytes > 0 {
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            if !wasm::pack_activations(x, self.cols, self.block, packed, scales) {
                y.fill(f32::NAN);
                return;
            }
            #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
            {
                let fill = |(i, (row, scales)): (usize, (&mut [i8], &mut [f32]))| {
                    let input = &x[i * self.cols..(i + 1) * self.cols];
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
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            wasm::apply(self, x, packed, scales, y, bias, rows);
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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
    use std::arch::x86_64::*;
    unsafe {
        let count = dst.len() / w.rows;
        let groups = w.groups();
        let one = _mm256_set1_epi16(1);
        for panel in 0..w.rows.div_ceil(8) {
            let mut sums = [_mm256_setzero_ps(); 4];
            for g in 0..groups {
                let base = (panel * groups + g) * w.block * 8;
                let sw = _mm256_loadu_ps(w.scales.as_ptr().add((panel * groups + g) * 8));
                if w.mode == Mode::WeightsActivations {
                    let mut acc = [_mm256_setzero_si256(); 4];
                    for k in (0..w.block).step_by(4) {
                        let b = _mm256_loadu_si256(w.data.as_ptr().add(base + k * 8).cast());
                        for r in 0..count {
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
                    for r in 0..count {
                        let s = _mm256_mul_ps(sw, _mm256_set1_ps(sa[(first + r) * groups + g]));
                        sums[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[r]), s, sums[r]);
                    }
                } else {
                    let mut acc = [_mm256_setzero_ps(); 4];
                    for k in 0..w.block.min(w.cols - g * w.block) {
                        let b = _mm_loadl_epi64(w.data.as_ptr().add(base + k * 8).cast());
                        let b = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
                        for r in 0..count {
                            let av = _mm256_set1_ps(x[(first + r) * w.cols + g * w.block + k]);
                            acc[r] = _mm256_fmadd_ps(av, b, acc[r]);
                        }
                    }
                    for r in 0..count {
                        sums[r] = _mm256_fmadd_ps(acc[r], sw, sums[r]);
                    }
                }
            }
            for r in 0..count {
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
    fn tails_extrema_zero_and_outliers_match_portable() {
        for (m, n, k) in [(1, 1, 1), (3, 11, 33), (7, 17, 127), (8, 16, 64)] {
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
                    if super::super::packed_supported() {
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
