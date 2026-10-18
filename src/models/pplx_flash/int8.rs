use anyhow::{Result, ensure};
use rayon::prelude::*;

pub(super) const GROUP: usize = 64;

#[derive(Default)]
pub(super) struct Scratch {
    values: Vec<i8>,
    scales: Vec<f32>,
    sums: Vec<i32>,
}

pub(super) struct Matrix {
    input: usize,
    output: usize,
    data: Vec<i8>,
    scales: Vec<f32>,
    avx2: bool,
    weight_bits: u8,
    activation_bits: u8,
    offsets: Vec<f32>,
}

impl Matrix {
    #[cfg(test)]
    pub fn from_bf16(bytes: &[u8], input: usize, output: usize, avx2: bool) -> Result<Self> {
        Self::from_bf16_quantized(bytes, input, output, avx2, 8, 8)
    }

    #[cfg(test)]
    pub fn from_bf16_quantized(
        bytes: &[u8],
        input: usize,
        output: usize,
        avx2: bool,
        weight_bits: u8,
        activation_bits: u8,
    ) -> Result<Self> {
        Self::from_bf16_optimized(
            bytes,
            input,
            output,
            avx2,
            weight_bits,
            activation_bits,
            false,
            false,
        )
    }

    pub fn from_bf16_optimized(
        bytes: &[u8],
        input: usize,
        output: usize,
        avx2: bool,
        weight_bits: u8,
        activation_bits: u8,
        optimized: bool,
        affine: bool,
    ) -> Result<Self> {
        ensure!(
            [4, 8].contains(&weight_bits) && [4, 8].contains(&activation_bits),
            "unsupported quantization bits"
        );
        ensure!(
            !avx2 || super::cpu::avx2_available(),
            "INT8 AVX2/FMA unavailable"
        );
        ensure!(
            input > 0
                && output > 0
                && input.checked_mul(output).and_then(|n| n.checked_mul(2)) == Some(bytes.len()),
            "invalid INT8 matrix shape"
        );
        let groups = input.div_ceil(GROUP);
        let divisor = if weight_bits == 4 { 2 } else { 1 };
        let mut data = vec![0; output.div_ceil(8) * groups * GROUP * 8 / divisor];
        let mut scales = vec![0.; output.div_ceil(8) * groups * 8];
        let mut offsets = vec![0.; if affine { scales.len() } else { 0 }];
        // Use one temporary per-panel offset slot even for symmetric modes so
        // the packing traversal and final resident layout remain shared.
        let mut panel_offsets = vec![vec![]; output.div_ceil(8)];
        data.par_chunks_mut(groups * GROUP * 8 / divisor)
            .zip(scales.par_chunks_mut(groups * 8))
            .zip(panel_offsets.par_iter_mut())
            .enumerate()
            .try_for_each(|(panel, ((data, scales), offsets))| -> Result<()> {
                if affine {
                    offsets.resize(groups * 8, 0.);
                }
                for c in 0..8.min(output - panel * 8) {
                    for g in 0..groups {
                        let count = GROUP.min(input - g * GROUP);
                        let mut values = [0.; GROUP];
                        for (k, v) in values[..count].iter_mut().enumerate() {
                            let at = ((panel * 8 + c) * input + g * GROUP + k) * 2;
                            *v = f32::from_bits(
                                (u16::from_le_bytes([bytes[at], bytes[at + 1]]) as u32) << 16,
                            );
                        }
                        ensure!(
                            values.iter().all(|v| v.is_finite()),
                            "non-finite INT8 weight"
                        );
                        let mut codes = [0; GROUP];
                        let s = if affine {
                            ensure!(weight_bits == 4, "affine fitting requires Q4");
                            let (s, b) = super::quantization::affine(
                                &values[..count],
                                &mut codes[..count],
                                avx2,
                            );
                            offsets[g * 8 + c] = b;
                            s
                        } else if optimized && weight_bits == 4 {
                            super::quantization::mse(&values[..count], &mut codes[..count], avx2)
                        } else {
                            let s = group_scale_bits(&values, avx2, weight_bits);
                            pack_bits(&values, &mut codes, s, avx2, weight_bits);
                            s
                        };
                        scales[g * 8 + c] = s;
                        for k in (0..GROUP).step_by(4) {
                            let at = g * GROUP * 8 + k / 4 * 32 + c * 4;
                            if weight_bits == 4 {
                                for j in 0..2 {
                                    data[at / 2 + j] = ((codes[k + 2 * j] + 8) as u8
                                        | ((codes[k + 2 * j + 1] + 8) as u8) << 4)
                                        as i8;
                                }
                            } else {
                                data[at..at + 4].copy_from_slice(&codes[k..k + 4]);
                            }
                        }
                    }
                }
                Ok(())
            })?;
        if affine {
            for (to, from) in offsets.chunks_mut(groups * 8).zip(panel_offsets) {
                to.copy_from_slice(&from);
            }
        }
        Ok(Self {
            input,
            output,
            data,
            scales,
            avx2,
            weight_bits,
            activation_bits,
            offsets,
        })
    }

    fn code(&self, index: usize) -> i8 {
        if self.weight_bits == 4 {
            ((self.data[index / 2] as u8 >> (index % 2 * 4)) & 15) as i8 - 8
        } else {
            self.data[index]
        }
    }

    pub fn bytes(&self) -> usize {
        self.data.len() + (self.scales.len() + self.offsets.len()) * 4
    }

    pub fn apply(&self, x: &[f32], y: &mut Vec<f32>, scratch: &mut Scratch) {
        assert_eq!(x.len() % self.input, 0);
        let tokens = x.len() / self.input;
        y.resize(tokens * self.output, 0.);
        if !x.iter().all(|v| v.is_finite()) {
            y.fill(f32::NAN);
            return;
        }
        self.prepare(x, scratch);
        self.apply_prepared(tokens, y, scratch);
    }

    fn prepare(&self, x: &[f32], scratch: &mut Scratch) {
        let tokens = x.len() / self.input;
        let groups = self.input.div_ceil(GROUP);
        scratch.values.resize(tokens * groups * GROUP, 0);
        scratch.scales.resize(tokens * groups, 0.);
        scratch.sums.resize(tokens * groups, 0);
        scratch
            .values
            .par_chunks_mut(groups * GROUP)
            .zip(scratch.scales.par_chunks_mut(groups))
            .zip(scratch.sums.par_chunks_mut(groups))
            .zip(x.par_chunks(self.input))
            .for_each(|(((out, scales), sums), row)| {
                for (g, values) in row.chunks(GROUP).enumerate() {
                    let s = group_scale_bits(values, self.avx2, self.activation_bits);
                    scales[g] = s;
                    let out = &mut out[g * GROUP..(g + 1) * GROUP];
                    out.fill(0);
                    pack_bits(values, out, s, self.avx2, self.activation_bits);
                    if !self.offsets.is_empty() {
                        sums[g] = out.iter().map(|&v| v as i32).sum();
                    }
                }
            });
    }

    pub fn apply_pair(
        matrices: [&Self; 2],
        x: &[f32],
        outputs: [&mut Vec<f32>; 2],
        scratch: &mut Scratch,
    ) {
        let [a, b] = matrices;
        assert_eq!(a.input, b.input);
        assert_eq!(a.activation_bits, b.activation_bits);
        assert_eq!(a.offsets.is_empty(), b.offsets.is_empty());
        assert_eq!(x.len() % a.input, 0);
        let tokens = x.len() / a.input;
        if !x.iter().all(|v| v.is_finite()) {
            for (matrix, out) in matrices.into_iter().zip(outputs) {
                out.resize(tokens * matrix.output, f32::NAN);
                out.fill(f32::NAN);
            }
            return;
        }
        a.prepare(x, scratch);
        for (matrix, out) in matrices.into_iter().zip(outputs) {
            matrix.apply_prepared(tokens, out, scratch);
        }
    }

    fn apply_prepared(&self, tokens: usize, y: &mut Vec<f32>, scratch: &Scratch) {
        let groups = self.input.div_ceil(GROUP);
        y.resize(tokens * self.output, 0.);
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            let dst = Columns(y.as_mut_ptr());
            (0..self.output.div_ceil(32)).into_par_iter().for_each_init(
                Vec::new,
                |expanded, tile| {
                    let first = tile * 4;
                    let end = (first + 4).min(self.output.div_ceil(8));
                    unsafe {
                        if self.weight_bits == 4 && tokens >= 16 {
                            project_buffered(
                                self,
                                scratch,
                                dst.pointer(),
                                tokens,
                                first,
                                end,
                                expanded,
                            );
                        } else {
                            project::<4>(self, scratch, dst.pointer(), tokens, first, end);
                        }
                    }
                },
            );
            return;
        }
        let _ = self.avx2;
        y.par_chunks_mut(self.output)
            .enumerate()
            .for_each(|(r, out)| {
                for (c, out) in out.iter_mut().enumerate() {
                    let mut sum = 0.;
                    for g in 0..groups {
                        let mut dot = 0_i32;
                        for k in 0..GROUP {
                            dot += scratch.values[(r * groups + g) * GROUP + k] as i32
                                * self.code(
                                    (c / 8 * groups + g) * GROUP * 8
                                        + k / 4 * 32
                                        + c % 8 * 4
                                        + k % 4,
                                ) as i32;
                        }
                        sum += dot as f32
                            * (scratch.scales[r * groups + g]
                                * self.scales[(c / 8 * groups + g) * 8 + c % 8]);
                        if !self.offsets.is_empty() {
                            sum += self.offsets[(c / 8 * groups + g) * 8 + c % 8]
                                * (scratch.scales[r * groups + g]
                                    * scratch.sums[r * groups + g] as f32);
                        }
                    }
                    *out = sum;
                }
            });
    }
}

#[cfg(test)]
fn scale(x: &[f32]) -> f32 {
    (x.iter().fold(0_f32, |a, v| a.max(v.abs())) / 127.).max(f32::MIN_POSITIVE)
}
#[cfg(test)]
fn quantize(x: f32, scale: f32) -> i8 {
    (x * scale.recip()).round_ties_even().clamp(-127., 127.) as i8
}

#[cfg(test)]
fn group_scale(x: &[f32], avx2: bool) -> f32 {
    group_scale_bits(x, avx2, 8)
}

pub(super) fn group_scale_bits(x: &[f32], avx2: bool, bits: u8) -> f32 {
    let limit = if bits == 4 { 7. } else { 127. };
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return unsafe { scale_avx2(x, limit) };
    }
    let _ = avx2;
    (x.iter().fold(0_f32, |a, v| a.max(v.abs())) / limit).max(f32::MIN_POSITIVE)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2(x: &[f32], limit: f32) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let end = x.len() / 8 * 8;
        let mut maximum = _mm256_setzero_ps();
        for k in (0..end).step_by(8) {
            maximum = _mm256_max_ps(
                maximum,
                _mm256_and_ps(_mm256_loadu_ps(x.as_ptr().add(k)), mask),
            );
        }
        let half = _mm_max_ps(
            _mm256_castps256_ps128(maximum),
            _mm256_extractf128_ps::<1>(maximum),
        );
        let pairs = _mm_max_ps(half, _mm_movehl_ps(half, half));
        let mut maximum = _mm_cvtss_f32(_mm_max_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs)));
        for &v in &x[end..] {
            maximum = maximum.max(v.abs());
        }
        (maximum / limit).max(f32::MIN_POSITIVE)
    }
}

#[cfg(test)]
fn pack(x: &[f32], out: &mut [i8], scale: f32, avx2: bool) {
    pack_bits(x, out, scale, avx2, 8)
}

pub(super) fn pack_bits(x: &[f32], out: &mut [i8], scale: f32, avx2: bool, bits: u8) {
    let limit = if bits == 4 { 7. } else { 127. };
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        unsafe {
            pack_avx2(x, out, scale, limit);
        }
        return;
    }
    let _ = avx2;
    for (out, &v) in out.iter_mut().zip(x) {
        *out = (v * scale.recip()).round_ties_even().clamp(-limit, limit) as i8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_avx2(x: &[f32], out: &mut [i8], scale: f32, limit: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let inverse = _mm256_set1_ps(scale.recip());
        let end = x.len() / 8 * 8;
        for k in (0..end).step_by(8) {
            let value = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(k)), inverse);
            let value = _mm256_min_ps(
                _mm256_max_ps(value, _mm256_set1_ps(-limit)),
                _mm256_set1_ps(limit),
            );
            let value = _mm256_cvttps_epi32(_mm256_round_ps::<
                { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
            >(value));
            let half = _mm_packs_epi32(
                _mm256_castsi256_si128(value),
                _mm256_extracti128_si256::<1>(value),
            );
            let bytes = _mm_packs_epi16(half, _mm_setzero_si128());
            _mm_storel_epi64(out.as_mut_ptr().add(k).cast(), bytes);
        }
        for k in end..x.len() {
            out[k] = (x[k] * scale.recip())
                .round_ties_even()
                .clamp(-limit, limit) as i8;
        }
    }
}

// Each job owns disjoint output columns and joins before the mutable borrow ends.
#[cfg(target_arch = "x86_64")]
struct Columns(*mut f32);
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for Columns {}
#[cfg(target_arch = "x86_64")]
impl Columns {
    fn pointer(&self) -> *mut f32 {
        self.0
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn project<const R: usize>(
    w: &Matrix,
    x: &Scratch,
    out: *mut f32,
    tokens: usize,
    start: usize,
    end: usize,
) {
    unsafe {
        if w.weight_bits == 4 {
            project_packed::<R, true>(w, x, out, tokens, start, end);
        } else {
            project_packed::<R, false>(w, x, out, tokens, start, end);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn project_packed<const R: usize, const Q4: bool>(
    w: &Matrix,
    x: &Scratch,
    out: *mut f32,
    tokens: usize,
    start: usize,
    end: usize,
) {
    unsafe {
        let groups = w.input.div_ceil(GROUP);
        for panel in start..end {
            let data = w
                .data
                .as_ptr()
                .add(panel * groups * GROUP * 8 / if Q4 { 2 } else { 1 });
            let scales = w.scales.as_ptr().add(panel * groups * 8);
            project_panel::<R, Q4>(w, x, out, tokens, panel, data, scales);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn project_buffered(
    w: &Matrix,
    x: &Scratch,
    out: *mut f32,
    tokens: usize,
    start: usize,
    end: usize,
    expanded: &mut Vec<i8>,
) {
    use std::arch::x86_64::*;
    unsafe {
        let groups = w.input.div_ceil(GROUP);
        let count = groups * GROUP * 8;
        // At most 72 KiB for the checkpoint's largest reduction. Reuse the panel
        // across tokens and Rayon jobs without expanding the resident matrix.
        expanded.resize(count, 0);
        for panel in start..end {
            for at in (0..count).step_by(32) {
                _mm256_storeu_si256(
                    expanded.as_mut_ptr().add(at).cast(),
                    unpack(w.data.as_ptr().add((panel * count + at) / 2)),
                );
            }
            project_panel::<4, false>(
                w,
                x,
                out,
                tokens,
                panel,
                expanded.as_ptr(),
                w.scales.as_ptr().add(panel * groups * 8),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack(data: *const i8) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    unsafe {
        let packed = _mm256_cvtepu8_epi16(_mm_loadu_si128(data.cast()));
        let low = _mm256_and_si256(packed, _mm256_set1_epi16(0x000f));
        let high = _mm256_and_si256(_mm256_slli_epi16::<4>(packed), _mm256_set1_epi16(0x0f00));
        _mm256_sub_epi8(_mm256_or_si256(low, high), _mm256_set1_epi8(8))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn project_panel<const R: usize, const Q4: bool>(
    w: &Matrix,
    x: &Scratch,
    out: *mut f32,
    tokens: usize,
    panel: usize,
    data: *const i8,
    scales: *const f32,
) {
    unsafe {
        let full = tokens / R * R;
        for r in (0..full).step_by(R) {
            rows::<R, Q4>(w, x, out, r, panel, data, scales);
        }
        match tokens - full {
            0 => {}
            1 => rows::<1, Q4>(w, x, out, full, panel, data, scales),
            2 => rows::<2, Q4>(w, x, out, full, panel, data, scales),
            3 => rows::<3, Q4>(w, x, out, full, panel, data, scales),
            4 => rows::<4, Q4>(w, x, out, full, panel, data, scales),
            5 => rows::<5, Q4>(w, x, out, full, panel, data, scales),
            6 => rows::<6, Q4>(w, x, out, full, panel, data, scales),
            7 => rows::<7, Q4>(w, x, out, full, panel, data, scales),
            _ => unreachable!(),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows<const R: usize, const Q4: bool>(
    w: &Matrix,
    x: &Scratch,
    out: *mut f32,
    first: usize,
    panel: usize,
    data: *const i8,
    scales: *const f32,
) {
    use std::arch::x86_64::*;
    unsafe {
        let groups = w.input.div_ceil(GROUP);
        let one = _mm256_set1_epi16(1);
        let mut sums = [_mm256_setzero_ps(); R];
        for g in 0..groups {
            let mut acc = [_mm256_setzero_si256(); R];
            let base = g * GROUP * 8;
            for k in (0..GROUP).step_by(4) {
                let b = if Q4 {
                    unpack(data.add((base + k * 8) / 2))
                } else {
                    _mm256_loadu_si256(data.add(base + k * 8).cast())
                };
                let magnitude = _mm256_abs_epi8(b);
                for r in 0..R {
                    let a = _mm256_set1_epi32(
                        x.values
                            .as_ptr()
                            .add(((first + r) * groups + g) * GROUP + k)
                            .cast::<i32>()
                            .read_unaligned(),
                    );
                    // Both codes are in [-127,127]: each pair fits i16, including extrema.
                    let pair = _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(a, b));
                    acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(pair, one));
                }
            }
            let sw = _mm256_loadu_ps(scales.add(g * 8));
            for r in 0..R {
                let s = _mm256_mul_ps(sw, _mm256_set1_ps(x.scales[(first + r) * groups + g]));
                sums[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[r]), s, sums[r]);
                if !w.offsets.is_empty() {
                    let offset = _mm256_loadu_ps(w.offsets.as_ptr().add((panel * groups + g) * 8));
                    let sx = x.scales[(first + r) * groups + g]
                        * x.sums[(first + r) * groups + g] as f32;
                    sums[r] = _mm256_fmadd_ps(offset, _mm256_set1_ps(sx), sums[r]);
                }
            }
        }
        for r in 0..R {
            let dst = out.add((first + r) * w.output + panel * 8);
            if panel * 8 + 8 <= w.output {
                _mm256_storeu_ps(dst, sums[r]);
            } else {
                let mut tail = [0.; 8];
                _mm256_storeu_ps(tail.as_mut_ptr(), sums[r]);
                std::ptr::copy_nonoverlapping(tail.as_ptr(), dst, w.output - panel * 8);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_offsets_match_independent_dequantized_dot_products() -> Result<()> {
        let (k, n) = (65usize, 9usize);
        let bytes: Vec<_> = (0..k * n)
            .flat_map(|i| {
                half::bf16::from_f32(if i / k % 3 == 0 {
                    (i / k) as f32 + 3.
                } else {
                    ((i * 31 % 97) as f32 - 48.) / 17. + 9.
                })
                .to_bits()
                .to_le_bytes()
            })
            .collect();
        for avx2 in [false, super::super::cpu::avx2_available()] {
            let w = Matrix::from_bf16_optimized(&bytes, k, n, avx2, 4, 8, false, true)?;
            assert_eq!(
                w.bytes(),
                n.div_ceil(8) * 8 * k.div_ceil(GROUP) * (GROUP / 2 + 8)
            );
            for m in [1, 15, 16, 17] {
                let x: Vec<_> = (0..m * k)
                    .map(|i| ((i * 13 % 255) as f32 - 127.) / 17. + 3.)
                    .collect();
                let mut out = Vec::new();
                let mut scratch = Scratch::default();
                w.apply(&x, &mut out, &mut scratch);
                for r in 0..m {
                    for c in 0..n {
                        let groups = k.div_ceil(GROUP);
                        let expected: f64 = (0..k)
                            .map(|i| {
                                let (g, j) = (i / GROUP, i % GROUP);
                                let index = (c / 8 * groups + g) * GROUP * 8
                                    + j / 4 * 32
                                    + c % 8 * 4
                                    + j % 4;
                                let sg = (c / 8 * groups + g) * 8 + c % 8;
                                let weight = w.code(index) as f64 * w.scales[sg] as f64
                                    + w.offsets[sg] as f64;
                                weight
                                    * scratch.values[(r * groups + g) * GROUP + j] as f64
                                    * scratch.scales[r * groups + g] as f64
                            })
                            .sum();
                        assert!(
                            (out[r * n + c] as f64 - expected).abs() < 1e-3 + expected.abs() * 4e-6
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn fitted_q4_signed_scales_and_full_code_range_match_scalar() -> Result<()> {
        if !super::super::cpu::avx2_available() {
            return Ok(());
        }
        for (k, n) in [(65, 9), (128, 33)] {
            let bytes: Vec<_> = (0..k * n)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 37 % 251) as f32 - 125.) / 31.)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect();
            let mut w = Matrix::from_bf16_optimized(&bytes, k, n, true, 4, 8, true, false)?;
            assert!(w.scales.iter().any(|&s| s < 0.));
            assert!((0..w.data.len() * 2).any(|i| w.code(i) == -8));
            for m in [1, 15, 16, 17] {
                let x: Vec<_> = (0..m * k)
                    .map(|i| ((i * 13 % 255) as f32 - 127.) / 17.)
                    .collect();
                let mut scalar = Vec::new();
                let mut simd = Vec::new();
                w.avx2 = false;
                w.apply(&x, &mut scalar, &mut Scratch::default());
                w.avx2 = true;
                w.apply(&x, &mut simd, &mut Scratch::default());
                for (a, b) in scalar.iter().zip(simd) {
                    assert!((a - b).abs() < 1e-4 + a.abs() * 2e-6, "{a} != {b}");
                }
            }
        }
        Ok(())
    }
    #[test]
    fn q4_and_q8_match_independent_grouped_dots() -> Result<()> {
        // Odd widths and panels exercise both nibble order and zero padding.
        for (wb, ab) in [(8, 8), (4, 8), (4, 4)] {
            for (m, k, n) in [
                (0usize, 7usize, 1usize),
                (1, 1, 1),
                (3, 63, 7),
                (7, 65, 9),
                (9, 129, 33),
                (15, 65, 9),
                (16, 65, 9),
                (17, 127, 32),
                (33, 1025, 65),
            ] {
                let weights: Vec<f32> = (0..k * n)
                    .map(|i| half::bf16::from_f32(((i * 37 % 257) as f32 - 128.) / 31.).to_f32())
                    .collect();
                let bytes: Vec<_> = weights
                    .iter()
                    .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                    .collect();
                let x: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 13 % 255) as f32 - 127.) / 17.)
                    .collect();
                let expected: Vec<f64> = (0..m * n)
                    .map(|i| {
                        let (r, c) = (i / n, i % n);
                        (0..k.div_ceil(GROUP))
                            .map(|g| {
                                let end = (g * GROUP + GROUP).min(k);
                                let xv = &x[r * k + g * GROUP..r * k + end];
                                let wv = &weights[c * k + g * GROUP..c * k + end];
                                let quant = |v: &[f32], bits| {
                                    let limit = if bits == 4 { 7. } else { 127. };
                                    let scale = (v.iter().map(|v| v.abs()).fold(0., f32::max)
                                        / limit)
                                        .max(f32::MIN_POSITIVE);
                                    (
                                        v.iter()
                                            .map(|&v| {
                                                (v * scale.recip())
                                                    .round_ties_even()
                                                    .clamp(-limit, limit)
                                                    as i32
                                            })
                                            .collect::<Vec<_>>(),
                                        scale,
                                    )
                                };
                                let (a, sa) = quant(xv, ab);
                                let (b, sb) = quant(wv, wb);
                                a.iter().zip(b).map(|(&a, b)| (a * b) as f64).sum::<f64>()
                                    * (sa * sb) as f64
                            })
                            .sum()
                    })
                    .collect();
                for avx2 in [false, super::super::cpu::avx2_available()] {
                    let w = Matrix::from_bf16_quantized(&bytes, k, n, avx2, wb, ab)?;
                    assert_eq!(
                        w.bytes(),
                        n.div_ceil(8) * 8 * k.div_ceil(GROUP) * (GROUP * wb as usize / 8 + 4)
                    );
                    for threads in [1, 3] {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(threads)
                            .build()?;
                        let mut a = Vec::new();
                        let mut b = Vec::new();
                        let mut scratch = Scratch::default();
                        pool.install(|| {
                            Matrix::apply_pair([&w, &w], &x, [&mut a, &mut b], &mut scratch)
                        });
                        assert_eq!(a, b);
                        #[cfg(target_arch = "x86_64")]
                        if avx2 {
                            let mut fused = vec![0.; m * n];
                            unsafe {
                                project::<4>(&w, &scratch, fused.as_mut_ptr(), m, 0, n.div_ceil(8));
                            }
                            assert_eq!(a, fused, "panel cache changed W{wb}A{ab} outputs");
                        }
                        for (&actual, &expected) in a.iter().zip(&expected) {
                            assert!(
                                (actual as f64 - expected).abs() <= 1e-4 + expected.abs() * 2e-6,
                                "W{wb}A{ab} {m}x{k}x{n}: {actual} != {expected}"
                            );
                        }
                        // Reusing buffers with different input must never reuse stale activation codes.
                        pool.install(|| w.apply(&vec![0.; m * k], &mut a, &mut scratch));
                        assert!(a.iter().all(|&v| v == 0.));
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn activation_rounding_matches_portable_at_boundaries() {
        if !super::super::cpu::avx2_available() {
            return;
        }
        let mut values = vec![0., -0., f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
        for i in -127..127 {
            let half = i as f32 + 0.5;
            values.extend([half.next_down(), half, half.next_up()]);
        }
        for s in [1., 0.0037, f32::MIN_POSITIVE, 1e30] {
            let scaled: Vec<_> = values.iter().map(|v| v * s).collect();
            for n in [1, 7, 8, 9, 63, 64, 65, scaled.len()] {
                let mut a = vec![0; n];
                let mut b = vec![0; n];
                assert_eq!(
                    group_scale(&scaled[..n], true),
                    group_scale(&scaled[..n], false)
                );
                pack(&scaled[..n], &mut a, s, false);
                pack(&scaled[..n], &mut b, s, true);
                assert_eq!(a, b, "scale={s}, n={n}");
                assert!(a.iter().all(|&v| v != -128));
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "manual AVX2 projection benchmark; excludes model quality and weight load time"]
    fn projection_benchmark() -> Result<()> {
        use super::super::{
            cpu::{Linear, ProjectionScratch},
            weights::Weights,
        };
        use safetensors::{Dtype, tensor::TensorView};
        use std::{hint::black_box, time::Instant};
        if !super::super::cpu::avx2_available() {
            return Ok(());
        }
        let m = std::env::var("DECISIVE_PPLX_BENCH_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(117);
        for (k, n) in [(2560, 9216), (9216, 2560)] {
            let bytes: Vec<_> = (0..k * n)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 17 % 113) as f32 - 56.) / 256.)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect();
            let w = Matrix::from_bf16(&bytes, k, n, true)?;
            let weights = Weights(std::collections::HashMap::from([(
                "p.weight".into(),
                TensorView::new(Dtype::BF16, vec![n, k], &bytes)?,
            )]));
            let bf16 = Linear::load(&weights, "p", k, n, true)?;
            let x: Vec<_> = (0..m * k)
                .map(|i| ((i * 23 % 127) as f32 - 63.) / 64.)
                .collect();
            for threads in [1, 3] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()?;
                pool.install(|| {
                    let mut scratch = Scratch::default(); let mut y = Vec::new();
                    w.apply(&x, &mut y, &mut scratch);
                    let expected=y.clone();
                    for tile_rows in [4,5,6,8] {
                        let mut times = Vec::new();
                        for _ in 0..5 {
                            let dst = Columns(y.as_mut_ptr());
                            let start = Instant::now();
                            (0..n.div_ceil(32)).into_par_iter().for_each(|tile| unsafe {
                                let first=tile*4; let end=(first+4).min(n.div_ceil(8));
                                match tile_rows {
                                    2=>project::<2>(&w,&scratch,dst.pointer(),m,first,end),
                                    3=>project::<3>(&w,&scratch,dst.pointer(),m,first,end),
                                    4=>project::<4>(&w,&scratch,dst.pointer(),m,first,end),
                                    5=>project::<5>(&w,&scratch,dst.pointer(),m,first,end),
                                    6=>project::<6>(&w,&scratch,dst.pointer(),m,first,end),
                                    8=>project::<8>(&w,&scratch,dst.pointer(),m,first,end),
                                    _=>unreachable!(),
                                }
                            });
                            times.push(start.elapsed().as_secs_f64()*1000.);
                            assert_eq!(y,expected,"projection tile variant {tile_rows}");
                            black_box(&y);
                        }
                        println!("{}", serde_json::json!({"kind":"int8-kernel", "m":m,"k":k,"n":n,"threads":threads,"rows":tile_rows,"ms":times}));
                    }
                    let mut scratch = ProjectionScratch::default();
                    let mut times = Vec::new();
                    for _ in 0..5 {
                        let start=Instant::now();bf16.apply(&x,&mut y,6,&mut scratch);
                        times.push(start.elapsed().as_secs_f64()*1000.);black_box(&y);
                    }
                    println!("{}", serde_json::json!({"kind":"bf16-projection", "m":m,"k":k,"n":n,"threads":threads,"ms":times}));
                });
            }
        }
        Ok(())
    }
    #[test]
    fn integer_extrema_tails_and_threads_match_scalar() -> Result<()> {
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?;
            for (m, k, n) in [
                (0, 1, 1),
                (1, 7, 1),
                (3, 64, 8),
                (4, 65, 9),
                (7, 127, 33),
                (13, 256, 65),
                (17, 1025, 9),
            ] {
                let weights: Vec<_> = (0..k * n)
                    .map(|i| match i % 5 {
                        0 => -127.,
                        1 => 127.,
                        2 => 0.,
                        _ => (i % 31) as f32 / 8.,
                    })
                    .collect();
                let bytes: Vec<_> = weights
                    .iter()
                    .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                    .collect();
                let x: Vec<_> = (0..m * k)
                    .map(|i| match i % 7 {
                        0 => 127.,
                        1 => -127.,
                        2 => 0.,
                        _ => (i % 13) as f32 - 6.,
                    })
                    .collect();
                let mut w = Matrix::from_bf16(&bytes, k, n, false)?;
                let mut a = Vec::new();
                pool.install(|| w.apply(&x, &mut a, &mut Scratch::default()));
                for r in 0..m {
                    for c in 0..n {
                        let expected: f64 = (0..k.div_ceil(GROUP))
                            .map(|g| {
                                let end = (g * GROUP + GROUP).min(k);
                                let xv = &x[r * k + g * GROUP..r * k + end];
                                let wv = &weights[c * k + g * GROUP..c * k + end];
                                let sx = scale(xv);
                                let sw = scale(wv);
                                xv.iter()
                                    .zip(wv)
                                    .map(|(&x, &w)| quantize(x, sx) as f64 * quantize(w, sw) as f64)
                                    .sum::<f64>()
                                    * (sx * sw) as f64
                            })
                            .sum();
                        assert!(
                            (a[r * n + c] as f64 - expected).abs() <= 0.01 + expected.abs() * 1e-6
                        );
                    }
                }
                if super::super::cpu::avx2_available() {
                    w.avx2 = true;
                    let mut b = Vec::new();
                    pool.install(|| w.apply(&x, &mut b, &mut Scratch::default()));
                    for (a, b) in a.iter().zip(b) {
                        assert!(
                            (a - b).abs() <= 0.02 + a.abs() * 1e-6,
                            "{m}x{k}x{n}: {a} != {b}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn invalid_weights_and_nonfinite_activations_are_rejected() -> Result<()> {
        assert!(Matrix::from_bf16(&[], 0, 1, false).is_err());
        assert!(Matrix::from_bf16(&half::bf16::NAN.to_bits().to_le_bytes(), 1, 1, false).is_err());
        let w = Matrix::from_bf16(&[0, 0], 1, 1, false)?;
        let mut y = Vec::new();
        w.apply(&[f32::NAN], &mut y, &mut Scratch::default());
        assert!(y[0].is_nan());
        w.apply(&[0.], &mut y, &mut Scratch::default());
        assert_eq!(y, [0.]);
        Ok(())
    }
}
