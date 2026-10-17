#[cfg(feature = "int8")]
use super::int8;
use super::{Kernel, Precision, weights::Weights};
use anyhow::Result;
#[cfg(feature = "cpu")]
use rayon::prelude::*;

pub fn packed_supported() -> bool {
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    {
        return std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    }
    #[allow(unreachable_code)]
    false
}

enum Weight {
    Float(Vec<f32>),
    #[cfg_attr(not(all(feature = "cpu", target_arch = "x86_64")), allow(dead_code))]
    Packed(Vec<f32>),
    #[cfg(feature = "int8")]
    Int8(int8::Matrix),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Wasm(Vec<f32>),
}
pub struct Linear {
    input: usize,
    output: usize,
    weight: Weight,
    bias: Vec<f32>,
}
impl Linear {
    #[cfg(not(feature = "int8"))]
    pub fn load(
        w: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        precision: Precision,
        kernel: Kernel,
    ) -> Result<Self> {
        anyhow::ensure!(
            precision == Precision::F32,
            "INT8 requires the int8 Cargo feature"
        );
        let data = w.get(&format!("{name}.weight"), &[output, input])?;
        let weight = if matches!(kernel, Kernel::Avx2) {
            let mut packed = vec![0.; output.div_ceil(16) * 16 * input];
            for n in 0..output {
                for k in 0..input {
                    packed[(n / 16 * input + k) * 16 + n % 16] = data[n * input + k];
                }
            }
            Weight::Packed(packed)
        } else {
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            if matches!(kernel, Kernel::WasmSimd) {
                return Ok(Self {
                    input,
                    output,
                    weight: Weight::Wasm(super::wasm_simd::pack(&data, input, output)),
                    bias: w.get(&format!("{name}.bias"), &[output])?,
                });
            }
            Weight::Float(data)
        };
        Ok(Self {
            input,
            output,
            weight,
            bias: w.get(&format!("{name}.bias"), &[output])?,
        })
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[cfg(feature = "int8")]
    pub(super) fn try_encoder_ffn(
        first: &Self,
        second: &Self,
        norm: &Norm,
        hidden: &mut [f32],
        plan: super::specialization::CpuPlan,
        scratch: &mut super::wasm_ffn::Scratch,
    ) -> bool {
        let (Weight::Int8(a), Weight::Int8(b)) = (&first.weight, &second.weight) else {
            return false;
        };
        if a.mode != int8::Mode::WeightsActivations
            || b.mode != int8::Mode::WeightsActivations
            || first.input != second.output
            || first.output != second.input
            || hidden.len() / first.input < 64
            || !hidden.iter().all(|v| v.is_finite())
        {
            return false;
        }
        assert_eq!(hidden.len() % first.input, 0);
        assert_eq!(norm.weight.len(), first.input);
        super::wasm_ffn::apply(
            hidden,
            a,
            &first.bias,
            b,
            &second.bias,
            norm,
            plan.int8_projection_rows
                .unwrap_or(plan.projection_rows)
                .min(4),
            scratch,
        );
        true
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[cfg(feature = "int8")]
    pub(super) fn apply_shared_int8(
        linears: [&Self; 3],
        x: &[f32],
        outputs: [&mut Vec<f32>; 3],
        workspace: &mut Vec<f32>,
        plan: super::specialization::CpuPlan,
    ) -> bool {
        if let (Weight::Int8(q), Weight::Int8(k), Weight::Int8(v)) =
            (&linears[0].weight, &linears[1].weight, &linears[2].weight)
        {
            int8::Matrix::apply_shared(
                [q, k, v],
                x,
                outputs,
                [&linears[0].bias, &linears[1].bias, &linears[2].bias],
                workspace,
                plan.int8_projection_rows
                    .unwrap_or(plan.projection_rows)
                    .min(4),
            )
        } else {
            false
        }
    }

    #[cfg(feature = "int8")]
    pub fn load(
        w: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        precision: Precision,
        kernel: Kernel,
    ) -> Result<Self> {
        let avx2 = matches!(kernel, Kernel::Avx2);
        let half = w.f16(&format!("{name}.weight"), &[output, input])?;
        anyhow::ensure!(
            half.is_none() || precision == Precision::Int8Compact,
            "mixed compact Decima projections require compact precision"
        );
        let stored_half = half.is_some();
        let imported = if stored_half {
            None
        } else {
            w.int8(&format!("{name}.weight"), &[output, input])?
        };
        let weight = if let Some(q) = imported {
            anyhow::ensure!(
                matches!(precision, Precision::Int8Full | Precision::Int8Compact),
                "compact Decima projections require int8-full or int8-compact"
            );
            Weight::Int8(int8::Matrix::from_quantized(
                q.values,
                &q.scales,
                output,
                input,
                q.group,
                if precision == Precision::Int8Compact {
                    int8::Mode::Weights
                } else {
                    int8::Mode::WeightsActivations
                },
                avx2,
            )?)
        } else {
            let data = match half {
                Some(values) => values,
                None => w.get(&format!("{name}.weight"), &[output, input])?,
            };
            let execution_precision = if stored_half {
                Precision::F32
            } else {
                precision
            };
            match execution_precision {
                Precision::F32 if avx2 => {
                    let mut packed = vec![0.; output.div_ceil(16) * 16 * input];
                    for n in 0..output {
                        for k in 0..input {
                            packed[(n / 16 * input + k) * 16 + n % 16] = data[n * input + k];
                        }
                    }
                    Weight::Packed(packed)
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                Precision::F32 if matches!(kernel, Kernel::WasmSimd) => {
                    Weight::Wasm(super::wasm_simd::pack(&data, input, output))
                }
                Precision::F32 => Weight::Float(data),
                Precision::Int8
                | Precision::Int8Weights
                | Precision::Int8Full
                | Precision::Int8Compact => Weight::Int8(int8::Matrix::new(
                    &data,
                    output,
                    input,
                    32,
                    if matches!(precision, Precision::Int8 | Precision::Int8Full) {
                        int8::Mode::WeightsActivations
                    } else {
                        int8::Mode::Weights
                    },
                    avx2,
                )?),
            }
        };
        Ok(Self {
            input,
            output,
            weight,
            bias: w.get(&format!("{name}.bias"), &[output])?,
        })
    }
    #[cfg(feature = "webgpu")]
    pub(super) fn into_float_parts(self) -> Result<(usize, usize, Vec<f32>, Vec<f32>)> {
        let data = match self.weight {
            Weight::Float(v) => v,
            Weight::Packed(p) => {
                let mut v = vec![0.; self.input * self.output];
                for n in 0..self.output {
                    for k in 0..self.input {
                        v[n * self.input + k] = p[(n / 16 * self.input + k) * 16 + n % 16];
                    }
                }
                v
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Weight::Wasm(p) => super::wasm_simd::unpack(&p, self.input, self.output),
            #[cfg(feature = "int8")]
            Weight::Int8(_) => {
                anyhow::bail!("WebGPU requires float Decima projections")
            }
        };
        Ok((self.input, self.output, data, self.bias))
    }
    pub fn apply(&self, x: &[f32], y: &mut Vec<f32>, workspace: &mut Vec<f32>) {
        self.apply_with_rows(x, y, workspace, 32);
    }
    #[inline]
    pub fn apply_with_plan(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        workspace: &mut Vec<f32>,
        plan: super::specialization::CpuPlan,
    ) {
        #[cfg(feature = "int8")]
        let rows = if matches!(self.weight, Weight::Int8(_)) {
            plan.int8_projection_rows.unwrap_or(plan.projection_rows)
        } else {
            plan.projection_rows
        };
        #[cfg(not(feature = "int8"))]
        let rows = plan.projection_rows;
        self.apply_with_rows(x, y, workspace, rows);
    }
    pub fn apply_with_rows(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        workspace: &mut Vec<f32>,
        rows: usize,
    ) {
        #[cfg(not(feature = "int8"))]
        let _ = workspace;
        assert!((1..=32).contains(&rows));
        assert_eq!(x.len() % self.input, 0);
        #[cfg(feature = "int8")]
        if let Weight::Int8(w) = &self.weight {
            w.apply_with_rows(x, y, Some(&self.bias), workspace, rows.min(4));
            return;
        }
        y.resize(x.len() / self.input * self.output, 0.);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if let Weight::Wasm(w) = &self.weight {
            #[cfg(feature = "decima-wasm-threads")]
            if super::wasm_simd::multiply_parallel(
                x,
                w,
                y,
                self.input,
                self.output,
                Some(&self.bias),
                rows.min(6),
            ) {
                return;
            }
            super::wasm_simd::multiply(
                x,
                w,
                y,
                self.input,
                self.output,
                Some(&self.bias),
                rows.min(6),
            );
            return;
        }
        let tile_rows = if matches!(self.weight, Weight::Packed(_)) {
            rows.min(6)
        } else {
            rows
        };
        let run = |(tile, dst): (usize, &mut [f32])| {
            let row = tile * tile_rows;
            match &self.weight {
                #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
                Weight::Packed(w) => unsafe {
                    multiply(x, w, dst, row, self.input, self.output, &self.bias);
                },
                Weight::Float(w) => unsafe {
                    matrixmultiply::sgemm(
                        dst.len() / self.output,
                        self.input,
                        self.output,
                        1.,
                        x.as_ptr().add(row * self.input),
                        self.input as isize,
                        1,
                        w.as_ptr(),
                        1,
                        self.input as isize,
                        0.,
                        dst.as_mut_ptr(),
                        self.output as isize,
                        1,
                    );
                    for out in dst.chunks_mut(self.output) {
                        for (a, b) in out.iter_mut().zip(&self.bias) {
                            *a += b;
                        }
                    }
                },
                #[cfg(any(feature = "int8", not(all(feature = "cpu", target_arch = "x86_64"))))]
                _ => unreachable!("validated kernel at load"),
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

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn multiply(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    first: usize,
    k: usize,
    n: usize,
    bias: &[f32],
) {
    unsafe {
        match y.len() / n {
            1 => multiply_rows::<1>(x, w, y, first, k, n, bias),
            2 => multiply_rows::<2>(x, w, y, first, k, n, bias),
            3 => multiply_rows::<3>(x, w, y, first, k, n, bias),
            4 => multiply_rows::<4>(x, w, y, first, k, n, bias),
            5 => multiply_rows::<5>(x, w, y, first, k, n, bias),
            6 => multiply_rows::<6>(x, w, y, first, k, n, bias),
            _ => unreachable!("six-row projection tile"),
        }
    }
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn multiply_rows<const R: usize>(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    first: usize,
    k: usize,
    n: usize,
    bias: &[f32],
) {
    use std::arch::x86_64::*;
    unsafe {
        for panel in 0..n.div_ceil(16) {
            let mut acc = [[_mm256_setzero_ps(); 2]; R];
            for j in 0..k {
                let b = w.as_ptr().add((panel * k + j) * 16);
                let b0 = _mm256_loadu_ps(b);
                let b1 = _mm256_loadu_ps(b.add(8));
                for i in 0..R {
                    let a = _mm256_set1_ps(*x.get_unchecked((first + i) * k + j));
                    acc[i][0] = _mm256_fmadd_ps(a, b0, acc[i][0]);
                    acc[i][1] = _mm256_fmadd_ps(a, b1, acc[i][1]);
                }
            }
            for i in 0..R {
                if panel * 16 + 16 <= n {
                    let b = bias.as_ptr().add(panel * 16);
                    let dst = y.as_mut_ptr().add(i * n + panel * 16);
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
                        y[i * n + panel * 16 + c] = tmp[c] + bias[panel * 16 + c];
                    }
                }
            }
        }
    }
}

pub struct Norm {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) eps: f32,
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    feature = "kernel-checks"
))]
#[cfg(feature = "int8")]
pub(super) fn check_ffn_wasm() -> Result<usize> {
    let linear = |input: usize, output: usize| -> Result<Linear> {
        let values: Vec<_> = (0..input * output)
            .map(|i| ((i * 43 % 137) as f32 - 68.) / 512.)
            .collect();
        Ok(Linear {
            input,
            output,
            weight: Weight::Int8(int8::Matrix::new(
                &values,
                output,
                input,
                32,
                int8::Mode::WeightsActivations,
                false,
            )?),
            bias: (0..output).map(|i| (i % 29) as f32 / 131.).collect(),
        })
    };
    let first = linear(384, 1536)?;
    let second = linear(1536, 384)?;
    let norm = Norm {
        weight: (0..384).map(|i| 0.8 + (i % 17) as f32 / 100.).collect(),
        bias: vec![0.01; 384],
        eps: 1e-12,
    };
    let mut checked = 0;
    for count in [63, 64, 65, 127, 512] {
        let input: Vec<_> = (0..count * 384)
            .map(|i| ((i * 31 % 149) as f32 - 74.) / 64.)
            .collect();
        for tile in [1, 3, 4] {
            let mut plan = super::specialization::default_plan(super::Kernel::WasmSimd);
            plan.int8_projection_rows = Some(tile);
            let (mut expanded, mut projected, mut work) = (vec![], vec![], vec![]);
            first.apply_with_plan(&input, &mut expanded, &mut work, plan);
            gelu(&mut expanded);
            second.apply_with_plan(&expanded, &mut projected, &mut work, plan);
            let mut expected = input.clone();
            add(&mut expected, &projected);
            norm.in_place(&mut expected);
            let mut actual = vec![999.; input.len() + 2];
            actual[1..=input.len()].copy_from_slice(&input);
            let ran = Linear::try_encoder_ffn(
                &first,
                &second,
                &norm,
                &mut actual[1..=input.len()],
                plan,
                &mut super::wasm_ffn::Scratch::default(),
            );
            anyhow::ensure!(ran == (count >= 64), "FFN length dispatch");
            let reference = if ran { &expected } else { &input };
            anyhow::ensure!(
                actual[1..=input.len()]
                    .iter()
                    .zip(reference)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "FFN arithmetic changed at rows={count}, tile={tile}"
            );
            anyhow::ensure!(
                actual[0] == 999. && actual[input.len() + 1] == 999.,
                "FFN output guard"
            );
            checked += input.len();
        }
    }
    Ok(checked)
}
impl Norm {
    pub fn load(w: &Weights<'_>, name: &str, d: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{name}.weight"), &[d])?,
            bias: w.get(&format!("{name}.bias"), &[d])?,
            eps,
        })
    }
    pub fn apply(&self, x: &[f32], y: &mut Vec<f32>) {
        y.clear();
        y.extend_from_slice(x);
        self.in_place(y);
    }
    pub fn in_place(&self, x: &mut [f32]) {
        let d = self.weight.len();
        for row in x.chunks_exact_mut(d) {
            let mean = row.iter().sum::<f32>() / d as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = 1. / (var + self.eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = (*v - mean) * inv * self.weight[i] + self.bias[i];
            }
        }
    }
}

pub fn gelu(x: &mut [f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        super::activation_wasm::apply(x);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let run = |v: &mut f32| {
            *v = 0.5 * *v * (1. + libm::erff(*v * std::f32::consts::FRAC_1_SQRT_2));
        };
        #[cfg(feature = "cpu")]
        x.par_iter_mut().for_each(run);
        #[cfg(not(feature = "cpu"))]
        x.iter_mut().for_each(run);
    }
}
pub fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}
pub fn softmax(x: &mut [f32]) {
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
pub fn mean(x: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.; d];
    for row in x.chunks_exact(d) {
        add(&mut out, row);
    }
    let n = (x.len() / d) as f32;
    for v in &mut out {
        *v /= n;
    }
    out
}

#[cfg(test)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    heads: usize,
    out: &mut Vec<f32>,
    scores: &mut Vec<f32>,
    work: &mut Vec<f32>,
    avx2: bool,
) {
    assert!(d > 0 && heads > 0 && d % heads == 0);
    assert!(!q.is_empty() && !k.is_empty());
    assert!(q.len() % d == 0 && k.len() % d == 0 && v.len() == k.len());
    assert!(q.len() / d <= 512 && k.len() / d <= 512);
    let tile = (16 * 1024 / (k.len() / d) / 8 * 8).max(8);
    attention_with_tile(q, k, v, d, heads, out, scores, work, tile, avx2);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) fn attention_with_tile(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    heads: usize,
    out: &mut Vec<f32>,
    scores: &mut Vec<f32>,
    work: &mut Vec<f32>,
    tile: usize,
    avx2: bool,
) {
    let _ = avx2;
    super::wasm_attention::apply(q, k, v, d, heads, out, scores, work, tile);
}
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub(super) fn attention_with_tile(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    heads: usize,
    out: &mut Vec<f32>,
    scores: &mut Vec<f32>,
    work: &mut Vec<f32>,
    tile: usize,
    avx2: bool,
) {
    let nq = q.len() / d;
    let nk = k.len() / d;
    let hd = d / heads;
    let rows = nq.min(tile);
    scores.resize(heads * rows * nk, 0.);
    let packed = nq > rows;
    let head_work = (nq + if packed { 2 * nk } else { 0 }) * hd;
    work.resize(heads * head_work, 0.);
    out.resize(nq * d, 0.);
    let run = |(h, (score, ctx)): (usize, (&mut [f32], &mut [f32]))| unsafe {
        let (ctx, kv) = ctx.split_at_mut(nq * hd);
        let (key, value, stride) = if packed {
            let (key, value) = kv.split_at_mut(nk * hd);
            for row in 0..nk {
                key[row * hd..(row + 1) * hd]
                    .copy_from_slice(&k[row * d + h * hd..row * d + (h + 1) * hd]);
                value[row * hd..(row + 1) * hd]
                    .copy_from_slice(&v[row * d + h * hd..row * d + (h + 1) * hd]);
            }
            (key.as_ptr(), value.as_ptr(), hd as isize)
        } else {
            (k.as_ptr().add(h * hd), v.as_ptr().add(h * hd), d as isize)
        };
        for first in (0..nq).step_by(rows) {
            let nr = rows.min(nq - first);
            let score = &mut score[..nr * nk];
            matrixmultiply::sgemm(
                nr,
                hd,
                nk,
                1. / (hd as f32).sqrt(),
                q.as_ptr().add(first * d + h * hd),
                d as isize,
                1,
                key,
                1,
                stride,
                0.,
                score.as_mut_ptr(),
                nk as isize,
                1,
            );
            for row in score.chunks_mut(nk) {
                #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
                if avx2 {
                    super::softmax_avx2::softmax(row);
                    continue;
                }
                let _ = avx2;
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                super::softmax_wasm::softmax(row);
                #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
                softmax(row);
            }
            matrixmultiply::sgemm(
                nr,
                nk,
                hd,
                1.,
                score.as_ptr(),
                nk as isize,
                1,
                value,
                stride,
                1,
                0.,
                ctx.as_mut_ptr().add(first * hd),
                hd as isize,
                1,
            );
        }
    };
    #[cfg(feature = "cpu")]
    scores
        .par_chunks_mut(rows * nk)
        .zip(work.par_chunks_mut(head_work))
        .enumerate()
        .for_each(run);
    #[cfg(not(feature = "cpu"))]
    scores
        .chunks_mut(rows * nk)
        .zip(work.chunks_mut(head_work))
        .enumerate()
        .for_each(run);
    for r in 0..nq {
        for h in 0..heads {
            out[r * d + h * hd..r * d + (h + 1) * hd]
                .copy_from_slice(&work[h * head_work + r * hd..h * head_work + (r + 1) * hd]);
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    fn data(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 41 + seed) % 127) as f32 / 63. - 1.)
            .collect()
    }
    #[test]
    fn packed_projection_tails_match_scalar_fma() {
        if !packed_supported() {
            return;
        }
        for (k, n) in [(35, 17), (384, 384), (384, 1536), (1536, 384)] {
            let w = data(k * n, 11);
            let bias = data(n, 17);
            let mut packed = vec![0.; n.div_ceil(16) * 16 * k];
            for c in 0..n {
                for j in 0..k {
                    packed[(c / 16 * k + j) * 16 + c % 16] = w[c * k + j];
                }
            }
            let linear = Linear {
                input: k,
                output: n,
                weight: Weight::Packed(packed),
                bias: bias.clone(),
            };
            for m in 1..=8 {
                let x = data(m * k, 3);
                let mut actual = vec![];
                linear.apply(&x, &mut actual, &mut vec![]);
                for r in 0..m {
                    for c in 0..n {
                        let expected = (0..k)
                            .fold(0_f32, |sum, j| x[r * k + j].mul_add(w[c * k + j], sum))
                            + bias[c];
                        assert_eq!(
                            actual[r * n + c],
                            expected,
                            "{m}x{k}x{n} row {r}, column {c}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn tiled_attention_matches_dense() {
        for (nq, nk, heads) in [
            (1, 1, 12),
            (31, 33, 12),
            (33, 31, 12),
            (64, 512, 6),
            (127, 127, 12),
            (512, 512, 12),
        ] {
            let (q, k, v) = (data(nq * 384, 1), data(nk * 384, 3), data(nk * 384, 7));
            let (mut dense, mut tile, mut scores, mut work) = (vec![], vec![], vec![], vec![]);
            attention_with_tile(
                &q,
                &k,
                &v,
                384,
                heads,
                &mut dense,
                &mut scores,
                &mut work,
                nq,
                false,
            );
            attention(
                &q,
                &k,
                &v,
                384,
                heads,
                &mut tile,
                &mut scores,
                &mut work,
                packed_supported(),
            );
            assert!(scores.len() <= heads * 32 * 512);
            let error = dense
                .iter()
                .zip(&tile)
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            assert!(error < 2e-6, "{nq}x{nk}: {error}");
        }
    }
    #[test]
    #[ignore = "kernel timing; run alone in release mode"]
    fn attention_tiles_benchmark() {
        for threads in [1, 6] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                for (nq,nk,heads) in [(32,32,12),(128,128,12),(512,512,12),(64,512,6)] {
                    let (q,k,v)=(data(nq*384,1),data(nk*384,3),data(nk*384,7));
                    for (label,tile,vector) in [("dense-scalar",nq,false),("dense-vector",nq,packed_supported()),("bounded-vector",(16*1024/nk/8*8).max(8),packed_supported())] {
                        let (mut out,mut scores,mut work)=(vec![],vec![],vec![]);
                        for _ in 0..3 {attention_with_tile(&q,&k,&v,384,heads,&mut out,&mut scores,&mut work,tile,vector);}
                        let mut times=vec![];
                        for _ in 0..15 {
                            let t=std::time::Instant::now();
                            attention_with_tile(&q,&k,&v,384,heads,&mut out,&mut scores,&mut work,tile,vector);
                            std::hint::black_box(&out);
                            times.push(t.elapsed().as_secs_f64()*1000.);
                        }
                        times.sort_by(f64::total_cmp);
                        eprintln!("threads={threads} q={nq} k={nk} heads={heads} path={label} tile={tile} median_ms={:.3}",times[7]);
                    }
                }
            });
        }
    }
}
