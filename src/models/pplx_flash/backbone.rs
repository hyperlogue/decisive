use super::{
    Plan,
    cpu::{self, Linear},
    weights::Weights,
};
use anyhow::{Result, ensure};
#[cfg(feature = "cpu")]
use rayon::prelude::*;
use serde_json::Value;

pub(super) const D: usize = 2560;
const FF: usize = 9216;
pub(super) const V: usize = 248320;
const LH: usize = 32;
const LK: usize = 16;
const LKV: usize = LK * LD;
const LD: usize = 128;
const LV: usize = LH * LD;
const CONV: usize = 2 * LKV + LV;
const AH: usize = 16;
const AD: usize = 256;
const KV: usize = 4 * AD;

struct Delta {
    qkv: Linear,
    z: Linear,
    a: Linear,
    b: Linear,
    out: Linear,
    conv: Vec<f32>,
    decay: Vec<f32>,
    dt: Vec<f32>,
    norm: Vec<f32>,
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    qnorm: Vec<f32>,
    knorm: Vec<f32>,
}
enum Mixer {
    Delta(Delta),
    Attention(Attention),
}
pub(super) struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mix: Mixer,
    gate: Linear,
    up: Linear,
    down: Linear,
}
#[derive(Default)]
pub(super) struct Work {
    pub linear_scratch: cpu::ProjectionScratch,
    pub hidden: Vec<f32>,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    z: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    conv: Vec<f32>,
    context: Vec<f32>,
    heads: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}
impl Layer {
    pub(super) fn load(w: &Weights<'_>, config: &Value, i: usize, avx2: bool) -> Result<Self> {
        Self::load_precision(w, config, i, avx2, false)
    }
    #[cfg(all(feature = "cpu", feature = "int8"))]
    pub(super) fn load_int8(w: &Weights<'_>, config: &Value, i: usize, avx2: bool) -> Result<Self> {
        Self::load_precision(w, config, i, avx2, true)
    }
    fn load_precision(
        w: &Weights<'_>,
        config: &Value,
        i: usize,
        avx2: bool,
        _int8: bool,
    ) -> Result<Self> {
        ensure!(i < 32, "layer index out of range");
        let prefix = "language_model";
        let p = format!("{prefix}.layers.{i}");
        let linear = |name: &str, input, output| {
            #[cfg(all(feature = "cpu", feature = "int8"))]
            if _int8 && output >= 512 {
                return Linear::load_int8(w, &format!("{p}.{name}"), input, output, avx2);
            }
            Linear::load(w, &format!("{p}.{name}"), input, output, avx2)
        };
        let norm = |name: &str, width| w.get(&format!("{p}.{name}.weight"), &[width]);
        let mix = if i % 4 == 3 {
            ensure!(
                config["text_config"]["layer_types"][i] == "full_attention",
                "unsupported layer layout"
            );
            Mixer::Attention(Attention {
                q: linear("self_attn.q_proj", D, AH * AD * 2)?,
                k: linear("self_attn.k_proj", D, KV)?,
                v: linear("self_attn.v_proj", D, KV)?,
                out: linear("self_attn.o_proj", AH * AD, D)?,
                qnorm: norm("self_attn.q_norm", AD)?,
                knorm: norm("self_attn.k_norm", AD)?,
            })
        } else {
            ensure!(
                config["text_config"]["layer_types"][i] == "linear_attention",
                "unsupported layer layout"
            );
            Mixer::Delta(Delta {
                qkv: linear("linear_attn.in_proj_qkv", D, CONV)?,
                z: linear("linear_attn.in_proj_z", D, LV)?,
                a: linear("linear_attn.in_proj_a", D, LH)?,
                b: linear("linear_attn.in_proj_b", D, LH)?,
                out: linear("linear_attn.out_proj", LV, D)?,
                conv: {
                    let original =
                        w.get(&format!("{p}.linear_attn.conv1d.weight"), &[CONV, 1, 4])?;
                    let mut packed = vec![0.; 4 * CONV];
                    for tap in 0..4 {
                        for c in 0..CONV {
                            packed[tap * CONV + c] = original[c * 4 + tap];
                        }
                    }
                    packed
                },
                decay: w
                    .get(&format!("{p}.linear_attn.A_log"), &[LH])?
                    .into_iter()
                    .map(f32::exp)
                    .collect(),
                dt: w.get(&format!("{p}.linear_attn.dt_bias"), &[LH])?,
                norm: norm("linear_attn.norm", LD)?,
            })
        };
        Ok(Self {
            input_norm: norm("input_layernorm", D)?,
            post_norm: norm("post_attention_layernorm", D)?,
            mix,
            gate: linear("mlp.gate_proj", D, FF)?,
            up: linear("mlp.up_proj", D, FF)?,
            down: linear("mlp.down_proj", FF, D)?,
        })
    }
    #[cfg(feature = "cpu")]
    pub(super) fn projection_bytes(&self) -> usize {
        let mixer = match &self.mix {
            Mixer::Delta(l) => {
                l.qkv.bytes() + l.z.bytes() + l.a.bytes() + l.b.bytes() + l.out.bytes()
            }
            Mixer::Attention(l) => l.q.bytes() + l.k.bytes() + l.v.bytes() + l.out.bytes(),
        };
        mixer + self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
    pub(super) fn forward(
        &self,
        work: &mut Work,
        plan: Plan,
        avx2: bool,
        markers: Option<&[usize]>,
    ) -> Result<()> {
        let n = work.hidden.len() / D;
        cpu::rms_from(&work.hidden, &mut work.norm, &self.input_norm, 1e-6, true);
        match &self.mix {
            Mixer::Delta(l) => {
                ensure!(markers.is_none(), "cannot prune a recurrent layer");
                let mut history = vec![0.; 3 * CONV];
                let mut state = vec![0.; LH * LD * LD];
                l.forward(n, &mut history, &mut state, work, plan, avx2);
            }
            Mixer::Attention(l) => {
                let inv_freq: Vec<_> = (0..32)
                    .map(|i| 1. / 10_000_000f32.powf((2 * i) as f32 / 64.))
                    .collect();
                l.forward(n, &inv_freq, work, plan, avx2, markers);
                if let Some(markers) = markers {
                    let mut selected = Vec::with_capacity(markers.len() * D);
                    for &m in markers {
                        selected.extend_from_slice(&work.hidden[m * D..(m + 1) * D]);
                    }
                    work.hidden = selected;
                }
            }
        }
        cpu::add(&mut work.hidden, &work.projected);
        cpu::rms_from(&work.hidden, &mut work.norm, &self.post_norm, 1e-6, true);
        self.gate.apply(
            &work.norm,
            &mut work.gate,
            plan.projection_rows,
            &mut work.linear_scratch,
        );
        self.up.apply(
            &work.norm,
            &mut work.up,
            plan.projection_rows,
            &mut work.linear_scratch,
        );
        cpu::silu_mul(&mut work.gate, &work.up, avx2);
        self.down.apply(
            &work.gate,
            &mut work.projected,
            plan.projection_rows,
            &mut work.linear_scratch,
        );
        cpu::add(&mut work.hidden, &work.projected);
        ensure!(
            work.hidden.iter().all(|v| v.is_finite()),
            "non-finite hidden state"
        );
        Ok(())
    }
}
impl Delta {
    fn forward(
        &self,
        n: usize,
        history: &mut Vec<f32>,
        state: &mut [f32],
        w: &mut Work,
        p: Plan,
        avx2: bool,
    ) {
        self.qkv.apply(
            &w.norm,
            &mut w.qkv,
            p.projection_rows,
            &mut w.linear_scratch,
        );
        self.z
            .apply(&w.norm, &mut w.z, p.projection_rows, &mut w.linear_scratch);
        self.a
            .apply(&w.norm, &mut w.a, p.projection_rows, &mut w.linear_scratch);
        self.b
            .apply(&w.norm, &mut w.b, p.projection_rows, &mut w.linear_scratch);
        w.conv.resize(n * CONV, 0.);
        let qkv = &w.qkv;
        let before = history.as_slice();
        let run = |(t, dst): (usize, &mut [f32])| {
            let rows = std::array::from_fn(|tap| {
                let i = t + tap;
                if i < 3 {
                    &before[i * CONV..(i + 1) * CONV]
                } else {
                    &qkv[(i - 3) * CONV..(i - 2) * CONV]
                }
            });
            cpu::conv_silu(rows, &self.conv, dst, avx2);
        };
        #[cfg(feature = "cpu")]
        if n >= 8 && rayon::current_num_threads() > 1 {
            w.conv.par_chunks_mut(CONV).enumerate().for_each(run);
        } else {
            w.conv.chunks_mut(CONV).enumerate().for_each(run);
        }
        #[cfg(not(feature = "cpu"))]
        w.conv.chunks_mut(CONV).enumerate().for_each(run);
        w.heads.resize(LH * n * LD, 0.);
        let run = |(h, (s, out)): (usize, (&mut [f32], &mut [f32]))| {
            #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
            if avx2 {
                unsafe { delta_avx2(n, h, &w.conv, &w.a, &w.b, self.decay[h], self.dt[h], s, out) };
                return;
            }
            delta_scan(n, h, &w.conv, &w.a, &w.b, self.decay[h], self.dt[h], s, out);
        };
        #[cfg(feature = "cpu")]
        state
            .par_chunks_mut(LD * LD)
            .zip(w.heads.par_chunks_mut(n * LD))
            .enumerate()
            .for_each(run);
        #[cfg(not(feature = "cpu"))]
        state
            .chunks_mut(LD * LD)
            .zip(w.heads.chunks_mut(n * LD))
            .enumerate()
            .for_each(run);
        w.context.resize(n * LV, 0.);
        let heads = &w.heads;
        let z = &w.z;
        let run = |(t, dst): (usize, &mut [f32])| {
            for h in 0..LH {
                let src = &heads[(h * n + t) * LD..(h * n + t + 1) * LD];
                let scale = 1. / (src.iter().map(|v| v * v).sum::<f32>() / LD as f32 + 1e-6).sqrt();
                cpu::delta_gate(
                    src,
                    &self.norm,
                    scale,
                    &z[t * LV + h * LD..t * LV + (h + 1) * LD],
                    &mut dst[h * LD..(h + 1) * LD],
                    avx2,
                );
            }
        };
        #[cfg(feature = "cpu")]
        if n >= 8 && rayon::current_num_threads() > 1 {
            w.context.par_chunks_mut(LV).enumerate().for_each(run);
        } else {
            w.context.chunks_mut(LV).enumerate().for_each(run);
        }
        #[cfg(not(feature = "cpu"))]
        w.context.chunks_mut(LV).enumerate().for_each(run);
        self.out.apply(
            &w.context,
            &mut w.projected,
            p.projection_rows,
            &mut w.linear_scratch,
        );
    }
}
#[inline(always)]
fn delta_scan(
    n: usize,
    h: usize,
    qkv: &[f32],
    a: &[f32],
    b: &[f32],
    decay: f32,
    dt: f32,
    state: &mut [f32],
    out: &mut [f32],
) {
    let mut q = [0.; LD];
    let mut k = [0.; LD];
    let mut memory = [0.; LD];
    let mut delta = [0.; LD];
    for t in 0..n {
        q.copy_from_slice(&qkv[t * CONV + (h / 2) * LD..t * CONV + (h / 2 + 1) * LD]);
        k.copy_from_slice(&qkv[t * CONV + LKV + (h / 2) * LD..t * CONV + LKV + (h / 2 + 1) * LD]);
        let qi = 1. / (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        let ki = 1. / (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        for i in 0..LD {
            q[i] = (q[i] * qi) * (1. / (LD as f32).sqrt());
            k[i] *= ki;
        }
        let x = a[t * LH + h] + dt;
        let softplus = if x > 20. { x } else { x.exp().ln_1p() };
        let g = (-decay * softplus).exp();
        let beta = cpu::sigmoid(b[t * LH + h]);
        memory.fill(0.);
        for i in 0..LD {
            let row = &mut state[i * LD..(i + 1) * LD];
            for j in 0..LD {
                row[j] *= g;
                memory[j] += row[j] * k[i];
            }
        }
        for j in 0..LD {
            delta[j] = (qkv[t * CONV + 2 * LKV + h * LD + j] - memory[j]) * beta;
        }
        let dst = &mut out[t * LD..(t + 1) * LD];
        dst.fill(0.);
        for i in 0..LD {
            let row = &mut state[i * LD..(i + 1) * LD];
            for j in 0..LD {
                row[j] += k[i] * delta[j];
                dst[j] += row[j] * q[i];
            }
        }
    }
}
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn delta_avx2(
    n: usize,
    h: usize,
    qkv: &[f32],
    a: &[f32],
    b: &[f32],
    decay: f32,
    dt: f32,
    state: &mut [f32],
    out: &mut [f32],
) {
    const BLOCK: usize = 8;
    use std::arch::x86_64::*;
    let mut q = [0.; LD];
    let mut k = [0.; LD];
    for t in 0..n {
        q.copy_from_slice(&qkv[t * CONV + (h / 2) * LD..t * CONV + (h / 2 + 1) * LD]);
        k.copy_from_slice(&qkv[t * CONV + LKV + (h / 2) * LD..t * CONV + LKV + (h / 2 + 1) * LD]);
        let qi = 1. / (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        let ki = 1. / (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        for i in 0..LD {
            q[i] = (q[i] * qi) * (1. / (LD as f32).sqrt());
            k[i] *= ki;
        }
        let x = a[t * LH + h] + dt;
        let softplus = if x > 20. { x } else { x.exp().ln_1p() };
        let g = (-decay * softplus).exp();
        let beta = cpu::sigmoid(b[t * LH + h]);
        unsafe {
            let gv = _mm256_set1_ps(g);
            let bv = _mm256_set1_ps(beta);
            for column in (0..LD).step_by(BLOCK * 8) {
                let mut memory = [_mm256_setzero_ps(); BLOCK];
                for i in 0..LD {
                    let kv = _mm256_set1_ps(k[i]);
                    let row = state.as_mut_ptr().add(i * LD + column);
                    for j in 0..BLOCK {
                        let value = _mm256_mul_ps(_mm256_loadu_ps(row.add(j * 8)), gv);
                        _mm256_storeu_ps(row.add(j * 8), value);
                        memory[j] = _mm256_add_ps(memory[j], _mm256_mul_ps(value, kv));
                    }
                }
                let mut delta = [_mm256_setzero_ps(); BLOCK];
                let mut sum = [_mm256_setzero_ps(); BLOCK];
                let v = qkv.as_ptr().add(t * CONV + 2 * LKV + h * LD + column);
                for j in 0..BLOCK {
                    delta[j] =
                        _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v.add(j * 8)), memory[j]), bv);
                }
                for i in 0..LD {
                    let kv = _mm256_set1_ps(k[i]);
                    let qv = _mm256_set1_ps(q[i]);
                    let row = state.as_mut_ptr().add(i * LD + column);
                    for j in 0..BLOCK {
                        let value = _mm256_add_ps(
                            _mm256_loadu_ps(row.add(j * 8)),
                            _mm256_mul_ps(kv, delta[j]),
                        );
                        _mm256_storeu_ps(row.add(j * 8), value);
                        sum[j] = _mm256_add_ps(sum[j], _mm256_mul_ps(value, qv));
                    }
                }
                for j in 0..BLOCK {
                    _mm256_storeu_ps(out.as_mut_ptr().add(t * LD + column + j * 8), sum[j]);
                }
            }
        }
    }
}

fn rope(x: &mut [f32], heads: usize, offset: usize, freq: &[f32]) {
    for (t, row) in x.chunks_exact_mut(heads * AD).enumerate() {
        for head in row.chunks_exact_mut(AD) {
            for i in 0..32 {
                let angle = (offset + t) as f32 * freq[i];
                let (c, s) = (angle.cos(), angle.sin());
                let (a, b) = (head[i], head[i + 32]);
                head[i] = a * c - b * s;
                head[i + 32] = b * c + a * s;
            }
        }
    }
}
impl Attention {
    fn forward(
        &self,
        n: usize,
        freq: &[f32],
        w: &mut Work,
        p: Plan,
        avx2: bool,
        markers: Option<&[usize]>,
    ) {
        self.k.apply(
            &w.norm,
            &mut w.key,
            p.projection_rows,
            &mut w.linear_scratch,
        );
        self.v.apply(
            &w.norm,
            &mut w.value,
            p.projection_rows,
            &mut w.linear_scratch,
        );
        let nk = n;
        if let Some(markers) = markers {
            let mut selected = Vec::with_capacity(markers.len() * D);
            for &m in markers {
                selected.extend_from_slice(&w.norm[m * D..(m + 1) * D]);
            }
            w.norm = selected;
        }
        let n = markers.map_or(n, |m| m.len());
        self.q.apply(
            &w.norm,
            &mut w.qkv,
            p.projection_rows,
            &mut w.linear_scratch,
        );
        w.query.resize(n * AH * AD, 0.);
        w.z.resize(n * AH * AD, 0.);
        for i in 0..n * AH {
            w.query[i * AD..(i + 1) * AD].copy_from_slice(&w.qkv[i * 2 * AD..i * 2 * AD + AD]);
            w.z[i * AD..(i + 1) * AD].copy_from_slice(&w.qkv[i * 2 * AD + AD..(i + 1) * 2 * AD]);
        }
        cpu::rms(&mut w.query, &self.qnorm, 1e-6, true);
        cpu::rms(&mut w.key, &self.knorm, 1e-6, true);
        if let Some(markers) = markers {
            for (row, &position) in w.query.chunks_mut(AH * AD).zip(markers) {
                rope(row, AH, position, freq);
            }
        } else {
            rope(&mut w.query, AH, 0, freq);
        }
        rope(&mut w.key, 4, 0, freq);
        let keys = &w.key;
        let values = &w.value;
        w.heads.resize(AH * n * AD, 0.);
        let run = |(head, out): (usize, &mut [f32])| {
            let kh = head / 4;
            let tile = p.attention_queries;
            let mut scores = vec![0.; tile * nk];
            for start in (0..n).step_by(tile) {
                let rows = tile.min(n - start);
                unsafe {
                    matrixmultiply::sgemm(
                        rows,
                        AD,
                        nk,
                        1. / 16.,
                        w.query.as_ptr().add(start * AH * AD + head * AD),
                        (AH * AD) as isize,
                        1,
                        keys.as_ptr().add(kh * AD),
                        1,
                        KV as isize,
                        0.,
                        scores.as_mut_ptr(),
                        nk as isize,
                        1,
                    );
                }
                for r in 0..rows {
                    let row = &mut scores[r * nk..(r + 1) * nk];
                    cpu::softmax(row);
                }
                unsafe {
                    matrixmultiply::sgemm(
                        rows,
                        nk,
                        AD,
                        1.,
                        scores.as_ptr(),
                        nk as isize,
                        1,
                        values.as_ptr().add(kh * AD),
                        KV as isize,
                        1,
                        0.,
                        out.as_mut_ptr().add(start * AD),
                        AD as isize,
                        1,
                    );
                }
            }
        };
        #[cfg(feature = "cpu")]
        w.heads.par_chunks_mut(n * AD).enumerate().for_each(run);
        #[cfg(not(feature = "cpu"))]
        w.heads.chunks_mut(n * AD).enumerate().for_each(run);
        w.context.resize(n * AH * AD, 0.);
        let heads = &w.heads;
        let z = &w.z;
        let run = |(t, dst): (usize, &mut [f32])| {
            for h in 0..AH {
                let i = t * AH * AD + h * AD;
                cpu::attention_gate(
                    &heads[(h * n + t) * AD..(h * n + t + 1) * AD],
                    &z[i..i + AD],
                    &mut dst[h * AD..(h + 1) * AD],
                    avx2,
                );
            }
        };
        #[cfg(feature = "cpu")]
        if n >= 8 && rayon::current_num_threads() > 1 {
            w.context.par_chunks_mut(AH * AD).enumerate().for_each(run);
        } else {
            w.context.chunks_mut(AH * AD).enumerate().for_each(run);
        }
        #[cfg(not(feature = "cpu"))]
        w.context.chunks_mut(AH * AD).enumerate().for_each(run);
        self.out.apply(
            &w.context,
            &mut w.projected,
            p.projection_rows,
            &mut w.linear_scratch,
        );
    }
}
