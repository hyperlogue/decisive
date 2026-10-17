use super::{
    Plan,
    cpu::{self, Linear},
    weights::Weights,
};
use anyhow::{Result, ensure};
#[cfg(feature = "cpu")]
use rayon::prelude::*;
use serde_json::Value;

const D: usize = 1024;
const FF: usize = 3584;
const V: usize = 248320;
const LH: usize = 16;
const LD: usize = 128;
const LV: usize = LH * LD;
const CONV: usize = 3 * LV;
const AH: usize = 8;
const AD: usize = 256;
const KV: usize = 2 * AD;

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
struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mix: Mixer,
    gate: Linear,
    up: Linear,
    down: Linear,
}
pub(super) struct Backbone {
    embedding: Vec<u16>,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    inv_freq: Vec<f32>,
    avx2: bool,
}

#[derive(Clone)]
enum LayerState {
    Delta { conv: Vec<f32>, recurrent: Vec<f32> },
    Attention { keys: Vec<f32>, values: Vec<f32> },
}
#[derive(Clone)]
pub(super) struct State {
    layers: Vec<LayerState>,
    pub position: usize,
}
impl State {
    pub fn bytes_at(&self, position: usize) -> usize {
        self.layers
            .iter()
            .map(|layer| match layer {
                LayerState::Delta { conv, recurrent } => (conv.len() + recurrent.len()) * 4,
                LayerState::Attention { .. } => position * KV * 2 * 4,
            })
            .sum()
    }
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Delta { conv, recurrent } => (conv.len() + recurrent.len()) * 4,
                LayerState::Attention { keys, values } => (keys.len() + values.len()) * 4,
            })
            .sum()
    }
}
#[derive(Default)]
pub(super) struct Work {
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
impl Backbone {
    pub fn load(w: &Weights<'_>, config: &Value, avx2: bool) -> Result<Self> {
        let c = config.get("text_config").unwrap_or(config);
        for (key, value) in [
            ("hidden_size", D),
            ("intermediate_size", FF),
            ("vocab_size", V),
            ("num_hidden_layers", 24),
            ("num_attention_heads", AH),
            ("num_key_value_heads", 2),
            ("head_dim", AD),
            ("linear_num_key_heads", LH),
            ("linear_num_value_heads", LH),
            ("linear_key_head_dim", LD),
            ("linear_value_head_dim", LD),
            ("linear_conv_kernel_dim", 4),
        ] {
            ensure!(
                c[key].as_u64() == Some(value as u64),
                "unsupported Qwen3.5 configuration: {key}"
            );
        }
        ensure!(
            c["hidden_act"] == "silu"
                && c["attention_bias"] == false
                && c["rms_norm_eps"].as_f64() == Some(1e-6),
            "unsupported activation/norm/attention configuration"
        );
        ensure!(
            c["rope_parameters"]["rope_theta"].as_f64() == Some(10_000_000.)
                && c["rope_parameters"]["partial_rotary_factor"].as_f64() == Some(0.25),
            "unsupported rotary configuration"
        );
        ensure!(
            c["model_type"] == "qwen3_5_text"
                && c["attn_output_gate"] == true
                && c["tie_word_embeddings"] == true
                && c["rope_parameters"]["rope_type"] == "default"
                && c["layer_types"].as_array().is_some_and(|v| v.len() == 24),
            "unsupported decoder, gate, embedding or rotary variant"
        );
        let prefix = "model.language_model";
        let mut layers = Vec::new();
        for i in 0..24 {
            let p = format!("{prefix}.layers.{i}");
            let linear = |name: &str, input, output| {
                Linear::load(w, &format!("{p}.{name}"), input, output, avx2)
            };
            let norm = |name: &str, width| w.get(&format!("{p}.{name}.weight"), &[width]);
            let mix = if i % 4 == 3 {
                ensure!(
                    c["layer_types"][i] == "full_attention",
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
                    c["layer_types"][i] == "linear_attention",
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
            layers.push(Layer {
                input_norm: norm("input_layernorm", D)?,
                post_norm: norm("post_attention_layernorm", D)?,
                mix,
                gate: linear("mlp.gate_proj", D, FF)?,
                up: linear("mlp.up_proj", D, FF)?,
                down: linear("mlp.down_proj", FF, D)?,
            });
        }
        Ok(Self {
            embedding: w.embedding(&format!("{prefix}.embed_tokens.weight"), V, D)?,
            layers,
            norm: w.get(&format!("{prefix}.norm.weight"), &[D])?,
            inv_freq: (0..32)
                .map(|i| 1. / 10_000_000f32.powf((2 * i) as f32 / 64.))
                .collect(),
            avx2,
        })
    }
    pub fn empty_state(&self) -> State {
        State {
            position: 0,
            layers: self
                .layers
                .iter()
                .map(|l| match l.mix {
                    Mixer::Delta(_) => LayerState::Delta {
                        conv: vec![0.; 3 * CONV],
                        recurrent: vec![0.; LH * LD * LD],
                    },
                    Mixer::Attention(_) => LayerState::Attention {
                        keys: Vec::new(),
                        values: Vec::new(),
                    },
                })
                .collect(),
        }
    }
    pub fn forward(
        &self,
        ids: &[u32],
        state: &mut State,
        work: &mut Work,
        plan: Plan,
    ) -> Result<()> {
        self.forward_saving(ids, state, work, plan, [None, None])?;
        Ok(())
    }
    pub fn forward_saving(
        &self,
        ids: &[u32],
        state: &mut State,
        work: &mut Work,
        plan: Plan,
        positions: [Option<usize>; 2],
    ) -> Result<[Option<State>; 2]> {
        ensure!(
            positions
                .iter()
                .flatten()
                .all(|&p| p > state.position && p <= state.position + ids.len()),
            "snapshot outside decoder chunk"
        );
        ensure!(
            !ids.is_empty() && ids.len() <= 1024 && state.position + ids.len() <= 262144,
            "invalid decoder chunk/position"
        );
        ensure!(
            ids.iter().all(|t| (*t as usize) < V),
            "token outside vocabulary"
        );
        let n = ids.len();
        let mut saved = positions.map(|p| p.map(|_| state.clone()));
        work.hidden.clear();
        work.hidden.reserve(n * D);
        for &t in ids {
            work.hidden.extend(
                self.embedding[t as usize * D..(t as usize + 1) * D]
                    .iter()
                    .map(|v| f32::from_bits((*v as u32) << 16)),
            );
        }
        for (index, (layer, cache)) in self.layers.iter().zip(&mut state.layers).enumerate() {
            cpu::rms_from(&work.hidden, &mut work.norm, &layer.input_norm, 1e-6, true);
            let [first, second] = &mut saved;
            let saves = [
                first
                    .as_mut()
                    .map(|s| (positions[0].unwrap() - state.position, &mut s.layers[index])),
                second
                    .as_mut()
                    .map(|s| (positions[1].unwrap() - state.position, &mut s.layers[index])),
            ];
            match (&layer.mix, cache) {
                (Mixer::Delta(l), LayerState::Delta { conv, recurrent }) => {
                    l.forward(n, conv, recurrent, work, plan, self.avx2, saves)
                }
                (Mixer::Attention(l), LayerState::Attention { keys, values }) => {
                    l.forward(
                        n,
                        state.position,
                        keys,
                        values,
                        &self.inv_freq,
                        work,
                        plan,
                        self.avx2,
                    );
                    for (at, saved) in saves.into_iter().flatten() {
                        let LayerState::Attention { keys, values } = saved else {
                            unreachable!()
                        };
                        keys.reserve_exact(at * KV);
                        values.reserve_exact(at * KV);
                        keys.extend_from_slice(&work.key[..at * KV]);
                        values.extend_from_slice(&work.value[..at * KV]);
                    }
                }
                _ => anyhow::bail!("cache layer kind mismatch"),
            }
            cpu::add(&mut work.hidden, &work.projected);
            cpu::rms_from(&work.hidden, &mut work.norm, &layer.post_norm, 1e-6, true);
            layer
                .gate
                .apply(&work.norm, &mut work.gate, plan.projection_rows);
            layer
                .up
                .apply(&work.norm, &mut work.up, plan.projection_rows);
            cpu::silu_mul(&mut work.gate, &work.up, self.avx2);
            layer
                .down
                .apply(&work.gate, &mut work.projected, plan.projection_rows);
            cpu::add(&mut work.hidden, &work.projected);
        }
        cpu::rms(&mut work.hidden, &self.norm, 1e-6, true);
        state.position += n;
        ensure!(
            work.hidden.iter().all(|v| v.is_finite()),
            "non-finite decoder hidden state"
        );
        for (saved, position) in saved.iter_mut().zip(positions) {
            if let Some(saved) = saved {
                saved.position = position.unwrap();
            }
        }
        Ok(saved)
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
        saves: [Option<(usize, &mut LayerState)>; 2],
    ) {
        self.qkv.apply(&w.norm, &mut w.qkv, p.projection_rows);
        self.z.apply(&w.norm, &mut w.z, p.projection_rows);
        self.a.apply(&w.norm, &mut w.a, p.projection_rows);
        self.b.apply(&w.norm, &mut w.b, p.projection_rows);
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
        let mut saves = saves.map(|s| {
            s.map(|(at, layer)| {
                let LayerState::Delta { conv, recurrent } = layer else {
                    unreachable!()
                };
                if at >= 3 {
                    conv.copy_from_slice(&w.qkv[(at - 3) * CONV..at * CONV]);
                } else {
                    conv.copy_within(at * CONV.., 0);
                    conv[(3 - at) * CONV..].copy_from_slice(&w.qkv[..at * CONV]);
                }
                (at, recurrent.chunks_mut(LD * LD))
            })
        });
        saves.sort_by_key(|s| s.as_ref().map_or(usize::MAX, |(at, _)| *at));
        if n >= 3 {
            history.copy_from_slice(&w.qkv[(n - 3) * CONV..]);
        } else {
            history.copy_within(n * CONV.., 0);
            history[(3 - n) * CONV..].copy_from_slice(&w.qkv);
        }
        w.heads.resize(LH * n * LD, 0.);
        let mut states = state.chunks_mut(LD * LD);
        let mut outputs = w.heads.chunks_mut(n * LD);
        let tasks: [_; LH] = std::array::from_fn(|h| {
            let [first, second] = &mut saves;
            (
                h,
                states.next().unwrap(),
                outputs.next().unwrap(),
                [
                    first.as_mut().map(|(at, rows)| (*at, rows.next().unwrap())),
                    second
                        .as_mut()
                        .map(|(at, rows)| (*at, rows.next().unwrap())),
                ],
            )
        });
        let run = |(h, s, out, saves): (
            usize,
            &mut [f32],
            &mut [f32],
            [Option<(usize, &mut [f32])>; 2],
        )| {
            let mut start = 0;
            for save in saves.into_iter().chain(std::iter::once(None)) {
                let end = save.as_ref().map_or(n, |(at, _)| *at);
                if end > start {
                    let qkv = &w.conv[start * CONV..];
                    let a = &w.a[start * LH..];
                    let b = &w.b[start * LH..];
                    let dst = &mut out[start * LD..end * LD];
                    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
                    if avx2 {
                        unsafe {
                            delta_avx2(end - start, h, qkv, a, b, self.decay[h], self.dt[h], s, dst)
                        };
                    } else {
                        delta_scan(end - start, h, qkv, a, b, self.decay[h], self.dt[h], s, dst);
                    }
                    #[cfg(not(all(feature = "cpu", target_arch = "x86_64")))]
                    {
                        let _ = avx2;
                        delta_scan(end - start, h, qkv, a, b, self.decay[h], self.dt[h], s, dst);
                    }
                }
                if let Some((_, snapshot)) = save {
                    snapshot.copy_from_slice(s);
                }
                start = end;
            }
        };
        #[cfg(feature = "cpu")]
        tasks.into_par_iter().for_each(run);
        #[cfg(not(feature = "cpu"))]
        tasks.into_iter().for_each(run);
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
        self.out
            .apply(&w.context, &mut w.projected, p.projection_rows);
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
        q.copy_from_slice(&qkv[t * CONV + h * LD..t * CONV + (h + 1) * LD]);
        k.copy_from_slice(&qkv[t * CONV + LV + h * LD..t * CONV + LV + (h + 1) * LD]);
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
            delta[j] = (qkv[t * CONV + 2 * LV + h * LD + j] - memory[j]) * beta;
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
        q.copy_from_slice(&qkv[t * CONV + h * LD..t * CONV + (h + 1) * LD]);
        k.copy_from_slice(&qkv[t * CONV + LV + h * LD..t * CONV + LV + (h + 1) * LD]);
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
                let v = qkv.as_ptr().add(t * CONV + 2 * LV + h * LD + column);
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
        offset: usize,
        keys: &mut Vec<f32>,
        values: &mut Vec<f32>,
        freq: &[f32],
        w: &mut Work,
        p: Plan,
        avx2: bool,
    ) {
        self.q.apply(&w.norm, &mut w.qkv, p.projection_rows);
        self.k.apply(&w.norm, &mut w.key, p.projection_rows);
        self.v.apply(&w.norm, &mut w.value, p.projection_rows);
        w.query.resize(n * AH * AD, 0.);
        w.z.resize(n * AH * AD, 0.);
        for i in 0..n * AH {
            w.query[i * AD..(i + 1) * AD].copy_from_slice(&w.qkv[i * 2 * AD..i * 2 * AD + AD]);
            w.z[i * AD..(i + 1) * AD].copy_from_slice(&w.qkv[i * 2 * AD + AD..(i + 1) * 2 * AD]);
        }
        cpu::rms(&mut w.query, &self.qnorm, 1e-6, true);
        cpu::rms(&mut w.key, &self.knorm, 1e-6, true);
        rope(&mut w.query, AH, offset, freq);
        rope(&mut w.key, 2, offset, freq);
        keys.extend_from_slice(&w.key);
        values.extend_from_slice(&w.value);
        let nk = offset + n;
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
                    let end = offset + start + r + 1;
                    cpu::softmax(&mut row[..end]);
                    row[end..].fill(0.);
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
        self.out
            .apply(&w.context, &mut w.projected, p.projection_rows);
    }
}

#[cfg(test)]
pub(super) fn assert_same_state(a: &State, b: &State) {
    assert_eq!(a.position, b.position);
    for (index, (a, b)) in a.layers.iter().zip(&b.layers).enumerate() {
        let pairs = match (a, b) {
            (
                LayerState::Delta {
                    conv: a,
                    recurrent: ar,
                },
                LayerState::Delta {
                    conv: b,
                    recurrent: br,
                },
            ) => [(a, b), (ar, br)],
            (
                LayerState::Attention {
                    keys: a,
                    values: av,
                },
                LayerState::Attention {
                    keys: b,
                    values: bv,
                },
            ) => [(a, b), (av, bv)],
            _ => panic!("state kind"),
        };
        for (a, b) in pairs {
            assert_eq!(a.len(), b.len());
            let error = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(error <= 1e-4, "layer {index} snapshot error {error}");
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod snapshot_tests {
    use super::*;
    #[test]
    fn delta_blocks_preserve_state_and_output_bits() {
        #[cfg(target_arch = "x86_64")]
        if cpu::avx2_available() {
            let value = |i: usize| (((i as u64 * 48271 + 731) % 65521) as f32 - 32760.) / 32768.;
            for n in [1, 2, 3, 17, 65, 256] {
                let qkv: Vec<_> = (0..n * CONV).map(value).collect();
                let a: Vec<_> = (0..n * LH).map(|i| value(i + 19) * 32.).collect();
                let b: Vec<_> = (0..n * LH).map(|i| value(i + 73)).collect();
                for h in [0, 7, 15] {
                    let mut expected: Vec<_> = (0..LD * LD).map(|i| value(i + h) * 0.1).collect();
                    let mut actual = expected.clone();
                    let mut reference = vec![0.; n * LD];
                    let mut output = reference.clone();
                    delta_scan(n, h, &qkv, &a, &b, 0.3, -0.4, &mut expected, &mut reference);
                    unsafe {
                        delta_avx2(n, h, &qkv, &a, &b, 0.3, -0.4, &mut actual, &mut output);
                    }
                    assert!(
                        actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "state n={n}, h={h}"
                    );
                    assert!(
                        output
                            .iter()
                            .zip(&reference)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "output n={n}, h={h}"
                    );
                }
            }
        }
    }
}
