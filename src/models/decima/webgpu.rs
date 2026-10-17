mod device;
mod readout;
mod shaders;
use super::{D, Kind, Model as CpuModel, Precision, Question, cpu, input};
use crate::length::{Band, Rules};
use anyhow::{Result, ensure};
use device::Device;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::VecDeque, sync::Arc};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub matrix_tile: usize,
    pub attention_query_tile: usize,
}
impl Plan {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.matrix_tile, 8 | 16),
            "Decima GPU matrix tile must be 8 or 16"
        );
        ensure!(
            matches!(self.attention_query_tile, 8 | 16 | 32 | 64),
            "Decima GPU attention tile must be 8, 16, 32 or 64"
        );
        Ok(())
    }
}
pub fn default_rules() -> Rules<Plan> {
    Rules::new(
        vec![Band {
            through: 64,
            settings: Plan {
                matrix_tile: 16,
                attention_query_tile: 16,
            },
        }],
        Plan {
            matrix_tile: 16,
            attention_query_tile: 32,
        },
    )
    .unwrap()
}
struct Linear {
    input: usize,
    output: usize,
    weight: wgpu::Buffer,
    bias: wgpu::Buffer,
}
struct Norm {
    width: usize,
    eps: f32,
    weight: wgpu::Buffer,
    bias: wgpu::Buffer,
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}
struct EncoderLayer {
    attention: Attention,
    n1: Norm,
    n2: Norm,
    ff1: Linear,
    ff2: Linear,
}
struct ScorerLayer {
    self_attn: Attention,
    cross_attn: Attention,
    n1: Norm,
    n2: Norm,
    n3: Norm,
    ff1: Linear,
    ff2: Linear,
}
struct Pipelines {
    gemm8: wgpu::ComputePipeline,
    gemm16: wgpu::ComputePipeline,
    norm: wgpu::ComputePipeline,
    pointwise: wgpu::ComputePipeline,
    scores: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    values: wgpu::ComputePipeline,
}
pub struct Model {
    device: Device,
    config: super::Config,
    input: input::Input,
    embeddings: super::embedding::Embeddings,
    positions: Vec<f32>,
    types: Vec<f32>,
    readout: readout::Readout,
    encoder: Vec<EncoderLayer>,
    scorer: Vec<ScorerLayer>,
    embedding_norm: Norm,
    out_norm: Norm,
    pipelines: Pipelines,
    dummy: wgpu::Buffer,
    rules: Rules<Plan>,
    identity: Arc<()>,
}
pub struct Session {
    identity: Option<Arc<()>>,
    cache: VecDeque<(Vec<u32>, wgpu::Buffer)>,
    bytes: usize,
    limit: usize,
    pub choice_cache_hits: u64,
}
impl Default for Session {
    fn default() -> Self {
        Self {
            identity: None,
            cache: VecDeque::new(),
            bytes: 0,
            limit: 32 * 1024 * 1024,
            choice_cache_hits: 0,
        }
    }
}
impl Session {
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.bytes = 0;
    }
    pub fn set_cache_bytes(&mut self, bytes: usize) {
        self.limit = bytes;
        self.clear_cache();
    }
    pub fn cache_bytes(&self) -> usize {
        self.bytes
    }
}
struct Buffers {
    hidden: wgpu::Buffer,
    norm: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    context: wgpu::Buffer,
    update: wgpu::Buffer,
    ff: wgpu::Buffer,
    activated: wgpu::Buffer,
    scores: wgpu::Buffer,
    probabilities: wgpu::Buffer,
}
fn parameters(values: &[u32]) -> [u32; 16] {
    let mut p = [0; 16];
    p[..values.len()].copy_from_slice(values);
    p
}
impl Model {
    pub async fn from_cpu(source: CpuModel) -> Result<Self> {
        ensure!(
            source.precision() == Precision::F32,
            "WebGPU currently requires float Decima projections"
        );
        let device = Device::new().await?;
        let scope = device
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 4 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 3,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Decima compute bindings"),
                entries: &entries,
            });
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Decima compute layout"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
        let pipeline = |name: &str, source: String| {
            let module = device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let pipelines = Pipelines {
            gemm8: pipeline("Decima GEMM 8", shaders::gemm(8)),
            gemm16: pipeline("Decima GEMM 16", shaders::gemm(16)),
            norm: pipeline("Decima norm", shaders::source(shaders::NORM)),
            pointwise: pipeline("Decima pointwise", shaders::source(shaders::POINTWISE)),
            scores: pipeline("Decima attention scores", shaders::source(shaders::SCORES)),
            softmax: pipeline("Decima softmax", shaders::source(shaders::SOFTMAX)),
            values: pipeline("Decima attention values", shaders::source(shaders::VALUES)),
        };
        if let Some(error) = scope.pop().await {
            anyhow::bail!("Decima GPU pipeline: {error}");
        }
        let embedding_norm = Norm::upload(&device, &source.embedding_norm)?;
        let out_norm = Norm::upload(&device, &source.out_norm)?;
        let encoder = source
            .encoder
            .into_iter()
            .map(|l| EncoderLayer::upload(&device, l))
            .collect::<Result<_>>()?;
        let scorer = source
            .scorer
            .into_iter()
            .map(|l| ScorerLayer::upload(&device, l))
            .collect::<Result<_>>()?;
        let dummy = device.buffer(&[0.], "Decima unused input")?;
        Ok(Self {
            device,
            config: source.config,
            input: source.input,
            embeddings: source.embeddings,
            positions: source.positions,
            types: source.types,
            readout: readout::Readout::new(
                source.score,
                source.ord_g,
                source.ord_gap,
                source.sim_scale,
                source.sim_center,
            )?,
            encoder,
            scorer,
            embedding_norm,
            out_norm,
            pipelines,
            dummy,
            rules: default_rules(),
            identity: Arc::new(()),
        })
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.device.info
    }
    pub fn set_length_rules(&mut self, rules: Rules<Plan>) -> Result<()> {
        rules.validate(Plan::validate)?;
        self.rules = rules;
        self.identity = Arc::new(());
        Ok(())
    }
    fn run(
        &self,
        e: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        b: &wgpu::Buffer,
        y: &wgpu::Buffer,
        p: &[u32],
        groups: [u32; 3],
    ) -> Result<()> {
        self.device
            .dispatch(e, pipeline, &[x, w, b, y], &parameters(p), groups)
    }
    fn linear(
        &self,
        e: &mut wgpu::CommandEncoder,
        l: &Linear,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
        plan: Plan,
    ) -> Result<()> {
        let t = plan.matrix_tile;
        self.run(
            e,
            if t == 8 {
                &self.pipelines.gemm8
            } else {
                &self.pipelines.gemm16
            },
            x,
            &l.weight,
            &l.bias,
            y,
            &[rows as u32, l.output as u32, l.input as u32],
            [l.output.div_ceil(t) as u32, rows.div_ceil(t) as u32, 1],
        )
    }
    fn norm(
        &self,
        e: &mut wgpu::CommandEncoder,
        l: &Norm,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
    ) -> Result<()> {
        self.run(
            e,
            &self.pipelines.norm,
            x,
            &l.weight,
            &l.bias,
            y,
            &[rows as u32, l.width as u32, l.eps.to_bits()],
            [rows.div_ceil(64) as u32, 1, 1],
        )
    }
    fn add(
        &self,
        e: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n: usize,
    ) -> Result<()> {
        self.run(
            e,
            &self.pipelines.pointwise,
            x,
            w,
            &self.dummy,
            y,
            &[(n * D) as u32, 0],
            [(n * D).div_ceil(64) as u32, 1, 1],
        )
    }
    fn gelu(
        &self,
        e: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        elements: usize,
    ) -> Result<()> {
        self.run(
            e,
            &self.pipelines.pointwise,
            x,
            &self.dummy,
            &self.dummy,
            y,
            &[elements as u32, 4],
            [elements.div_ceil(64) as u32, 1, 1],
        )
    }
    fn mean(
        &self,
        e: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
        offset: usize,
    ) -> Result<()> {
        self.run(
            e,
            &self.pipelines.pointwise,
            x,
            &self.dummy,
            &self.dummy,
            y,
            &[D as u32, 5, D as u32, rows as u32, offset as u32],
            [D.div_ceil(64) as u32, 1, 1],
        )
    }
    fn attention(
        &self,
        e: &mut wgpu::CommandEncoder,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        s: &Buffers,
        nq: usize,
        nk: usize,
        heads: usize,
        plan: Plan,
    ) -> Result<()> {
        for first in (0..nq).step_by(plan.attention_query_tile) {
            let rows = plan.attention_query_tile.min(nq - first);
            let p = [
                nk as u32,
                heads as u32,
                (D / heads) as u32,
                rows as u32,
                first as u32,
            ];
            self.run(
                e,
                &self.pipelines.scores,
                q,
                k,
                &self.dummy,
                &s.scores,
                &p,
                [(rows * heads * nk).div_ceil(64) as u32, 1, 1],
            )?;
            self.run(
                e,
                &self.pipelines.softmax,
                &s.scores,
                &self.dummy,
                &self.dummy,
                &s.probabilities,
                &[(rows * heads) as u32, nk as u32],
                [(rows * heads).div_ceil(64) as u32, 1, 1],
            )?;
            self.run(
                e,
                &self.pipelines.values,
                &s.probabilities,
                v,
                &self.dummy,
                &s.context,
                &p,
                [(rows * D).div_ceil(64) as u32, 1, 1],
            )?;
        }
        Ok(())
    }
    fn encode(&self, e: &mut wgpu::CommandEncoder, ids: &[u32]) -> Result<wgpu::Buffer> {
        let n = ids.len();
        ensure!(
            n > 0 && n <= 512 && ids.iter().all(|&id| (id as usize) < 250037),
            "invalid Decima tokens"
        );
        let plan = *self.rules.select(n);
        let mut embedding = vec![0.; n * D];
        for (p, &id) in ids.iter().enumerate() {
            let row = &mut embedding[p * D..(p + 1) * D];
            self.embeddings.row(id as usize, row);
            for i in 0..D {
                row[i] = (row[i] + self.types[i]) + self.positions[p * D + i];
            }
        }
        let source = self.device.buffer(&embedding, "Decima embedded input")?;
        let s = Buffers::new(&self.device, n, n, plan.attention_query_tile)?;
        self.norm(e, &self.embedding_norm, &source, &s.hidden, n)?;
        for l in &self.encoder {
            self.linear(e, &l.attention.q, &s.hidden, &s.q, n, plan)?;
            self.linear(e, &l.attention.k, &s.hidden, &s.k, n, plan)?;
            self.linear(e, &l.attention.v, &s.hidden, &s.v, n, plan)?;
            self.attention(e, &s.q, &s.k, &s.v, &s, n, n, 12, plan)?;
            self.linear(e, &l.attention.o, &s.context, &s.update, n, plan)?;
            self.add(e, &s.hidden, &s.update, &s.norm, n)?;
            self.norm(e, &l.n1, &s.norm, &s.hidden, n)?;
            self.linear(e, &l.ff1, &s.hidden, &s.ff, n, plan)?;
            self.gelu(e, &s.ff, &s.activated, n * 1536)?;
            self.linear(e, &l.ff2, &s.activated, &s.update, n, plan)?;
            self.add(e, &s.hidden, &s.update, &s.norm, n)?;
            self.norm(e, &l.n2, &s.norm, &s.hidden, n)?;
        }
        Ok(s.hidden)
    }
    pub async fn decide(
        &self,
        state: &str,
        q: &Question,
        session: &mut Session,
    ) -> Result<Vec<f32>> {
        let result = self.decide_inner(state, q, session).await;
        if result.is_err() {
            session.clear_cache();
        }
        result
    }
    async fn decide_inner(
        &self,
        state: &str,
        q: &Question,
        session: &mut Session,
    ) -> Result<Vec<f32>> {
        ensure!((2..=255).contains(&q.choices.len()), "need 2–255 choices");
        if q.kind == Kind::Score {
            ensure!(q.choices.len() <= 10, "score allows at most 10 levels");
        }
        if !session
            .identity
            .as_ref()
            .is_some_and(|id| Arc::ptr_eq(id, &self.identity))
        {
            session.clear_cache();
            session.identity = Some(self.identity.clone());
        }
        let c = &self.config;
        let body = if c.question_in_state {
            format!("{}\n{state}", q.text)
        } else {
            state.into()
        };
        let text = format!("{}{}", c.state_prefix, input::normalize(&body, &q.lang));
        let ids = self.input.ids(&text)?;
        ensure!(
            ids.len() <= self.input.max_state,
            "STATE_TRUNCATED: state + question is {} tokens; maximum {}",
            ids.len(),
            self.input.max_state
        );
        let nk = ids.len();
        let state_plan = *self.rules.select(nk);
        let mut e = self.device.encoder();
        let state_h = self.encode(&mut e, &ids)?;
        let rows = self
            .input
            .choices(q, &c.choice_prefix, c.question_in_choices, c.choice_fit)?;
        let readout = self
            .device
            .empty((1 + 2 * rows.len()) * D, "Decima readout vectors")?;
        self.mean(&mut e, &state_h, &readout, nk, 0)?;
        let mut memory = Vec::new();
        for l in &self.scorer {
            let k = self.device.empty(nk * D, "Decima state keys")?;
            let v = self.device.empty(nk * D, "Decima state values")?;
            self.linear(&mut e, &l.cross_attn.k, &state_h, &k, nk, state_plan)?;
            self.linear(&mut e, &l.cross_attn.v, &state_h, &v, nk, state_plan)?;
            memory.push((k, v));
        }
        for (index, ids) in rows.into_iter().enumerate() {
            let n = ids.len();
            let plan = *self.rules.select(n);
            let choice = if let Some(i) = session.cache.iter().position(|(key, _)| key == &ids) {
                let entry = session.cache.remove(i).unwrap();
                let buffer = entry.1.clone();
                session.cache.push_back(entry);
                session.choice_cache_hits += 1;
                buffer
            } else {
                let buffer = self.encode(&mut e, &ids)?;
                let bytes = n * (D * 4 + 4);
                if bytes <= session.limit {
                    while session.bytes + bytes > session.limit {
                        if let Some((key, _)) = session.cache.pop_front() {
                            session.bytes -= key.len() * (D * 4 + 4);
                        }
                    }
                    session.bytes += bytes;
                    session.cache.push_back((ids, buffer.clone()));
                }
                buffer
            };
            self.mean(&mut e, &choice, &readout, n, (1 + 2 * index) * D)?;
            let mut s = Buffers::new(
                &self.device,
                n,
                n.max(nk),
                plan.attention_query_tile
                    .max(state_plan.attention_query_tile),
            )?;
            // Scorer mutations must not alias the cached choice encoding.
            e.copy_buffer_to_buffer(&choice, 0, &s.hidden, 0, (n * D * 4) as u64);
            for (l, (mk, mv)) in self.scorer.iter().zip(&memory) {
                self.norm(&mut e, &l.n1, &s.hidden, &s.norm, n)?;
                self.linear(&mut e, &l.self_attn.q, &s.norm, &s.q, n, plan)?;
                self.linear(&mut e, &l.self_attn.k, &s.norm, &s.k, n, plan)?;
                self.linear(&mut e, &l.self_attn.v, &s.norm, &s.v, n, plan)?;
                self.attention(&mut e, &s.q, &s.k, &s.v, &s, n, n, 6, plan)?;
                self.linear(&mut e, &l.self_attn.o, &s.context, &s.update, n, plan)?;
                self.add(&mut e, &s.hidden, &s.update, &s.norm, n)?;
                std::mem::swap(&mut s.hidden, &mut s.norm);
                self.norm(&mut e, &l.n2, &s.hidden, &s.norm, n)?;
                self.linear(&mut e, &l.cross_attn.q, &s.norm, &s.q, n, plan)?;
                self.attention(&mut e, &s.q, mk, mv, &s, n, nk, 6, state_plan)?;
                self.linear(&mut e, &l.cross_attn.o, &s.context, &s.update, n, plan)?;
                self.add(&mut e, &s.hidden, &s.update, &s.norm, n)?;
                std::mem::swap(&mut s.hidden, &mut s.norm);
                self.norm(&mut e, &l.n3, &s.hidden, &s.norm, n)?;
                self.linear(&mut e, &l.ff1, &s.norm, &s.ff, n, plan)?;
                self.gelu(&mut e, &s.ff, &s.activated, n * 1536)?;
                self.linear(&mut e, &l.ff2, &s.activated, &s.update, n, plan)?;
                self.add(&mut e, &s.hidden, &s.update, &s.norm, n)?;
                std::mem::swap(&mut s.hidden, &mut s.norm);
            }
            self.norm(&mut e, &self.out_norm, &s.hidden, &s.norm, n)?;
            self.mean(&mut e, &s.norm, &readout, n, (2 + 2 * index) * D)?;
        }
        let result = self
            .device
            .read(e, &readout, (1 + 2 * q.choices.len()) * D)
            .await?;
        self.readout.probabilities(&result, q.kind, c.temperature)
    }
}
impl Linear {
    fn upload(device: &Device, l: cpu::Linear) -> Result<Self> {
        let (input, output, weight, bias) = l.into_float_parts()?;
        Ok(Self {
            input,
            output,
            weight: device.buffer(&weight, "Decima projection")?,
            bias: device.buffer(&bias, "Decima bias")?,
        })
    }
}
impl Norm {
    fn upload(device: &Device, l: &cpu::Norm) -> Result<Self> {
        Ok(Self {
            width: l.weight.len(),
            eps: l.eps,
            weight: device.buffer(&l.weight, "Decima norm weight")?,
            bias: device.buffer(&l.bias, "Decima norm bias")?,
        })
    }
}
impl Attention {
    fn upload(device: &Device, l: super::Attention) -> Result<Self> {
        Ok(Self {
            q: Linear::upload(device, l.q)?,
            k: Linear::upload(device, l.k)?,
            v: Linear::upload(device, l.v)?,
            o: Linear::upload(device, l.o)?,
        })
    }
}
impl EncoderLayer {
    fn upload(device: &Device, l: super::EncoderLayer) -> Result<Self> {
        Ok(Self {
            attention: Attention {
                q: Linear::upload(device, l.q)?,
                k: Linear::upload(device, l.k)?,
                v: Linear::upload(device, l.v)?,
                o: Linear::upload(device, l.o)?,
            },
            n1: Norm::upload(device, &l.n1)?,
            n2: Norm::upload(device, &l.n2)?,
            ff1: Linear::upload(device, l.ff1)?,
            ff2: Linear::upload(device, l.ff2)?,
        })
    }
}
impl ScorerLayer {
    fn upload(device: &Device, l: super::ScorerLayer) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::upload(device, l.self_attn)?,
            cross_attn: Attention::upload(device, l.cross_attn)?,
            n1: Norm::upload(device, &l.n1)?,
            n2: Norm::upload(device, &l.n2)?,
            n3: Norm::upload(device, &l.n3)?,
            ff1: Linear::upload(device, l.ff1)?,
            ff2: Linear::upload(device, l.ff2)?,
        })
    }
}
impl Buffers {
    fn new(device: &Device, n: usize, nk: usize, tile: usize) -> Result<Self> {
        Ok(Self {
            hidden: device.empty(n * D, "Decima hidden")?,
            norm: device.empty(n * D, "Decima norm")?,
            q: device.empty(n * D, "Decima q")?,
            k: device.empty(n * D, "Decima k")?,
            v: device.empty(n * D, "Decima v")?,
            context: device.empty(n * D, "Decima context")?,
            update: device.empty(n * D, "Decima update")?,
            ff: device.empty(n * 1536, "Decima ff")?,
            activated: device.empty(n * 1536, "Decima activated")?,
            scores: device.empty(tile.min(n) * nk * 12, "Decima scores")?,
            probabilities: device.empty(tile.min(n) * nk * 12, "Decima probabilities")?,
        })
    }
}

impl Model {
    pub async fn predict_json(&self, request: &Value, s: &mut Session) -> Result<Value> {
        if let Some(rows) = request.as_array() {
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                out.push(Box::pin(self.predict_json(row, s)).await?);
            }
            return Ok(Value::Array(out));
        }
        ensure!(request.is_object(), "request must be an object");
        ensure!(
            request["long"].is_null(),
            "Decima long-document chunking is not implemented; maximum state + question is 512 tokens"
        );
        let state = &request["state"];
        ensure!(
            state.is_string() || state.is_array() || state.is_object(),
            "state must be a string, object or array"
        );
        let state = input::text(state);
        let lang = request.get("lang").map_or(Ok("en"), |v| {
            v.as_str()
                .ok_or_else(|| anyhow::anyhow!("lang must be a string"))
        })?;
        let questions = request["questions"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("questions must be an object"))?;
        ensure!((1..=256).contains(&questions.len()), "need 1–256 questions");
        let threshold = request
            .get("min_confidence")
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| anyhow::anyhow!("min_confidence must be numeric"))
            })
            .transpose()?;
        ensure!(
            threshold.is_none_or(|v| (0.0..=1.0).contains(&v)),
            "min_confidence must be in [0,1]"
        );
        let mut answers = serde_json::Map::new();
        let parsed = questions
            .iter()
            .map(|(id, spec)| Ok((id, input::parse_question(id, spec, lang)?)))
            .collect::<Result<Vec<_>>>()?;
        let r = |v: f64| (v * 10000.).round_ties_even() / 10000.;
        for (id, (q, keys)) in parsed {
            let p = self.decide(&state, &q, s).await?;
            let best = p
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                .unwrap()
                .0;
            let confidence =
                ((p.len() as f64 * p[best] as f64 - 1.) / (p.len() - 1) as f64).clamp(0., 1.);
            let probabilities: serde_json::Map<String, Value> = p
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        if q.kind == Kind::Choice {
                            keys[i].clone()
                        } else {
                            i.to_string()
                        },
                        json!(r(*v as f64)),
                    )
                })
                .collect();
            let mut answer = match q.kind {
                Kind::Choice => {
                    json!({"type":"choice","choice":keys[best],"confidence":r(confidence),"probabilities":probabilities})
                }
                Kind::Score => {
                    json!({"type":"score","score":r(p.iter().enumerate().map(|(i,v)|i as f64* *v as f64).sum()),"confidence":r(confidence),"legend":keys.iter().enumerate().map(|(i,v)|(i.to_string(),json!(v))).collect::<serde_json::Map<_,_>>(),"probabilities":probabilities})
                }
                _ => json!({"type":"noul","noul":r(p[0] as f64)}),
            };
            if let Some(t) = threshold {
                answer["abstain"] = json!(confidence < t);
            }
            answers.insert(id.clone(), answer);
        }
        Ok(
            json!({"model":"decima-small","answers":answers,"usage":{"input_tokens":self.input.ids(&state)?.len(),"output_tokens":0}}),
        )
    }
}

#[cfg(feature = "wasm")]
mod browser {
    use super::*;
    use wasm_bindgen::prelude::*;
    fn error(e: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&e.to_string())
    }
    #[wasm_bindgen(js_name=DecimaWebGpuModel)]
    pub struct BrowserModel {
        model: Model,
        session: Session,
    }
    #[wasm_bindgen(js_class=DecimaWebGpuModel)]
    impl BrowserModel {
        pub async fn create(
            config: Vec<u8>,
            encoder_config: Vec<u8>,
            tokenizer: Vec<u8>,
            encoder_weights: Vec<u8>,
            head_weights: Vec<u8>,
        ) -> Result<BrowserModel, JsValue> {
            let model = CpuModel::from_bytes(
                &config,
                &encoder_config,
                &tokenizer,
                &encoder_weights,
                &head_weights,
                super::super::Options {
                    precision: Precision::F32,
                    kernel: super::super::Kernel::Portable,
                },
            )
            .map_err(error)?;
            drop((
                config,
                encoder_config,
                tokenizer,
                encoder_weights,
                head_weights,
            ));
            Ok(Self {
                model: Model::from_cpu(model).await.map_err(error)?,
                session: Session::default(),
            })
        }
        pub async fn predict(&mut self, json: &str) -> Result<String, JsValue> {
            let request = serde_json::from_str(json).map_err(error)?;
            serde_json::to_string(
                &self
                    .model
                    .predict_json(&request, &mut self.session)
                    .await
                    .map_err(error)?,
            )
            .map_err(error)
        }
        pub async fn decide(
            &mut self,
            state: &str,
            question_json: &str,
        ) -> Result<String, JsValue> {
            let question = serde_json::from_str(question_json).map_err(error)?;
            serde_json::to_string(
                &self
                    .model
                    .decide(state, &question, &mut self.session)
                    .await
                    .map_err(error)?,
            )
            .map_err(error)
        }
        #[wasm_bindgen(js_name=setLengthRules)]
        pub fn set_length_rules(&mut self, json: &str) -> Result<(), JsValue> {
            self.model
                .set_length_rules(serde_json::from_str(json).map_err(error)?)
                .map_err(error)
        }
        #[wasm_bindgen(js_name=clearCache)]
        pub fn clear_cache(&mut self) {
            self.session.clear_cache();
        }
        #[wasm_bindgen(js_name=setCacheBytes)]
        pub fn set_cache_bytes(&mut self, bytes: usize) {
            self.session.set_cache_bytes(bytes);
        }
        #[wasm_bindgen(js_name=adapterInfo)]
        pub fn adapter_info(&self) -> String {
            let i = self.model.adapter_info();
            json!({"name":i.name,"backend":format!("{:?}",i.backend),"device_type":format!("{:?}",i.device_type)}).to_string()
        }
    }
}
