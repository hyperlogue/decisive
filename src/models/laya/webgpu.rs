use crate::length::{Band, Rules};
mod device;
mod shaders;
use super::{Layer, Model as CpuModel, Scratch};
use crate::laya::{cpu, decision, input::Encoded};
use anyhow::{Context, Result, ensure};
use device::Device;
use serde::{Deserialize, Serialize};

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
            "WebGPU matrix tile must be 8 or 16"
        );
        ensure!(
            matches!(self.attention_query_tile, 8 | 16 | 32 | 64),
            "WebGPU attention tile must be 8, 16, 32 or 64"
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
struct GpuLayer {
    norm1: Option<Norm>,
    qkv: Linear,
    attention_out: Linear,
    norm2: Norm,
    ff_in: Linear,
    ff_out: Linear,
    window: Option<usize>,
    rope: Option<f32>,
    gated: bool,
}
struct Pipelines {
    gemm8: wgpu::ComputePipeline,
    gemm16: wgpu::ComputePipeline,
    norm: wgpu::ComputePipeline,
    pointwise: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    scores: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    values: wgpu::ComputePipeline,
}
pub struct Model {
    device: Device,
    contract: CpuModel,
    encoder: Vec<GpuLayer>,
    head: Vec<GpuLayer>,
    embedding_norm: Norm,
    final_norm: Norm,
    types: wgpu::Buffer,
    pipelines: Pipelines,
    dummy: wgpu::Buffer,
    rules: Rules<Plan>,
}
struct Buffers {
    hidden: wgpu::Buffer,
    norm: wgpu::Buffer,
    qkv: wgpu::Buffer,
    rotated: wgpu::Buffer,
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
    pub async fn from_cpu(mut contract: CpuModel) -> Result<Self> {
        ensure!(
            contract
                .encoder
                .iter()
                .all(|l| !l.qkv.weight.is_quantized()),
            "WebGPU currently requires float projections; load a float checkpoint"
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
                label: Some("Laya compute bindings"),
                entries: &entries,
            });
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Laya compute layout"),
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
            gemm8: pipeline("Laya GEMM 8", shaders::gemm(8)),
            gemm16: pipeline("Laya GEMM 16", shaders::gemm(16)),
            norm: pipeline("Laya norm", shaders::source(shaders::NORM)),
            pointwise: pipeline("Laya pointwise", shaders::source(shaders::POINTWISE)),
            rope: pipeline("Laya RoPE", shaders::source(shaders::ROPE)),
            scores: pipeline("Laya attention scores", shaders::source(shaders::SCORES)),
            softmax: pipeline("Laya softmax", shaders::source(shaders::SOFTMAX)),
            values: pipeline("Laya attention values", shaders::source(shaders::VALUES)),
        };
        if let Some(error) = scope.pop().await {
            anyhow::bail!("Laya GPU pipeline: {error}");
        }
        let embedding_norm = Norm::upload(&device, &contract.embedding_norm)?;
        let final_norm = Norm::upload(&device, &contract.final_norm)?;
        let types = device.buffer(&contract.type_embedding, "Laya question types")?;
        let encoder = std::mem::take(&mut contract.encoder)
            .into_iter()
            .map(|l| GpuLayer::upload(&device, l))
            .collect::<Result<_>>()?;
        let head = std::mem::take(&mut contract.head)
            .into_iter()
            .map(|l| GpuLayer::upload(&device, l))
            .collect::<Result<_>>()?;
        let dummy = device.buffer(&[0.], "Laya unused input")?;
        Ok(Self {
            device,
            contract,
            encoder,
            head,
            embedding_norm,
            final_norm,
            types,
            pipelines,
            dummy,
            rules: default_rules(),
        })
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.device.info
    }
    pub fn set_length_rules(&mut self, rules: Rules<Plan>) -> Result<()> {
        rules.validate(Plan::validate)?;
        self.rules = rules;
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
        let mut groups = groups;
        let mut params = parameters(p);
        let limit = self
            .device
            .device
            .limits()
            .max_compute_workgroups_per_dimension;
        if groups[0] > limit && groups[1] == 1 {
            groups[1] = groups[0].div_ceil(limit);
            groups[0] = limit;
            params[15] = limit * 64;
        }
        self.device
            .dispatch(e, pipeline, &[x, w, b, y], &params, groups)
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
        let tile = plan.matrix_tile;
        let pipeline = if tile == 8 {
            &self.pipelines.gemm8
        } else {
            &self.pipelines.gemm16
        };
        self.run(
            e,
            pipeline,
            x,
            &l.weight,
            &l.bias,
            y,
            &[rows as u32, l.output as u32, l.input as u32],
            [
                l.output.div_ceil(tile) as u32,
                rows.div_ceil(tile) as u32,
                1,
            ],
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
    fn pointwise(
        &self,
        e: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        y: &wgpu::Buffer,
        size: usize,
        op: u32,
        width: usize,
        extra: u32,
    ) -> Result<()> {
        self.run(
            e,
            &self.pipelines.pointwise,
            x,
            w,
            &self.dummy,
            y,
            &[size as u32, op, width as u32, extra],
            [size.div_ceil(64) as u32, 1, 1],
        )
    }
    fn layer(
        &self,
        e: &mut wgpu::CommandEncoder,
        l: &GpuLayer,
        s: &mut Buffers,
        n: usize,
        plan: Plan,
    ) -> Result<()> {
        let d = self.contract.config.hidden_size;
        let heads = if l.rope.is_some() {
            self.contract.config.num_attention_heads
        } else {
            (d / 64).max(1)
        };
        if let Some(norm) = &l.norm1 {
            self.norm(e, norm, &s.hidden, &s.norm, n)?;
            self.linear(e, &l.qkv, &s.norm, &s.qkv, n, plan)?;
        } else {
            self.linear(e, &l.qkv, &s.hidden, &s.qkv, n, plan)?;
        }
        let qkv = if let Some(theta) = l.rope {
            let dim = d / heads;
            let half = dim / 2;
            let inverse: Vec<_> = (0..half)
                .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
                .collect();
            let angles: Vec<_> = (0..n)
                .flat_map(|t| {
                    inverse.iter().flat_map(move |&v| {
                        let (s, c) = (t as f32 * v).sin_cos();
                        [s, c]
                    })
                })
                .collect();
            let table = self.device.buffer(&angles, "Laya RoPE table")?;
            self.run(
                e,
                &self.pipelines.rope,
                &s.qkv,
                &table,
                &self.dummy,
                &s.rotated,
                &[n as u32, d as u32, dim as u32],
                [(n * 3 * d).div_ceil(64) as u32, 1, 1],
            )?;
            &s.rotated
        } else {
            &s.qkv
        };
        for first in (0..n).step_by(plan.attention_query_tile) {
            let rows = plan.attention_query_tile.min(n - first);
            let params = [
                n as u32,
                heads as u32,
                (d / heads) as u32,
                rows as u32,
                first as u32,
                l.window.map_or(0, |w| w + 1) as u32,
            ];
            self.run(
                e,
                &self.pipelines.scores,
                qkv,
                &self.dummy,
                &self.dummy,
                &s.scores,
                &params,
                [(rows * heads * n).div_ceil(64) as u32, 1, 1],
            )?;
            self.run(
                e,
                &self.pipelines.softmax,
                &s.scores,
                &self.dummy,
                &self.dummy,
                &s.probabilities,
                &[(rows * heads) as u32, n as u32],
                [(rows * heads).div_ceil(64) as u32, 1, 1],
            )?;
            self.run(
                e,
                &self.pipelines.values,
                &s.probabilities,
                qkv,
                &self.dummy,
                &s.context,
                &params,
                [(rows * d).div_ceil(64) as u32, 1, 1],
            )?;
        }
        self.linear(e, &l.attention_out, &s.context, &s.update, n, plan)?;
        self.pointwise(e, &s.hidden, &s.update, &s.norm, n * d, 0, 0, 0)?;
        std::mem::swap(&mut s.hidden, &mut s.norm);
        self.norm(e, &l.norm2, &s.hidden, &s.norm, n)?;
        self.linear(e, &l.ff_in, &s.norm, &s.ff, n, plan)?;
        self.pointwise(
            e,
            &s.ff,
            &self.dummy,
            &s.activated,
            n * l.ff_out.input,
            if l.gated { 2 } else { 3 },
            l.ff_out.input,
            0,
        )?;
        self.linear(e, &l.ff_out, &s.activated, &s.update, n, plan)?;
        self.pointwise(e, &s.hidden, &s.update, &s.norm, n * d, 0, 0, 0)?;
        std::mem::swap(&mut s.hidden, &mut s.norm);
        Ok(())
    }
    pub async fn forward(&self, encoded: &Encoded) -> Result<decision::RawDecision> {
        let n = encoded.ids.len();
        let d = self.contract.config.hidden_size;
        ensure!(
            n > 0 && n <= self.contract.config.max_position_embeddings,
            "invalid Laya sequence length"
        );
        ensure!(
            encoded.qtype < 3
                && !encoded.markers.is_empty()
                && encoded.markers.iter().all(|&m| m < n),
            "invalid Laya markers/type"
        );
        ensure!(
            encoded
                .ids
                .iter()
                .all(|&t| (t as usize) < self.contract.config.vocab_size),
            "token outside vocabulary"
        );
        let plan = *self.rules.select(n);
        let mut embedding = vec![0.; n * d];
        for (row, &id) in embedding.chunks_exact_mut(d).zip(&encoded.ids) {
            self.contract.embedding.copy_row(id as usize, row);
        }
        let source = self.device.buffer(&embedding, "Laya embedded input")?;
        let ff = self
            .encoder
            .iter()
            .chain(&self.head)
            .map(|l| l.ff_in.output)
            .max()
            .context("empty Laya graph")?;
        let heads = self.contract.config.num_attention_heads.max(d / 64);
        let mut s = Buffers::new(
            &self.device,
            n,
            d,
            ff,
            plan.attention_query_tile.min(n) * n * heads,
        )?;
        let mut e = self.device.encoder();
        self.norm(&mut e, &self.embedding_norm, &source, &s.hidden, n)?;
        for l in &self.encoder {
            self.layer(&mut e, l, &mut s, n, plan)?;
        }
        self.norm(&mut e, &self.final_norm, &s.hidden, &s.norm, n)?;
        self.pointwise(
            &mut e,
            &s.norm,
            &self.types,
            &s.hidden,
            n * d,
            1,
            d,
            encoded.qtype as u32,
        )?;
        for l in &self.head {
            self.layer(&mut e, l, &mut s, n, plan)?;
        }
        let hidden = self.device.read(e, &s.hidden, n * d).await?;
        let mut scratch = Scratch {
            hidden,
            ..Default::default()
        };
        Ok(self
            .contract
            .readout(std::slice::from_ref(encoded), &mut scratch, false)?
            .remove(0))
    }
    pub async fn predict(
        &self,
        request: &crate::laya::input::Request,
    ) -> Result<serde_json::Value> {
        let encoded = self.contract.encode(request)?;
        let mut results = Vec::with_capacity(encoded.len());
        for q in &encoded {
            results.push(self.forward(q).await?);
        }
        decision::decode(&self.contract.agent_config, request, &encoded, results)
    }
}
impl Linear {
    fn upload(device: &Device, l: cpu::Linear) -> Result<Self> {
        Ok(Self {
            input: l.input,
            output: l.output,
            weight: device.buffer(&l.weight.values(), "Laya projection")?,
            bias: device.buffer(&l.bias.unwrap_or_else(|| vec![0.; l.output]), "Laya bias")?,
        })
    }
}
impl Norm {
    fn upload(device: &Device, l: &cpu::Norm) -> Result<Self> {
        Ok(Self {
            width: l.weight.len(),
            eps: l.eps,
            weight: device.buffer(&l.weight, "Laya norm weight")?,
            bias: device.buffer(
                &l.bias.clone().unwrap_or_else(|| vec![0.; l.weight.len()]),
                "Laya norm bias",
            )?,
        })
    }
}
impl GpuLayer {
    fn upload(device: &Device, l: Layer) -> Result<Self> {
        Ok(Self {
            norm1: l
                .norm1
                .as_ref()
                .map(|n| Norm::upload(device, n))
                .transpose()?,
            qkv: Linear::upload(device, l.qkv)?,
            attention_out: Linear::upload(device, l.attention_out)?,
            norm2: Norm::upload(device, &l.norm2)?,
            ff_in: Linear::upload(device, l.ff_in)?,
            ff_out: Linear::upload(device, l.ff_out)?,
            window: l.window,
            rope: l.rope,
            gated: l.gated,
        })
    }
}
impl Buffers {
    fn new(device: &Device, n: usize, d: usize, ff: usize, scores: usize) -> Result<Self> {
        Ok(Self {
            hidden: device.empty(n * d, "Laya hidden")?,
            norm: device.empty(n * d, "Laya norm")?,
            qkv: device.empty(n * 3 * d, "Laya qkv")?,
            rotated: device.empty(n * 3 * d, "Laya rotated qkv")?,
            context: device.empty(n * d, "Laya context")?,
            update: device.empty(n * d, "Laya update")?,
            ff: device.empty(n * ff, "Laya ff")?,
            activated: device.empty(n * ff, "Laya activated")?,
            scores: device.empty(scores, "Laya scores")?,
            probabilities: device.empty(scores, "Laya probabilities")?,
        })
    }
}

impl Model {
    pub async fn predict_json(&self, mut value: serde_json::Value) -> Result<serde_json::Value> {
        use serde_json::Value;
        if let Some(rows) = value.as_array() {
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                out.push(Box::pin(self.predict_json(row.clone())).await?);
            }
            return Ok(Value::Array(out));
        }
        if let Some(states) = value.get("states").and_then(Value::as_array).cloned() {
            value.as_object_mut().unwrap().remove("states");
            let mut out = Vec::with_capacity(states.len());
            for state in states {
                let mut row = value.clone();
                row["state"] = state;
                out.push(Box::pin(self.predict_json(row)).await?);
            }
            return Ok(Value::Array(out));
        }
        if let Some(schema) = value.get("schema") {
            let plan = crate::laya::schema::Schema::parse(schema)?;
            let details = value["return_details"].as_bool().unwrap_or(false);
            value["questions"] = Value::Object(plan.questions.clone());
            value.as_object_mut().unwrap().remove("schema");
            return plan.project(&Box::pin(self.predict_json(value)).await?, details);
        }
        if value["long"].as_bool().unwrap_or(false) {
            ensure!(
                value.get("aggregate").is_none_or(|v| v == "auto"),
                "only aggregate=auto is supported"
            );
            let window = value
                .get("window")
                .filter(|v| !v.is_null())
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()?;
            let stride = value
                .get("stride")
                .filter(|v| !v.is_null())
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()?;
            return self
                .predict_long(&serde_json::from_value(value)?, window, stride)
                .await;
        }
        self.predict(&serde_json::from_value(value)?).await
    }
}

#[cfg(feature = "wasm")]
mod browser {
    use super::*;
    use wasm_bindgen::prelude::*;
    fn error(e: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&e.to_string())
    }
    #[wasm_bindgen(js_name=LayaWebGpuModel)]
    pub struct BrowserModel {
        model: Model,
    }
    #[wasm_bindgen(js_class=LayaWebGpuModel)]
    impl BrowserModel {
        pub async fn create(
            encoder_config: Vec<u8>,
            agent_config: Vec<u8>,
            tokenizer: Vec<u8>,
            tokenizer_config: Vec<u8>,
            weights: Vec<u8>,
            fp16_storage: bool,
        ) -> Result<BrowserModel, JsValue> {
            let model = CpuModel::from_bytes(
                &encoder_config,
                &agent_config,
                &tokenizer,
                &tokenizer_config,
                &weights,
                if fp16_storage {
                    crate::laya::weights::Storage::F16
                } else {
                    crate::laya::weights::Storage::F32
                },
            )
            .map_err(error)?;
            drop((
                encoder_config,
                agent_config,
                tokenizer,
                tokenizer_config,
                weights,
            ));
            Ok(Self {
                model: Model::from_cpu(model).await.map_err(error)?,
            })
        }
        pub async fn predict(&self, request_json: &str) -> Result<String, JsValue> {
            let value = serde_json::from_str(request_json).map_err(error)?;
            serde_json::to_string(&self.model.predict_json(value).await.map_err(error)?)
                .map_err(error)
        }
        #[wasm_bindgen(js_name=setLengthRules)]
        pub fn set_length_rules(&mut self, json: &str) -> Result<(), JsValue> {
            self.model
                .set_length_rules(serde_json::from_str(json).map_err(error)?)
                .map_err(error)
        }
        #[wasm_bindgen(js_name=adapterInfo)]
        pub fn adapter_info(&self) -> String {
            let i = self.model.adapter_info();
            serde_json::json!({"name":i.name,"backend":format!("{:?}",i.backend),"device_type":format!("{:?}",i.device_type)}).to_string()
        }
    }
}

use crate::laya::input::{Request, render};
use serde_json::{Map, Value, json};
impl Model {
    pub async fn predict_long(
        &self,
        request: &Request,
        window: Option<usize>,
        stride: Option<usize>,
    ) -> Result<Value> {
        ensure!(
            !request.state.is_null(),
            "long input state must not be null"
        );
        ensure!(
            request.min_confidence.is_none(),
            "long-document aggregation does not accept min_confidence"
        );
        let max_len = request.max_len.unwrap_or(
            self.contract.agent_config["max_len"]
                .as_u64()
                .unwrap_or(512) as usize,
        );
        let head_max_len = request.head_max_len.unwrap_or(
            self.contract.agent_config["head_max_len"]
                .as_u64()
                .unwrap_or(192) as usize,
        );
        let requested = window
            .filter(|&w| w > 0)
            .unwrap_or(max_len.saturating_sub(head_max_len + 8).max(64));
        let mut empty = request.clone();
        empty.state = json!("");
        let heads = self.contract.encode(&empty)?;
        let room = heads
            .iter()
            .map(|q| max_len.saturating_sub(q.ids.len()))
            .min()
            .unwrap_or(requested);
        let (size, step) = crate::laya::long::budget(requested, room, stride)?;
        let tokens = self.contract.input.tokens(&render(&request.state))?;
        let single_limit = if window.is_some_and(|w| w > 0) {
            size
        } else {
            room
        };
        if tokens.len() <= single_limit {
            let mut result = self.predict(request).await?;
            result["usage"]["windows"] = json!(1);
            return Ok(result);
        }
        let mut starts = Vec::new();
        let mut results = Vec::new();
        for start in (0..tokens.len()).step_by(step) {
            let end = (start + size).min(tokens.len());
            let mut item = request.clone();
            item.state = json!(self.contract.input.text(&tokens[start..end])?);
            starts.push(start);
            results.push(self.predict(&item).await?);
            if end == tokens.len() {
                break;
            }
        }
        let mut answers = Map::new();
        for (id, question) in &request.questions {
            let key = if question["type"] == "noul" {
                "noul"
            } else {
                "answer_confidence"
            };
            let mut best = 0;
            for i in 1..results.len() {
                if results[i]["answers"][id][key].as_f64()
                    > results[best]["answers"][id][key].as_f64()
                {
                    best = i;
                }
            }
            let mut answer = results[best]["answers"][id].clone();
            answer["window"] = json!({"index":best,"token_start":starts[best],"token_end":(starts[best]+size).min(tokens.len()),"count":results.len()});
            answers.insert(id.clone(), answer);
        }
        let mut usage = Map::new();
        for result in &results {
            for (key, value) in result["usage"].as_object().unwrap() {
                match value {
                    Value::Bool(v) => {
                        let prev = usage.get(key).and_then(Value::as_u64).unwrap_or(0);
                        usage.insert(key.clone(), json!(prev + u64::from(*v)));
                    }
                    Value::Number(v) => {
                        let prev = usage.get(key).and_then(Value::as_u64).unwrap_or(0);
                        usage.insert(key.clone(), json!(prev + v.as_u64().unwrap_or(0)));
                    }
                    Value::Object(values) => {
                        let entry = usage.entry(key.clone()).or_insert_with(|| json!({}));
                        entry.as_object_mut().unwrap().extend(values.clone());
                    }
                    _ => {
                        usage.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        usage.insert("windows".into(), json!(results.len()));
        Ok(json!({"model":"laya-rl-agent","answers":answers,"usage":usage}))
    }
}
