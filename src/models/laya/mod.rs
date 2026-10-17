pub mod cpu;
pub mod decision;
pub mod input;
pub mod language;
pub mod long;
pub mod router;
pub mod schema;
pub mod specialization;
#[cfg(feature = "wasm")]
mod web;
pub mod weights;

use crate::laya::{
    cpu::{Linear, Norm},
    input::{Encoded, Input, Request},
    weights::{Matrix, Storage, Weights},
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
mod token_cache;
#[cfg(feature = "webgpu")]
pub mod webgpu;
pub use token_cache::TokenCacheStats;
mod state_cache;
pub use state_cache::StateCacheStats;

#[derive(Clone, Debug, Deserialize)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_window")]
    pub local_attention: usize,
    #[serde(default = "default_global")]
    pub global_attn_every_n_layers: usize,
    #[serde(default = "default_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_global_theta")]
    pub global_rope_theta: f32,
    #[serde(default = "default_local_theta")]
    pub local_rope_theta: f32,
    #[serde(default)]
    pub rope_parameters: Value,
    #[serde(default)]
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
}
fn default_window() -> usize {
    128
}
fn default_global() -> usize {
    3
}
fn default_eps() -> f32 {
    1e-5
}
fn default_global_theta() -> f32 {
    160000.0
}
fn default_local_theta() -> f32 {
    10000.0
}

struct Layer {
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

#[derive(Default)]
pub struct Scratch {
    hidden: Vec<f32>,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    context: Vec<f32>,
    update: Vec<f32>,
    ff: Vec<f32>,
    activated: Vec<f32>,
    attention_tile: Option<std::num::NonZeroUsize>,
    attention: cpu::AttentionScratch,
    reuse_attention: bool,
    fuse_gelu: bool,
    panel_group: usize,
    projection: cpu::ProjectionStrategy,
    length_rules: Option<crate::length::Rules<crate::laya::specialization::CpuPlan>>,
    worker_loop: bool,
    simd_gelu: bool,
    fused_projection_norm: bool,
    fused_activation_pack: bool,
    projection_input: Vec<f32>,
    token_cache: token_cache::TokenCache,
    state_cache: state_cache::StateCache,
    prune_head: bool,
    pub trace: Option<Vec<(String, Vec<f32>)>>,
    pub profile: Option<std::collections::BTreeMap<&'static str, u128>>,
    pub layer_profile: Option<std::collections::BTreeMap<String, u128>>,
    profile_layer: Option<(&'static str, usize)>,
}
impl Scratch {
    pub fn set_length_rules(
        &mut self,
        rules: Option<crate::length::Rules<crate::laya::specialization::CpuPlan>>,
    ) -> Result<()> {
        if let Some(rules) = &rules {
            rules.validate(crate::laya::specialization::CpuPlan::validate)?;
        }
        self.length_rules = rules;
        self.token_cache.invalidate();
        self.state_cache.invalidate();
        Ok(())
    }
    fn projection_for(&self, rows: usize) -> cpu::ProjectionStrategy {
        self.length_rules
            .as_ref()
            .map_or(self.projection, |r| r.select(rows).projection)
    }
    fn attention_tile_for(&self, tokens: usize) -> usize {
        self.length_rules.as_ref().map_or_else(
            || self.attention_tile(),
            |r| r.select(tokens).attention_query_tile,
        )
    }
    pub fn set_optimizations(
        &mut self,
        reuse_attention: bool,
        fuse_gelu: bool,
        panel_group: usize,
    ) -> Result<()> {
        ensure!(
            (1..=64).contains(&panel_group),
            "panel group must be in 1..=64"
        );
        self.reuse_attention = reuse_attention;
        self.fuse_gelu = fuse_gelu;
        self.panel_group = panel_group;
        Ok(())
    }
    pub fn set_projection_block(&mut self, block: usize) -> Result<()> {
        ensure!(
            block == 0 || ((32..=1024).contains(&block) && block.is_power_of_two()),
            "projection block must be zero or a power of two from 32 to 1024"
        );
        self.projection = if block == 0 {
            cpu::ProjectionStrategy::Default
        } else {
            cpu::ProjectionStrategy::ReductionBlocks(block)
        };
        Ok(())
    }
    pub fn enable_avx2_pipeline(&mut self) {
        self.projection = cpu::ProjectionStrategy::PackedInput;
        self.fused_projection_norm = true;
        self.simd_gelu = true;
        self.worker_loop = true;
        self.reuse_attention = true;
        self.fuse_gelu = false;
        self.attention.set_packed_avx2(true);
    }
    pub fn enable_avx2_long_pipeline(&mut self) {
        self.enable_avx2_pipeline();
        self.attention.set_long_attention(true);
        self.attention.set_math(cpu::AttentionMath::TableExpSum);
        self.projection = cpu::ProjectionStrategy::LongRows(24);
        self.prune_head = true;
    }
    pub fn projection_retained_bytes(&self) -> usize {
        self.projection_input.capacity() * 4
    }
    pub fn set_fused_projection_norm(&mut self, enabled: bool) {
        self.fused_projection_norm = enabled;
    }
    pub fn set_fused_activation_pack(&mut self, enabled: bool) {
        self.fused_activation_pack = enabled;
    }
    pub fn set_packed_attention(&mut self, enabled: bool) {
        self.attention.set_packed_avx2(enabled);
    }
    pub fn set_long_attention(&mut self, enabled: bool) {
        self.attention.set_long_attention(enabled);
    }
    pub fn set_simd_softmax(&mut self, enabled: bool) {
        self.attention.set_simd_softmax(enabled);
    }
    pub fn set_attention_math(&mut self, math: cpu::AttentionMath) {
        self.attention.set_math(math);
    }
    pub fn set_packed_normalization(&mut self, enabled: bool) {
        self.attention.set_packed_normalization(enabled);
    }
    pub fn set_row_parallel_attention(&mut self, enabled: bool) {
        self.attention.set_row_parallel_attention(enabled);
    }
    pub fn set_attention_key_tile(&mut self, rows: usize) -> Result<()> {
        self.attention.set_key_tile(rows)
    }
    pub fn set_prune_head(&mut self, enabled: bool) {
        self.prune_head = enabled;
    }
    pub fn set_token_cache_capacity(&mut self, tokens: usize) {
        self.token_cache.configure(tokens);
    }
    pub fn token_cache_stats(&self) -> TokenCacheStats {
        self.token_cache.stats()
    }
    pub fn clear_token_cache(&mut self) {
        self.token_cache.clear();
    }
    pub fn reset_token_cache_stats(&mut self) {
        self.token_cache.reset_stats();
    }
    pub fn set_state_cache_capacity(&mut self, tokens: usize) {
        self.state_cache.configure(tokens);
    }
    pub fn state_cache_stats(&self) -> StateCacheStats {
        self.state_cache.stats()
    }
    pub fn clear_state_cache(&mut self) {
        self.state_cache.clear();
    }
    pub fn reset_state_cache_stats(&mut self) {
        self.state_cache.reset_stats();
    }
    pub fn set_simd_gelu(&mut self, enabled: bool) {
        self.simd_gelu = enabled;
    }
    pub fn set_projection_strategy(&mut self, strategy: cpu::ProjectionStrategy) {
        self.projection = strategy;
    }
    pub fn set_worker_loop(&mut self, enabled: bool) {
        self.worker_loop = enabled;
    }
    pub fn set_tiled_projection_tails(&mut self, enabled: bool) {
        self.projection = if enabled {
            cpu::ProjectionStrategy::TiledTails
        } else {
            cpu::ProjectionStrategy::Default
        };
    }
    pub fn attention_retained_bytes(&self) -> usize {
        self.attention.retained_bytes()
    }
    pub fn set_attention_tile(&mut self, rows: usize) -> Result<()> {
        ensure!(
            (16..=256).contains(&rows) && rows.is_power_of_two(),
            "attention tile must be one of 16, 32, 64, 128, 256"
        );
        self.attention_tile = std::num::NonZeroUsize::new(rows);
        Ok(())
    }
    pub fn attention_tile(&self) -> usize {
        self.attention_tile.map_or(64, std::num::NonZeroUsize::get)
    }
    fn mark(&mut self, name: &'static str, time: &mut Option<std::time::Instant>) {
        if let Some(last) = time {
            let now = std::time::Instant::now();
            let duration = now.duration_since(*last).as_nanos();
            *self.profile.as_mut().unwrap().entry(name).or_default() += duration;
            if let Some(profile) = &mut self.layer_profile {
                let key = self
                    .profile_layer
                    .map_or_else(|| name.to_owned(), |(kind, i)| format!("{kind}.{i}.{name}"));
                *profile.entry(key).or_default() += duration;
            }
            *last = now;
        }
    }
    fn record(&mut self, label: &str, layer: Option<usize>) {
        if let Some(v) = &mut self.trace {
            let name = layer.map_or_else(|| label.to_owned(), |i| format!("{label}.{i}"));
            v.push((name, self.hidden.clone()));
        }
    }
}

pub struct Model {
    cache_identity: std::sync::Arc<()>,
    pub config: EncoderConfig,
    pub agent_config: Value,
    pub input: Input,
    kernel: cpu::Kernel,
    embedding: Matrix,
    embedding_norm: Norm,
    encoder: Vec<Layer>,
    final_norm: Norm,
    type_embedding: Vec<f32>,
    head: Vec<Layer>,
    scorer_norm: Norm,
    scorer_in: Linear,
    scorer_out: Linear,
    act_in: Linear,
    act_out: Linear,
}

impl Model {
    #[cfg(feature = "int8")]
    pub fn quantize_encoder(&mut self, mode: cpu::int8::Mode, block: usize) -> Result<usize> {
        #[cfg(feature = "cpu")]
        ensure!(
            self.encoder.iter().all(|l| !l.qkv.uses_blas()),
            "disable BLAS before quantizing"
        );
        anyhow::ensure!(
            self.encoder.iter().all(|l| !l.qkv.weight.is_quantized()),
            "encoder already quantized; reload float weights to change the recipe"
        );
        anyhow::ensure!(
            matches!(block, 16 | 32 | 64 | 128),
            "invalid INT8 block size"
        );
        self.cache_identity = std::sync::Arc::new(());
        let mut bytes = 0;
        for layer in &mut self.encoder {
            for linear in [
                &mut layer.qkv,
                &mut layer.attention_out,
                &mut layer.ff_in,
                &mut layer.ff_out,
            ] {
                bytes += linear.quantize(mode, block, self.kernel == cpu::Kernel::X86Avx2)?;
            }
        }
        Ok(bytes)
    }
    pub fn kernel(&self) -> cpu::Kernel {
        self.kernel
    }

    #[cfg(feature = "cpu")]
    pub fn set_blas(&mut self, blas: Option<std::sync::Arc<cpu::Blas>>) -> Result<()> {
        ensure!(
            blas.is_none()
                || (self.kernel == cpu::Kernel::Portable
                    && self.encoder.iter().all(|l| !l.qkv.weight.is_quantized())),
            "BLAS requires a model loaded with portable float projections"
        );
        for l in self.encoder.iter_mut().chain(&mut self.head) {
            for linear in [
                &mut l.qkv,
                &mut l.attention_out,
                &mut l.ff_in,
                &mut l.ff_out,
            ] {
                linear.set_blas(blas.clone())?;
            }
        }
        for linear in [
            &mut self.scorer_in,
            &mut self.scorer_out,
            &mut self.act_in,
            &mut self.act_out,
        ] {
            linear.set_blas(blas.clone())?;
        }
        self.cache_identity = std::sync::Arc::new(());
        Ok(())
    }

    pub fn predict_json(&self, mut value: Value, scratch: &mut Scratch) -> Result<Value> {
        if let Some(requests) = value.as_array() {
            return requests
                .iter()
                .map(|r| self.predict_json(r.clone(), scratch))
                .collect::<Result<Vec<_>>>()
                .map(Value::Array);
        }
        if let Some(states) = value.get("states").and_then(Value::as_array).cloned() {
            value.as_object_mut().unwrap().remove("states");
            return states
                .into_iter()
                .map(|state| {
                    let mut request = value.clone();
                    request["state"] = state;
                    self.predict_json(request, scratch)
                })
                .collect::<Result<Vec<_>>>()
                .map(Value::Array);
        }
        if let Some(schema) = value.get("schema") {
            let plan = crate::laya::schema::Schema::parse(schema)?;
            let details = value["return_details"].as_bool().unwrap_or(false);
            value["questions"] = Value::Object(plan.questions.clone());
            value.as_object_mut().unwrap().remove("schema");
            let result = self.predict_json(value, scratch)?;
            return plan.project(&result, details);
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
            return self.predict_long(&serde_json::from_value(value)?, window, stride, scratch);
        }
        self.predict(&serde_json::from_value(value)?, scratch)
    }

    #[cfg(feature = "cpu")]
    pub fn load(path: impl AsRef<std::path::Path>, storage: Storage) -> Result<Self> {
        Self::load_with_kernel(path, storage, cpu::Kernel::Portable)
    }
    #[cfg(feature = "cpu")]
    pub fn load_with_kernel(
        path: impl AsRef<std::path::Path>,
        storage: Storage,
        kernel: cpu::Kernel,
    ) -> Result<Self> {
        let p = path.as_ref();
        Self::from_weights(
            &std::fs::read(p.join("encoder/config.json"))?,
            &std::fs::read(p.join("rl_agent_config.json"))?,
            &std::fs::read(p.join("tokenizer/tokenizer.json"))?,
            &std::fs::read(p.join("tokenizer/tokenizer_config.json"))?,
            Weights::open(&p.join("model.safetensors"))?,
            storage,
            kernel,
        )
    }
    pub fn from_bytes(
        encoder_config: &[u8],
        agent_config: &[u8],
        tokenizer: &[u8],
        tokenizer_config: &[u8],
        weights: &[u8],
        storage: Storage,
    ) -> Result<Self> {
        Self::from_bytes_with_kernel(
            encoder_config,
            agent_config,
            tokenizer,
            tokenizer_config,
            weights,
            storage,
            cpu::Kernel::Portable,
        )
    }
    pub fn from_bytes_with_kernel(
        encoder_config: &[u8],
        agent_config: &[u8],
        tokenizer: &[u8],
        tokenizer_config: &[u8],
        weights: &[u8],
        storage: Storage,
        kernel: cpu::Kernel,
    ) -> Result<Self> {
        Self::from_weights(
            encoder_config,
            agent_config,
            tokenizer,
            tokenizer_config,
            Weights::new(weights)?,
            storage,
            kernel,
        )
    }
    fn from_weights(
        encoder_config: &[u8],
        agent_config: &[u8],
        tokenizer: &[u8],
        tokenizer_config: &[u8],
        w: Weights<'_>,
        storage: Storage,
        kernel: cpu::Kernel,
    ) -> Result<Self> {
        kernel.validate()?;
        let mut c: EncoderConfig = serde_json::from_slice(encoder_config)?;
        ensure!(
            c.hidden_size > 0
                && c.num_attention_heads > 0
                && c.hidden_size % c.num_attention_heads == 0
                && (c.hidden_size / c.num_attention_heads) % 2 == 0,
            "invalid head dimensions"
        );
        ensure!(
            c.global_attn_every_n_layers > 0,
            "global attention interval must be positive"
        );
        if let Some(v) = c.rope_parameters["full_attention"]["rope_theta"].as_f64() {
            c.global_rope_theta = v as f32;
        }
        if let Some(v) = c.rope_parameters["sliding_attention"]["rope_theta"].as_f64() {
            c.local_rope_theta = v as f32;
        }
        let cfg: Value = serde_json::from_slice(agent_config)?;
        ensure!(
            cfg.get("option_layout").is_none_or(|v| v == "sequential"),
            "parallel option checkpoints require a different attention contract"
        );
        let d = c.hidden_size;
        let mut encoder = Vec::new();
        for i in 0..c.num_hidden_layers {
            let prefix = format!("encoder.layers.{i}");
            let local = if c.layer_types.is_empty() {
                i % c.global_attn_every_n_layers != 0
            } else {
                c.layer_types[i] == "sliding_attention"
            };
            encoder.push(Layer {
                norm1: if i == 0 {
                    None
                } else {
                    Some(Norm::load(
                        &w,
                        &format!("{prefix}.attn_norm"),
                        d,
                        c.norm_eps,
                    )?)
                },
                qkv: Linear::load_with_kernel(
                    &w,
                    &format!("{prefix}.attn.Wqkv"),
                    d,
                    3 * d,
                    storage,
                    kernel,
                )?,
                attention_out: Linear::load_with_kernel(
                    &w,
                    &format!("{prefix}.attn.Wo"),
                    d,
                    d,
                    storage,
                    kernel,
                )?,
                norm2: Norm::load(&w, &format!("{prefix}.mlp_norm"), d, c.norm_eps)?,
                ff_in: Linear::load_with_kernel(
                    &w,
                    &format!("{prefix}.mlp.Wi"),
                    d,
                    2 * c.intermediate_size,
                    storage,
                    kernel,
                )?,
                ff_out: Linear::load_with_kernel(
                    &w,
                    &format!("{prefix}.mlp.Wo"),
                    c.intermediate_size,
                    d,
                    storage,
                    kernel,
                )?,
                window: if local {
                    Some(c.local_attention / 2)
                } else {
                    None
                },
                rope: Some(if local {
                    c.local_rope_theta
                } else {
                    c.global_rope_theta
                }),
                gated: true,
            });
        }
        let mut head = Vec::new();
        for i in 0..cfg["head_layers"].as_u64().unwrap_or(2) {
            let p = format!("head.layers.{i}");
            head.push(Layer {
                norm1: Some(Norm::load(&w, &format!("{p}.norm1"), d, 1e-5)?),
                qkv: Linear::load_named_with_kernel(
                    &w,
                    &format!("{p}.self_attn.in_proj_weight"),
                    &format!("{p}.self_attn.in_proj_bias"),
                    d,
                    3 * d,
                    storage,
                    kernel,
                )?,
                attention_out: Linear::load_with_kernel(
                    &w,
                    &format!("{p}.self_attn.out_proj"),
                    d,
                    d,
                    storage,
                    kernel,
                )?,
                norm2: Norm::load(&w, &format!("{p}.norm2"), d, 1e-5)?,
                ff_in: Linear::load_with_kernel(
                    &w,
                    &format!("{p}.linear1"),
                    d,
                    4 * d,
                    storage,
                    kernel,
                )?,
                ff_out: Linear::load_with_kernel(
                    &w,
                    &format!("{p}.linear2"),
                    4 * d,
                    d,
                    storage,
                    kernel,
                )?,
                window: None,
                rope: None,
                gated: false,
            });
        }
        Ok(Self {
            cache_identity: std::sync::Arc::new(()),
            kernel,
            input: Input::new(tokenizer, tokenizer_config)?,
            embedding: w.matrix(
                "encoder.embeddings.tok_embeddings.weight",
                c.vocab_size,
                d,
                storage,
            )?,
            embedding_norm: Norm::load(&w, "encoder.embeddings.norm", d, c.norm_eps)?,
            final_norm: Norm::load(&w, "encoder.final_norm", d, c.norm_eps)?,
            type_embedding: w.tensor("type_emb.weight", &[3, d])?,
            scorer_norm: Norm::load(&w, "scorer.0", d, 1e-5)?,
            scorer_in: Linear::load_with_kernel(&w, "scorer.1", d, d, storage, kernel)?,
            scorer_out: Linear::load_with_kernel(&w, "scorer.3", d, 1, storage, kernel)?,
            act_in: Linear::load_with_kernel(&w, "act_head.0", d + 4, 256, storage, kernel)?,
            act_out: Linear::load_with_kernel(&w, "act_head.2", 256, 2, storage, kernel)?,
            config: c,
            agent_config: cfg,
            encoder,
            head,
        })
    }
    fn layer(
        &self,
        l: &Layer,
        lengths: &[usize],
        s: &mut Scratch,
        selected: Option<&[Vec<usize>]>,
        state_starts: Option<&[usize]>,
    ) {
        let mut clock = s.profile.as_ref().map(|_| std::time::Instant::now());
        let group = s.panel_group.max(1);
        let projection = s.projection_for(s.hidden.len() / self.config.hidden_size);
        if s.fused_projection_norm && l.norm1.is_some() {
            l.qkv.apply_normalized(
                &s.hidden,
                l.norm1.as_ref().unwrap(),
                &mut s.qkv,
                &mut s.norm,
                &mut s.projection_input,
                group,
                projection,
            );
            s.mark("norm_attention_qkv", &mut clock);
        } else {
            if let Some(norm) = &l.norm1 {
                norm.apply(&s.hidden, &mut s.norm);
            } else {
                s.norm.clone_from(&s.hidden);
            }
            s.mark("norm_attention", &mut clock);
            l.qkv.apply_with_workspace(
                &s.norm,
                &mut s.qkv,
                group,
                projection,
                &mut s.projection_input,
            );
            s.mark("qkv", &mut clock);
        }
        self.layer_after_qkv(l, lengths, s, clock, selected, state_starts);
    }
    fn layer_after_qkv(
        &self,
        l: &Layer,
        lengths: &[usize],
        s: &mut Scratch,
        mut clock: Option<std::time::Instant>,
        selected: Option<&[Vec<usize>]>,
        state_starts: Option<&[usize]>,
    ) {
        let group = s.panel_group.max(1);
        let projection = s.projection_for(s.hidden.len() / self.config.hidden_size);
        let heads = if l.rope.is_some() {
            self.config.num_attention_heads
        } else {
            (self.config.hidden_size / 64).max(1)
        };
        let d = self.config.hidden_size;
        if let Some(selected) = selected {
            assert!(l.rope.is_none() && l.window.is_none() && selected.len() == lengths.len());
        }
        let output_rows =
            selected.map_or(s.hidden.len() / d, |rows| rows.iter().map(Vec::len).sum());
        s.context.resize(output_rows * d, 0.0);
        let mut offset = 0;
        let mut output_offset = 0;
        for (sequence, &length) in lengths.iter().enumerate() {
            let attention_tile = s.attention_tile_for(length);
            if let Some(selected) = selected {
                let queries = &selected[sequence];
                cpu::attention_selected(
                    &s.qkv[offset * 3 * d..(offset + length) * 3 * d],
                    &mut s.context[output_offset * d..(output_offset + queries.len()) * d],
                    heads,
                    d / heads,
                    queries,
                    attention_tile,
                    &mut s.attention,
                    s.reuse_attention,
                );
                output_offset += queries.len();
            } else {
                let reused = state_starts.is_some_and(|starts| {
                    s.state_cache.apply(
                        &self.cache_identity,
                        &mut s.qkv[offset * 3 * d..(offset + length) * 3 * d],
                        &mut s.context[offset * d..(offset + length) * d],
                        heads,
                        d / heads,
                        l.rope.expect("first encoder rotary attention"),
                        starts[sequence],
                        attention_tile,
                        &mut s.attention,
                    )
                });
                if !reused {
                    cpu::attention_with_scratch(
                        &mut s.qkv[offset * 3 * d..(offset + length) * 3 * d],
                        &mut s.context[offset * d..(offset + length) * d],
                        heads,
                        d / heads,
                        l.window,
                        l.rope,
                        attention_tile,
                        &mut s.attention,
                        s.reuse_attention,
                    );
                }
            }
            offset += length;
        }
        s.mark("attention", &mut clock);
        if let Some(selected) = selected {
            s.norm.clear();
            let mut offset = 0;
            for (rows, &length) in selected.iter().zip(lengths) {
                for &row in rows {
                    s.norm
                        .extend_from_slice(&s.hidden[(offset + row) * d..][..d]);
                }
                offset += length;
            }
            std::mem::swap(&mut s.hidden, &mut s.norm);
        }
        l.attention_out.apply_with_workspace(
            &s.context,
            &mut s.update,
            group,
            projection,
            &mut s.projection_input,
        );
        s.mark("attention_out", &mut clock);
        cpu::residual(&mut s.hidden, &s.update);
        let norm_into_projection = s.fused_projection_norm && !(l.gated && s.fuse_gelu);
        if norm_into_projection {
            l.ff_in.apply_normalized(
                &s.hidden,
                &l.norm2,
                &mut s.ff,
                &mut s.norm,
                &mut s.projection_input,
                group,
                projection,
            );
            s.mark("norm_ff_in", &mut clock);
        } else {
            l.norm2.apply(&s.hidden, &mut s.norm);
            s.mark("norm_ff", &mut clock);
        }
        let width = l.ff_out.input;
        let rows = s.hidden.len() / self.config.hidden_size;
        if s.fused_activation_pack && !(l.gated && s.fuse_gelu) {
            if !norm_into_projection {
                l.ff_in.apply_with_workspace(
                    &s.norm,
                    &mut s.ff,
                    group,
                    projection,
                    &mut s.projection_input,
                );
                s.mark("ff_in", &mut clock);
            }
            l.ff_out.apply_activated(
                &s.ff,
                l.gated,
                &mut s.update,
                &mut s.activated,
                &mut s.projection_input,
                group,
                projection,
            );
            s.mark("activation_ff_out", &mut clock);
            cpu::residual(&mut s.hidden, &s.update);
            s.mark("residual", &mut clock);
            return;
        }
        s.activated.resize(rows * width, 0.0);
        if l.gated && s.fuse_gelu {
            l.ff_in
                .apply_gated(&s.norm, &mut s.activated, &mut s.ff, group);
            s.mark("ff_in_gelu", &mut clock);
        } else {
            if !norm_into_projection {
                l.ff_in.apply_with_workspace(
                    &s.norm,
                    &mut s.ff,
                    group,
                    projection,
                    &mut s.projection_input,
                );
                s.mark("ff_in", &mut clock);
            }
            if l.gated {
                if s.simd_gelu {
                    cpu::gated_gelu_simd(&s.ff, &mut s.activated, width);
                } else {
                    cpu::gated_gelu(&s.ff, &mut s.activated, width);
                }
            } else {
                for (a, &v) in s.activated.iter_mut().zip(&s.ff) {
                    *a = v.max(0.0);
                }
            }
            s.mark("activation", &mut clock);
        }
        l.ff_out.apply_with_workspace(
            &s.activated,
            &mut s.update,
            group,
            projection,
            &mut s.projection_input,
        );
        s.mark("ff_out", &mut clock);
        cpu::residual(&mut s.hidden, &s.update);
        s.mark("residual", &mut clock);
    }
    pub fn forward(
        &self,
        encoded: &Encoded,
        s: &mut Scratch,
    ) -> Result<crate::laya::decision::RawDecision> {
        Ok(self
            .forward_batch(std::slice::from_ref(encoded), s)?
            .remove(0))
    }

    pub fn forward_batch(
        &self,
        encoded: &[Encoded],
        s: &mut Scratch,
    ) -> Result<Vec<crate::laya::decision::RawDecision>> {
        #[cfg(feature = "cpu")]
        if s.worker_loop && rayon::current_thread_index().is_none() {
            return rayon::scope(|_| self.forward_batch_inner(encoded, s));
        }
        self.forward_batch_inner(encoded, s)
    }
    fn forward_batch_inner(
        &self,
        encoded: &[Encoded],
        s: &mut Scratch,
    ) -> Result<Vec<crate::laya::decision::RawDecision>> {
        let d = self.config.hidden_size;
        let lengths: Vec<_> = encoded.iter().map(|q| q.ids.len()).collect();
        let state_starts: Vec<_> = if s.state_cache.enabled()
            && self
                .encoder
                .first()
                .is_some_and(|l| l.window.is_none() && l.rope.is_some())
        {
            encoded
                .iter()
                .map(|q| q.ids.len().saturating_sub(q.state_tokens_used + 1))
                .collect()
        } else {
            Vec::new()
        };
        let n = lengths.iter().sum::<usize>();
        for q in encoded {
            ensure!(
                !q.ids.is_empty() && q.ids.len() <= self.config.max_position_embeddings,
                "invalid sequence length {}",
                q.ids.len()
            );
            ensure!(
                q.qtype < 3 && !q.markers.is_empty() && q.markers.iter().all(|&m| m < q.ids.len()),
                "invalid question markers/type"
            );
        }
        s.hidden.resize(n * d, 0.0);
        ensure!(
            encoded
                .iter()
                .flat_map(|q| &q.ids)
                .all(|&id| (id as usize) < self.config.vocab_size),
            "token outside vocabulary"
        );
        let cached_prelude =
            s.token_cache.enabled() && self.encoder.first().is_some_and(|l| l.norm1.is_none());
        let projection = s.projection_for(n);
        s.profile_layer = None;
        let mut embedding_clock = s.profile.as_ref().map(|_| std::time::Instant::now());
        if cached_prelude {
            s.token_cache.prepare(
                &self.cache_identity,
                encoded.iter().flat_map(|q| q.ids.iter().copied()),
                &self.embedding,
                &self.embedding_norm,
                &self.encoder[0].qkv,
                &mut s.hidden,
                &mut s.qkv,
                &mut s.projection_input,
                s.panel_group.max(1),
                projection,
            );
            s.mark("cached_token_prelude", &mut embedding_clock);
        } else {
            for (row, &id) in s
                .hidden
                .chunks_exact_mut(d)
                .zip(encoded.iter().flat_map(|q| &q.ids))
            {
                self.embedding.copy_row(id as usize, row);
            }
            self.embedding_norm.apply(&s.hidden, &mut s.norm);
            std::mem::swap(&mut s.hidden, &mut s.norm);
            s.mark("embedding", &mut embedding_clock);
        }
        s.record("embedding", None);
        for (i, l) in self.encoder.iter().enumerate() {
            if s.layer_profile.is_some() {
                s.profile_layer = Some(("encoder", i));
            }
            if i == 0 && cached_prelude {
                self.layer_after_qkv(
                    l,
                    &lengths,
                    s,
                    s.profile.as_ref().map(|_| std::time::Instant::now()),
                    None,
                    (!state_starts.is_empty()).then_some(state_starts.as_slice()),
                );
            } else {
                self.layer(
                    l,
                    &lengths,
                    s,
                    None,
                    (i == 0 && !state_starts.is_empty()).then_some(state_starts.as_slice()),
                );
            }
            s.record("encoder", Some(i));
        }
        s.profile_layer = None;
        let mut post_clock = s.profile.as_ref().map(|_| std::time::Instant::now());
        self.final_norm.apply(&s.hidden, &mut s.norm);
        std::mem::swap(&mut s.hidden, &mut s.norm);
        s.record("encoder.final", None);
        let mut offset = 0;
        for q in encoded {
            for row in s.hidden[offset * d..(offset + q.ids.len()) * d].chunks_exact_mut(d) {
                for (v, t) in row.iter_mut().zip(&self.type_embedding[q.qtype * d..][..d]) {
                    *v += t;
                }
            }
            offset += q.ids.len();
        }
        let pruned = s.prune_head && s.trace.is_none() && !self.head.is_empty();
        let selected: Vec<Vec<usize>> = if pruned {
            encoded
                .iter()
                .map(|q| {
                    std::iter::once(0)
                        .chain(q.markers.iter().copied())
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };
        s.mark("encoder_postprocessing", &mut post_clock);
        for (i, l) in self.head.iter().enumerate() {
            if s.layer_profile.is_some() {
                s.profile_layer = Some(("head", i));
            }
            let rows = if pruned && i + 1 == self.head.len() {
                Some(selected.as_slice())
            } else {
                None
            };
            self.layer(l, &lengths, s, rows, None);
            s.record("head", Some(i));
        }
        s.profile_layer = None;
        self.readout(encoded, s, pruned)
    }
    fn readout(
        &self,
        encoded: &[Encoded],
        s: &mut Scratch,
        pruned: bool,
    ) -> Result<Vec<crate::laya::decision::RawDecision>> {
        let d = self.config.hidden_size;
        let mut readout_clock = s.profile.as_ref().map(|_| std::time::Instant::now());
        let mut markers = Vec::new();
        let mut offset = 0;
        for q in encoded {
            for (i, &marker) in q.markers.iter().enumerate() {
                let m = if pruned { i + 1 } else { marker };
                markers.extend_from_slice(&s.hidden[(offset + m) * d..][..d]);
            }
            offset += if pruned {
                q.markers.len() + 1
            } else {
                q.ids.len()
            };
        }
        self.scorer_norm.apply(&markers, &mut s.norm);
        self.scorer_in.apply(&s.norm, &mut s.update);
        for v in &mut s.update {
            *v = cpu::gelu(*v);
        }
        let mut logits = Vec::new();
        self.scorer_out.apply(&s.update, &mut logits);
        let mut results = Vec::new();
        let mut logit_offset = 0;
        let mut token_offset = 0;
        for q in encoded {
            let logits = logits[logit_offset..logit_offset + q.markers.len()].to_vec();
            let mut p = logits.clone();
            cpu::softmax(&mut p);
            let k = p.len().max(2) as f32;
            let entropy = -p.iter().map(|&v| v * v.max(1e-9).ln()).sum::<f32>() / k.ln();
            p.sort_by(|a, b| b.total_cmp(a));
            let mut act_input = s.hidden[token_offset * d..][..d].to_vec();
            act_input.extend([
                p[0],
                p[0] - p.get(1).copied().unwrap_or(0.0),
                entropy,
                k / 255.0,
            ]);
            self.act_in.apply(&act_input, &mut s.update);
            for v in &mut s.update {
                *v = cpu::gelu(*v);
            }
            let mut act = Vec::new();
            self.act_out.apply(&s.update, &mut act);
            cpu::softmax(&mut act);
            ensure!(
                logits.iter().all(|v| v.is_finite()) && act.iter().all(|v| v.is_finite()),
                "non-finite model output"
            );
            results.push((logits, act[0]));
            logit_offset += q.markers.len();
            token_offset += if pruned {
                q.markers.len() + 1
            } else {
                q.ids.len()
            };
        }
        s.mark("readout", &mut readout_clock);
        Ok(results)
    }
    pub fn encode(&self, request: &Request) -> Result<Vec<Encoded>> {
        crate::laya::decision::prepare(
            &self.input,
            &self.agent_config,
            self.config.max_position_embeddings,
            request,
        )
    }
    pub fn predict(&self, request: &Request, scratch: &mut Scratch) -> Result<Value> {
        let encoded = self.encode(request)?;
        let mut results = Vec::new();
        let mut start = 0;
        while start < encoded.len() {
            let mut end = start + 1;
            let mut total = encoded[start].ids.len();
            while end < encoded.len() && total + encoded[end].ids.len() <= 1024 {
                total += encoded[end].ids.len();
                end += 1;
            }
            results.extend(
                self.forward_batch(&encoded[start..end], scratch)
                    .context("question batch")?,
            );
            start = end;
        }
        crate::laya::decision::decode(&self.agent_config, request, &encoded, results)
    }
}
