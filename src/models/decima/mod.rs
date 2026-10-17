#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod activation_wasm;
mod cpu;
mod embedding;
mod input;
#[cfg(feature = "int8")]
mod int8;
#[cfg(all(feature = "cpu", feature = "int8"))]
pub mod pack;
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
mod softmax_avx2;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod softmax_wasm;
pub mod specialization;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_attention;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_encode;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[cfg(feature = "int8")]
mod wasm_ffn;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_request;
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    feature = "decima-wasm-threads"
))]
mod wasm_score;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd;
#[cfg(all(target_arch = "wasm32", feature = "decima-wasm-threads"))]
mod wasm_threads;
#[cfg(feature = "wasm")]
mod web;
#[cfg(feature = "webgpu")]
pub mod webgpu;
mod weights;
use anyhow::{Result, ensure};
use cpu::{Linear, Norm};
pub use input::{Kind, Question};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::VecDeque, sync::Arc};
use weights::Weights;

pub const REPOSITORY: &str = "amyrmahdy/decima-small";
pub const REVISION: &str = "14d09cabc1b4e23106ae4413eef852a568ced77b";
pub const SDK_REVISION: &str = "0ab6f2164abb34caa749f31ac643801d1b4b03b7";
const D: usize = 384;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Precision {
    #[default]
    F32,
    Int8,
    Int8Weights,
    Int8Full,
    Int8Compact,
}
#[derive(Clone, Copy, Debug, Default)]
pub enum Kernel {
    #[default]
    Auto,
    Portable,
    Avx2,
    WasmSimd,
}
impl Kernel {
    pub fn resolve(self) -> Result<Self> {
        let wasm = cfg!(all(target_arch = "wasm32", target_feature = "simd128"));
        match self {
            Self::Auto => Ok(if wasm {
                Self::WasmSimd
            } else if cpu::packed_supported() {
                Self::Avx2
            } else {
                Self::Portable
            }),
            Self::Avx2 => {
                ensure!(cpu::packed_supported(), "AVX2/FMA unavailable");
                Ok(self)
            }
            Self::WasmSimd => {
                ensure!(wasm, "Wasm SIMD128 requires a simd128 build");
                Ok(self)
            }
            Self::Portable => Ok(self),
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub precision: Precision,
    pub kernel: Kernel,
}

#[derive(Deserialize)]
struct Config {
    backbone: String,
    scorer_layers: usize,
    heads: usize,
    ffn: usize,
    max_state_tokens: usize,
    max_choice_tokens: usize,
    temperature: f32,
    question_in_state: bool,
    question_in_choices: bool,
    #[serde(default)]
    choice_fit: bool,
    state_prefix: String,
    choice_prefix: String,
}
struct EncoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n1: Norm,
    n2: Norm,
    ff1: Linear,
    ff2: Linear,
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}
impl Attention {
    fn load(w: &Weights<'_>, name: &str, kernel: Kernel) -> Result<Self> {
        let l = |suffix: &str| {
            Linear::load(w, &format!("{name}.{suffix}"), D, D, Precision::F32, kernel)
        };
        Ok(Self {
            q: l("q")?,
            k: l("k")?,
            v: l("v")?,
            o: l("o")?,
        })
    }
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
impl ScorerLayer {
    #[cfg_attr(feature = "cpu", inline(always))]
    fn prepare_query(&self, x: &mut Scratch, plan: specialization::CpuPlan, avx2: bool) {
        self.n1.apply(&x.hidden, &mut x.norm);
        self.self_attn
            .q
            .apply_with_plan(&x.norm, &mut x.q, &mut x.work, plan);
        self.self_attn
            .k
            .apply_with_plan(&x.norm, &mut x.k, &mut x.work, plan);
        self.self_attn
            .v
            .apply_with_plan(&x.norm, &mut x.v, &mut x.work, plan);
        cpu::attention_with_tile(
            &x.q,
            &x.k,
            &x.v,
            D,
            6,
            &mut x.ctx,
            &mut x.scores,
            &mut x.heads,
            plan.query_tile(x.hidden.len() / D),
            avx2,
        );
        self.self_attn
            .o
            .apply_with_plan(&x.ctx, &mut x.tmp, &mut x.work, plan);
        cpu::add(&mut x.hidden, &x.tmp);
        self.n2.apply(&x.hidden, &mut x.norm);
        self.cross_attn
            .q
            .apply_with_plan(&x.norm, &mut x.q, &mut x.work, plan);
    }
}
#[derive(Default)]
struct Scratch {
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        feature = "decima-wasm-threads"
    ))]
    segment_attention: Vec<wasm_encode::AttentionScratch>,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[cfg(feature = "int8")]
    ffn: wasm_ffn::Scratch,
    hidden: Vec<f32>,
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    ctx: Vec<f32>,
    tmp: Vec<f32>,
    ff: Vec<f32>,
    work: Vec<f32>,
    scores: Vec<f32>,
    heads: Vec<f32>,
}

// Each worker owns a Session; choice-token keys and model identity bound cache reuse.
pub struct Model {
    identity: Arc<()>,
    config: Config,
    input: input::Input,
    options: Options,
    avx2: bool,
    embeddings: embedding::Embeddings,
    positions: Vec<f32>,
    types: Vec<f32>,
    embedding_norm: Norm,
    encoder: Vec<EncoderLayer>,
    scorer: Vec<ScorerLayer>,
    out_norm: Norm,
    score: Linear,
    ord_g: Linear,
    ord_gap: Linear,
    sim_scale: f32,
    sim_center: f32,
}
#[derive(Default, Clone, Debug, Serialize)]
pub struct Stats {
    pub state_tokens: u64,
    pub choice_tokens: u64,
    pub choice_cache_hits: u64,
}
// Cache only before state cross-attention; later scorer activations depend on the request.
struct CachedChoice {
    hidden: Box<[f32]>,
    query: Box<[f32]>,
    mean: Box<[f32]>,
    norm: f32,
}
impl CachedChoice {
    fn bytes(&self, tokens: usize) -> usize {
        (tokens + self.hidden.len() + self.query.len() + self.mean.len() + 1) * 4
    }
}
pub struct Session {
    identity: Option<Arc<()>>,
    encoder: Scratch,
    scorer: Scratch,
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        feature = "decima-wasm-threads"
    ))]
    choices: Vec<wasm_score::Choice>,
    cache: VecDeque<(Vec<u32>, CachedChoice)>,
    cache_bytes: usize,
    cache_limit: usize,
    memory: Vec<(Vec<f32>, Vec<f32>)>,
    rules: Option<crate::length::Rules<specialization::CpuPlan>>,
    pub stats: Stats,
}
impl Default for Session {
    fn default() -> Self {
        Self {
            identity: None,
            encoder: Scratch::default(),
            scorer: Scratch::default(),
            #[cfg(all(
                target_arch = "wasm32",
                target_feature = "simd128",
                feature = "decima-wasm-threads"
            ))]
            choices: vec![],
            cache: VecDeque::new(),
            cache_bytes: 0,
            cache_limit: 32 * 1024 * 1024,
            memory: vec![],
            rules: None,
            stats: Stats::default(),
        }
    }
}
impl Session {
    pub fn set_length_rules(
        &mut self,
        rules: Option<crate::length::Rules<specialization::CpuPlan>>,
    ) -> Result<()> {
        if let Some(rules) = &rules {
            rules.validate(specialization::CpuPlan::validate)?;
        }
        self.rules = rules;
        // Tiling can change rounding; invalidate cached choices when the plan changes.
        self.clear_cache();
        Ok(())
    }
    fn plan(&self, tokens: usize, kernel: Kernel) -> specialization::CpuPlan {
        self.rules.as_ref().map_or_else(
            || specialization::default_plan(kernel),
            |r| *r.select(tokens),
        )
    }
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_bytes = 0;
    }
    pub fn set_cache_bytes(&mut self, bytes: usize) {
        self.cache_limit = bytes;
        self.clear_cache();
    }
    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }
}
impl Model {
    #[cfg(feature = "cpu")]
    pub fn load(path: impl AsRef<std::path::Path>, options: Options) -> Result<Self> {
        let root = path.as_ref();
        let root = if root.join("pytorch").is_dir() {
            root.join("pytorch")
        } else {
            root.to_owned()
        };
        let encoder = std::fs::File::open(root.join("encoder/model.safetensors"))?;
        let head = std::fs::File::open(root.join("head.safetensors"))?;
        // Checkpoint files must remain immutable until loading finishes.
        let encoder = unsafe { memmap2::Mmap::map(&encoder)? };
        let head = unsafe { memmap2::Mmap::map(&head)? };
        Self::from_bytes(
            &std::fs::read(root.join("decima.json"))?,
            &std::fs::read(root.join("encoder/config.json"))?,
            &std::fs::read(root.join("encoder/tokenizer.json"))?,
            &encoder,
            &head,
            options,
        )
    }
    pub fn from_bytes(
        config: &[u8],
        encoder_config: &[u8],
        tokenizer: &[u8],
        encoder: &[u8],
        head: &[u8],
        options: Options,
    ) -> Result<Self> {
        ensure!(
            options.precision == Precision::F32 || cfg!(feature = "int8"),
            "INT8 requires the int8 Cargo feature"
        );
        let config: Config = serde_json::from_slice(config)?;
        let ec: Value = serde_json::from_slice(encoder_config)?;
        ensure!(
            config.backbone == "intfloat/multilingual-e5-small"
                && config.scorer_layers == 2
                && config.heads == 6
                && config.ffn == 1536,
            "only the Decima-small architecture is supported"
        );
        for (k, v) in [
            ("hidden_size", 384),
            ("intermediate_size", 1536),
            ("num_hidden_layers", 12),
            ("num_attention_heads", 12),
            ("max_position_embeddings", 512),
            ("vocab_size", 250037),
            ("type_vocab_size", 2),
        ] {
            ensure!(ec[k].as_u64() == Some(v), "unsupported Decima encoder {k}");
        }
        ensure!(
            ec["model_type"] == "bert"
                && ec["hidden_act"] == "gelu"
                && ec["position_embedding_type"] == "absolute"
                && ec["is_decoder"] == false
                && ec["add_cross_attention"] == false,
            "unsupported Decima encoder semantics"
        );
        ensure!(
            ec["layer_norm_eps"].as_f64() == Some(1e-12),
            "unsupported encoder norm epsilon"
        );
        ensure!(
            (2..=512).contains(&config.max_state_tokens)
                && (2..=512).contains(&config.max_choice_tokens)
                && config.temperature.is_finite()
                && config.temperature > 0.,
            "invalid Decima limits/temperature"
        );
        let kernel = options.kernel.resolve()?;
        let avx2 = matches!(kernel, Kernel::Avx2);
        let options = Options { kernel, ..options };
        let w = Weights::new(encoder)?;
        let h = Weights::new(head)?;
        let mut layers = vec![];
        for i in 0..12 {
            let p = format!("encoder.layer.{i}");
            let l = |n: &str, a, b| {
                Linear::load(&w, &format!("{p}.{n}"), a, b, options.precision, kernel)
            };
            layers.push(EncoderLayer {
                q: l("attention.self.query", D, D)?,
                k: l("attention.self.key", D, D)?,
                v: l("attention.self.value", D, D)?,
                o: l("attention.output.dense", D, D)?,
                n1: Norm::load(&w, &format!("{p}.attention.output.LayerNorm"), D, 1e-12)?,
                n2: Norm::load(&w, &format!("{p}.output.LayerNorm"), D, 1e-12)?,
                ff1: l("intermediate.dense", D, 1536)?,
                ff2: l("output.dense", 1536, D)?,
            });
        }
        let mut scorer = vec![];
        for i in 0..2 {
            let p = format!("layers.{i}");
            let norm = |n: &str| Norm::load(&h, &format!("{p}.{n}"), D, 1e-5);
            scorer.push(ScorerLayer {
                self_attn: Attention::load(&h, &format!("{p}.self_attn"), kernel)?,
                cross_attn: Attention::load(&h, &format!("{p}.cross_attn"), kernel)?,
                n1: norm("n1")?,
                n2: norm("n2")?,
                n3: norm("n3")?,
                ff1: Linear::load(&h, &format!("{p}.ff.0"), D, 1536, Precision::F32, kernel)?,
                ff2: Linear::load(&h, &format!("{p}.ff.3"), 1536, D, Precision::F32, kernel)?,
            });
        }
        Ok(Self {
            identity: Arc::new(()),
            input: input::Input::new(tokenizer, config.max_state_tokens, config.max_choice_tokens)?,
            config,
            options,
            avx2,
            embeddings: embedding::Embeddings::load(&w, options.precision)?,
            positions: w.get("embeddings.position_embeddings.weight", &[512, D])?,
            types: w.get("embeddings.token_type_embeddings.weight", &[2, D])?,
            embedding_norm: Norm::load(&w, "embeddings.LayerNorm", D, 1e-12)?,
            encoder: layers,
            scorer,
            out_norm: Norm::load(&h, "out_norm", D, 1e-5)?,
            score: Linear::load(&h, "score", D, 1, Precision::F32, Kernel::Portable)?,
            ord_g: Linear::load(&h, "ord_g", D, 1, Precision::F32, Kernel::Portable)?,
            ord_gap: Linear::load(&h, "ord_gap", D, 1, Precision::F32, Kernel::Portable)?,
            sim_scale: h.get("sim_scale", &[])?[0],
            sim_center: h.get("sim_center", &[])?[0],
        })
    }
    pub fn precision(&self) -> Precision {
        self.options.precision
    }
    fn encode(
        &self,
        ids: &[u32],
        s: &mut Scratch,
        plan: specialization::CpuPlan,
    ) -> Result<Arc<[f32]>> {
        ensure!(
            !ids.is_empty() && ids.len() <= 512 && ids.iter().all(|i| (*i as usize) < 250037),
            "invalid encoder token sequence"
        );
        s.hidden.resize(ids.len() * D, 0.);
        for (p, &id) in ids.iter().enumerate() {
            let row = &mut s.hidden[p * D..(p + 1) * D];
            self.embeddings.row(id as usize, row);
            for i in 0..D {
                row[i] = (row[i] + self.types[i]) + self.positions[p * D + i];
            }
        }
        self.embedding_norm.in_place(&mut s.hidden);
        for l in &self.encoder {
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            #[cfg(feature = "int8")]
            let shared = Linear::apply_shared_int8(
                [&l.q, &l.k, &l.v],
                &s.hidden,
                [&mut s.q, &mut s.k, &mut s.v],
                &mut s.work,
                plan,
            );
            #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128", feature = "int8")))]
            let shared = false;
            if !shared {
                l.q.apply_with_plan(&s.hidden, &mut s.q, &mut s.work, plan);
                l.k.apply_with_plan(&s.hidden, &mut s.k, &mut s.work, plan);
                l.v.apply_with_plan(&s.hidden, &mut s.v, &mut s.work, plan);
            }
            cpu::attention_with_tile(
                &s.q,
                &s.k,
                &s.v,
                D,
                12,
                &mut s.ctx,
                &mut s.scores,
                &mut s.heads,
                plan.query_tile(ids.len()),
                self.avx2,
            );
            l.o.apply_with_plan(&s.ctx, &mut s.tmp, &mut s.work, plan);
            cpu::add(&mut s.hidden, &s.tmp);
            l.n1.in_place(&mut s.hidden);
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            #[cfg(feature = "int8")]
            if Linear::try_encoder_ffn(&l.ff1, &l.ff2, &l.n2, &mut s.hidden, plan, &mut s.ffn) {
                continue;
            }
            l.ff1
                .apply_with_plan(&s.hidden, &mut s.ff, &mut s.work, plan);
            cpu::gelu(&mut s.ff);
            l.ff2.apply_with_plan(&s.ff, &mut s.tmp, &mut s.work, plan);
            cpu::add(&mut s.hidden, &s.tmp);
            l.n2.in_place(&mut s.hidden);
        }
        Ok(Arc::from(s.hidden.as_slice()))
    }
    // Reject state+question above 512 tokens; silent truncation changes the contract.
    pub fn decide(&self, state: &str, q: &Question, s: &mut Session) -> Result<Vec<f32>> {
        #[cfg(feature = "cpu")]
        if rayon::current_thread_index().is_none() {
            return rayon::scope(|_| self.decide_inner(state, q, s));
        }
        #[cfg(all(target_arch = "wasm32", feature = "decima-wasm-threads"))]
        if wasm_threads::count() > 1 && rayon::current_thread_index().is_none() {
            return rayon::scope(|_| self.decide_inner(state, q, s));
        }
        self.decide_inner(state, q, s)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn decide_inner(&self, state: &str, q: &Question, s: &mut Session) -> Result<Vec<f32>> {
        ensure!((2..=255).contains(&q.choices.len()), "need 2–255 choices");
        if q.kind == Kind::Score {
            ensure!(q.choices.len() <= 10, "score allows at most 10 levels");
        }
        if !s
            .identity
            .as_ref()
            .is_some_and(|id| Arc::ptr_eq(id, &self.identity))
        {
            s.clear_cache();
            s.identity = Some(self.identity.clone());
        }
        let body = if self.config.question_in_state {
            format!("{}\n{state}", q.text)
        } else {
            state.into()
        };
        let text = format!(
            "{}{}",
            self.config.state_prefix,
            input::normalize(&body, &q.lang)
        );
        let ids = self.input.ids(&text)?;
        ensure!(
            ids.len() <= self.input.max_state,
            "STATE_TRUNCATED: state + question is {} tokens; maximum {}",
            ids.len(),
            self.input.max_state
        );
        let state_plan = s.plan(ids.len(), self.options.kernel);
        let state_h = self.encode(&ids, &mut s.encoder, state_plan)?;
        s.stats.state_tokens += ids.len() as u64;
        let state_mean = cpu::mean(&state_h, D);
        let norm = |v: &[f32]| v.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let state_norm = norm(&state_mean);
        let cosine = |choice_mean: &[f32], choice_norm: f32| {
            state_mean
                .iter()
                .zip(choice_mean)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                / (state_norm * choice_norm)
        };
        s.memory.resize_with(self.scorer.len(), Default::default);
        for (l, (k, v)) in self.scorer.iter().zip(&mut s.memory) {
            l.cross_attn
                .k
                .apply_with_plan(&state_h, k, &mut s.scorer.work, state_plan);
            l.cross_attn
                .v
                .apply_with_plan(&state_h, v, &mut s.scorer.work, state_plan);
        }
        let rows = self.input.choices(
            q,
            &self.config.choice_prefix,
            self.config.question_in_choices,
            self.config.choice_fit,
        )?;
        let mut logits = vec![];
        let mut zs = vec![];
        for ids in rows {
            let plan = s.plan(ids.len(), self.options.kernel);
            let similarity;
            if let Some(index) = s.cache.iter().position(|(key, _)| key == &ids) {
                let entry = s.cache.remove(index).unwrap();
                let x = &mut s.scorer;
                x.hidden.clear();
                x.hidden.extend_from_slice(&entry.1.hidden);
                x.q.clear();
                x.q.extend_from_slice(&entry.1.query);
                similarity = cosine(&entry.1.mean, entry.1.norm);
                s.cache.push_back(entry);
                s.stats.choice_cache_hits += 1;
            } else {
                let h = self.encode(&ids, &mut s.encoder, plan)?;
                s.stats.choice_tokens += ids.len() as u64;
                let mean = cpu::mean(&h, D);
                let choice_norm = norm(&mean);
                similarity = cosine(&mean, choice_norm);
                let x = &mut s.scorer;
                x.hidden.clear();
                x.hidden.extend_from_slice(&h);
                self.scorer[0].prepare_query(x, plan, self.avx2);
                let bytes = (ids.len() + x.hidden.len() + x.q.len() + mean.len() + 1) * 4;
                if bytes <= s.cache_limit {
                    while s.cache_bytes + bytes > s.cache_limit {
                        if let Some((key, old)) = s.cache.pop_front() {
                            s.cache_bytes -= old.bytes(key.len());
                        }
                    }
                    s.cache_bytes += bytes;
                    s.cache.push_back((
                        ids,
                        CachedChoice {
                            hidden: x.hidden.as_slice().into(),
                            query: x.q.as_slice().into(),
                            mean: mean.into_boxed_slice(),
                            norm: choice_norm,
                        },
                    ));
                }
            }
            let x = &mut s.scorer;
            for (index, (l, (mk, mv))) in self.scorer.iter().zip(&s.memory).enumerate() {
                if index > 0 {
                    l.prepare_query(x, plan, self.avx2);
                }
                cpu::attention_with_tile(
                    &x.q,
                    mk,
                    mv,
                    D,
                    6,
                    &mut x.ctx,
                    &mut x.scores,
                    &mut x.heads,
                    state_plan.query_tile(state_h.len() / D),
                    self.avx2,
                );
                l.cross_attn
                    .o
                    .apply_with_plan(&x.ctx, &mut x.tmp, &mut x.work, plan);
                cpu::add(&mut x.hidden, &x.tmp);
                l.n3.apply(&x.hidden, &mut x.norm);
                l.ff1.apply_with_plan(&x.norm, &mut x.ff, &mut x.work, plan);
                cpu::gelu(&mut x.ff);
                l.ff2.apply_with_plan(&x.ff, &mut x.tmp, &mut x.work, plan);
                cpu::add(&mut x.hidden, &x.tmp);
            }
            self.out_norm.in_place(&mut x.hidden);
            let z = cpu::mean(&x.hidden, D);
            self.score.apply(&z, &mut x.tmp, &mut x.work);
            logits.push(
                (x.tmp[0] + self.sim_scale * (similarity - self.sim_center))
                    / self.config.temperature,
            );
            zs.extend(z);
        }
        self.finish_probabilities(logits, &zs, q.kind, &mut s.scorer)
    }
    fn finish_probabilities(
        &self,
        mut logits: Vec<f32>,
        zs: &[f32],
        kind: Kind,
        scratch: &mut Scratch,
    ) -> Result<Vec<f32>> {
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite Decima scores"
        );
        match kind {
            Kind::Score => {
                cpu::softmax(&mut logits);
                let expected = logits
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f32 * p)
                    .sum::<f32>()
                    - (logits.len() - 1) as f32 / 2.;
                let zm = cpu::mean(&zs, D);
                self.ord_g.apply(&zm, &mut scratch.tmp, &mut scratch.work);
                let g = scratch.tmp[0] + expected;
                self.ord_gap.apply(&zs, &mut scratch.tmp, &mut scratch.work);
                let gaps: Vec<_> = scratch
                    .tmp
                    .iter()
                    .map(|v| {
                        if *v > 20. {
                            *v + 1e-3
                        } else {
                            v.exp().ln_1p() + 1e-3
                        }
                    })
                    .collect();
                let center = gaps.iter().sum::<f32>() / 2.;
                let mut theta = 0.;
                let mut previous = 1.;
                for i in 0..logits.len() - 1 {
                    theta += gaps[i];
                    let sig = 1. / (1. + (-(g - theta + center)).exp());
                    logits[i] = (previous - sig).max(1e-7);
                    previous = sig;
                }
                *logits.last_mut().unwrap() = previous.max(1e-7);
                let sum = logits.iter().sum::<f32>();
                for v in &mut logits {
                    *v /= sum;
                }
            }
            Kind::Rank => {
                for v in &mut logits {
                    *v = 1. / (1. + (-*v).exp());
                }
            }
            _ => cpu::softmax(&mut logits),
        }
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite Decima probabilities"
        );
        Ok(logits)
    }
    pub fn predict_json(&self, request: &Value, s: &mut Session) -> Result<Value> {
        if let Some(rows) = request.as_array() {
            return rows
                .iter()
                .map(|r| self.predict_json(r, s))
                .collect::<Result<Vec<_>>>()
                .map(Value::Array);
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
            let p = self.decide(&state, &q, s)?;
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

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    fn question(v: &Value) -> Question {
        Question {
            text: v["text"].as_str().unwrap().into(),
            choices: serde_json::from_value(v["choices"].clone()).unwrap(),
            lang: v["lang"].as_str().unwrap().into(),
            kind: match v["kind"].as_str().unwrap() {
                "score" => Kind::Score,
                "rank" => Kind::Rank,
                "noul" => Kind::Noul,
                _ => Kind::Choice,
            },
        }
    }
    fn difference(a: &Value, b: &Value) {
        match (a, b) {
            (Value::Number(a), Value::Number(b)) => assert!(
                (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 1.000001e-4,
                "{a} != {b}"
            ),
            (Value::Object(a), Value::Object(b)) => {
                assert_eq!(a.len(), b.len());
                for (k, v) in a {
                    difference(v, &b[k]);
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    difference(a, b);
                }
            }
            _ => assert_eq!(a, b),
        }
    }
    #[test]
    #[ignore = "requires models/decima-small; run in release mode"]
    fn choice_cache_varied_requests() -> Result<()> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        pool.install(|| -> Result<()> {
            let model = Model::load("models/decima-small", Options::default())?;
            let fixtures: Vec<Value> = include_str!("../../../benches/decima-reference.jsonl")
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            let find = |id: &str| fixtures.iter().find(|v| v["id"] == id).unwrap();
            let mut cached = Session::default();
            let mut fresh = Session::default();
            fresh.set_cache_bytes(0);
            for id in [
                "en-choice-0",
                "en-choice-1",
                "en-choice-0",
                "ordinal-0",
                "ordinal-1",
                "ordinal-2",
                "zh-choice-0",
                "zh-choice-1",
                "rank",
            ] {
                let v = find(id);
                let q = question(&v["question"]);
                let state = v["state"].as_str().unwrap();
                let result = model.decide(state, &q, &mut cached)?;
                assert_eq!(
                    result,
                    model.decide(state, &q, &mut fresh)?,
                    "changing-state cache: {id}"
                );
                for (actual, expected) in result.iter().zip(v["expected"].as_array().unwrap()) {
                    assert!(
                        (*actual as f64 - expected.as_f64().unwrap()).abs() < 1e-4,
                        "SDK: {id}"
                    );
                }
            }
            assert!(cached.stats.choice_cache_hits >= 3 + 3 + 3 + 3 + 3);
            assert_eq!(fresh.cache_bytes(), 0);
            assert_eq!(fresh.stats.choice_cache_hits, 0);

            let v = find("en-choice-0");
            let mut q = question(&v["question"]);
            let state = v["state"].as_str().unwrap();
            let mut changing_outputs = vec![];
            for evidence in [
                state,
                "The application crashes whenever I log in.",
                "My package is missing. Please track the delivery.",
                state,
            ] {
                let hits = cached.stats.choice_cache_hits;
                let result = model.decide(evidence, &q, &mut cached)?;
                assert_eq!(result, model.decide(evidence, &q, &mut fresh)?);
                assert_eq!(
                    cached.stats.choice_cache_hits - hits,
                    q.choices.len() as u64
                );
                changing_outputs.push(result);
            }
            assert_ne!(
                changing_outputs[0], changing_outputs[1],
                "evidence must affect scoring"
            );
            assert_eq!(changing_outputs[0], changing_outputs[3]);
            for change in 0..4 {
                match change {
                    0 => q.choices.reverse(),
                    1 => q.text = "Which department can solve the customer's problem?".into(),
                    2 => q.choices[0] = "account security".into(),
                    _ => q.kind = Kind::Rank,
                }
                assert_eq!(
                    model.decide(state, &q, &mut cached)?,
                    model.decide(state, &q, &mut fresh)?
                );
            }

            let q = question(&v["question"]);
            cached.clear_cache();
            let expected = model.decide(state, &q, &mut fresh)?;
            model.decide(state, &q, &mut cached)?;
            let all_bytes = cached.cache_bytes();
            for budget in [0, 1, all_bytes / 2, all_bytes - 1, all_bytes] {
                cached.set_cache_bytes(budget);
                let start_hits = cached.stats.choice_cache_hits;
                for _ in 0..2 {
                    assert_eq!(expected, model.decide(state, &q, &mut cached)?);
                    assert!(cached.cache_bytes() <= budget);
                }
                if budget <= 1 {
                    assert_eq!(cached.stats.choice_cache_hits, start_hits);
                }
                if budget == all_bytes {
                    assert_eq!(
                        cached.stats.choice_cache_hits - start_hits,
                        q.choices.len() as u64
                    );
                }
            }
            Ok(())
        })
    }
    #[test]
    #[ignore = "requires models/decima-small; run in release mode"]
    fn official_checkpoint_contract() -> Result<()> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        pool.install(||->Result<()> {
            let model=Model::load("models/decima-small",Options::default())?;
            let mut s=Session::default();
            for line in include_str!("../../../benches/decima-reference.jsonl").lines() {
                let v:Value=serde_json::from_str(line)?;let q=question(&v["question"]);
                let ids=model.input.choices(&q,&model.config.choice_prefix,model.config.question_in_choices,model.config.choice_fit)?;
                assert_eq!(serde_json::to_value(ids)?,v["choice_ids"],"tokenization {}",v["id"]);
                let p=model.decide(v["state"].as_str().unwrap(),&q,&mut s)?;
                let hits=s.stats.choice_cache_hits;
                let cached=model.decide(v["state"].as_str().unwrap(),&q,&mut s)?;
                assert_eq!(p,cached,"cache changes outputs");assert!(s.stats.choice_cache_hits>hits);
                for (p,e) in p.iter().zip(v["expected"].as_array().unwrap()) {assert!((*p as f64-e.as_f64().unwrap()).abs()<1e-4,"{}: {p} != {e}",v["id"]);}
            }
            let policy_fixture: Value = serde_json::from_str(include_str!("../../../benches/decima-reference.jsonl").lines().next().unwrap())?;
            let policy_question = question(&policy_fixture["question"]);
            let policy_state = policy_fixture["state"].as_str().unwrap();
            s.set_cache_bytes(1024 * 1024);
            let before_policy = model.decide(policy_state, &policy_question, &mut s)?;
            assert!(s.cache_bytes() > 0);
            let rules = serde_json::from_str(r#"{"bands":[{"through":64,"settings":{"projection_rows":3,"attention_scores":1024}}],"fallback":{"projection_rows":2,"int8_projection_rows":4,"attention_scores":4096}}"#)?;
            s.set_length_rules(Some(rules))?;
            assert_eq!(s.cache_bytes(), 0);
            assert_eq!(s.cache_limit, 1024 * 1024);
            let after_policy = model.decide(policy_state, &policy_question, &mut s)?;
            for (a, b) in before_policy.iter().zip(&after_policy) {
                assert!((a - b).abs() <= 1e-4, "length policy changes probability beyond the float bound");
            }
            let cached_bytes = s.cache_bytes();
            let invalid = serde_json::from_str(r#"{"bands":[],"fallback":{"projection_rows":0,"attention_scores":4096}}"#)?;
            assert!(s.set_length_rules(Some(invalid)).is_err());
            assert_eq!(s.cache_bytes(), cached_bytes, "rejected policy must not mutate the session");
            assert_eq!(after_policy, model.decide(policy_state, &policy_question, &mut s)?);
            s.set_length_rules(None)?;
            assert_eq!(s.cache_bytes(), 0);
            assert_eq!(s.cache_limit, 1024 * 1024);
            for line in include_str!("../../../benches/decima-systemone.jsonl").lines() {
                let v:Value=serde_json::from_str(line)?;let out=model.predict_json(&v["request"],&mut s)?;difference(&out,&v["expected"]);
            }
            let mut req=json!({"state":"record ".repeat(520),"questions":{"q":{"type":"choice","criteria":["yes","no"]}}});
            assert!(model.predict_json(&req,&mut s).unwrap_err().to_string().contains("STATE_TRUNCATED"));
            req["state"]=json!("hello");req["long"]=json!("chunk");assert!(model.predict_json(&req,&mut s).is_err());
            let fixture: Value = serde_json::from_str(include_str!("../../../benches/decima-reference.jsonl").lines().next().unwrap())?;
            let q = question(&fixture["question"]);
            model.decide(fixture["state"].as_str().unwrap(), &q, &mut s)?;
            drop(model);
            let precision = if cfg!(feature = "int8") { Precision::Int8 } else { Precision::F32 };
            let replacement = Model::load("models/decima-small", Options { precision, kernel:Kernel::Auto })?;
            let hits = s.stats.choice_cache_hits;
            let reused = replacement.decide(fixture["state"].as_str().unwrap(), &q, &mut s)?;
            let fresh = replacement.decide(fixture["state"].as_str().unwrap(), &q, &mut Session::default())?;
            assert_eq!(reused, fresh, "a session must not reuse another model's encodings");
            assert_eq!(hits, s.stats.choice_cache_hits);
            s.set_cache_bytes(0);assert_eq!(s.cache_bytes(),0);
            Ok(())
        })
    }
}

#[cfg(all(
    feature = "kernel-checks",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
pub(crate) fn wasm_kernel_checks() -> anyhow::Result<serde_json::Value> {
    #[cfg_attr(not(feature = "int8"), allow(unused_mut))]
    let mut checks = serde_json::json!({"projection_values":wasm_simd::check()?,"softmax":softmax_wasm::check()?,"gelu_values":activation_wasm::check()?,"attention":wasm_attention::check()?});
    #[cfg(feature = "int8")]
    {
        checks["int8_values"] = serde_json::json!(int8::wasm_check()?);
        checks["ffn_values"] = serde_json::json!(cpu::check_ffn_wasm()?);
    }
    Ok(checks)
}
