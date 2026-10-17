mod backbone;
mod checkpoint;
mod cpu;
mod input;
mod weights;
use anyhow::Result;
use backbone::{Backbone, State, Work};
use input::Input;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::Arc;

pub const REPOSITORY: &str = "jaredpalmer/kev-0.8b";
pub const REVISION: &str = "bf75a6a8848ea6960ff2ed108d9ed44c2941174f";
pub const BASE_REPOSITORY: &str = "Qwen/Qwen3.5-0.8B-Base";
pub const BASE_REVISION: &str = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68";
use anyhow::ensure;
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub projection_rows: usize,
    pub attention_queries: usize,
    pub prefill_tokens: usize,
}
impl Default for Plan {
    fn default() -> Self {
        Self {
            projection_rows: 6,
            attention_queries: 32,
            prefill_tokens: 256,
        }
    }
}
impl Plan {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=6).contains(&self.projection_rows)
                && (1..=64).contains(&self.attention_queries)
                && (1..=1024).contains(&self.prefill_tokens),
            "invalid Kev CPU length plan"
        );
        Ok(())
    }
}
#[derive(Default, Debug, Clone, Serialize)]
pub struct CacheStats {
    pub hits: usize,
    pub prefix_tokens_computed: usize,
    pub prefix_tokens_reused: usize,
    pub question_tokens_computed: usize,
}
// Snapshots include KV, convolution and recurrent state; capture during prefill, never rewind.
struct CachedPrefix {
    ids: Vec<u32>,
    state: State,
    checkpoint: Option<State>,
}
impl CachedPrefix {
    fn bytes(&self) -> usize {
        self.ids.len() * 4 + self.state.bytes() + self.checkpoint.as_ref().map_or(0, State::bytes)
    }
}
pub struct Session {
    work: Work,
    prefix: Option<CachedPrefix>,
    cache_bytes: usize,
    model: Option<Arc<()>>,
    rules: Option<crate::length::Rules<Plan>>,
    pub stats: CacheStats,
}
impl Default for Session {
    fn default() -> Self {
        Self {
            work: Work::default(),
            prefix: None,
            cache_bytes: 128 * 1024 * 1024,
            model: None,
            rules: None,
            stats: CacheStats::default(),
        }
    }
}
impl Session {
    pub fn clear_cache(&mut self) {
        self.prefix = None;
    }
    pub fn set_cache_bytes(&mut self, bytes: usize) {
        self.cache_bytes = bytes;
        if self.prefix.as_ref().is_some_and(|p| p.bytes() > bytes) {
            self.clear_cache();
        }
    }
    pub fn cached_bytes(&self) -> usize {
        self.prefix.as_ref().map_or(0, CachedPrefix::bytes)
    }
    pub fn set_length_rules(&mut self, rules: Option<crate::length::Rules<Plan>>) -> Result<()> {
        if let Some(r) = &rules {
            r.validate(Plan::validate)?;
        }
        self.rules = rules;
        self.clear_cache();
        Ok(())
    }
    fn plan(&self, n: usize) -> Plan {
        self.rules
            .as_ref()
            .map_or_else(Plan::default, |r| *r.select(n))
    }
}
pub struct Model {
    backbone: Backbone,
    input: Input,
    query: cpu::Linear,
    key: cpu::Linear,
    temperature: f32,
    identity: Arc<()>,
}
impl Model {
    #[cfg(feature = "cpu")]
    pub fn load(
        path: impl AsRef<std::path::Path>,
        base: impl AsRef<std::path::Path>,
        portable: bool,
    ) -> Result<Self> {
        let (p, b) = (path.as_ref(), base.as_ref());
        let file = std::fs::File::open(b.join("model.safetensors-00001-of-00001.safetensors"))?;
        let bytes = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let adapter = std::fs::File::open(p.join("adapter_model.safetensors"))?;
        let adapter = unsafe { memmap2::MmapOptions::new().map(&adapter)? };
        Self::from_bytes(
            &std::fs::read(b.join("config.json"))?,
            &std::fs::read(p.join("tokenizer.json"))?,
            &std::fs::read(p.join("adapter_config.json"))?,
            &adapter,
            &std::fs::read(p.join("head.pt"))?,
            &bytes,
            portable,
        )
    }
    pub fn from_bytes(
        config: &[u8],
        tokenizer: &[u8],
        adapter_config: &[u8],
        adapter: &[u8],
        head: &[u8],
        base: &[u8],
        portable: bool,
    ) -> Result<Self> {
        let head = checkpoint::Head::load(head)?;
        let avx2 = !portable && cpu::avx2_available();
        let weights =
            weights::Weights::new(base, adapter, &serde_json::from_slice(adapter_config)?)?;
        let backbone = Backbone::load(&weights, &serde_json::from_slice(config)?, avx2)?;
        weights.validate_consumed()?;
        Ok(Self {
            backbone,
            input: Input::new(tokenizer)?,
            query: cpu::Linear::new(head.q, head.qb, 1024, 256, avx2)?,
            key: cpu::Linear::new(head.k, head.kb, 1024, 256, avx2)?,
            temperature: head.temperature as f32,
            identity: Arc::new(()),
        })
    }
    fn prefix(&self, ids: &[u32], s: &mut Session) -> Result<State> {
        if !s
            .model
            .as_ref()
            .is_some_and(|m| Arc::ptr_eq(m, &self.identity))
        {
            s.clear_cache();
            s.model = Some(self.identity.clone());
        }
        let matched = s.prefix.as_ref().and_then(|cached| {
            if ids.starts_with(&cached.ids) {
                Some(&cached.state)
            } else {
                cached.checkpoint.as_ref().filter(|state| {
                    ids.len() >= state.position
                        && ids[..state.position] == cached.ids[..state.position]
                })
            }
        });
        let reused = matched.map_or(0, |state| state.position);
        let mut state = matched.map_or_else(|| self.backbone.empty_state(), State::clone);
        if reused > 0 {
            s.stats.hits += 1;
            s.stats.prefix_tokens_reused += reused;
            if reused == ids.len() {
                return Ok(state);
            }
        }
        let plan = s.plan(ids.len());
        let full_bytes = ids.len() * 4 + state.bytes_at(ids.len());
        let fixed_bytes = state.bytes_at(0);
        let per_token = state.bytes_at(1) - fixed_bytes;
        let checkpoint_at = s
            .cache_bytes
            .checked_sub(full_bytes + fixed_bytes)
            .map(|available| (available / per_token).min(ids.len().saturating_sub(8)))
            .filter(|&position| position > 0 && position >= reused);
        let mut checkpoint = if checkpoint_at == Some(reused) {
            Some(state.clone())
        } else if checkpoint_at.is_none() && full_bytes <= s.cache_bytes {
            s.prefix
                .as_ref()
                .and_then(|p| {
                    p.checkpoint.as_ref().filter(|old| {
                        old.position < ids.len()
                            && full_bytes + old.bytes() <= s.cache_bytes
                            && ids[..old.position] == p.ids[..old.position]
                    })
                })
                .cloned()
        } else {
            None
        };
        for chunk in ids[reused..].chunks(plan.prefill_tokens) {
            let capture = checkpoint_at
                .filter(|&at| at > state.position && at <= state.position + chunk.len());
            let [saved, _] = self.backbone.forward_saving(
                chunk,
                &mut state,
                &mut s.work,
                plan,
                [capture, None],
            )?;
            if saved.is_some() {
                checkpoint = saved;
            }
        }
        s.stats.prefix_tokens_computed += ids.len() - reused;
        s.clear_cache();
        if full_bytes <= s.cache_bytes {
            s.prefix = Some(CachedPrefix {
                ids: ids.to_vec(),
                state: state.clone(),
                checkpoint,
            });
        }
        Ok(state)
    }

    pub fn encode(&self, request: &Value) -> Result<Value> {
        self.input.describe(request)
    }
    pub fn decide_json(&self, request: &Value, s: &mut Session) -> Result<Value> {
        let encoded = self.input.encode(request)?;
        let prefix = self.prefix(&encoded.prefix, s)?;
        let mut answers = Map::new();
        let mut distributions = Map::new();
        let mut tokens = encoded.prefix.len();
        for q in &encoded.questions {
            let mut state = prefix.clone();
            let plan = s.plan(prefix.position + q.ids.len());
            let mut options = Vec::with_capacity(q.options.len() * 1024);
            let mut decision = Vec::new();
            let mut slot = 0;
            for (chunk_index, chunk) in q.ids.chunks(plan.prefill_tokens).enumerate() {
                self.backbone
                    .forward(chunk, &mut state, &mut s.work, plan)?;
                let begin = chunk_index * plan.prefill_tokens;
                while slot < q.options.len() && q.options[slot] < begin + chunk.len() {
                    let offset = (q.options[slot] - begin) * 1024;
                    options.extend_from_slice(&s.work.hidden[offset..offset + 1024]);
                    slot += 1;
                }
                if (begin..begin + chunk.len()).contains(&q.decide) {
                    let offset = (q.decide - begin) * 1024;
                    decision.extend_from_slice(&s.work.hidden[offset..offset + 1024]);
                }
            }
            let (mut query, mut keys) = (Vec::new(), Vec::new());
            self.query
                .apply(&decision, &mut query, plan.projection_rows);
            self.key.apply(&options, &mut keys, plan.projection_rows);
            let mut logits: Vec<f32> = keys
                .chunks_exact(256)
                .map(|k| {
                    k.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>() * (1. / 16.)
                        / self.temperature
                })
                .collect();
            cpu::softmax(&mut logits);
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "non-finite Kev probabilities"
            );
            let p: Vec<f64> = logits.into_iter().map(f64::from).collect();
            let best = (0..p.len())
                .max_by(|&i, &j| p[i].total_cmp(&p[j]).then_with(|| j.cmp(&i)))
                .unwrap();
            tokens += q.ids.len();
            s.stats.question_tokens_computed += q.ids.len();
            distributions.insert(q.id.clone(), json!(p));
            let dist = q
                .names
                .iter()
                .cloned()
                .zip(p.iter().map(|x| json!(round(*x))))
                .collect::<Map<_, _>>();
            let total: f64 = p.iter().sum();
            let confidence = if p.len() == 1 {
                1.
            } else {
                (p[best] / total - 1. / p.len() as f64) / (1. - 1. / p.len() as f64)
            };
            let answer = match q.kind.as_str() {
                "noul" => json!({"type":"noul","noul":round(p[1])}),
                "choice" => {
                    json!({"type":"choice","choice":q.names[best],"confidence":round(confidence),"probabilities":dist})
                }
                _ => {
                    let confidence = if p.len() == 1 {
                        1.
                    } else {
                        let center = (p.len() - 1) as f64 / 2.;
                        let d = (0..p.len()).map(|i| (i as f64 - center).abs()).sum::<f64>()
                            / p.len() as f64;
                        (1. - p
                            .iter()
                            .enumerate()
                            .map(|(i, x)| x / total * i.abs_diff(best) as f64)
                            .sum::<f64>()
                            / d)
                            .max(0.)
                    };
                    json!({"type":"score","score":round(p.iter().enumerate().map(|(i,x)|i as f64*x).sum()),"legend":q.legend,"probabilities":dist,"confidence":round(confidence)})
                }
            };
            answers.insert(q.id.clone(), answer);
        }
        let output_tokens = self.input.output_tokens(&json!(answers))?;
        Ok(
            json!({"model":"kev-0.8b","answers":answers,"probabilities":distributions,"usage":{"input_tokens":tokens,"state_tokens":encoded.prefix.len(),"output_tokens":output_tokens},"backend":"rust-cpu"}),
        )
    }
    pub fn predict_json(&self, request: &Value, s: &mut Session) -> Result<Value> {
        let mut out = self.decide_json(request, s)?;
        out.as_object_mut().unwrap().shift_remove("probabilities");
        Ok(out)
    }
}
fn round(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap()
}

#[cfg(all(test, feature = "cpu"))]
mod tests;
