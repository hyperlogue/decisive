mod backbone;
mod cpu;
mod input;
mod weights;
use anyhow::{Context, Result, ensure};
use backbone::{Backbone, State, Work};
use input::{Input, Question};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::Arc;

pub const REPOSITORY: &str = "chaoliangUNSW/Jev-Style-0.8B-Decision-v3";
pub const REVISION: &str = "b023d1f9c7858fbf01504577a3bfc349ea5c7385";
pub const MAX_TOKENS: usize = 25_600;
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
            "invalid Jev CPU length plan"
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
    readout: Value,
    direction: Vec<f32>,
    identity: Arc<()>,
}
impl Model {
    #[cfg(feature = "cpu")]
    pub fn load(path: impl AsRef<std::path::Path>, portable: bool) -> Result<Self> {
        let p = path.as_ref();
        let file = std::fs::File::open(p.join("model.safetensors"))?;
        let bytes = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Self::from_bytes(
            &std::fs::read(p.join("config.json"))?,
            &std::fs::read(p.join("tokenizer.json"))?,
            &std::fs::read(p.join("readout_config.json"))?,
            &bytes,
            portable,
        )
    }
    pub fn from_bytes(
        config: &[u8],
        tokenizer: &[u8],
        readout: &[u8],
        weights: &[u8],
        portable: bool,
    ) -> Result<Self> {
        let readout: Value = serde_json::from_slice(readout)?;
        let input = Input::new(tokenizer, &readout)?;
        let backbone = Backbone::load(
            &weights::Weights::new(weights)?,
            &serde_json::from_slice(config)?,
            !portable && cpu::avx2_available(),
        )?;
        let yes = backbone.embedding_row(
            readout["slot_tokens"]["yes"]["id"]
                .as_u64()
                .context("yes token")? as usize,
        )?;
        let no = backbone.embedding_row(
            readout["slot_tokens"]["no"]["id"]
                .as_u64()
                .context("no token")? as usize,
        )?;
        let direction = yes.into_iter().zip(no).map(|(a, b)| a - b).collect();
        Ok(Self {
            backbone,
            input,
            readout,
            direction,
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
    fn temperature(
        &self,
        q: &Question,
        category: Option<&str>,
        override_t: Option<f64>,
    ) -> Result<f64> {
        if let Some(t) = override_t {
            ensure!(
                t.is_finite() && t > 0.,
                "temperature must be finite and positive"
            );
            return Ok(t);
        }
        let temps = &self.readout["temperatures"];
        let mut t = temps["global"].as_f64().context("global temperature")?;
        if let Some(category) = category.filter(|c| !c.is_empty()) {
            let family = [
                ("typed_official", "typed"),
                ("typed_synthetic", "typed_synth"),
                ("general_", "general"),
                ("intent", "intent"),
                ("nli", "nli"),
                ("theme_", "theme"),
                ("mac_", "mac"),
                ("long_", "long"),
            ]
            .iter()
            .find(|(p, _)| category.starts_with(p))
            .map_or("other", |(_, f)| *f);
            let bucket = match q.names.len() {
                0..=2 => "2",
                3..=5 => "3-5",
                6..=10 => "6-10",
                11..=20 => "11-20",
                _ => "21+",
            };
            if let Some(group) =
                temps["groups"][format!("{family}|{}|{bucket}", q.kind)]["T"].as_f64()
            {
                t = group;
            }
        }
        let lo = temps["clamp"][0].as_f64().context("temperature clamp")?;
        let hi = temps["clamp"][1].as_f64().context("temperature clamp")?;
        ensure!(
            lo > 0. && lo <= hi && t.is_finite(),
            "invalid temperature table"
        );
        Ok(t.clamp(lo, hi))
    }
    fn score(
        &self,
        q: &Question,
        prefix: &State,
        s: &mut Session,
        category: Option<&str>,
        temperature: Option<f64>,
        head_max: usize,
    ) -> Result<Value> {
        let parts = self.input.render(q, head_max)?;
        ensure!(
            parts
                .iter()
                .all(|r| prefix.position + r.ids.len() <= MAX_TOKENS),
            "input exceeds {MAX_TOKENS} tokens; nothing truncated"
        );
        let mut scores = Vec::new();
        let mut names = Vec::new();
        let mut head_tokens = 0;
        for r in &parts {
            let mut state = prefix.clone();
            let plan = s.plan(prefix.position + r.ids.len());
            let mut slot = 0;
            for (chunk_index, chunk) in r.ids.chunks(plan.prefill_tokens).enumerate() {
                self.backbone
                    .forward(chunk, &mut state, &mut s.work, plan)?;
                let begin = chunk_index * plan.prefill_tokens;
                while slot < r.slots.len() && r.slots[slot] < begin + chunk.len() {
                    let row = &s.work.hidden
                        [(r.slots[slot] - begin) * 1024..(r.slots[slot] - begin + 1) * 1024];
                    let score = row
                        .iter()
                        .zip(&self.direction)
                        .map(|(h, w)| (*h as f64) * (*w as f64))
                        .sum::<f64>() as f32;
                    scores.push(score as f64);
                    slot += 1;
                }
            }
            names.extend(r.names.clone());
            head_tokens += r.ids.len();
            s.stats.question_tokens_computed += r.ids.len();
        }
        let t = self.temperature(q, category, temperature)?;
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = scores.iter().map(|v| ((v - max) / t).exp()).collect();
        let sum: f64 = probs.iter().sum();
        ensure!(sum.is_finite() && sum > 0., "non-finite decision scores");
        for p in &mut probs {
            *p /= sum;
        }
        let best = (0..probs.len())
            .max_by(|&i, &j| probs[i].total_cmp(&probs[j]).then_with(|| j.cmp(&i)))
            .unwrap();
        let concentration = if probs.len() < 2 {
            1.
        } else {
            (1. + probs
                .iter()
                .map(|p| p * p.clamp(1e-12, 1.).ln())
                .sum::<f64>()
                / (probs.len() as f64).ln())
            .clamp(0., 1.)
        };
        let mut out = json!({"answer":names[best],"probabilities":names.iter().cloned().zip(probs.iter().copied().map(Value::from)).collect::<Map<_,_>>(),"scores":names.iter().cloned().zip(scores.into_iter().map(Value::from)).collect::<Map<_,_>>(),"temperature":t,"top_probability":probs[best],"entropy_concentration":concentration,"input_tokens":prefix.position*parts.len()+head_tokens,"head_tokens":head_tokens,"model":"Jev-Style-0.8B-Decision-v3","backend":"rust-cpu"});
        if parts.len() > 1 {
            out["option_chunks"] = json!(parts.len());
        }
        Ok(out)
    }
    pub fn decide(
        &self,
        state: &Value,
        question: &Value,
        category: Option<&str>,
        temperature: Option<f64>,
        session: &mut Session,
    ) -> Result<Value> {
        let q = Question::raw(question)?;
        let ids = self.input.prefix(state)?;
        let parts = self.input.render(&q, 2048)?;
        ensure!(
            parts.iter().all(|p| ids.len() + p.ids.len() <= MAX_TOKENS),
            "input budget exceeded; nothing truncated"
        );
        let prefix = self.prefix(&ids, session)?;
        self.score(&q, &prefix, session, category, temperature, 2048)
    }
    pub fn encode(&self, state: &Value, question: &Value) -> Result<Value> {
        let prefix = self.input.prefix(state)?;
        let q = Question::raw(question)?;
        let parts = self.input.render(&q, 2048)?;
        Ok(
            json!({"prefix":prefix,"parts":parts.into_iter().map(|p|json!({"ids":p.ids,"slots":p.slots,"names":p.names})).collect::<Vec<_>>() }),
        )
    }
    pub fn predict_json(&self, request: &Value, s: &mut Session) -> Result<Value> {
        let state = request.get("state").context("missing state")?;
        ensure!(
            state.is_string() || state.is_object() || state.is_array(),
            "state must be a string, object or array"
        );
        let questions = request["questions"]
            .as_object()
            .context("questions must be an object")?;
        ensure!(!questions.is_empty(), "empty questions");
        let questions = questions
            .iter()
            .map(|(id, q)| Ok((id.clone(), Question::systemone(q)?)))
            .collect::<Result<Vec<_>>>()?;
        let ids = self.input.prefix(state)?;
        for (_, q) in &questions {
            ensure!(
                self.input
                    .render(q, 2048)?
                    .iter()
                    .all(|p| ids.len() + p.ids.len() <= MAX_TOKENS),
                "input budget exceeded; nothing truncated"
            );
        }
        let prefix = self.prefix(&ids, s)?;
        let mut answers = Map::new();
        let mut heads = 0;
        let mut reported_state_tokens = None;
        for (id, q) in questions {
            let r = self.score(&q, &prefix, s, Some("typed_official"), None, 2048)?;
            heads += r["head_tokens"].as_u64().unwrap();
            reported_state_tokens.get_or_insert(
                r["input_tokens"].as_u64().unwrap() - r["head_tokens"].as_u64().unwrap(),
            );
            let probs = &r["probabilities"];
            let p: Vec<f64> = q.names.iter().map(|n| probs[n].as_f64().unwrap()).collect();
            let total: f64 = p.iter().sum();
            let p: Vec<f64> = p.iter().map(|x| x / total).collect();
            let confidence = if p.len() == 1 {
                1.
            } else {
                ((p.len() as f64 * p.iter().copied().fold(0., f64::max) - 1.)
                    / (p.len() - 1) as f64)
                    .clamp(0., 1.)
            };
            let dist = q
                .names
                .iter()
                .cloned()
                .zip(p.iter().copied().map(Value::from))
                .collect::<Map<_, _>>();
            let answer = match q.kind.as_str() {
                "noul" => json!({"type":"noul","noul":p[1]}),
                "choice" => {
                    json!({"type":"choice","choice":r["answer"],"confidence":confidence,"probabilities":dist})
                }
                _ => {
                    json!({"type":"score","score":p.iter().enumerate().map(|(i,p)|i as f64*p).sum::<f64>(),"confidence":confidence,"legend":q.legend,"probabilities":dist})
                }
            };
            answers.insert(id, answer);
        }
        Ok(
            json!({"model":"jev-style-0.8b-v3","answers":answers,"usage":{"input_tokens":reported_state_tokens.unwrap()+heads,"state_tokens":reported_state_tokens.unwrap(),"output_tokens":0},"backend":"rust-cpu"}),
        )
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests;
