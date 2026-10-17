use anyhow::{Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

#[derive(Clone, Deserialize, Serialize)]
pub struct Request {
    pub state: Value,
    pub questions: Map<String, Value>,
    #[serde(default)]
    pub max_len: Option<usize>,
    #[serde(default)]
    pub head_max_len: Option<usize>,
    #[serde(default)]
    pub lang: Option<String>,
    #[serde(default)]
    pub min_confidence: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: usize,
    pub labels: Vec<Value>,
    pub descriptions: Vec<String>,
    pub order: Vec<usize>,
    pub state_tokens: usize,
    pub state_tokens_used: usize,
    pub options_distinct: usize,
    pub tokens_per_option: Option<usize>,
}

pub struct Input {
    tokenizer: Tokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
    mask_text: String,
    cleanup: bool,
}

pub fn render(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => json_spaced(v),
    }
}
fn json_spaced(v: &Value) -> String {
    match v {
        Value::Array(v) => format!(
            "[{}]",
            v.iter().map(json_spaced).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(v) => format!(
            "{{{}}}",
            v.iter()
                .map(|(k, v)| format!("{}: {}", json!(k), json_spaced(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => v.to_string(),
    }
}
fn label_text(v: &Value) -> String {
    match v {
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        _ => render(v),
    }
}

fn same_label(a: &Value, b: &Value) -> bool {
    let integer = |v: &Value| match v {
        Value::Bool(v) => Some(i128::from(*v)),
        Value::Number(n) => n
            .as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from)),
        _ => None,
    };
    if a == b {
        return true;
    }
    match (integer(a), integer(b)) {
        (Some(a), Some(b)) => a == b,
        (Some(a), None) => b
            .as_f64()
            .is_some_and(|b| b.fract() == 0.0 && b as i128 == a),
        (None, Some(b)) => a
            .as_f64()
            .is_some_and(|a| a.fract() == 0.0 && a as i128 == b),
        _ => a.as_f64().zip(b.as_f64()).is_some_and(|(a, b)| a == b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_label_equality_preserves_integer_precision() {
        assert!(same_label(&json!(true), &json!(1)));
        assert!(same_label(&json!(1), &json!(1.0)));
        assert!(!same_label(&json!("1"), &json!(1)));
        assert!(!same_label(
            &json!(9007199254740993_u64),
            &json!(9007199254740992_f64)
        ));
    }
    #[test]
    fn confidence_thresholds_validate_and_gate_by_option_count() {
        assert!(validate_threshold(&json!(true)).is_err());
        assert!(validate_threshold(&json!({"choice:bad":0.5})).is_err());
        let thresholds = json!({"choice:3-5":0.7,"default":0.0});
        validate_threshold(&thresholds).unwrap();
        let mut answer = json!({"answer_confidence":0.65});
        gate(&mut answer, &thresholds, 0, 3);
        assert_eq!(answer["abstention"], "abstained");
        assert_eq!(answer["abstention_threshold"], 0.7);
    }
}

pub fn validate_threshold(value: &Value) -> Result<()> {
    let valid = |v: &Value| v.as_f64().is_some_and(|v| (0.0..=1.0).contains(&v));
    if let Some(map) = value.as_object() {
        ensure!(!map.is_empty(), "min_confidence map must not be empty");
        for (key, threshold) in map {
            ensure!(
                key == "default"
                    || (0..3).any(|q| [2, 3, 6, 11].iter().any(|&k| bucket(q, k) == *key)),
                "unknown confidence bucket {key}"
            );
            ensure!(
                valid(threshold),
                "confidence threshold must be a number in [0, 1]"
            );
        }
    } else {
        ensure!(
            valid(value),
            "confidence threshold must be a number in [0, 1]"
        );
    }
    Ok(())
}

pub fn gate(answer: &mut Value, thresholds: &Value, qtype: usize, k: usize) {
    let threshold = if thresholds.is_object() {
        thresholds
            .get(bucket(qtype, k))
            .or_else(|| thresholds.get("default"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    } else {
        thresholds.as_f64().unwrap()
    };
    let low = answer["answer_confidence"].as_f64().unwrap() < threshold;
    answer["abstention"] = json!(if low { "abstained" } else { "passed" });
    answer["abstention_threshold"] = json!(threshold);
    if low {
        answer["low_confidence"] = json!(true);
    }
}

impl Input {
    pub fn new(tokenizer: &[u8], config: &[u8]) -> Result<Self> {
        #[cfg(target_arch = "wasm32")]
        tokenizers::utils::parallelism::set_parallelism(false);
        let mut tokenizer =
            Tokenizer::from_bytes(tokenizer).map_err(|e| anyhow!("tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let cfg: Value = serde_json::from_slice(config)?;
        let token = |name: &str| -> Result<(String, u32)> {
            let v = &cfg[name];
            let text = v
                .as_str()
                .or_else(|| v["content"].as_str())
                .ok_or_else(|| anyhow!("missing {name}"))?
                .to_owned();
            let id = tokenizer
                .token_to_id(&text)
                .ok_or_else(|| anyhow!("unknown {name}: {text}"))?;
            Ok((text, id))
        };
        let (_, cls) = token("cls_token")?;
        let (_, sep) = token("sep_token")?;
        let (mask_text, mask) = token("mask_token")?;
        Ok(Self {
            tokenizer,
            cls,
            sep,
            mask,
            mask_text,
            cleanup: cfg["clean_up_tokenization_spaces"]
                .as_bool()
                .unwrap_or(false),
        })
    }
    pub fn text(&self, ids: &[u32]) -> Result<String> {
        let mut text = self
            .tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow!("decode: {e}"))?;
        if self.cleanup {
            for (from, to) in [
                (" .", "."),
                (" ?", "?"),
                (" !", "!"),
                (" ,", ","),
                (" ' ", "'"),
                (" n't", "n't"),
                (" 'm", "'m"),
                (" 's", "'s"),
                (" 've", "'ve"),
                (" 're", "'re"),
            ] {
                text = text.replace(from, to);
            }
        }
        Ok(text)
    }
    pub fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        let sanitized = text.replace(&self.mask_text, " ");
        Ok(self
            .tokenizer
            .encode(sanitized, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn encode(
        &self,
        state: &Value,
        questions: &Map<String, Value>,
        max_len: usize,
        head_max_len: usize,
    ) -> Result<Vec<Encoded>> {
        ensure!(
            max_len > 0 && head_max_len > 0,
            "token budgets must be positive"
        );
        let state_tokens = self.tokens(&render(state))?;
        questions
            .iter()
            .map(|(qid, q)| {
                self.encode_question(
                    qid,
                    q,
                    &state_tokens,
                    state.is_array(),
                    max_len,
                    head_max_len,
                )
            })
            .collect()
    }
    pub fn encode_question(
        &self,
        qid: &str,
        q: &Value,
        state: &[u32],
        truncate_left: bool,
        max_len: usize,
        head_max_len: usize,
    ) -> Result<Encoded> {
        ensure!(!qid.trim().is_empty(), "question id is empty");
        let t = q["type"]
            .as_str()
            .ok_or_else(|| anyhow!("question {qid}: missing type"))?;
        let qtype = match t {
            "choice" => 0,
            "score" => 1,
            "noul" => 2,
            _ => bail!("question {qid}: unknown type {t}"),
        };
        ensure!(
            !q["instructions"].is_null(),
            "question {qid}: missing instructions"
        );
        let instructions = render(&q["instructions"]);
        ensure!(
            !instructions.trim().is_empty()
                && !q["instructions"].as_array().is_some_and(Vec::is_empty)
                && !q["instructions"].as_object().is_some_and(Map::is_empty),
            "question {qid}: empty instructions"
        );
        ensure!(
            qtype == 2 || q.get("labels").is_none(),
            "labels only apply to noul"
        );
        let mut labels = Vec::new();
        let mut descriptions = Vec::new();
        let mut texts = Vec::new();
        let normalized;
        let crit = if qtype == 2 && q["criteria"].is_object() {
            normalized = Value::Object(
                q["criteria"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.to_lowercase(), v.clone()))
                    .collect(),
            );
            &normalized
        } else {
            &q["criteria"]
        };
        match qtype {
            0 => {
                let entries: Vec<(Value, Value)> = if let Some(a) = crit.as_array() {
                    a.iter().map(|v| (v.clone(), Value::Null)).collect()
                } else if let Some(o) = crit.as_object() {
                    o.iter().map(|(k, v)| (json!(k), v.clone())).collect()
                } else {
                    bail!("question {qid}: choice criteria must be an object or array")
                };
                for (label, description) in entries {
                    ensure!(
                        label.is_string() || label.is_number() || label.is_boolean(),
                        "invalid choice label"
                    );
                    ensure!(
                        !labels.iter().any(|v| same_label(v, &label)),
                        "duplicate choice label"
                    );
                    let text = label_text(&label);
                    texts.push(if description.is_null() || description == "" {
                        text
                    } else {
                        format!("{text}: {}", render(&description))
                    });
                    labels.push(label);
                    descriptions.push(render(&description));
                }
            }
            1 => {
                let a = crit
                    .as_array()
                    .ok_or_else(|| anyhow!("question {qid}: score criteria must be an array"))?;
                for (i, v) in a.iter().enumerate() {
                    ensure!(!v.is_null(), "null score level");
                    let desc = render(v);
                    texts.push(format!("level {i}: {desc}"));
                    labels.push(json!(i.to_string()));
                    descriptions.push(desc);
                }
            }
            _ => {
                ensure!(
                    crit.is_null() || crit.is_object(),
                    "noul criteria must be an object"
                );
                if let Some(o) = crit.as_object() {
                    ensure!(
                        o.keys().all(|k| k == "false" || k == "true"),
                        "unknown noul criterion"
                    );
                }
                let custom_labels = q.get("labels").filter(|v| !v.is_null());
                if let Some(l) = custom_labels {
                    ensure!(
                        l.as_object().is_some_and(|o| o.len() == 2
                            && o.contains_key("false")
                            && o.contains_key("true")),
                        "noul labels must contain false and true"
                    );
                }
                for (key, fallback) in [
                    ("false", "no, the statement does not hold"),
                    ("true", "yes, the statement holds"),
                ] {
                    let label = if custom_labels.is_some() {
                        q["labels"][key]
                            .as_str()
                            .ok_or_else(|| anyhow!("noul label must be a string"))?
                            .trim()
                    } else {
                        key
                    };
                    ensure!(!label.is_empty(), "empty noul label");
                    let desc = if crit[key].is_null() || crit[key] == "" {
                        fallback.into()
                    } else {
                        render(&crit[key])
                    };
                    texts.push(format!("{label}: {desc}"));
                    labels.push(json!(key));
                    descriptions.push(desc);
                }
                if custom_labels.is_some() {
                    ensure!(
                        q["labels"]["false"].as_str().unwrap().trim()
                            != q["labels"]["true"].as_str().unwrap().trim(),
                        "duplicate noul labels"
                    );
                }
            }
        }
        ensure!(!texts.is_empty(), "question {qid}: no options");
        let order = if let Some(v) = q.get("option_order") {
            let order: Vec<usize> = serde_json::from_value(v.clone())?;
            let mut sorted = order.clone();
            sorted.sort_unstable();
            ensure!(
                sorted == (0..texts.len()).collect::<Vec<_>>(),
                "invalid option permutation"
            );
            order
        } else {
            (0..texts.len()).collect()
        };
        let mut head = self.tokens(&format!("{t} question: {instructions}"))?;
        let mut options: Vec<Vec<u32>> = order
            .iter()
            .map(|&i| {
                let mut v = vec![self.mask];
                v.extend(self.tokens(&format!(" {}", texts[i]))?.into_iter().take(48));
                Ok(v)
            })
            .collect::<Result<_>>()?;
        let mut budget =
            head_max_len as isize - options.iter().map(Vec::len).sum::<usize>() as isize;
        let mut tokens_per_option = None;
        if budget < 16 {
            let per =
                ((head_max_len as isize - 16).div_euclid(options.len() as isize)).max(4) as usize;
            tokens_per_option = Some(per);
            for v in &mut options {
                v.truncate(per);
            }
            budget = head_max_len as isize - options.iter().map(Vec::len).sum::<usize>() as isize;
        }
        head.truncate(budget.max(8) as usize);
        let options_distinct = options
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        let mut ids = vec![self.cls];
        ids.extend(head);
        ids.push(self.sep);
        let mut markers = Vec::new();
        for o in options {
            markers.push(ids.len());
            ids.extend(o);
        }
        ids.push(self.sep);
        let room = max_len.saturating_sub(ids.len() + 1).min(state.len());
        let start = if truncate_left { state.len() - room } else { 0 };
        ids.extend_from_slice(&state[start..start + room]);
        ids.push(self.sep);
        ids.truncate(max_len);
        ensure!(
            markers.iter().all(|&p| p < ids.len()),
            "question {qid}: options exceed max_len={max_len}"
        );
        Ok(Encoded {
            ids,
            markers,
            qtype,
            labels,
            descriptions,
            order,
            state_tokens: state.len(),
            state_tokens_used: room,
            options_distinct,
            tokens_per_option,
        })
    }
}

pub fn bucket(qtype: usize, k: usize) -> String {
    let kind = ["choice", "score", "noul"][qtype];
    let count = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{kind}:{count}")
}
pub fn round4(x: f64) -> f64 {
    (x * 10000.0).round_ties_even() / 10000.0
}
pub fn decode(encoded: &Encoded, logits: &[f32], act: f32, temperature: f32) -> Value {
    let mut p: Vec<f32> = logits.iter().map(|v| v / temperature).collect();
    crate::laya::cpu::softmax(&mut p);
    let mut probabilities = vec![0.0; p.len()];
    for (slot, &option) in encoded.order.iter().enumerate() {
        probabilities[option] = p[slot];
    }
    let p = probabilities;
    let best = p
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .unwrap()
        .0;
    let confidence = if p.len() == 1 {
        1.0
    } else {
        let ent = -p.iter().map(|&v| v * v.clamp(1e-12, 1.0).ln()).sum::<f32>();
        (1.0 - ent / (p.len() as f32).ln()).clamp(0.0, 1.0)
    };
    let mut probs = Map::new();
    for (k, &v) in encoded.labels.iter().zip(&p) {
        probs.insert(render(k), json!(round4(v as f64)));
    }
    let kind = ["choice", "score", "noul"][encoded.qtype];
    let mut answer = json!({"type":kind,"confidence":round4(confidence as f64),"answer_confidence":round4(p[best] as f64),"action":{"act_probability":round4(act as f64)}});
    match encoded.qtype {
        0 => {
            answer["choice"] = encoded.labels[best].clone();
            answer["probabilities"] = json!(probs);
        }
        1 => {
            answer["score"] = json!(round4(
                p.iter()
                    .enumerate()
                    .map(|(i, &v)| i as f64 * v as f64)
                    .sum()
            ));
            answer["probabilities"] = json!(probs);
            answer["legend"] = Value::Object(
                encoded
                    .descriptions
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (i.to_string(), json!(s)))
                    .collect(),
            );
        }
        _ => {
            answer["noul"] = json!(round4(p[1] as f64));
            answer["confidence"] = json!(round4((p[1] as f64).max(1.0 - p[1] as f64)));
        }
    }
    answer
}
