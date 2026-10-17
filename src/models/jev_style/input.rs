use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

pub(super) struct Input {
    tok: Tokenizer,
    dash: Vec<u32>,
    newline: Vec<u32>,
    judge: Vec<u32>,
    arrow: u32,
}
#[derive(Clone)]
pub(super) struct Question {
    pub kind: String,
    pub instruction: String,
    pub names: Vec<String>,
    pub options: Vec<String>,
    pub legend: Map<String, Value>,
}
pub(super) struct Rendered {
    pub ids: Vec<u32>,
    pub slots: Vec<usize>,
    pub names: Vec<String>,
}

pub(super) fn python_json(value: &Value, compact: bool) -> String {
    let (comma, colon) = if compact { (",", ":") } else { (", ", ": ") };
    match value {
        Value::Array(v) => format!(
            "[{}]",
            v.iter()
                .map(|v| python_json(v, compact))
                .collect::<Vec<_>>()
                .join(comma)
        ),
        Value::Object(v) => format!(
            "{{{}}}",
            v.iter()
                .map(|(k, v)| format!(
                    "{}{colon}{}",
                    serde_json::to_string(k).unwrap(),
                    python_json(v, compact)
                ))
                .collect::<Vec<_>>()
                .join(comma)
        ),
        Value::Number(n) if n.is_f64() => {
            let s = format!("{:?}", n.as_f64().unwrap());
            if let Some((m, e)) = s.split_once('e') {
                let e: i32 = e.parse().unwrap();
                format!(
                    "{m}e{}{abs:02}",
                    if e < 0 { "-" } else { "+" },
                    abs = e.abs()
                )
            } else {
                s
            }
        }
        _ => value.to_string(),
    }
}
fn text(v: &Value) -> String {
    v.as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| python_json(v, false))
}
fn description(v: &Value) -> bool {
    v.is_null() || v.is_string() || v.is_object() || v.is_array()
}
impl Question {
    pub fn raw(q: &Value) -> Result<Self> {
        let kind = q["t"].as_str().context("question.t")?.to_owned();
        let instruction = q["ins"]
            .as_str()
            .context("question.ins must be a string")?
            .to_owned();
        ensure!(!instruction.trim().is_empty(), "question text is empty");
        let c = &q["crit"];
        let mut legend = Map::new();
        let (names, options) = match kind.as_str() {
            "choice" => {
                let c = c.as_object().context("choice criteria must be an object")?;
                ensure!(!c.is_empty(), "empty choices");
                (
                    c.keys().cloned().collect(),
                    c.iter()
                        .map(|(k, v)| {
                            if v.is_null() || v == "" {
                                k.clone()
                            } else {
                                format!("{k}: {}", text(v))
                            }
                        })
                        .collect(),
                )
            }
            "score" => {
                let c = c.as_array().context("score criteria must be an array")?;
                ensure!((2..=10).contains(&c.len()), "score requires 2..10 levels");
                for (i, v) in c.iter().enumerate() {
                    legend.insert(i.to_string(), json!(text(v)));
                }
                (
                    (0..c.len()).map(|i| i.to_string()).collect(),
                    c.iter()
                        .enumerate()
                        .map(|(i, v)| format!("level {i}: {}", text(v)))
                        .collect(),
                )
            }
            "noul" => {
                ensure!(
                    c.is_null() || c.is_object(),
                    "noul criteria must be an object or null"
                );
                let opts = [
                    ("false", "no, the statement does not hold"),
                    ("true", "yes, the statement holds"),
                ]
                .map(|(key, default)| {
                    format!(
                        "{key}: {}",
                        if c[key].is_null() || c[key] == "" {
                            default.to_owned()
                        } else {
                            text(&c[key])
                        }
                    )
                });
                (vec!["false".into(), "true".into()], opts.into())
            }
            _ => anyhow::bail!("unsupported question type {kind}"),
        };
        Ok(Self {
            kind,
            instruction,
            names,
            options,
            legend,
        })
    }
    pub fn systemone(q: &Value) -> Result<Self> {
        ensure!(q.is_object(), "question must be an object");
        let kind = q["type"].as_str().context("question.type")?;
        let ins = &q["instructions"];
        let instruction = if let Some(s) = ins.as_str() {
            s.to_owned()
        } else {
            ensure!(
                ins.as_object().is_some_and(|v| !v.is_empty())
                    || ins.as_array().is_some_and(|v| !v.is_empty()),
                "instructions must be a nonempty string, object or array"
            );
            python_json(ins, true)
        };
        let mut criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
        let mut legend = Map::new();
        match kind {
            "choice" => {
                let c = criteria.as_object().context("choice criteria")?;
                ensure!(
                    (1..=255).contains(&c.len()),
                    "choice requires 1..255 options"
                );
                ensure!(
                    c.iter()
                        .all(|(k, v)| !k.trim().is_empty() && description(v)),
                    "invalid choice name/description"
                );
            }
            "noul" => {
                ensure!(
                    criteria.is_null()
                        || criteria.as_object().is_some_and(|c| c
                            .iter()
                            .all(|(k, v)| (k == "false" || k == "true") && description(v))),
                    "invalid noul criteria"
                );
            }
            "score" => {
                let c = criteria.as_array_mut().context("score criteria")?;
                for (i, v) in c.iter_mut().enumerate() {
                    ensure!(
                        v.as_str().is_some_and(|s| !s.trim().is_empty())
                            || v.as_array().is_some_and(|v| !v.is_empty())
                            || v.as_object().is_some_and(|v| !v.is_empty()),
                        "invalid score level"
                    );
                    let label = if let Some(o) = v.as_object() {
                        if o.contains_key("label") {
                            let l = o["label"]
                                .as_str()
                                .context("score label must be a string")?;
                            ensure!(
                                !l.trim().is_empty()
                                    && (o
                                        .get("description")
                                        .is_none_or(|d| d.is_null() || d.is_string())),
                                "invalid score label/description"
                            );
                            l.to_owned()
                        } else {
                            python_json(v, true)
                        }
                    } else {
                        v.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| python_json(v, true))
                    };
                    legend.insert(i.to_string(), json!(label));
                    if let Some(o) = v.as_object() {
                        if o.contains_key("label")
                            && o.keys().all(|k| k == "label" || k == "description")
                        {
                            let label = o["label"].as_str().unwrap();
                            let d = o.get("description").and_then(Value::as_str).unwrap_or("");
                            *v = json!(if d.is_empty() {
                                label.to_owned()
                            } else {
                                format!("{label}: {d}")
                            });
                        }
                    }
                }
            }
            _ => anyhow::bail!("unsupported question type {kind}"),
        }
        let mut q = Self::raw(&json!({"t":kind,"ins":instruction,"crit":criteria}))?;
        if kind == "score" {
            q.legend = legend;
        }
        Ok(q)
    }
}
impl Input {
    pub fn new(bytes: &[u8], cfg: &Value) -> Result<Self> {
        ensure!(
            cfg["format"] == "macjev-readout-v1"
                && cfg["template"] == "macjev-render-v1"
                && cfg["readout"] == "verdict",
            "unsupported Jev readout"
        );
        let mut tok = Tokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        tok.set_encode_special_tokens(true);
        tok.with_padding(None);
        tok.with_truncation(None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(tok
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec())
        };
        for (key, text) in [("yes", " yes"), ("no", " no"), ("verdict_slot", " ->")] {
            let id = cfg["slot_tokens"][key]["id"]
                .as_u64()
                .context("slot token ID")? as u32;
            ensure!(encode(text)? == [id], "readout token mismatch: {key}");
        }
        let (dash, newline, judge) = (
            encode("- ")?,
            encode("\n")?,
            encode("Judge each option:\n")?,
        );
        let arrow = cfg["slot_tokens"]["verdict_slot"]["id"].as_u64().unwrap() as u32;
        Ok(Self {
            tok,
            dash,
            newline,
            judge,
            arrow,
        })
    }
    pub fn encode(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn prefix(&self, state: &Value) -> Result<Vec<u32>> {
        let mut ids = self.encode("State:\n")?;
        ids.extend(self.encode(&text(state))?);
        ids.extend(self.encode("\n\n")?);
        Ok(ids)
    }
    pub fn render(&self, q: &Question, head_max: usize) -> Result<Vec<Rendered>> {
        ensure!(
            (1..=2048).contains(&head_max),
            "head budget must be 1..2048"
        );
        let options = q
            .options
            .iter()
            .map(|s| self.encode(s))
            .collect::<Result<Vec<_>>>()?;
        let introduction = self.encode(&format!(
            "Question [{}]: {}\nOptions:\n",
            q.kind, q.instruction
        ))?;
        let fixed = introduction.len() + self.judge.len();
        let costs: Vec<usize> = options
            .iter()
            .map(|o| self.dash.len() + 2 * o.len() + 1 + 2 * self.newline.len())
            .collect();
        let pack = |cap: usize| {
            let mut chunks = Vec::new();
            let (mut begin, mut used) = (0, 0);
            for (i, &cost) in costs.iter().enumerate() {
                if i > begin && used + cost > cap {
                    chunks.push(begin..i);
                    begin = i;
                    used = 0;
                }
                used += cost;
            }
            chunks.push(begin..costs.len());
            chunks
        };
        let chunks = if fixed + costs.iter().sum::<usize>() <= head_max {
            vec![0..options.len()]
        } else {
            ensure!(
                q.kind == "choice",
                "only choice questions can split over option chunks"
            );
            let room = head_max
                .checked_sub(fixed)
                .context("question exceeds head budget")?;
            let max = *costs.iter().max().unwrap();
            ensure!(
                max <= room,
                "one option exceeds head budget; nothing truncated"
            );
            let count = pack(room).len();
            let lower = max.max(costs.iter().sum::<usize>().div_ceil(count));
            (lower..=room)
                .map(pack)
                .find(|chunks| chunks.len() <= count)
                .unwrap()
        };
        let mut out = Vec::new();
        for range in chunks {
            let mut ids = introduction.clone();
            for o in &options[range.clone()] {
                ids.extend(&self.dash);
                ids.extend(o);
                ids.extend(&self.newline);
            }
            ids.extend(&self.judge);
            let mut slots = Vec::new();
            for o in &options[range.clone()] {
                ids.extend(o);
                ids.push(self.arrow);
                slots.push(ids.len() - 1);
                ids.extend(&self.newline);
            }
            ensure!(ids.len() <= head_max, "head budget exceeded");
            out.push(Rendered {
                ids,
                slots,
                names: q.names[range].to_vec(),
            });
        }
        Ok(out)
    }
}
