use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;
pub(super) struct Input {
    tok: Tokenizer,
    special: [u32; 5],
}
pub(super) struct Question {
    pub id: String,
    pub kind: String,
    pub names: Vec<String>,
    pub legend: Map<String, Value>,
    pub ids: Vec<u32>,
    pub options: Vec<usize>,
    pub decide: usize,
}
pub(super) struct Encoded {
    pub prefix: Vec<u32>,
    pub questions: Vec<Question>,
}
fn number(n: &serde_json::Number) -> String {
    if !n.is_f64() {
        return n.to_string();
    }
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
fn quoted(s: &str, ascii: bool) -> String {
    let s = serde_json::to_string(s).unwrap();
    if !ascii {
        return s;
    }
    let mut out = String::new();
    for c in s.chars() {
        let n = c as u32;
        if n < 128 {
            out.push(c);
        } else if n <= 0xffff {
            out.push_str(&format!("\\u{n:04x}"));
        } else {
            let n = n - 0x10000;
            out.push_str(&format!(
                "\\u{:04x}\\u{:04x}",
                0xd800 + (n >> 10),
                0xdc00 + (n & 1023)
            ));
        }
    }
    out
}
fn python_json(v: &Value) -> String {
    match v {
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", quoted(k, true), python_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::String(s) => quoted(s, true),
        Value::Number(n) => number(n),
        _ => v.to_string(),
    }
}
fn render(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => number(n),
        Value::Array(a) => a
            .iter()
            .map(|v| format!("{pad}- {}", render(v, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(o) => o
            .iter()
            .map(|(k, v)| {
                if v.is_object() || v.is_array() {
                    format!("{pad}{k}:\n{}", render(v, indent + 1))
                } else {
                    format!("{pad}{k}: {}", render(v, 0))
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}
fn option(name: &str, v: &Value) -> String {
    if v.is_null() || v == "" {
        name.to_owned()
    } else {
        format!("{name}: {}", render(v, 0))
    }
}
fn escape_special(text: &str) -> String {
    let (mut out, mut rest) = (String::new(), text);
    while let Some(start) = rest.find("<|") {
        out.push_str(&rest[..start]);
        rest = &rest[start + 2..];
        if let Some(end) = rest.find("|>") {
            let name = &rest[..end];
            if !name.is_empty() && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
                out.push_str("<¦");
                out.push_str(name);
                out.push_str("¦>");
                rest = &rest[end + 2..];
                continue;
            }
        }
        out.push_str("<|");
    }
    out.push_str(rest);
    out
}
impl Input {
    pub fn new(bytes: &[u8]) -> Result<Self> {
        let mut tok = Tokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        tok.with_padding(None);
        tok.with_truncation(None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut special = [0; 5];
        for (i, name) in [
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|fim_suffix|>",
        ]
        .iter()
        .enumerate()
        {
            special[i] = tok
                .token_to_id(name)
                .with_context(|| format!("missing Kev delimiter {name}"))?;
        }
        Ok(Self { tok, special })
    }
    fn user_tokens(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(escape_special(s), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn output_tokens(&self, answers: &Value) -> Result<usize> {
        Ok(self
            .tok
            .encode(python_json(answers), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .len())
    }
    pub fn encode(&self, request: &Value) -> Result<Encoded> {
        let state = request.get("state").context("missing state")?;
        let mut prefix = vec![self.special[0]];
        prefix.extend(self.user_tokens(&render(state, 0))?);
        ensure!(
            prefix.len() <= 65536,
            "state exceeds 65536 tokens; nothing truncated"
        );
        let questions = request["questions"]
            .as_object()
            .context("questions must be an object")?;
        ensure!(!questions.is_empty(), "empty questions");
        let mut encoded = Vec::new();
        for (id, q) in questions {
            let kind = q["type"].as_str().context("question.type")?.to_owned();
            let criteria = q.get("criteria").unwrap_or(&Value::Null);
            let mut legend = Map::new();
            let (names, texts): (Vec<String>, Vec<String>) = match kind.as_str() {
                "choice" => {
                    let c = criteria
                        .as_object()
                        .context("choice criteria must be an object")?;
                    ensure!(
                        (1..=255).contains(&c.len()),
                        "choice requires 1..255 options"
                    );
                    (
                        c.keys().cloned().collect(),
                        c.iter().map(|(k, v)| option(k, v)).collect(),
                    )
                }
                "noul" => {
                    ensure!(
                        criteria.is_null() || criteria.is_object(),
                        "noul criteria must be an object or null"
                    );
                    (
                        vec!["false".into(), "true".into()],
                        vec![
                            option("no", &criteria["false"]),
                            option("yes", &criteria["true"]),
                        ],
                    )
                }
                "score" => {
                    let c = criteria
                        .as_array()
                        .context("score criteria must be an array")?;
                    ensure!((1..=255).contains(&c.len()), "score requires 1..255 levels");
                    let texts: Vec<String> = c.iter().map(|v| render(v, 0)).collect();
                    for (i, s) in texts.iter().enumerate() {
                        legend.insert(i.to_string(), json!(s));
                    }
                    ((0..c.len()).map(|i| i.to_string()).collect(), texts)
                }
                _ => anyhow::bail!("unsupported question type {kind}"),
            };
            let mut ids = vec![self.special[1]];
            ids.extend(self.user_tokens(&render(&q["instructions"], 0))?);
            let mut options = Vec::new();
            for text in texts {
                ids.push(self.special[2]);
                ids.extend(self.user_tokens(&text)?);
                ids.push(self.special[3]);
                options.push(ids.len() - 1);
            }
            ids.push(self.special[4]);
            let decide = ids.len() - 1;
            ensure!(
                prefix.len() + ids.len() <= 73728,
                "question row exceeds 73728 tokens; nothing truncated"
            );
            encoded.push(Question {
                id: id.clone(),
                kind,
                names,
                legend,
                ids,
                options,
                decide,
            });
        }
        Ok(Encoded {
            prefix,
            questions: encoded,
        })
    }
    pub fn describe(&self, request: &Value) -> Result<Value> {
        let encoded = self.encode(request)?;
        let mut ids = encoded.prefix.clone();
        let mut pos: Vec<usize> = (0..ids.len()).collect();
        let mut decide_idx = Vec::new();
        let mut opt_idx = Vec::new();
        for q in &encoded.questions {
            let begin = ids.len();
            ids.extend(&q.ids);
            pos.extend(encoded.prefix.len()..encoded.prefix.len() + q.ids.len());
            decide_idx.push(begin + q.decide);
            opt_idx.push(q.options.iter().map(|i| begin + i).collect::<Vec<_>>());
        }
        Ok(
            json!({"ids":ids,"pos":pos,"decide_idx":decide_idx,"opt_idx":opt_idx,"state_tokens":encoded.prefix.len()}),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn special_text_cannot_inject_control_tokens() {
        assert_eq!(escape_special("x<|box_end|>y"), "x<¦box_end¦>y");
        assert_eq!(
            escape_special("<|bad-name|> <|ok_1|>"),
            "<|bad-name|> <¦ok_1¦>"
        );
        assert_eq!(escape_special("<|bad<|yes|>"), "<|bad<¦yes¦>");
    }
}
