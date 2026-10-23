use super::input::{self, Input, Kind, Question};
use anyhow::{Result, ensure};
use serde_json::{Map, Value, json};
use std::ops::Range;

// Measured on 700–1,100-token drafts with one planted detail. Without overlap, 32-token chunks
// found 34 of 48 at 46 chunks per draft; 48 found 31 at 25, 64 found 22 and 128 found 6.
// Overlapping by one sentence found one more for 12% more chunks.
pub const DEFAULT_WINDOW: usize = 32;
pub const DEFAULT_OVERLAP: usize = 0;

pub(crate) struct Parsed {
    pub states: Vec<String>,
    pub questions: Vec<(String, Question, Vec<String>)>,
    pub threshold: Option<f64>,
    pub input_tokens: usize,
    pub chunks: Option<Vec<Range<usize>>>,
}

pub(crate) fn parse(input: &Input, request: &Value) -> Result<Parsed> {
    ensure!(request.is_object(), "request must be an object");
    let long = match &request["long"] {
        Value::Null => false,
        Value::Bool(v) => *v,
        _ => anyhow::bail!("long must be true or false"),
    };
    let state = &request["state"];
    ensure!(
        state.is_string() || state.is_array() || state.is_object(),
        "state must be a string, object or array"
    );
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
    ensure!(
        !long || threshold.is_none(),
        "long requests do not accept min_confidence"
    );
    for key in ["window", "overlap"] {
        ensure!(long || request[key].is_null(), "{key} requires long: true");
    }
    let size = |key: &str, default: usize| -> Result<usize> {
        match &request[key] {
            Value::Null => Ok(default),
            v => v
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| anyhow::anyhow!("{key} must be a non-negative integer")),
        }
    };
    let window = size("window", DEFAULT_WINDOW)?;
    let overlap = size("overlap", DEFAULT_OVERLAP)?;
    ensure!((1..=512).contains(&window), "window must be 1–512 tokens");
    let questions = questions
        .iter()
        .map(|(id, spec)| {
            let (q, keys) = input::parse_question(id, spec, lang)?;
            Ok((id.clone(), q, keys))
        })
        .collect::<Result<Vec<_>>>()?;
    let rendered = input::text(state);
    let input_tokens = input.ids(&rendered)?.len();
    if !long {
        return Ok(Parsed {
            states: vec![rendered],
            questions,
            threshold,
            input_tokens,
            chunks: None,
        });
    }
    let mut copy = state.clone();
    let text = last_text(&mut copy).map_or_else(String::new, |s| s.clone());
    let ranges = chunks(input, &text, window, overlap)?;
    let mut states = Vec::with_capacity(ranges.len());
    for r in &ranges {
        if let Some(leaf) = last_text(&mut copy) {
            *leaf = slice(&text, r.clone()).into();
        }
        states.push(input::text(&copy));
    }
    Ok(Parsed {
        states,
        questions,
        threshold,
        input_tokens,
        chunks: Some(ranges),
    })
}

fn last_text(v: &mut Value) -> Option<&mut String> {
    match v {
        Value::String(s) => Some(s),
        Value::Array(a) => a.iter_mut().rev().find_map(last_text),
        Value::Object(o) => o.values_mut().rev().find_map(last_text),
        _ => None,
    }
}

fn slice(text: &str, chars: Range<usize>) -> &str {
    let byte = |c: usize| text.char_indices().nth(c).map_or(text.len(), |(b, _)| b);
    &text[byte(chars.start)..byte(chars.end)]
}

pub(crate) fn sentences(text: &str) -> Vec<Range<usize>> {
    let chars: Vec<char> = text.chars().collect();
    let terminal = |c: char| matches!(c, '.' | '!' | '?' | '…' | '؟' | '।');
    let wide = |c: char| matches!(c, '。' | '！' | '？');
    let closing = |c: char| matches!(c, '"' | '\'' | ')' | ']' | '”' | '’' | '»' | '」' | '』');
    let mut out = Vec::new();
    let mut push = |first: usize, end: usize| {
        if let Some(k) = (first..end).rev().find(|&k| !chars[k].is_whitespace()) {
            out.push(first..k + 1);
        }
    };
    let mut start = None;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() && start.is_none() {
            i += 1;
            continue;
        }
        let first = *start.get_or_insert(i);
        let mut end = None;
        if c == '\n' {
            end = Some(i);
        } else if terminal(c) || wide(c) {
            let mut j = i + 1;
            while j < chars.len() && (terminal(chars[j]) || wide(chars[j]) || closing(chars[j])) {
                j += 1;
            }
            if wide(c) || j == chars.len() || chars[j].is_whitespace() {
                end = Some(j);
            }
            i = j - 1;
        }
        if let Some(e) = end {
            push(first, e);
            start = None;
        }
        i += 1;
    }
    if let Some(first) = start {
        push(first, chars.len());
    }
    out
}

pub(crate) fn chunks(
    input: &Input,
    text: &str,
    window: usize,
    overlap: usize,
) -> Result<Vec<Range<usize>>> {
    let mut units = Vec::new();
    for s in sentences(text) {
        let part = slice(text, s.clone());
        let starts = input.token_starts(part)?;
        if starts.len() <= window {
            units.push((s, starts.len()));
            continue;
        }
        for (k, &first) in starts.iter().enumerate().step_by(window) {
            let end = starts.get(k + window).copied().unwrap_or(s.end - s.start);
            if end > first {
                units.push((
                    s.start + first..s.start + end,
                    (starts.len() - k).min(window),
                ));
            }
        }
    }
    if units.is_empty() {
        return Ok(vec![0..text.chars().count()]);
    }
    let mut out = Vec::new();
    let mut i = 0;
    loop {
        let (mut j, mut tokens) = (i, 0);
        while j < units.len() && (j == i || tokens + units[j].1 <= window) {
            tokens += units[j].1;
            j += 1;
        }
        out.push(units[i].0.start..units[j - 1].0.end);
        if j == units.len() {
            return Ok(out);
        }
        i = (i + 1).max(j.saturating_sub(overlap));
    }
}

pub(crate) fn answer(q: &Question, keys: &[String], p: &[f32], threshold: Option<f64>) -> Value {
    let r = |v: f64| (v * 10000.).round_ties_even() / 10000.;
    let best = p
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .unwrap()
        .0;
    let confidence = ((p.len() as f64 * p[best] as f64 - 1.) / (p.len() - 1) as f64).clamp(0., 1.);
    let probabilities: Map<String, Value> = p
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
            json!({"type":"score","score":r(p.iter().enumerate().map(|(i,v)|i as f64* *v as f64).sum()),"confidence":r(confidence),"legend":keys.iter().enumerate().map(|(i,v)|(i.to_string(),json!(v))).collect::<Map<_,_>>(),"probabilities":probabilities})
        }
        _ => json!({"type":"noul","noul":r(p[0] as f64)}),
    };
    if let Some(t) = threshold {
        answer["abstain"] = json!(confidence < t);
    }
    answer
}

pub(crate) fn respond(p: &Parsed, scores: &[Vec<Vec<f32>>]) -> Value {
    let mut answers = Map::new();
    for (k, (id, q, keys)) in p.questions.iter().enumerate() {
        let all: Vec<Value> = scores
            .iter()
            .map(|s| answer(q, keys, &s[k], p.threshold))
            .collect();
        let Some(ranges) = &p.chunks else {
            answers.insert(id.clone(), all.into_iter().next().unwrap());
            continue;
        };
        let strength = |a: &Value| {
            a["noul"]
                .as_f64()
                .or(a["confidence"].as_f64())
                .unwrap_or(0.)
        };
        let mut best = 0;
        for i in 1..all.len() {
            if strength(&all[i]) > strength(&all[best]) {
                best = i;
            }
        }
        let mut chosen = all[best].clone();
        if let Some(probabilities) = chosen["probabilities"].as_object() {
            let max: Map<String, Value> = probabilities
                .keys()
                .map(|key| {
                    let m = all
                        .iter()
                        .filter_map(|a| a["probabilities"][key].as_f64())
                        .fold(0., f64::max);
                    (key.clone(), json!(m))
                })
                .collect();
            chosen["max_probabilities"] = Value::Object(max);
        }
        chosen["chunk"] = json!({"index":best,"count":all.len(),"start":ranges[best].start,"end":ranges[best].end});
        answers.insert(id.clone(), chosen);
    }
    let mut usage = json!({"input_tokens":p.input_tokens,"output_tokens":0});
    if p.chunks.is_some() {
        usage["chunks"] = json!(p.states.len());
    }
    json!({"model":"decima-small","answers":answers,"usage":usage})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sentences_cover_punctuation_scripts_and_lines() {
        let text = "Hi, I'm Sam.  My code is 4821! Is 3.5 ok?\n第一句。第二句！ \"Quoted.\" Last";
        let got: Vec<String> = sentences(text)
            .into_iter()
            .map(|r| slice(text, r).to_owned())
            .collect();
        assert_eq!(
            got,
            [
                "Hi, I'm Sam.",
                "My code is 4821!",
                "Is 3.5 ok?",
                "第一句。",
                "第二句！",
                "\"Quoted.\"",
                "Last"
            ]
        );
        assert!(sentences("   ").is_empty());
    }
    #[test]
    fn last_text_is_the_newest_message() {
        let mut v =
            json!({"context":[{"role":"assistant","content":"a"},{"role":"user","content":"b"}]});
        *last_text(&mut v).unwrap() = "c".into();
        assert_eq!(v["context"][1]["content"], "c");
        assert_eq!(v["context"][0]["content"], "a");
    }
}
#[cfg(all(test, feature = "cpu"))]
mod checkpoint_tests {
    use super::*;
    #[test]
    #[ignore = "requires models/decima-small"]
    fn chunks_pack_sentences_and_cut_long_ones() -> Result<()> {
        let bytes = std::fs::read("models/decima-small/pytorch/encoder/tokenizer.json")?;
        let input = Input::new(&bytes, 512, 64)?;
        let text = "One two. Three four five. Six.";
        let pick = |w, o| -> Result<Vec<String>> {
            Ok(chunks(&input, text, w, o)?
                .into_iter()
                .map(|r| slice(text, r).into())
                .collect())
        };
        assert_eq!(pick(512, 0)?, [text]);
        let tiny = chunks(&input, text, 1, 0)?;
        assert!(tiny.len() > 3 && tiny.windows(2).all(|w| w[0].end <= w[1].start));
        let sentences = pick(6, 0)?;
        assert!(sentences.iter().all(|s| text.contains(s.as_str())));
        assert_eq!(pick(6, 1)?.first(), sentences.first());
        let key = "k".repeat(300);
        let pieces = chunks(&input, &key, 32, 0)?;
        assert!(pieces.len() > 1 && pieces.windows(2).all(|w| w[0].end == w[1].start));
        assert_eq!((pieces[0].start, pieces.last().unwrap().end), (0, 300));
        assert_eq!(chunks(&input, "", 32, 0)?, [0..0]);
        Ok(())
    }
}
