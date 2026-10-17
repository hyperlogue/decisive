use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value};
use tokenizers::{
    NormalizedString, Normalizer, PostProcessor, Tokenizer, normalizers::unicode::NFC,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Choice,
    Noul,
    Score,
    Rank,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Question {
    pub text: String,
    pub choices: Vec<String>,
    pub kind: Kind,
    pub lang: String,
}

pub fn normalize(text: &str, lang: &str) -> String {
    let mut n = NormalizedString::from(text);
    NFC.normalize(&mut n).expect("NFC cannot fail");
    if lang != "fa" && lang != "ar" {
        return n.get().trim().to_owned();
    }
    let mut out = String::new();
    let mut space = false;
    for mut c in n.get().chars() {
        let code = c as u32;
        if matches!(code,0x0610..=0x061a|0x064b..=0x065f|0x0670|0x06d6..=0x06ed|0x0640) {
            continue;
        }
        if (0x660..=0x669).contains(&code) {
            c = char::from_u32(code - 0x660 + 48).unwrap();
        }
        if (0x6f0..=0x6f9).contains(&code) {
            c = char::from_u32(code - 0x6f0 + 48).unwrap();
        }
        c = match (lang, c) {
            ("fa", 'ي' | 'ى') => 'ی',
            ("fa", 'ك') => 'ک',
            ("fa", 'ة') => 'ه',
            ("fa", 'ؤ') => 'و',
            ("fa", 'إ' | 'أ') => 'ا',
            ("ar", 'ی') => 'ي',
            ("ar", 'ک') => 'ك',
            _ => c,
        };
        if c == ' ' || c == '\t' {
            if !space {
                out.push(' ');
            }
            space = true;
        } else {
            out.push(c);
            space = false;
        }
    }
    out.trim().to_owned()
}

pub struct Input {
    tok: Tokenizer,
    choice_payload: usize,
    pub max_state: usize,
    pub max_choice: usize,
}
impl Input {
    pub fn new(bytes: &[u8], max_state: usize, max_choice: usize) -> Result<Self> {
        type UnigramTokenizer = tokenizers::TokenizerImpl<
            tokenizers::models::unigram::Unigram,
            tokenizers::normalizers::NormalizerWrapper,
            tokenizers::pre_tokenizers::PreTokenizerWrapper,
            tokenizers::processors::PostProcessorWrapper,
            tokenizers::decoders::DecoderWrapper,
        >;
        let mut tok: Tokenizer = match serde_json::from_slice::<UnigramTokenizer>(bytes) {
            Ok(tokenizer) => tokenizer.into(),
            Err(_) => Tokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("{e}"))?,
        };
        tok.with_padding(None);
        tok.with_truncation(None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let special = tok
            .get_post_processor()
            .map_or(0, |p| p.added_tokens(false));
        let choice_payload = max_choice
            .checked_sub(special)
            .ok_or_else(|| anyhow::anyhow!("choice limit is shorter than its special tokens"))?;
        Ok(Self {
            tok,
            choice_payload,
            max_state,
            max_choice,
        })
    }
    fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
        Ok(tok
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn ids(&self, text: &str) -> Result<Vec<u32>> {
        Self::encode(&self.tok, text)
    }
    pub fn choice_ids(&self, text: &str) -> Result<Vec<u32>> {
        let mut encoding = self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        encoding.truncate(
            self.choice_payload,
            0,
            tokenizers::utils::truncation::TruncationDirection::Right,
        );
        Ok(self
            .tok
            .post_process(encoding, None, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn choices(
        &self,
        q: &Question,
        prefix: &str,
        question_in_choices: bool,
        fit: bool,
    ) -> Result<Vec<Vec<u32>>> {
        let question = if question_in_choices {
            q.text.as_str()
        } else {
            ""
        };
        let full: Vec<String> = q
            .choices
            .iter()
            .map(|c| {
                format!(
                    "{prefix}{}",
                    normalize(format!("{question} {c}").trim(), &q.lang)
                )
            })
            .collect();
        if !fit {
            let rows: Vec<_> = full
                .iter()
                .map(|t| self.choice_ids(t))
                .collect::<Result<_>>()?;
            let unique: std::collections::HashSet<_> = rows.iter().collect();
            if unique.len() == rows.len() {
                return Ok(rows);
            }
        }
        full.iter()
            .zip(&q.choices)
            .map(|(full, c)| {
                if full.chars().count() <= self.max_choice - 2
                    || self.ids(full)?.len() <= self.max_choice
                {
                    return self.choice_ids(full);
                }
                let opt = normalize(c, &q.lang);
                let text = normalize(question, &q.lang);
                let words: Vec<_> = text.split_whitespace().collect();
                let (mut lo, mut hi) = (0, words.len());
                while lo < hi {
                    let mid = (lo + hi).div_ceil(2);
                    if self
                        .ids(&format!("{prefix}{} … {opt}", words[..mid].join(" ")))?
                        .len()
                        <= self.max_choice
                    {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                self.choice_ids(&if lo == 0 {
                    format!("{prefix}{opt}")
                } else {
                    format!("{prefix}{} … {opt}", words[..lo].join(" "))
                })
            })
            .collect()
    }
}

pub fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        _ => json_text(v),
    }
}
fn json_text(v: &Value) -> String {
    match v {
        Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().unwrap();
            if x != 0.0 && (x.abs() < 1e-4 || x.abs() >= 1e16) {
                let scientific = format!("{x:e}");
                let (mantissa, exponent) = scientific.split_once('e').unwrap();
                let exponent: i32 = exponent.parse().unwrap();
                format!("{mantissa}e{exponent:+03}")
            } else {
                let mut out = x.to_string();
                if !out.contains('.') {
                    out.push_str(".0");
                }
                out
            }
        }
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(json_text).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), json_text(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => serde_json::to_string(v).unwrap(),
    }
}
pub fn parse_question(id: &str, spec: &Value, lang: &str) -> Result<(Question, Vec<String>)> {
    ensure!(spec.is_object(), "questions.{id} must be an object");
    let mut instructions = if spec["instructions"].is_null() {
        id.into()
    } else {
        text(&spec["instructions"])
    };
    let mut keys = vec![];
    let mut choices = vec![];
    let kind = match spec["type"].as_str() {
        Some("choice") => {
            let owned: Map<String, Value>;
            let criteria = if let Some(list) = spec["criteria"].as_array() {
                owned = list
                    .iter()
                    .map(|v| {
                        Ok((
                            v.as_str()
                                .ok_or_else(|| {
                                    anyhow::anyhow!("choice list labels must be strings")
                                })?
                                .to_owned(),
                            Value::Null,
                        ))
                    })
                    .collect::<Result<_>>()?;
                &owned
            } else {
                spec["criteria"].as_object().ok_or_else(|| {
                    anyhow::anyhow!("choice criteria must be an object or string list")
                })?
            };
            ensure!(
                (2..=255).contains(&criteria.len()),
                "choice requires 2–255 labels"
            );
            for (k, v) in criteria {
                keys.push(k.clone());
                let desc = text(v);
                choices.push(if desc.trim().is_empty() {
                    k.clone()
                } else {
                    format!("{k}: {desc}")
                });
            }
            Kind::Choice
        }
        Some("score") => {
            let levels = spec["criteria"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("score criteria must be an array"))?;
            ensure!(
                (2..=10).contains(&levels.len()),
                "score requires 2–10 levels"
            );
            keys = levels.iter().map(text).collect();
            choices = keys.clone();
            Kind::Score
        }
        Some("noul") => {
            let t = text(&spec["criteria"]["true"]);
            let f = text(&spec["criteria"]["false"]);
            if !t.trim().is_empty() || !f.trim().is_empty() {
                instructions = format!(
                    "{instructions}\nTrue if: {}\nFalse if: {}",
                    if t.trim().is_empty() { "—" } else { t.trim() },
                    if f.trim().is_empty() { "—" } else { f.trim() }
                );
            }
            keys = vec!["yes".into(), "no".into()];
            choices = keys.clone();
            Kind::Noul
        }
        _ => bail!("questions.{id}.type must be choice, score or noul"),
    };
    Ok((
        Question {
            text: instructions,
            choices,
            kind,
            lang: lang.into(),
        },
        keys,
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn non_unigram_tokenizer_uses_the_generic_fallback() {
        let bytes=br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}}"#;
        let input = super::Input::new(bytes, 512, 64).unwrap();
        assert_eq!(input.ids("hello").unwrap(), [1]);
    }
    #[test]
    #[ignore = "requires the pinned Decima tokenizer"]
    fn typed_loader_matches_the_complete_pinned_tokenizer() {
        let bytes = std::fs::read("models/decima-small/pytorch/encoder/tokenizer.json").unwrap();
        let input = super::Input::new(&bytes, 512, 64).unwrap();
        let mut reference = tokenizers::Tokenizer::from_bytes(&bytes).unwrap();
        reference.with_padding(None);
        reference.with_truncation(None).unwrap();
        assert_eq!(
            input.tok.to_string(false).unwrap(),
            reference.to_string(false).unwrap()
        );
        for text in [
            "hello world",
            "کارت من ۲ بار شارژ شده",
            "退款给我",
            "ช่วยตรวจสอบ",
            "café café\n",
            "👩🏽‍💻 <s> </s>",
            "",
            "अतिरिक्त भुगतान",
        ] {
            assert_eq!(
                input.ids(text).unwrap(),
                reference.encode(text, true).unwrap().get_ids()
            );
        }
    }
    use super::*;
    #[cfg(feature = "cpu")]
    #[test]
    #[ignore = "requires the pinned Decima tokenizer in models/decima-small"]
    fn checkpoint_choice_truncation_matches_library() -> Result<()> {
        let bytes = std::fs::read("models/decima-small/pytorch/encoder/tokenizer.json")?;
        let mut input = Input::new(&bytes, 512, 512)?;
        let mut reference = input.tok.clone();
        let special = reference
            .get_post_processor()
            .map_or(0, |p| p.added_tokens(false));
        let texts = [
            "".to_owned(),
            " ".into(),
            "refund".into(),
            "<s>literal</s>".into(),
            "我被重复扣款了，请退款。".into(),
            "كِتابـ ۱۲\t ي".into(),
            "e\u{301} 👩\u{200d}💻\0עברית русский".into(),
            "record ".repeat(1025),
            "قیمت ۱۲۳ 中文 é ".repeat(200),
        ];
        for limit in [2, 3, 16, 64, 128, 511, 512] {
            input.max_choice = limit;
            input.choice_payload = limit - special;
            reference
                .with_truncation(Some(tokenizers::utils::truncation::TruncationParams {
                    max_length: limit,
                    ..Default::default()
                }))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            for text in &texts {
                let expected = Input::encode(&reference, text)?;
                assert_eq!(input.choice_ids(text)?, expected, "choice limit {limit}");
                assert!(expected.len() <= limit);
            }
        }
        Ok(())
    }
    #[test]
    fn rendering_is_not_laya_rendering() {
        assert_eq!(
            text(&serde_json::json!([1e-7, 1e16, -0.0, 1.0, 0.0001, 1e15])),
            "[1e-07, 1e+16, -0.0, 1.0, 0.0001, 1000000000000000.0]"
        );
        assert_eq!(normalize("  كِتابـ ۱۲\t ي ", "fa"), "کتاب 12 ی");
        assert_eq!(normalize("ی ک ١٢", "ar"), "ي ك 12");
        assert_eq!(normalize("e\u{301}", "en"), "é");
        assert_eq!(
            text(&serde_json::json!({"x":[1,"é",null]})),
            "{\"x\": [1, \"é\", null]}"
        );
        let (q, k) = parse_question(
            "urgency",
            &serde_json::json!({"type":"noul","criteria":{"true":"today"}}),
            "en",
        )
        .unwrap();
        assert_eq!(q.text, "urgency\nTrue if: today\nFalse if: —");
        assert_eq!(k, ["yes", "no"]);
    }
}
