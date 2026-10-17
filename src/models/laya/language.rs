use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
};

#[derive(Deserialize)]
struct Rules {
    scripts: Vec<(String, Vec<[u32; 2]>)>,
    #[serde(deserialize_with = "read_stops")]
    stop: Vec<(String, HashSet<String>)>,
    short_swedish: HashSet<String>,
    diacritics: String,
    shared: HashSet<String>,
    nordic: HashSet<String>,
    english_only: HashSet<String>,
    collisions: HashSet<String>,
    alpha: Vec<[u32; 2]>,
    upper: Vec<[u32; 2]>,
    lower: Vec<[u32; 2]>,
    title: Vec<[u32; 2]>,
    alnum: Vec<[u32; 2]>,
    word: Vec<[u32; 2]>,
    combining: Vec<[u32; 2]>,
    space: Vec<[u32; 2]>,
}
fn read_stops<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Vec<(String, HashSet<String>)>, D::Error> {
    let values = Map::<String, Value>::deserialize(d)?;
    values
        .into_iter()
        .map(|(key, value)| {
            serde_json::from_value(value)
                .map(|v| (key, v))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}
fn rules() -> &'static Rules {
    static RULES: OnceLock<Rules> = OnceLock::new();
    RULES.get_or_init(|| {
        serde_json::from_str(include_str!("language_data.json")).expect("pinned routing data")
    })
}
fn has(ranges: &[[u32; 2]], c: char) -> bool {
    let cp = c as u32;
    let i = ranges.partition_point(|r| r[0] <= cp);
    i > 0 && cp <= ranges[i - 1][1]
}
fn alpha(c: char) -> bool {
    has(&rules().alpha, c)
}
fn upper(c: char) -> bool {
    has(&rules().upper, c)
}
fn lower(c: char) -> bool {
    has(&rules().lower, c)
}
fn word(c: char) -> bool {
    has(&rules().word, c)
}
fn alnum(c: char) -> bool {
    has(&rules().alnum, c)
}
fn space(c: char) -> bool {
    has(&rules().space, c)
}
fn take(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}
fn words(text: &str) -> Vec<&str> {
    text.split(|c| !word(c)).filter(|s| !s.is_empty()).collect()
}
fn shouted(text: &str) -> bool {
    text.chars().any(upper) && !text.chars().any(lower)
}
fn all_upper(text: &str) -> bool {
    text.chars().any(upper)
        && text
            .chars()
            .all(|c| upper(c) || !(lower(c) || has(&rules().title, c)))
}
fn leaves<'a>(state: &'a Value, depth: usize, out: &mut Vec<&'a str>) {
    if depth > 6 {
        return;
    }
    match state {
        Value::String(s) => out.push(s),
        Value::Array(v) => v.iter().for_each(|v| leaves(v, depth + 1, out)),
        Value::Object(v) => v.values().for_each(|v| leaves(v, depth + 1, out)),
        _ => (),
    }
}
fn state_text(leaves: &[&str]) -> String {
    let mut parts = Vec::new();
    let mut budget = 4000;
    for leaf in leaves {
        if budget == 0 {
            break;
        }
        let length = leaf.chars().count();
        if length > budget {
            parts.push(take(leaf, budget));
            break;
        }
        parts.push(leaf.to_string());
        budget = budget.saturating_sub(length + 1);
    }
    take(&parts.join(" "), 4000)
}
fn script_of(c: char) -> Option<&'static str> {
    let cp = c as u32;
    if cp < 592
        || (7680..=7935).contains(&cp)
        || (65313..=65338).contains(&cp)
        || (65345..=65370).contains(&cp)
    {
        return None;
    }
    rules()
        .scripts
        .iter()
        .find(|(_, r)| r.iter().any(|r| (r[0]..=r[1]).contains(&cp)))
        .map(|(name, _)| name.as_str())
}
fn non_latin_words(text: &str) -> bool {
    let mut script = None;
    let mut length = 0;
    let mut first_upper = false;
    for c in text.chars() {
        if has(&rules().combining, c) {
            continue;
        }
        let next = script_of(c);
        if next.is_some() && next == script {
            length += 1;
            continue;
        }
        if length >= 2 && !first_upper {
            return true;
        }
        script = next;
        length = usize::from(next.is_some());
        first_upper = upper(c);
    }
    length >= 2 && !first_upper
}
fn non_latin_carries(fraction: f64, letters: f64) -> bool {
    fraction >= 0.2 || (fraction >= 0.1 && letters >= 10.0)
}
fn identifier(c: char) -> bool {
    alnum(c) || c == '_' || c == '-'
}
fn remove_identifiers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if i == 0 || !identifier(chars[i - 1]) {
            let mut j = i;
            while j < chars.len() && identifier(chars[j]) {
                j += 1;
            }
            let mut end = None;
            while j + 1 < chars.len() && matches!(chars[j], '.' | '@') && identifier(chars[j + 1]) {
                j += 1;
                while j < chars.len() && identifier(chars[j]) {
                    j += 1;
                }
                end = Some(j);
            }
            if let Some(end) = end {
                out.push(' ');
                i = end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}
struct Latin {
    language: Option<String>,
    diacritic_rate: f64,
    foreign: bool,
}
fn latin_profile(text: &str) -> Latin {
    let cleaned = remove_identifiers(text).replace('İ', "i").to_lowercase();
    let tokens = words(&cleaned);
    let unique: HashSet<&str> = tokens.iter().copied().collect();
    let lowered = text.to_lowercase();
    let rate = lowered
        .chars()
        .filter(|&c| rules().diacritics.contains(c))
        .count() as f64
        / lowered.chars().count().max(1) as f64;
    let foreign = rate >= 0.02;
    let nordic = unique.iter().any(|w| rules().nordic.contains(*w))
        && !unique.iter().any(|w| rules().english_only.contains(*w));
    if tokens.len() > 1
        && tokens.len() < 4
        && unique.iter().any(|w| rules().short_swedish.contains(*w))
    {
        return Latin {
            language: Some("sv".into()),
            diacritic_rate: rate,
            foreign,
        };
    }
    if tokens.len() < 4 {
        return Latin {
            language: None,
            diacritic_rate: rate,
            foreign: foreign || nordic,
        };
    }
    let mut counts = HashMap::new();
    for token in &tokens {
        *counts.entry(*token).or_insert(0_usize) += 1;
    }
    let mut en = 0;
    let mut best = 0;
    let mut best_language = None;
    for (language, stop) in &rules().stop {
        let score = counts
            .iter()
            .filter(|(w, _)| stop.contains(**w))
            .map(|(w, n)| {
                if rules().collisions.contains(*w) {
                    1
                } else {
                    *n
                }
            })
            .sum::<usize>();
        if language == "en" {
            en = score;
            continue;
        }
        let evidence = unique
            .iter()
            .any(|w| stop.contains(*w) && !rules().shared.contains(*w));
        if evidence && (best_language.is_none() || score > best) {
            best = score;
            best_language = Some(language.clone());
        }
    }
    let mut language = None;
    if best_language.is_some() && best >= (en + 2).max(2) {
        language = best_language.clone();
    } else if best_language.as_deref() == Some("sv")
        && unique.contains("inte")
        && unique.contains("kan")
        && matches!(tokens[0], "kan" | "jag" | "vi")
        && en <= 1
    {
        language = best_language.clone();
    } else if best_language.is_some() && foreign && best >= en.max(2) {
        language = best_language;
    } else if en > 0 {
        let rescued = rate < 0.06
            && unique
                .iter()
                .filter(|w| rules().english_only.contains(**w))
                .count()
                >= 2
            && unique
                .iter()
                .filter(|w| w.chars().any(|c| rules().diacritics.contains(c)))
                .count()
                <= 1;
        if !foreign || rescued {
            language = Some("en".into());
        }
    }
    let undecided = language.is_none();
    Latin {
        language,
        diacritic_rate: rate,
        foreign: foreign || (undecided && nordic),
    }
}
fn code(text: &str) -> bool {
    let mut previous = None;
    for c in text.chars() {
        if "=;{}[]".contains(c) || (c == '(' && previous.is_some_and(|c| alnum(c) || c == '_')) {
            return true;
        }
        previous = Some(c);
    }
    false
}
fn joined(text: &str) -> bool {
    let c: Vec<_> = text.chars().collect();
    c.windows(3)
        .any(|v| alnum(v[0]) && "._/\\".contains(v[1]) && alnum(v[2]))
}
fn blank_capitals(text: &str) -> String {
    let mut result = String::new();
    let mut start = None;
    let mut copied = 0;
    for (index, c) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        if word(c) {
            start.get_or_insert(index);
        } else if let Some(begin) = start.take() {
            let token = &text[begin..index];
            if token.chars().count() >= 2 && all_upper(token) {
                result.push_str(&text[copied..begin]);
                result.push(' ');
                copied = index;
            }
        }
    }
    result.push_str(&text[copied..]);
    result
}
fn shouted_evidence(tokens: &[&str], language: &str, rate: f64) -> bool {
    if !tokens.iter().any(|t| t.chars().count() >= 5) {
        return false;
    }
    if rate >= 0.02 {
        return true;
    }
    let stop = &rules()
        .stop
        .iter()
        .find(|(name, _)| name == language)
        .unwrap()
        .1;
    tokens
        .iter()
        .any(|w| w.chars().count() >= 4 && stop.contains(&w.to_lowercase()))
}
fn named_prose(segment: &str) -> Option<String> {
    if segment.trim_matches(space).is_empty() || code(segment) {
        return None;
    }
    let mut prose = segment
        .split(space)
        .filter(|t| !t.is_empty() && !joined(t))
        .collect::<Vec<_>>()
        .join(" ");
    let caps = shouted(&prose);
    if !caps {
        prose = blank_capitals(&prose);
    }
    let tokens = words(&prose);
    if tokens.len() < 4 {
        return None;
    }
    let profile = latin_profile(&prose);
    let language = profile.language?;
    if language == "en" || (caps && !shouted_evidence(&tokens, &language, profile.diacritic_rate)) {
        return None;
    }
    let unique: HashSet<_> = tokens.iter().map(|w| w.to_lowercase()).collect();
    let stop = &rules()
        .stop
        .iter()
        .find(|(name, _)| name == &language)
        .unwrap()
        .1;
    if unique.intersection(stop).count() < 2 {
        return None;
    }
    Some(language)
}

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub script: String,
    pub script_profile: Map<String, Value>,
    pub language: Option<String>,
    pub is_english: bool,
    pub language_undecided: bool,
    pub diacritic_rate: f64,
    pub non_latin_fraction: f64,
    pub mixed_segment: Option<String>,
}
fn analyse_text(text: &str) -> Detection {
    let mut counts: Vec<(String, usize)> = Vec::new();
    let mut latin = 0;
    for c in text.chars().filter(|&c| alpha(c)) {
        let cp = c as u32;
        if cp < 688
            || (7680..=7935).contains(&cp)
            || (65313..=65338).contains(&cp)
            || (65345..=65370).contains(&cp)
        {
            latin += 1;
            continue;
        }
        let name = script_of(c).unwrap_or("other");
        if let Some((_, count)) = counts.iter_mut().find(|(key, _)| key == name) {
            *count += 1;
        } else {
            counts.push((name.into(), 1));
        }
    }
    counts.push(("latin".into(), latin));
    let total = counts.iter().map(|(_, n)| n).sum::<usize>();
    let mut script = "unknown".to_owned();
    let mut best = 0;
    for (name, n) in &counts {
        if *n > best {
            script = name.clone();
            best = *n;
        }
    }
    let mut profile = Map::new();
    if latin > 0 {
        profile.insert("latin".into(), json!(latin as f64 / total as f64));
    }
    for (name, n) in &counts {
        if name != "latin" && *n > 0 {
            profile.insert(name.clone(), json!(*n as f64 / total as f64));
        }
    }
    let non_latin = if total == 0 {
        0.0
    } else {
        crate::laya::input::round4(1.0 - latin as f64 / total as f64)
    };
    let letters = (non_latin * total as f64).round_ties_even();
    if script == "latin" && non_latin_words(text) && non_latin_carries(non_latin, letters) {
        let mut best = 0.0;
        for (name, value) in &profile {
            let n = value.as_f64().unwrap();
            if name != "latin" && n > best {
                script = name.clone();
                best = n;
            }
        }
    }
    let mut detection = Detection {
        is_english: script == "unknown",
        script,
        script_profile: profile,
        language: None,
        language_undecided: true,
        diacritic_rate: 0.0,
        non_latin_fraction: non_latin,
        mixed_segment: None,
    };
    if detection.script != "latin" {
        return detection;
    }
    let profile = latin_profile(text);
    let mut language = profile.language;
    if language.as_deref().is_some_and(|l| l != "en") && !non_latin_carries(non_latin, letters) {
        if shouted(text) {
            if !shouted_evidence(
                &words(text),
                language.as_deref().unwrap(),
                profile.diacritic_rate,
            ) {
                language = None;
            }
        } else {
            let caps: Vec<_> = words(text)
                .into_iter()
                .filter(|w| w.chars().count() >= 2 && all_upper(w))
                .collect();
            if !caps.is_empty() && !caps.iter().any(|w| w.chars().count() >= 5) {
                language = latin_profile(&blank_capitals(text)).language;
            }
        }
    }
    detection.language_undecided = language.is_none();
    detection.is_english =
        language.as_deref() == Some("en") || (language.is_none() && !profile.foreign);
    detection.language = language;
    detection.diacritic_rate = crate::laya::input::round4(profile.diacritic_rate);
    detection
}
fn foreign_leaf(leaf: &str) -> Option<Detection> {
    let mut best = None;
    let mut best_n = 0;
    for line in leaf.split('\n') {
        if line.chars().count() < 7 {
            continue;
        }
        let sample = take(line, 4000);
        if sample.trim_matches(space).is_empty() || code(&sample) {
            continue;
        }
        let det = analyse_text(&sample);
        if det.is_english {
            continue;
        }
        let named =
            det.language.as_deref().is_some_and(|l| l != "en") && named_prose(&sample).is_some();
        let n = sample.chars().filter(|&c| alpha(c)).count();
        if !named {
            if !matches!(det.script.as_str(), "latin" | "unknown") {
                if !non_latin_words(&sample) || n < 10 {
                    continue;
                }
            } else if !(det.diacritic_rate >= 0.02 && words(&sample).len() >= 4) {
                continue;
            }
        }
        if best.is_none() || n > best_n {
            best = Some(det);
            best_n = n;
        }
    }
    best
}

pub fn analyse(state: &Value) -> Detection {
    let mut text_leaves = Vec::new();
    leaves(state, 0, &mut text_leaves);
    let mut result = analyse_text(&state_text(&text_leaves));
    if result.script == "latin"
        && result.is_english
        && (text_leaves.len() > 1 || text_leaves.iter().any(|s| s.contains('\n')))
    {
        let mut seen = 0;
        let mut found = None;
        'scan: for leaf in &text_leaves {
            for segment in leaf.split('\n') {
                if seen >= 4000 {
                    break 'scan;
                }
                let segment = take(segment, 4000 - seen);
                seen += segment.chars().count();
                if let Some(lang) = named_prose(&segment) {
                    found = Some((lang, segment.trim_matches(space).to_owned()));
                    break 'scan;
                }
            }
        }
        if let Some((language, segment)) = found {
            result.language = Some(language);
            result.is_english = false;
            result.language_undecided = false;
            result.mixed_segment = Some(segment);
        }
    }
    if state.is_string() || state.is_null() || !result.is_english {
        return result;
    }
    let mut best = None;
    let mut best_n = 0;
    for leaf in text_leaves {
        if let Some(det) = foreign_leaf(leaf) {
            let n = leaf.chars().take(4000).filter(|&c| alpha(c)).count();
            if best.is_none() || n > best_n {
                best = Some(det);
                best_n = n;
            }
        }
    }
    if let Some(best) = best {
        result.language = best.language;
        result.is_english = false;
        result.language_undecided = best.language_undecided;
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn agrees_with_pinned_sdk_language_rules() {
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../../tests/fixtures/language.json")).unwrap();
        for (i, case) in cases.iter().enumerate() {
            let actual = serde_json::to_value(super::analyse(&case["state"])).unwrap();
            assert_eq!(
                actual, case["expected"],
                "routing case {i}: {}",
                case["state"]
            );
        }
    }
}
