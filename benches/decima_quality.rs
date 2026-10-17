use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Serialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long, default_value_t = 0.02)]
    bound: f64,
    #[arg(required = true)]
    results: Vec<PathBuf>,
}
#[derive(Default, Serialize)]
struct Metrics {
    cases: usize,
    labeled: usize,
    correct: usize,
    reference_correct: usize,
    changed_decisions: usize,
    above_probability_bound: usize,
    maximum_probability_error: f64,
    brier: f64,
    reference_brier: f64,
    nll: f64,
    reference_nll: f64,
}
fn best(p: &[f64]) -> usize {
    (1..p.len()).fold(0, |best, i| if p[i] > p[best] { i } else { best })
}
impl Metrics {
    fn add(&mut self, p: &[f64], r: &[f64], gold: Option<usize>, bound: f64) {
        self.cases += 1;
        let error = p
            .iter()
            .zip(r)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        self.maximum_probability_error = self.maximum_probability_error.max(error);
        self.above_probability_bound += usize::from(error > bound);
        self.changed_decisions += usize::from(best(p) != best(r));
        if let Some(g) = gold {
            self.labeled += 1;
            self.correct += usize::from(best(p) == g);
            self.reference_correct += usize::from(best(r) == g);
            self.brier += p
                .iter()
                .enumerate()
                .map(|(i, v)| (v - (if i == g { 1. } else { 0. })).powi(2))
                .sum::<f64>();
            self.reference_brier += r
                .iter()
                .enumerate()
                .map(|(i, v)| (v - (if i == g { 1. } else { 0. })).powi(2))
                .sum::<f64>();
            self.nll -= p[g].max(1e-12).ln();
            self.reference_nll -= r[g].max(1e-12).ln();
        }
    }
    fn normalize(&mut self) {
        let n = self.labeled.max(1) as f64;
        self.brier /= n;
        self.reference_brier /= n;
        self.nll /= n;
        self.reference_nll /= n;
    }
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    ensure!(
        args.bound.is_finite() && args.bound >= 0.,
        "invalid probability bound"
    );
    let corpus: BTreeMap<String, Value> = fs::read_to_string(&args.corpus)?
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|s| -> Result<_> {
            let v: Value = serde_json::from_str(s)?;
            Ok((v["id"].as_str().context("corpus id")?.to_owned(), v))
        })
        .collect::<Result<_>>()?;
    ensure!(!corpus.is_empty(), "empty corpus");
    let mut reports = vec![];
    for path in &args.results {
        let result: Value = serde_json::from_slice(&fs::read(path)?)?;
        let cases = result["cases"].as_array().context("evaluation cases")?;
        ensure!(
            cases.len() == corpus.len(),
            "evaluation does not cover the complete corpus"
        );
        let mut overall = Metrics::default();
        let mut languages = BTreeMap::<String, Metrics>::new();
        let mut clusters = BTreeMap::<String, (i64, usize)>::new();
        let mut errors = vec![];
        let mut seen = std::collections::BTreeSet::new();
        for case in cases {
            let id = case["id"].as_str().context("case id")?;
            ensure!(seen.insert(id), "duplicate evaluation case {id}");
            let source = corpus.get(id).context("evaluation id outside corpus")?;
            let p: Vec<f64> = serde_json::from_value(case["probabilities"].clone())?;
            let r: Vec<f64> = serde_json::from_value(source["expected"].clone())?;
            ensure!(
                !p.is_empty() && p.len() == r.len() && p.iter().chain(&r).all(|v| v.is_finite()),
                "malformed probabilities"
            );
            let gold = source["gold"].as_u64().map(|g| g as usize);
            ensure!(gold.is_none_or(|g| g < p.len()), "invalid gold index");
            overall.add(&p, &r, gold, args.bound);
            let language = source["language"]
                .as_str()
                .or_else(|| source["question"]["lang"].as_str())
                .unwrap_or("unspecified");
            languages
                .entry(language.into())
                .or_default()
                .add(&p, &r, gold, args.bound);
            let error = p
                .iter()
                .zip(&r)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            errors.push(error);
            if let Some(g) = gold {
                let key = source["pair_id"].as_str().unwrap_or(id);
                let cluster = clusters.entry(key.into()).or_default();
                cluster.0 += i64::from(best(&p) == g) - i64::from(best(&r) == g);
                cluster.1 += 1;
            }
        }
        overall.normalize();
        for m in languages.values_mut() {
            m.normalize();
        }
        errors.sort_by(f64::total_cmp);
        let clusters: Vec<_> = clusters.into_values().collect();
        let confidence = if clusters.is_empty() {
            Value::Null
        } else {
            let mut seed = 0xdec1_0008_0123_4567_u64;
            let mut deltas = Vec::with_capacity(10000);
            for _ in 0..10000 {
                let (mut delta, mut count) = (0_i64, 0_usize);
                for _ in 0..clusters.len() {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let c = clusters[seed as usize % clusters.len()];
                    delta += c.0;
                    count += c.1;
                }
                deltas.push(delta as f64 / count as f64);
            }
            deltas.sort_by(f64::total_cmp);
            json!({"lower":deltas[250],"upper":deltas[9749],"resamples":10000,"source_pairs":clusters.len(),"seed":"dec1000801234567"})
        };
        reports.push(json!({"file":path.file_name().context("result filename")?.to_string_lossy(),"precision":result["precision"],"probability_bound":args.bound,"gates":{"probability":overall.above_probability_bound==0,"observed_gold_accuracy":overall.correct>=overall.reference_correct},"p95_probability_error":errors[((errors.len()*95).div_ceil(100)).saturating_sub(1)],"overall":overall,"languages":languages,"paired_accuracy_delta_95_percent_bootstrap":confidence}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"scope":"Paired quantization comparison. Translations resample together by source pair. A confidence interval is not a proof of non-inferiority on other tasks.","results":reports})
        )?
    );
    Ok(())
}
