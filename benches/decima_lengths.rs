use anyhow::{Context, Result, ensure};
use clap::Parser;
use decisive::decima::{Kernel, Model, Options, Precision, Question, Session};
use serde_json::{Value, json};
use std::{fs, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/decima-small")]
    model: PathBuf,
    #[arg(long, default_value = "benches/decima-reference.jsonl")]
    source: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "32,64,128,256,512")]
    tokens: Vec<usize>,
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    ensure!(
        !args.tokens.is_empty() && args.tokens.iter().all(|&n| (32..=512).contains(&n)),
        "lengths must be 32..=512"
    );
    let source = fs::read_to_string(&args.source)?;
    let rows: Vec<Value> = source
        .lines()
        .filter(|s| !s.is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let seed = rows
        .iter()
        .find(|r| {
            r["state_tokens"] == 512
                && r["state"]
                    .as_str()
                    .is_some_and(|s| s.starts_with("record "))
        })
        .context("missing 512-token repeated-record seed")?;
    let seed_state = seed["state"].as_str().unwrap();
    let count = seed_state
        .split_whitespace()
        .take_while(|w| *w == "record")
        .count();
    let suffix = seed_state
        .strip_prefix(&"record ".repeat(count))
        .context("nonuniform seed padding")?;
    let overhead = 512usize
        .checked_sub(count)
        .context("seed has more words than tokens")?;
    let question: Question = serde_json::from_value(seed["question"].clone())?;
    let model = Model::load(
        &args.model,
        Options {
            precision: Precision::F32,
            kernel: Kernel::Portable,
        },
    )?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let mut result = Vec::new();
    for &tokens in &args.tokens {
        let repeats = tokens
            .checked_sub(overhead)
            .context("length shorter than question framing")?;
        let state = format!("{}{suffix}", "record ".repeat(repeats));
        let mut session = Session::default();
        session.set_cache_bytes(0);
        let expected = pool.install(|| model.decide(&state, &question, &mut session))?;
        ensure!(
            session.stats.state_tokens == tokens as u64,
            "requested {tokens} tokens, executed {}",
            session.stats.state_tokens
        );
        result.push(json!({"id":format!("timing-{tokens}"),"state":state,"question":question,"expected":expected,"state_tokens":tokens,"reference":{"implementation":"Decisive native Rust FP32 portable matrix path","checkpoint_revision":decisive::decima::REVISION,"purpose":"timing and numerical regression; not a labeled quality case"}}));
    }
    let mut out = fs::File::create_new(&args.output)?;
    for row in result {
        writeln!(out, "{}", serde_json::to_string(&row)?)?;
    }
    Ok(())
}
