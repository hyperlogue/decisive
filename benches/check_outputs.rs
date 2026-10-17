use anyhow::{Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
#[allow(dead_code)]
#[path = "../src/models/laya/benchmark.rs"]
mod benchmark;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(help = "Also reject any accuracy/Brier/NLL regression against these reference outputs")]
    #[arg(long)]
    quality: bool,
}
fn read(path: &PathBuf) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    let input = read(&args.input)?;
    let reference = read(&args.reference)?;
    let candidate = read(&args.candidate)?;
    ensure!(
        !input.is_empty() && input.len() == reference.len() && input.len() == candidate.len(),
        "fixture sizes differ"
    );
    let mut failures = Vec::new();
    let mut max = 0f64;
    for (i, (a, b)) in reference.iter().zip(&candidate).enumerate() {
        match benchmark::compare_outputs(a, b, 1e-4) {
            Ok(error) => max = max.max(error),
            Err(e) => failures.push(json!({"request":i,"error":e.to_string()})),
        }
    }
    let expected_quality = benchmark::quality(&input, &reference)?;
    let actual_quality = benchmark::quality(&input, &candidate)?;
    if args.quality && !expected_quality.is_null() {
        if actual_quality["correct"].as_u64() < expected_quality["correct"].as_u64() {
            failures.push(json!({"metric":"correct","error":"accuracy regressed"}));
        }
        for key in ["brier", "nll"] {
            if let (Some(a), Some(b)) =
                (actual_quality[key].as_f64(), expected_quality[key].as_f64())
            {
                if a > b + 1e-12 {
                    failures.push(json!({"metric":key,"reference":b,"candidate":a,"error":"quality regressed"}));
                }
            }
        }
    }
    println!(
        "{}",
        json!({"requests":input.len(),"numerical_bound":1e-4,"max_error_among_passing_requests":max,"reference_quality":expected_quality,"candidate_quality":actual_quality,"quality_gate_enabled":args.quality,"failures":failures})
    );
    ensure!(
        failures.is_empty(),
        "output/quality gate failed; details retained in JSON"
    );
    Ok(())
}
