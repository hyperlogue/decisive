use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
#[allow(dead_code)]
#[path = "../src/models/laya/benchmark.rs"]
mod benchmark;

#[derive(Parser)]
struct Args {
    input: PathBuf,
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    let mut result: Value = serde_json::from_slice(&fs::read(args.input)?)?;
    ensure!(
        result.get("error").is_none(),
        "browser capture failed: {result}"
    );
    let records = result["records"]
        .as_array()
        .context("missing browser records")?;
    ensure!(!records.is_empty(), "no predictions");
    let mut maximum = 0.0_f64;
    for record in records {
        maximum = maximum.max(benchmark::compare_outputs(
            &record["reference"],
            &record["output"],
            1e-4,
        )?);
    }
    result["checked_predictions"] = json!(records.len());
    result["maximum_output_error"] = json!(maximum);
    result["comparison"] = json!(
        "Rust benchmark::compare_outputs: identical discrete outputs and usage, unchanged argmax/binary thresholds, final output error <=1e-4 plus1e-12 representation slack. Finite browser correctness check only."
    );
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
