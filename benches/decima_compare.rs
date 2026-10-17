use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    process::Command,
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    cases: PathBuf,
    #[arg(long, default_value = "benches/decima-lengths.jsonl")]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "1,6")]
    threads: Vec<usize>,
    #[arg(long, default_value_t = 2)]
    repetitions: usize,
    #[arg(long, default_value_t = 7)]
    rounds: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(help = "Retain exact choice embeddings between timed requests; default clears them")]
    #[arg(long)]
    cached_choices: bool,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    executable: PathBuf,
    #[serde(default)]
    args: Vec<String>,
    model: PathBuf,
    precision: String,
}
fn median(x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    if v.len() % 2 == 0 {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.
    } else {
        v[v.len() / 2]
    }
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    ensure!(
        args.repetitions > 0
            && args.rounds > 0
            && !args.threads.is_empty()
            && args.threads.iter().all(|&n| n > 0),
        "invalid matrix dimensions"
    );
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(&args.cases)?)?;
    ensure!(!cases.is_empty(), "empty comparison");
    let mut names = BTreeSet::new();
    for case in &cases {
        ensure!(
            !case.name.is_empty()
                && case
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                && names.insert(&case.name),
            "case names must be unique and contain only letters, digits, - or _"
        );
        ensure!(
            ["f32", "int8", "int8-weights", "int8-full", "int8-compact"]
                .contains(&case.precision.as_str()),
            "unknown precision"
        );
        ensure!(
            case.executable.is_file() && case.model.is_dir(),
            "missing executable or model for {}",
            case.name
        );
    }
    ensure!(!args.output.exists(), "choose a new output directory");
    fs::create_dir_all(&args.output)?;
    let input = fs::read(&args.input)?;
    let mut labeled_cases = 0;
    for line in std::str::from_utf8(&input)?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let row: Value = serde_json::from_str(line)?;
        labeled_cases += usize::from(row["gold"].as_u64().is_some());
    }
    fs::write(args.output.join("input.jsonl"), &input)?;
    let identities=cases.iter().map(|case|Ok(json!({"case":case,"executable_sha256":format!("{:x}",Sha256::digest(fs::read(&case.executable)?))}))).collect::<Result<Vec<_>>>()?;
    let mut runs = vec![];
    let mut medians = BTreeMap::<(String, usize, String), Vec<f64>>::new();
    for repetition in 0..args.repetitions {
        let mut threads = args.threads.clone();
        if repetition % 2 == 1 {
            threads.reverse();
        }
        for workers in threads {
            for offset in 0..cases.len() {
                let index = if repetition % 2 == 0 {
                    offset
                } else {
                    cases.len() - 1 - offset
                };
                let case = &cases[index];
                let name = format!("r{repetition}-t{workers}-{}", case.name);
                eprintln!("{name}");
                let mut command = Command::new(&case.executable);
                command
                    .args(&case.args)
                    .arg("--model")
                    .arg(&case.model)
                    .args([
                        "--threads",
                        &workers.to_string(),
                        "--precision",
                        &case.precision,
                        "evaluate",
                        "--input",
                    ])
                    .arg(&args.input)
                    .args([
                        "--rounds",
                        &args.rounds.to_string(),
                        "--warmup",
                        &args.warmup.to_string(),
                        "--max-error",
                        if case.precision == "f32" {
                            "0.0001"
                        } else {
                            "0.02"
                        },
                    ]);
                if labeled_cases > 0 {
                    command.arg("--require-gold-parity");
                }
                if !args.cached_choices {
                    command.arg("--cold-choices");
                }
                let start = std::time::Instant::now();
                let result = command
                    .output()
                    .with_context(|| format!("run {}", case.name))?;
                fs::write(args.output.join(format!("{name}.json")), &result.stdout)?;
                fs::write(args.output.join(format!("{name}.log")), &result.stderr)?;
                if let Ok(value) = serde_json::from_slice::<Value>(&result.stdout) {
                    if let Some(records) = value["cases"].as_array() {
                        for record in records {
                            let samples: Vec<f64> =
                                serde_json::from_value(record["latency_ms"].clone())?;
                            ensure!(
                                !samples.is_empty()
                                    && samples.iter().all(|v| v.is_finite() && *v >= 0.),
                                "invalid latency samples"
                            );
                            let id = record["id"].as_str().context("fixture id")?.to_owned();
                            medians
                                .entry((case.name.clone(), workers, id))
                                .or_default()
                                .push(median(&samples));
                        }
                    }
                }
                runs.push(json!({"file":format!("{name}.json"),"exit_code":result.status.code(),"elapsed_seconds":start.elapsed().as_secs_f64()}));
                let summary:Vec<_>=medians.iter().map(|((name,threads,id),values)|json!({"case":name,"threads":threads,"fixture":id,"run_medians_ms":values,"median_of_run_medians_ms":median(values)})).collect();
                fs::write(
                    args.output.join("manifest.json"),
                    serde_json::to_vec_pretty(
                        &json!({"cases":identities,"input_sha256":format!("{:x}",Sha256::digest(&input)),"model_revision":decisive::decima::REVISION,"architecture":std::env::consts::ARCH,"available_parallelism":std::thread::available_parallelism().map(|v|v.get()).ok(),"repetitions":args.repetitions,"rounds":args.rounds,"warmup":args.warmup,"cached_choices":args.cached_choices,"labeled_cases":labeled_cases,"order":"Forward cases/threads on even repetitions, reverse on odd; sequential processes.","scope":"Warm resident CPU latency. Run on an otherwise idle host. Failed quality gates are retained, not excluded.","runs":runs,"summary":summary}),
                    )?,
                )?;
            }
        }
    }
    ensure!(
        runs.iter().all(|r| r["exit_code"] == 0),
        "one or more runs failed; inspect retained results and logs"
    );
    Ok(())
}
