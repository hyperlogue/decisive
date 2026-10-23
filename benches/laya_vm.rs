use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use decisive::laya::{Model, Scratch, cpu, weights::Storage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
#[path = "../src/models/laya/artifact.rs"]
mod artifact;
#[allow(dead_code)]
#[path = "../src/models/laya/benchmark.rs"]
mod benchmark;
use benchmark::paired_interval;

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Study {
    Strata,
    Projection,
    Tails,
    Avx2,
    Long,
}

#[derive(Parser)]
#[command(about = "Reproducible Laya CPU experiments; run from the repository root")]
pub(crate) struct Args {
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
    #[arg(
        help = "Choose an optimization study; long uses full 512/1024/2048-token English inputs"
    )]
    #[arg(long, value_enum, default_value = "strata")]
    study: Study,
    #[arg(help = "Must not exist. Raw results are retained if a later phase fails")]
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = cpu::hardware::available_threads())]
    threads: usize,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "english,multilingual,typed-decisions"
    )]
    variants: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "f16,f32")]
    storage: Vec<String>,
    #[arg(help = "Rounded up to complete balanced cycles in the exploratory sweep")]
    #[arg(long, default_value_t = 24)]
    rounds: usize,
    #[arg(long, default_value_t = 40)]
    confirmation_rounds: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long, default_value_t = 4)]
    startup_repeats: usize,
    #[arg(help = "Download the pinned assets before hashing and validation")]
    #[arg(long)]
    fetch: bool,
    #[arg(help = "Force portable Rust; unavailable packed-only experiments are skipped")]
    #[arg(long)]
    portable: bool,
    #[arg(
        help = "Smoke-test orchestration only: English FP16, two cases, two rounds, one startup pair"
    )]
    #[arg(long)]
    quick: bool,
    #[arg(help = "Print the resolved protocol without loading models or writing results")]
    #[arg(long)]
    plan: bool,
    #[arg(long, hide = true)]
    worker: Option<PathBuf>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Job {
    phase: String,
    model: PathBuf,
    variant: String,
    storage: String,
    packed: bool,
    threads: usize,
    cases: Vec<String>,
    rounds: usize,
    warmup: usize,
    quick: bool,
    reference: PathBuf,
    #[serde(default)]
    long: bool,
}
const WORKLOADS: &str = "benches/strata.jsonl";
const LONG_WORKLOADS: &str = "benches/long-inputs.jsonl";
fn baseline(study: Study) -> &'static str {
    if study == Study::Long {
        "pipeline"
    } else {
        "baseline"
    }
}
const ASSETS: [&str; 6] = [
    "beam-revision.json",
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];
fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
fn save(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn configure(name: &str) -> Result<Scratch> {
    if name.starts_with("long-") || name == "selected-head" {
        let mut s = Scratch::default();
        s.enable_avx2_pipeline();
        s.set_long_attention(true);
        match name {
            "long-attention" => {}
            "long-exp" => s.set_attention_math(cpu::AttentionMath::TableExp),
            "long-rows24" => s.set_projection_strategy(cpu::ProjectionStrategy::LongRows(24)),
            "selected-head" => s.set_prune_head(true),
            "long-combined" => s.enable_avx2_long_pipeline(),
            _ => anyhow::bail!("unknown long candidate"),
        }
        return Ok(s);
    }
    let (reuse, fused, group) = match name {
        "baseline" => (false, false, 1),
        "attention" => (true, false, 1),
        "fused" => (false, true, 1),
        "group4" => (false, false, 4),
        "group16" => (false, false, 16),
        "combined" => (true, true, 1),
        "block128"
        | "block256"
        | "block512"
        | "tails"
        | "pipeline"
        | "pipeline-no-worker"
        | "pipeline-no-attention" => (false, false, 1),
        _ => anyhow::bail!("unknown candidate"),
    };
    let mut scratch = Scratch::default();
    scratch.set_optimizations(reuse, fused, group)?;
    if name.starts_with("pipeline") {
        scratch.enable_avx2_pipeline();
        if name == "pipeline-no-attention" {
            scratch.set_packed_attention(false);
        }
        if name == "pipeline-no-worker" {
            scratch.set_worker_loop(false);
        }
    }
    if name == "tails" {
        scratch.set_tiled_projection_tails(true);
    }
    if let Some(block) = name.strip_prefix("block") {
        scratch.set_projection_block(block.parse()?)?;
    }
    Ok(scratch)
}
// Each candidate occupies every position twice per 2*N cycle (AB/BA for N=2).
fn order(n: usize, round: usize) -> Vec<usize> {
    let mut result: Vec<_> = (0..n).map(|i| (i + round / 2) % n).collect();
    if round % 2 == 1 {
        result.reverse();
    }
    result
}
fn rounded(rounds: usize, n: usize) -> usize {
    rounds.div_ceil(2 * n) * 2 * n
}
fn host_snapshot() -> Value {
    let load = fs::read_to_string("/proc/loadavg").ok().map(|s| {
        s.split_whitespace()
            .take(3)
            .map(str::to_owned)
            .collect::<Vec<_>>()
    });
    let cpu = fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|s| s.lines().find(|s| s.starts_with("cpu ")).map(str::to_owned));
    json!({"unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        "load_1_5_15m": load, "aggregate_cpu_ticks":cpu,
        "cpu_pressure":fs::read_to_string("/proc/pressure/cpu").ok()})
}
fn environment() -> Value {
    let selected = |file: &str, keys: &[&str]| -> Vec<String> {
        fs::read_to_string(file)
            .unwrap_or_default()
            .lines()
            .filter(|line| keys.iter().any(|key| line.starts_with(key)))
            .map(str::to_owned)
            .collect()
    };
    let mut cpu = selected("/proc/cpuinfo", &["model name", "Hardware"]);
    cpu.sort();
    cpu.dedup();
    json!({"cpu_models":cpu,
        "memory":selected("/proc/meminfo", &["MemTotal:", "MemAvailable:"]),
        "affinity":selected("/proc/self/status", &["Cpus_allowed_list:"]),
        "os_release":selected("/etc/os-release", &["NAME=", "VERSION_ID=", "PRETTY_NAME="]),
        "cgroup_cpu_max":fs::read_to_string("/sys/fs/cgroup/cpu.max").ok(),
        "cgroup_cpuset":fs::read_to_string("/sys/fs/cgroup/cpuset.cpus.effective").ok(),
        "cgroup_memory_max":fs::read_to_string("/sys/fs/cgroup/memory.max").ok(),
        "note":"No hostname, username, environment dump, process list or network identifiers collected. Cgroup files may be absent or refer to the visible hierarchy root."})
}
fn quality_gate(reference: &Value, actual: &Value) -> Result<()> {
    ensure!(
        actual["correct"] == reference["correct"],
        "labeled decisions regressed"
    );
    for metric in ["brier", "nll"] {
        if let (Some(a), Some(b)) = (actual[metric].as_f64(), reference[metric].as_f64()) {
            ensure!(a <= b + 1e-12, "{metric} regressed versus current baseline");
        }
    }
    Ok(())
}
fn process_memory() -> Value {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return Value::Null;
    };
    let get = |key: &str| {
        status.lines().find_map(|line| {
            line.strip_prefix(key)
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse::<u64>().ok())
        })
    };
    json!({"rss_kib":get("VmRSS:"),"peak_rss_kib":get("VmHWM:"),
        "scope":"Linux process RSS and lifetime high-water mark; includes model loading and allocator retention"})
}
fn worker(job: &Job) -> Result<Value> {
    let before = host_snapshot();
    let start = Instant::now();
    rayon::ThreadPoolBuilder::new()
        .num_threads(job.threads)
        .build_global()?;
    let pool_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    let model = Model::load_with_kernel(
        &job.model,
        if job.storage == "f16" {
            Storage::F16
        } else {
            Storage::F32
        },
        if job.packed {
            cpu::Kernel::X86Avx2
        } else {
            cpu::Kernel::Portable
        },
    )?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let memory_after_load = process_memory();
    let workloads = if job.quick {
        read_jsonl("benches/smoke.jsonl")?
    } else {
        read_jsonl(if job.long { LONG_WORKLOADS } else { WORKLOADS })?
    };
    let mut validation = Vec::new();
    if job.phase == "validate" {
        let mut fixtures = vec![(
            "benches/heldout.jsonl".to_owned(),
            format!("tests/fixtures/outputs/{}.jsonl", job.variant),
        )];
        if job.variant == "english" {
            fixtures.extend(["smoke", "contracts", "batch"].map(|s| {
                (
                    format!("benches/{s}.jsonl"),
                    format!("tests/fixtures/outputs/{s}.jsonl"),
                )
            }));
            if job.long {
                fixtures.push((
                    LONG_WORKLOADS.into(),
                    "tests/fixtures/outputs/long-inputs.jsonl".into(),
                ));
                fixtures.push((
                    "benches/long-quality.jsonl".into(),
                    "tests/fixtures/outputs/long-quality.jsonl".into(),
                ));
            }
        } else if job.variant == "multilingual" {
            fixtures.push((
                "benches/multilingual.jsonl".into(),
                "tests/fixtures/outputs/languages.jsonl".into(),
            ));
        }
        if job.quick {
            fixtures = vec![(
                "benches/smoke.jsonl".into(),
                "tests/fixtures/outputs/smoke.jsonl".into(),
            )];
        }
        for (input, expected) in fixtures {
            let requests = read_jsonl(&input)?;
            let reference = read_jsonl(&expected)?;
            ensure!(
                !requests.is_empty() && requests.len() == reference.len(),
                "fixture length differs"
            );
            let mut baseline_quality = Value::Null;
            for case in &job.cases {
                let mut scratch = configure(case)?;
                let mut outputs = Vec::new();
                let mut error = 0.0_f64;
                for (r, e) in requests.iter().zip(&reference) {
                    let output = model.predict_json(r.clone(), &mut scratch)?;
                    error = error.max(benchmark::compare_outputs(e, &output, 1e-4)?);
                    outputs.push(output);
                }
                let quality = benchmark::quality(&requests, &outputs)?;
                if case == &job.cases[0] {
                    baseline_quality = quality.clone();
                }
                quality_gate(&baseline_quality, &quality)?;
                validation.push(json!({"fixture":input,"candidate":case,"max_output_error":error,"quality":quality,"sdk_quality":benchmark::quality(&requests,&reference)?}));
            }
        }
        let mut scratch = configure(&job.cases[0])?;
        let mut file = fs::File::create(&job.reference)?;
        for r in &workloads {
            writeln!(file, "{}", model.predict_json(r.clone(), &mut scratch)?)?;
        }
        return Ok(
            json!({"validation":validation,"pool_ms":pool_ms,"load_ms":load_ms,"host_before":before,"host_after":host_snapshot(),
            "memory_after_load":memory_after_load,"memory_after_workloads":process_memory(),
            "reference_scope":"SDK fixtures gate correctness, including explicit English long lengths in the long study. Generated timing references detect changes from the first candidate; they are not independent oracles."}),
        );
    }
    let reference = read_jsonl(&job.reference)?;
    ensure!(
        workloads.len() == reference.len(),
        "timing reference length differs"
    );
    let mut scratch = job
        .cases
        .iter()
        .map(|s| configure(s))
        .collect::<Result<Vec<_>>>()?;
    let mut samples = Vec::new();
    let mut first_session = Vec::new();
    for phase in 0..2 {
        let rounds = if phase == 0 { job.warmup } else { job.rounds };
        for round in 0..rounds {
            let mut row = Vec::new();
            for ci in order(job.cases.len(), round) {
                let request_count = if job.phase == "startup" {
                    1
                } else {
                    workloads.len()
                };
                for j in 0..request_count {
                    let ri = (j + round) % request_count;
                    let request = workloads[ri].clone();
                    let start = Instant::now();
                    let result = model.predict_json(request, &mut scratch[ci])?;
                    let ms = start.elapsed().as_secs_f64() * 1000.;
                    let error = benchmark::compare_outputs(&reference[ri], &result, 1e-4)?;
                    let sample = json!({"round":round,"case":job.cases[ci],"request":ri,"ms":ms,"max_output_error":error});
                    if (phase == 0 || job.warmup == 0) && round == 0 {
                        first_session.push(sample.clone());
                    }
                    if phase == 1 {
                        row.push(sample);
                    }
                }
            }
            if phase == 1 {
                samples.extend(row);
            }
            eprintln!(
                "{} {} {} {}/{}",
                job.variant,
                job.storage,
                if phase == 0 { "warmup" } else { "measured" },
                round + 1,
                rounds
            );
        }
    }
    Ok(
        json!({"pool_ms":pool_ms,"load_ms":load_ms,"first_session":first_session,"samples":samples,"host_before":before,"host_after":host_snapshot(),
            "memory_after_load":memory_after_load,"memory_after_workloads":process_memory(),
            "retained_scratch":job.cases.iter().zip(&scratch).map(|(case,s)|json!({"candidate":case,"attention_bytes":s.attention_retained_bytes(),"projection_bytes":s.projection_retained_bytes()})).collect::<Vec<_>>()}),
    )
}
fn digest(path: &Path) -> Result<String> {
    let out = Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .context("install GNU coreutils (sha256sum)")?;
    ensure!(out.status.success(), "hash failed");
    let hash = String::from_utf8(out.stdout)?
        .split_whitespace()
        .next()
        .context("empty hash")?
        .to_owned();
    ensure!(
        hash.len() == 64 && hash.bytes().all(|c| c.is_ascii_hexdigit()),
        "invalid SHA256"
    );
    Ok(hash)
}
fn tree_hashes(root: &Path, result: &mut serde_json::Map<String, Value>) -> Result<()> {
    for e in fs::read_dir(root)? {
        let p = e?.path();
        if p.is_dir() {
            tree_hashes(&p, result)?;
        } else {
            result.insert(p.to_string_lossy().into_owned(), json!(digest(&p)?));
        }
    }
    Ok(())
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn summarize_run(raw: &Value, job: &Job) -> Result<Vec<Value>> {
    let samples = raw["samples"].as_array().context("missing samples")?;
    let count = samples
        .iter()
        .filter_map(|v| v["request"].as_u64())
        .max()
        .context("empty samples")?
        + 1;
    ensure!(
        samples.len() == job.rounds * job.cases.len() * count as usize,
        "incomplete schedule"
    );
    for (round, rows) in samples.chunks(job.cases.len() * count as usize).enumerate() {
        for (position, case_rows) in rows.chunks(count as usize).enumerate() {
            let candidate = &job.cases[order(job.cases.len(), round)[position]];
            for (j, sample) in case_rows.iter().enumerate() {
                ensure!(
                    sample["round"] == round
                        && sample["case"] == *candidate
                        && sample["request"] == (j + round) % count as usize,
                    "sample order differs from the balanced protocol"
                );
            }
        }
    }
    let mut result = Vec::new();
    for request in 0..count {
        let baseline: Vec<_> = samples
            .iter()
            .filter(|v| v["request"] == request && v["case"] == job.cases[0])
            .collect();
        for case in &job.cases {
            let rows: Vec<_> = samples
                .iter()
                .filter(|v| v["request"] == request && v["case"] == *case)
                .collect();
            ensure!(rows.len() == job.rounds, "incomplete sample series");
            let times: Vec<_> = rows.iter().map(|v| v["ms"].as_f64().unwrap()).collect();
            let paired = if baseline.len() == rows.len() && case != &job.cases[0] {
                let ratios: Vec<_> = rows
                    .iter()
                    .zip(&baseline)
                    .map(|(a, b)| {
                        assert_eq!(a["round"], b["round"]);
                        a["ms"].as_f64().unwrap() / b["ms"].as_f64().unwrap()
                    })
                    .collect();
                paired_interval(&ratios)
            } else {
                Value::Null
            };
            result.push(json!({"phase":job.phase,"variant":job.variant,"storage":job.storage,"candidate":case,"request":request,"timing":benchmark::summarize(&times),"paired":paired}));
        }
    }
    Ok(result)
}
fn child(out: &Path, name: &str, job: &Job) -> Result<Value> {
    eprintln!("Starting {name}");
    let job_file = out.join(format!("{name}.job.json"));
    save(&job_file, &serde_json::to_value(job)?)?;
    let raw_file = out.join(format!("{name}.json"));
    let start = Instant::now();
    let status = Command::new(std::env::current_exe()?)
        .arg("--worker")
        .arg(&job_file)
        .env("TOKENIZERS_PARALLELISM", "false")
        .stdout(fs::File::create(&raw_file)?)
        .stderr(fs::File::create(out.join(format!("{name}.log")))?)
        .stdin(Stdio::null())
        .status()?;
    let wall_ms = start.elapsed().as_secs_f64() * 1000.;
    fs::remove_file(job_file)?;
    ensure!(
        status.success(),
        "phase {name} failed; see its log (no successful report issued)"
    );
    let mut raw: Value = serde_json::from_slice(&fs::read(&raw_file)?)?;
    raw["child_wall_ms"] = json!(wall_ms);
    save(raw_file, &raw)?;
    Ok(raw)
}
fn report(out: &Path, summary: &Value) -> Result<()> {
    let mut rows = String::new();
    for r in summary["comparisons"].as_array().unwrap() {
        let p = &r["paired"];
        let interval = if let Some(ratio) = p["geometric_ratio"].as_f64() {
            format!(
                "{:+.1}% [{:+.1}%, {:+.1}%]",
                (ratio - 1.) * 100.,
                (p["lower_95"].as_f64().unwrap() - 1.) * 100.,
                (p["upper_95"].as_f64().unwrap() - 1.) * 100.
            )
        } else {
            "—".into()
        };
        rows += &format!(
            "<tr><td>{} / {}</td><td>{} · {}</td><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2}</td><td>{}</td></tr>",
            escape(r["variant"].as_str().unwrap()),
            escape(r["storage"].as_str().unwrap()),
            escape(r["phase"].as_str().unwrap()),
            r["request"],
            escape(r["candidate"].as_str().unwrap()),
            r["timing"]["requests"],
            r["timing"]["p50_ms"].as_f64().unwrap(),
            r["timing"]["p95_ms"].as_f64().unwrap(),
            r["timing"]["requests_per_second"].as_f64().unwrap(),
            interval
        );
    }
    let mut startup = String::new();
    for r in summary["startup"].as_array().unwrap() {
        startup += &format!(
            "<tr><td>{} / {}</td><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2}</td></tr>",
            escape(r["variant"].as_str().unwrap()),
            escape(r["storage"].as_str().unwrap()),
            escape(r["candidate"].as_str().unwrap()),
            r["repeat"],
            r["load_ms"].as_f64().unwrap(),
            r["first_ms"].as_f64().unwrap(),
            r["following"]["p50_ms"].as_f64().unwrap()
        );
    }
    let mut memory = String::new();
    for r in summary["memory"].as_array().into_iter().flatten() {
        let mib = |v: &Value| {
            v.as_f64()
                .map(|n| format!("{:.2}", n / 1024.))
                .unwrap_or("unavailable".into())
        };
        memory += &format!(
            "<tr><td>{} / {}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape(r["variant"].as_str().unwrap()),
            escape(r["storage"].as_str().unwrap()),
            escape(r["candidate"].as_str().unwrap()),
            mib(&r["after_load"]["rss_kib"]),
            mib(&r["after_workloads"]["rss_kib"]),
            mib(&r["after_workloads"]["peak_rss_kib"])
        );
    }
    let html = format!(
        r#"<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Decisive CPU benchmark</title><style>body{{font:16px/1.6 system-ui;background:#0d1721;color:#e3edf3;max-width:1400px;margin:40px auto;padding:0 20px}}h1{{font-size:36px}}a{{color:#7bdace}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #344352;white-space:nowrap}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.note{{background:#203444;padding:18px;border-radius:8px}}</style><h1>Decisive CPU benchmark</h1><p class="note">{}</p><p>Accuracy gates passed before timing. Every timed prediction is checked again. Warm latency includes tokenization and predict_json; excludes request cloning, JSON file I/O, validation, HTTP, queueing, and loading. Serial req/s is not concurrent serving capacity. No allocator counter is installed.</p><p>FP16 denotes weight storage with FP32 accumulation. Reference inputs and all source, model and executable hashes are recorded in manifest.json. Labeled quality covers a finite corpus. This report compares Decisive candidates, not llama.cpp.</p><h2>Warm results</h2><p>Nearest-rank percentiles. Relative change uses within-round geometric latency ratios and descriptive 95% bootstrap intervals; negative means faster. Confirmation is fixed in advance, not chosen from the sweep. Host load/pressure snapshots are in every raw phase file; they cannot certify an idle host.</p><div class="scroll"><table><thead><tr><th>Model / weights</th><th>Phase · request index</th><th>Candidate</th><th>Samples</th><th>p50 ms</th><th>p95 ms</th><th>Serial req/s</th><th>Paired change [95% interval]</th></tr></thead><tbody>{rows}</tbody></table></div><h2>Fresh-process startup</h2><p>Individual repetitions, first workload only. OS file cache is not purged. Model load and first inference are separate; pool creation and complete process wall time remain in raw JSON. Following p50 uses the next three predictions.</p><div class="scroll"><table><thead><tr><th>Model / weights</th><th>Candidate</th><th>Repeat</th><th>Load ms</th><th>First ms</th><th>Following p50 ms</th></tr></thead><tbody>{startup}</tbody></table></div><h2>Separate-process memory</h2><p>One candidate per fresh process; one warmup and one checked pass over all timed inputs. RSS includes allocator retention; peak includes loading. These runs are not timing samples.</p><div class="scroll"><table><thead><tr><th>Model / weights</th><th>Candidate</th><th>After load MiB</th><th>After inputs MiB</th><th>Lifetime peak MiB</th></tr></thead><tbody>{memory}</tbody></table></div><details><summary>Complete summary JSON</summary><pre>{}</pre></details></html>"#,
        if summary["quick"] == true {
            "SMOKE TEST ONLY — not performance evidence; no complete task-quality gate."
        } else {
            "Completed CPU protocol. Repeat on a quiet VM with a new output directory before accepting performance claims."
        },
        escape(&serde_json::to_string_pretty(summary)?)
    );
    fs::write(out.join("index.html"), html)?;
    Ok(())
}
fn run_study(args: &Args, out: &Path, packed: bool, cases: &[String], plan: &Value) -> Result<()> {
    let refs: Value = serde_json::from_str(include_str!("../references.json"))?;
    ensure!(
        serde_json::from_slice::<Value>(&fs::read("references.json")?)? == refs,
        "binary has different reference pins; rebuild"
    );
    for variant in &args.variants {
        let dir = args.models_dir.join(if variant == "english" {
            "laya".into()
        } else {
            format!("laya-{variant}")
        });
        if args.fetch {
            artifact::fetch(variant, &dir)?;
        }
        let stamp: Value = serde_json::from_slice(
            &fs::read(dir.join("beam-revision.json"))
                .context("fetch the pinned model first, or add --fetch")?,
        )?;
        ensure!(
            stamp == refs["models"][variant],
            "model revision differs from references.json"
        );
    }
    eprintln!("Hashing sources, executable and model assets before inference");
    let mut source = serde_json::Map::new();
    for name in ["src", "scripts", "benches", "examples", "tests/fixtures"] {
        tree_hashes(Path::new(name), &mut source)?;
    }
    for name in ["Cargo.toml", "Cargo.lock", "references.json", WORKLOADS] {
        source.insert(name.into(), json!(digest(Path::new(name))?));
    }
    let mut models = serde_json::Map::new();
    for variant in &args.variants {
        let dir = args.models_dir.join(if variant == "english" {
            "laya".into()
        } else {
            format!("laya-{variant}")
        });
        let mut hashes = serde_json::Map::new();
        for name in ASSETS {
            hashes.insert(name.into(), json!(digest(&dir.join(name))?));
        }
        models.insert(
            variant.clone(),
            json!({"pin":refs["models"][variant],"sha256":hashes}),
        );
    }
    let version = |program: &str, arg: &str| {
        Command::new(program)
            .arg(arg)
            .output()
            .ok()
            .filter(|r| r.status.success())
            .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned())
    };
    save(
        out.join("manifest.json"),
        &json!({"protocol_version":4,"plan":plan,"references":refs,"hardware":cpu::hardware::detect(),"environment":environment(),"os":std::env::consts::OS,"rustc":version("rustc","--version"),"cargo":version("cargo","--version"),"executable_sha256":digest(&std::env::current_exe()?)?,"source_sha256":source,"models":models,"host_before":host_snapshot(),"build_note":"Build separately with bash scripts/build_bench.sh laya_vm --features bench. Source hashes describe files at run time; executable hash identifies actual code. Tool versions are observed at run time. Build flags are not reconstructed from an executable."}),
    )?;
    let mut jobs = Vec::new();
    let mut summary = json!({"protocol_version":4,"quick":args.quick,"comparisons":[],"startup":[],"memory":[],"validation":[]});
    for variant in &args.variants {
        for storage in &args.storage {
            let name = format!("{variant}-{storage}");
            let job = Job {
                phase: "validate".into(),
                model: args.models_dir.join(if variant == "english" {
                    "laya".into()
                } else {
                    format!("laya-{variant}")
                }),
                variant: variant.clone(),
                storage: storage.clone(),
                packed,
                threads: args.threads,
                cases: cases.to_vec(),
                rounds: 0,
                warmup: 0,
                quick: args.quick,
                reference: out.join(format!("{name}-reference.jsonl")),
                long: args.study == Study::Long,
            };
            let raw = child(out, &format!("{name}-validate"), &job)?;
            summary["validation"]
                .as_array_mut()
                .unwrap()
                .push(json!({"variant":variant,"storage":storage,"result":raw}));
            jobs.push(job);
        }
    }
    let confirm = confirmation(args.study, packed);
    let baseline = baseline(args.study);
    for base in jobs {
        let name = format!("{}-{}", base.variant, base.storage);
        for phase in ["exploratory", "confirmation"] {
            let mut job = base.clone();
            job.phase = phase.into();
            if phase == "confirmation" || args.quick {
                job.cases = vec![baseline.into(), confirm.into()];
            }
            job.rounds = if args.quick {
                2
            } else if phase == "confirmation" {
                args.confirmation_rounds
            } else {
                rounded(args.rounds, job.cases.len())
            };
            job.warmup = args.warmup;
            let raw = child(out, &format!("{name}-{phase}"), &job)?;
            summary["comparisons"]
                .as_array_mut()
                .unwrap()
                .extend(summarize_run(&raw, &job)?);
        }
        for case in [baseline, confirm] {
            let mut job = base.clone();
            job.phase = "memory".into();
            job.cases = vec![case.into()];
            job.warmup = 1;
            job.rounds = 1;
            let raw = child(out, &format!("{name}-memory-{case}"), &job)?;
            summary["memory"].as_array_mut().unwrap().push(json!({"variant":job.variant,"storage":job.storage,"candidate":case,
                "after_load":raw["memory_after_load"],"after_workloads":raw["memory_after_workloads"],"retained_scratch":raw["retained_scratch"],
                "scope":"Separate single-candidate process: one warmup and one checked pass across every timed workload; Linux RSS if available. Not a timing sample."}));
        }
        for repeat in 0..args.startup_repeats {
            for ci in order(2, repeat) {
                let mut job = base.clone();
                job.phase = "startup".into();
                job.cases = vec![[baseline, confirm][ci].into()];
                job.rounds = 4;
                job.warmup = 0;
                let raw = child(
                    out,
                    &format!("{name}-startup-{repeat}-{}", job.cases[0]),
                    &job,
                )?;
                let samples = raw["samples"].as_array().unwrap();
                let following: Vec<_> = samples[1..]
                    .iter()
                    .map(|v| v["ms"].as_f64().unwrap())
                    .collect();
                summary["startup"].as_array_mut().unwrap().push(json!({"variant":job.variant,"storage":job.storage,"candidate":job.cases[0],"repeat":repeat,"pool_ms":raw["pool_ms"],"load_ms":raw["load_ms"],"first_ms":samples[0]["ms"],"following":benchmark::summarize(&following),"child_wall_ms":raw["child_wall_ms"]}));
            }
        }
    }
    save(out.join("summary.json"), &summary)?;
    report(out, &summary)?;
    save(
        out.join("status.json"),
        &json!({"status":"complete","performance_evidence":!args.quick,"host_after":host_snapshot()}),
    )?;
    Ok(())
}
fn confirmation(study: Study, packed: bool) -> &'static str {
    if study == Study::Long {
        "long-combined"
    } else if study == Study::Avx2 {
        "pipeline"
    } else if study == Study::Tails {
        "tails"
    } else if study == Study::Projection {
        "block256"
    } else if packed {
        "combined"
    } else {
        "attention"
    }
}
pub(crate) fn run(mut args: Args) -> Result<()> {
    if let Some(path) = &args.worker {
        println!(
            "{}",
            worker(&serde_json::from_slice::<Job>(&fs::read(path)?)?)?
        );
        return Ok(());
    }
    if args.quick {
        args.variants = vec!["english".into()];
        args.storage = vec!["f16".into()];
        args.warmup = 1;
        args.startup_repeats = 1;
    }
    ensure!(
        args.threads > 0
            && args.rounds > 0
            && args.confirmation_rounds > 0
            && args.warmup > 0
            && args.startup_repeats > 0,
        "counts must be positive"
    );
    ensure!(
        args.quick || (args.confirmation_rounds % 2 == 0 && args.startup_repeats % 2 == 0),
        "confirmation rounds and startup repeats must be even for balanced A/B ordering"
    );
    ensure!(
        !args.variants.is_empty()
            && args
                .variants
                .iter()
                .all(|s| ["english", "multilingual", "typed-decisions"].contains(&s.as_str())),
        "unknown variant"
    );
    ensure!(
        !args.storage.is_empty()
            && args
                .storage
                .iter()
                .all(|s| ["f16", "f32"].contains(&s.as_str())),
        "storage must be f16,f32"
    );
    for values in [&args.variants, &args.storage] {
        let mut unique = values.clone();
        unique.sort();
        unique.dedup();
        ensure!(unique.len() == values.len(), "duplicate matrix entry");
    }
    ensure!(
        args.study != Study::Long || args.variants == ["english"],
        "the long study uses exact-length English fixtures; pass --variants english"
    );
    let packed = !args.portable && cpu::packed_supported();
    ensure!(
        args.plan || args.study == Study::Strata || packed,
        "projection study requires packed AVX2/FMA/F16C; the strata study supports portable Rust"
    );
    let mut cases: Vec<String> = if args.study == Study::Long {
        vec![
            "pipeline",
            "long-attention",
            "long-exp",
            "long-rows24",
            "selected-head",
            "long-combined",
        ]
    } else if args.study == Study::Avx2 {
        vec![
            "baseline",
            "combined",
            "pipeline-no-worker",
            "pipeline-no-attention",
            "pipeline",
        ]
    } else if args.study == Study::Tails {
        vec!["baseline", "tails"]
    } else if args.study == Study::Projection {
        vec!["baseline", "block128", "block256", "block512"]
    } else if packed {
        vec![
            "baseline",
            "attention",
            "fused",
            "group4",
            "group16",
            "combined",
        ]
    } else {
        vec!["baseline", "attention"]
    }
    .into_iter()
    .map(str::to_owned)
    .collect();
    if args.quick {
        cases = vec![
            baseline(args.study).into(),
            confirmation(args.study, packed).into(),
        ];
    }
    let plan = json!({"study":match args.study {Study::Long=>"long",Study::Avx2=>"avx2",Study::Projection=>"projection",Study::Tails=>"tails",Study::Strata=>"strata"},"baseline_candidate":baseline(args.study),"confirmation_candidate":confirmation(args.study,packed),"variants":args.variants,"storage":args.storage,"threads":args.threads,"kernel":if packed {"packed"} else {"portable"},"candidates":cases,"warmup":args.warmup,"exploratory_rounds":if args.quick {2} else {rounded(args.rounds,cases.len())},"confirmation_rounds":if args.quick {2} else {args.confirmation_rounds},"startup_repeats_per_case":args.startup_repeats,"quick":args.quick,"timed_workloads":if args.quick {"English SDK smoke"} else if args.study==Study::Long {"512/1024/2048 full English-token sequences with explicit max_len; synthetic length workload, no result/token cache"} else {"31/112/512 English-token inputs, retokenized per checkpoint; three fixed requests"},"skipped":if packed {"none"} else {"packed-only fusion and panel grouping"},"scope":"native serial predict_json, startup and finite accuracy; CPU only, no HTTP/concurrency/llama.cpp/browser benchmarks"});
    if args.plan {
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }
    let out = args
        .out
        .as_ref()
        .context("--out is required (must be a new directory)")?;
    ensure!(
        !out.exists(),
        "output directory already exists; choose a new run directory"
    );
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(out)?;
    save(out.join("plan.json"), &plan)?;
    save(out.join("status.json"), &json!({"status":"running"}))?;
    if let Err(e) = run_study(&args, out, packed, &cases, &plan) {
        save(
            out.join("status.json"),
            &json!({"status":"failed","note":"Partial evidence retained; see phase logs. Do not treat this run as accepted performance evidence."}),
        )?;
        return Err(e);
    }
    eprintln!(
        "Finished. Open {}/index.html; share the entire output directory.",
        out.display()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn balanced_positions_and_pair_order() {
        for n in [2, 4, 6] {
            let mut counts = vec![vec![0; n]; n];
            for round in 0..2 * n {
                for (position, candidate) in order(n, round).into_iter().enumerate() {
                    counts[candidate][position] += 1;
                }
            }
            assert!(counts.iter().flatten().all(|&n| n == 2));
        }
        for round in (0..40).step_by(2) {
            let mut reversed = order(2, round);
            reversed.reverse();
            assert_eq!(order(2, round + 1), reversed);
        }
        assert_eq!(rounded(13, 6), 24);
    }
    #[test]
    fn numerical_and_quality_regressions_fail() {
        let a = json!({"correct":80,"brier":0.2,"nll":0.3});
        assert!(quality_gate(&a, &a).is_ok());
        assert!(quality_gate(&a, &json!({"correct":79,"brier":0.2,"nll":0.3})).is_err());
        assert!(quality_gate(&a, &json!({"correct":80,"brier":0.201,"nll":0.3})).is_err());
    }
    #[test]
    fn paired_stats_and_escaping() {
        let p = paired_interval(&[0.9; 20]);
        assert!((p["geometric_ratio"].as_f64().unwrap() - 0.9).abs() < 1e-12);
        assert!((p["upper_95"].as_f64().unwrap() - 0.9).abs() < 1e-12);
        assert_eq!(escape("<script>\"&"), "&lt;script&gt;&quot;&amp;");
    }
}

fn main() -> Result<()> {
    run(Args::parse_from(
        std::env::args().filter(|arg| arg != "--bench"),
    ))
}
