use anyhow::{Result, ensure};
use clap::Parser;
use decisive::laya::{Model, Scratch, cpu, weights::Storage};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    time::Instant,
};
#[allow(dead_code)]
#[path = "../src/models/laya/benchmark.rs"]
mod benchmark;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/laya")]
    model: PathBuf,
    #[arg(long, default_value = "benches/strata.jsonl")]
    input: PathBuf,
    #[arg(long, default_value_t = 1)]
    index: usize,
    #[arg(long, default_value_t = 6)]
    threads: usize,
    #[arg(long, default_value_t = 40)]
    rounds: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long)]
    baseline: bool,
    #[arg(help = "Enable the long-input AVX2 candidate; token caching remains disabled")]
    #[arg(long, conflicts_with = "baseline")]
    avx2_long: bool,
    #[arg(long)]
    control: Option<PathBuf>,
    #[arg(long, requires = "control")]
    ack: Option<PathBuf>,
    #[arg(help = "Collect a separate operator-timing pass after disabling perf")]
    #[arg(long)]
    operations: bool,
    #[arg(help = "Include each encoder/head layer in the separate operator pass")]
    #[arg(long, requires = "operations")]
    layer_operations: bool,
}
fn scheduler() -> BTreeMap<u32, [u64; 5]> {
    let mut records = BTreeMap::new();
    for entry in fs::read_dir("/proc/self/task")
        .into_iter()
        .flatten()
        .flatten()
    {
        let Some(id) = entry.file_name().to_str().and_then(|v| v.parse().ok()) else {
            continue;
        };
        let Ok(stat) = fs::read_to_string(entry.path().join("schedstat")) else {
            continue;
        };
        let values: Vec<u64> = stat
            .split_whitespace()
            .filter_map(|v| v.parse().ok())
            .collect();
        if values.len() != 3 {
            continue;
        }
        let mut r = [values[0], values[1], values[2], 0, 0];
        for line in fs::read_to_string(entry.path().join("status"))
            .unwrap_or_default()
            .lines()
        {
            if let Some((key, value)) = line.split_once(':') {
                if key == "voluntary_ctxt_switches" {
                    r[3] = value.trim().parse().unwrap_or(0);
                }
                if key == "nonvoluntary_ctxt_switches" {
                    r[4] = value.trim().parse().unwrap_or(0);
                }
            }
        }
        records.insert(id, r);
    }
    records
}
fn host() -> Value {
    json!({"load_1_5_15m":fs::read_to_string("/proc/loadavg").unwrap_or_default().split_whitespace().take(3).collect::<Vec<_>>(),
        "cpu_ticks":fs::read_to_string("/proc/stat").unwrap_or_default().lines().find(|s|s.starts_with("cpu ")).unwrap_or(""),
        "cpu_pressure":fs::read_to_string("/proc/pressure/cpu").ok()})
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    ensure!(
        args.rounds > 0 && args.warmup > 0 && args.threads > 0,
        "positive counts required"
    );
    let request: Value = serde_json::from_str(
        fs::read_to_string(&args.input)?
            .lines()
            .filter(|v| !v.trim().is_empty())
            .nth(args.index)
            .ok_or_else(|| anyhow::anyhow!("request index"))?,
    )?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let model = Model::load_with_kernel(&args.model, Storage::F16, cpu::Kernel::X86Avx2)?;
    let mut scratch = Scratch::default();
    if args.avx2_long {
        scratch.enable_avx2_long_pipeline();
    } else if !args.baseline {
        scratch.enable_avx2_pipeline();
    }
    let mut expected = Value::Null;
    for _ in 0..args.warmup {
        expected = model.predict_json(request.clone(), &mut scratch)?;
    }
    let mut ctl = args
        .control
        .as_ref()
        .map(|p| fs::OpenOptions::new().write(true).open(p))
        .transpose()?;
    let mut ack = args
        .ack
        .as_ref()
        .map(fs::File::open)
        .transpose()?
        .map(BufReader::new);
    let mut gate = |command: &str| -> Result<()> {
        if let Some(c) = &mut ctl {
            writeln!(c, "{command}")?;
            c.flush()?;
            if let Some(a) = &mut ack {
                let mut line = String::new();
                ensure!(
                    a.read_line(&mut line)? > 0
                        && line.trim_matches(|c: char| c.is_ascii_whitespace() || c == '\0')
                            == "ack",
                    "perf acknowledgement: {line:?}"
                );
            }
        }
        Ok(())
    };
    let mut times = Vec::with_capacity(args.rounds);
    let mut outputs = Vec::with_capacity(args.rounds);
    let host_before = host();
    let before = scheduler();
    gate("enable")?;
    let begin = Instant::now();
    for _ in 0..args.rounds {
        let start = Instant::now();
        let out = model.predict_json(request.clone(), &mut scratch)?;
        times.push(start.elapsed().as_secs_f64() * 1000.);
        outputs.push(out);
    }
    let seconds = begin.elapsed().as_secs_f64();
    gate("disable")?;
    let after = scheduler();
    let host_after = host();
    let mut sched = Vec::new();
    for (id, a) in after {
        if let Some(b) = before.get(&id) {
            let d: Vec<_> = a.iter().zip(b).map(|(a, b)| a.saturating_sub(*b)).collect();
            sched.push(json!({"role":if id==std::process::id(){"caller"}else{"worker"}, "scheduled_ns":d[0],"runqueue_wait_ns":d[1],"timeslices":d[2],"voluntary_switches":d[3],"involuntary_switches":d[4]}));
        }
    }
    let mut error = 0f64;
    for out in &outputs {
        error = error.max(benchmark::compare_outputs(&expected, out, 1e-4)?);
    }
    if args.operations {
        scratch.profile = Some(Default::default());
        if args.layer_operations {
            scratch.layer_profile = Some(Default::default());
        }
        for _ in 0..3 {
            model.predict_json(request.clone(), &mut scratch)?;
        }
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        json!({"scope":"Warm predict_json, FP16 weights / FP32 arithmetic. Repeated identical request, no result cache. Validation after the perf window; expected is the warm output, not an independent quality oracle. Operator timings are a separate three-request pass. Scheduler snapshots include FIFO acknowledgement overhead. Worker IDs are anonymized.",
        "model_shape":{"hidden_size":model.config.hidden_size,"intermediate_size":model.config.intermediate_size,"num_hidden_layers":model.config.num_hidden_layers,"num_attention_heads":model.config.num_attention_heads,"local_attention":model.config.local_attention,"global_attn_every_n_layers":model.config.global_attn_every_n_layers,"layer_types":model.config.layer_types,"head_layers":model.agent_config["head_layers"]},
        "pipeline":!args.baseline,"long_pipeline":args.avx2_long,"threads":args.threads,"request_index":args.index,"rounds":args.rounds,"warmup":args.warmup,"wall_seconds":seconds,"p50_ms":sorted[(args.rounds-1)/2],"p95_ms":sorted[(args.rounds as f64*0.95).ceil() as usize-1],"latencies_ms":times,"max_repeat_output_error":error,"output":expected,"scheduler":sched,"host_before":host_before,"host_after":host_after,"operation_nanoseconds_three_requests":scratch.profile,"layer_operation_nanoseconds_three_requests":scratch.layer_profile})
    );
    Ok(())
}
