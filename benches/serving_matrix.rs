use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};
#[allow(dead_code)]
#[path = "../src/models/laya/benchmark.rs"]
mod benchmark;
#[allow(dead_code)]
#[path = "framework_probe.rs"]
mod framework;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "target/release/decisive")]
    decisive: PathBuf,
    #[arg(long)]
    llama: Option<PathBuf>,
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long, default_value = "benches/serving-matrix-inputs.jsonl")]
    input: PathBuf,
    #[arg(long, default_value = "tests/fixtures/outputs/serving-matrix.jsonl")]
    reference: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "english,multilingual")]
    checkpoints: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "en,es,de,zh,ja,ar,hi")]
    languages: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "128,512,1024,2048")]
    lengths: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,6")]
    threads: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
    concurrency: Vec<usize>,
    #[arg(long, default_value = "focused", value_parser = ["focused", "full"])]
    matrix: String,
    #[arg(long, default_value_t = 4)]
    rounds: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(help = "At least this many requests per measured block; every client sends >=2")]
    #[arg(long, default_value_t = 8)]
    minimum_requests: usize,
    #[arg(help = "Zero uses the maximum requested client concurrency for all server groups")]
    #[arg(long, default_value_t = 0)]
    llama_slots: usize,
    #[arg(long, default_value = "f16", value_parser = ["f16", "f32"])]
    storage: String,
    #[arg(long, default_value = "native")]
    backend_label: String,
    #[arg(long, default_value_t = 18192)]
    llama_port: u16,
    #[arg(long, default_value_t = 18194)]
    decisive_port: u16,
    #[arg(help = "Emit the complete schedule without loading assets or starting servers")]
    #[arg(long)]
    plan: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct Cell {
    checkpoint: String,
    language: String,
    tokens: usize,
    threads: usize,
    concurrency: usize,
}
fn cells(args: &Args) -> Vec<Cell> {
    let mut result = Vec::new();
    for checkpoint in &args.checkpoints {
        for &threads in &args.threads {
            for language in &args.languages {
                if checkpoint == "english" && language != "en" {
                    continue;
                }
                for &tokens in &args.lengths {
                    for &concurrency in &args.concurrency {
                        let focused = if checkpoint == "english" {
                            concurrency == 1
                                || ([2, 6].contains(&threads) && [128, 512, 2048].contains(&tokens))
                        } else {
                            (concurrency == 1
                                && [1, 6].contains(&threads)
                                && [128, 512].contains(&tokens))
                                || (concurrency == 4 && threads == 6 && tokens == 512)
                        };
                        if args.matrix == "full" || focused {
                            result.push(Cell {
                                checkpoint: checkpoint.clone(),
                                language: language.clone(),
                                tokens,
                                threads,
                                concurrency,
                            });
                        }
                    }
                }
            }
        }
    }
    result
}
#[derive(Clone, Deserialize)]
struct Fixture {
    checkpoint: String,
    language: String,
    tokens: usize,
    scenario: String,
    request: Value,
}
fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}
fn save(path: &Path, data: &Value) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(data)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}
fn report(out: &Path) -> Result<()> {
    let mut data = serde_json::Map::new();
    for name in ["manifest", "summary", "status"] {
        data.insert(
            name.into(),
            serde_json::from_str::<Value>(&fs::read_to_string(out.join(format!("{name}.json")))?)?,
        );
    }
    let data = serde_json::to_string(&data)?.replace('<', "\\u003c");
    fs::write(
        out.join("index.html"),
        include_str!("../benches/serving_matrix_report.html")
            .replace("__DECISIVE_SERVING_EVIDENCE__", &data),
    )?;
    Ok(())
}
fn host() -> Value {
    json!({"load_1_5_15m":fs::read_to_string("/proc/loadavg").ok().map(|s|s.split_whitespace().take(3).map(str::to_owned).collect::<Vec<_>>()),"cpu_pressure":fs::read_to_string("/proc/pressure/cpu").ok(),"cpu_ticks":fs::read_to_string("/proc/stat").ok().and_then(|s|s.lines().find(|l|l.starts_with("cpu ")).map(str::to_owned))})
}
fn latency_summary(latencies: &[f64]) -> Value {
    if latencies.is_empty() {
        return Value::Null;
    }
    let mut summary = benchmark::summarize(latencies);
    summary
        .as_object_mut()
        .unwrap()
        .remove("requests_per_second");
    summary
}
fn tokens_match(fixture: &Fixture, output: &Value) -> Result<()> {
    ensure!(
        output["usage"]["input_tokens"].as_u64() == Some(fixture.tokens as u64),
        "missing or unequal token count"
    );
    ensure!(
        output["usage"]["truncated"].as_bool() != Some(true),
        "input truncated"
    );
    ensure!(
        output["usage"]["state_tokens_dropped"]
            .as_u64()
            .is_none_or(|n| n == 0),
        "state truncated"
    );
    Ok(())
}

fn burst(
    url: &str,
    bodies: &[String],
    concurrency: usize,
    per_client: usize,
    rotation: usize,
) -> Result<Value> {
    ensure!(
        !bodies.is_empty() && concurrency > 0 && per_client > 0,
        "invalid load"
    );
    let barrier = Arc::new(Barrier::new(concurrency + 1));
    let ready = Arc::new(Barrier::new(concurrency + 1));
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for client in 0..concurrency {
            let barrier = Arc::clone(&barrier);
            let ready = Arc::clone(&ready);
            handles.push(scope.spawn(move || {
                let agent: ureq::Agent = ureq::Agent::config_builder()
                    .timeout_global(Some(Duration::from_secs(1800)))
                    .http_status_as_error(false).build().into();
                ready.wait();
                barrier.wait();
                let mut samples = Vec::new();
                for ordinal in 0..per_client {
                    let fixture = (client + ordinal + rotation) % bodies.len();
                    let started = Instant::now();
                    let response: Result<(u16, Value)> = (|| {
                        let mut r = agent.post(url).header("Content-Type", "application/json").send(bodies[fixture].as_bytes())?;
                        let status = r.status().as_u16();
                        let output = serde_json::from_str(&r.body_mut().read_to_string()?)?;
                        Ok((status, output))
                    })();
                    let ms = started.elapsed().as_secs_f64() * 1000.;
                    samples.push(match response {
                        Ok((status, output)) => json!({"client":client,"ordinal":ordinal,"fixture":fixture,"latency_ms":ms,"status":status,"output":output}),
                        Err(error) => json!({"client":client,"ordinal":ordinal,"fixture":fixture,"latency_ms":ms,"error":error.to_string()}),
                    });
                }
                samples
            }));
        }
        ready.wait();
        let started = Instant::now();
        barrier.wait();
        let mut samples = Vec::new();
        for handle in handles {
            samples.extend(
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("HTTP client panicked"))?,
            );
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
        let successes = samples.iter().filter(|s| s["status"] == 200).count();
        let latencies: Vec<_> = samples
            .iter()
            .filter(|s| s["status"] == 200)
            .map(|s| s["latency_ms"].as_f64().unwrap())
            .collect();
        let timing = latency_summary(&latencies);
        Ok(
            json!({"elapsed_ms":elapsed_ms,"attempted":samples.len(),"successful":successes,"errors":samples.len()-successes,"successful_requests_per_second":1000.*successes as f64/elapsed_ms,"timing":timing,"samples":samples}),
        )
    })
}
fn sampled_burst(
    server: &framework::Server,
    url: &str,
    bodies: &[String],
    concurrency: usize,
    per_client: usize,
    rotation: usize,
    threads: usize,
) -> Result<Value> {
    let before = framework::process_cpu_seconds(server.0.id());
    let start = Instant::now();
    let mut result = burst(url, bodies, concurrency, per_client, rotation)?;
    let wall = start.elapsed().as_secs_f64();
    let after = framework::process_cpu_seconds(server.0.id());
    let cpu = before.zip(after).map(|(a, b)| b - a);
    let ratio = cpu.map(|v| v / wall);
    let valid = ratio.is_some_and(|r| r.is_finite() && r >= 0. && r <= threads as f64 * 1.05);
    result["server_cpu_seconds"] = json!(cpu);
    result["cpu_audit_wall_seconds"] = json!(wall);
    result["server_cpu_over_wall"] = json!(ratio);
    result["cpu_budget_passed"] = json!(if cfg!(target_os = "linux") {
        valid
    } else {
        ratio.is_none_or(|r| r <= threads as f64 * 1.05)
    });
    Ok(result)
}
fn common_answer(output: &Value) -> Value {
    let mut answers = output["answers"].clone();
    if let Some(answers) = answers.as_object_mut() {
        for answer in answers.values_mut() {
            if let Some(answer) = answer.as_object_mut() {
                for key in ["confidence", "answer_confidence", "action"] {
                    answer.remove(key);
                }
            }
        }
    }
    json!({"answers":answers})
}
fn validate(
    block: &Value,
    indices: &[usize],
    fixtures: &[Fixture],
    expected: &[Value],
    own: &BTreeMap<usize, Value>,
    decisive: bool,
) -> Vec<Value> {
    let mut errors = Vec::new();
    if block["cpu_budget_passed"] != true {
        errors.push(json!({"kind":"cpu_budget"}));
    }
    for sample in block["samples"].as_array().unwrap() {
        let local = sample["fixture"].as_u64().unwrap() as usize;
        let index = indices[local];
        let output = &sample["output"];
        let result = (|| -> Result<()> {
            ensure!(sample["status"] == 200, "HTTP request failed");
            tokens_match(&fixtures[index], output)?;
            benchmark::compare_outputs(&own[&index], output, 1e-4)
                .context("concurrent versus serial output")?;
            if decisive {
                benchmark::compare_outputs(&expected[index], output, 1e-4).context("SDK output")?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            errors.push(json!({"client":sample["client"],"ordinal":sample["ordinal"],"fixture":index,"error":format!("{error:#}")}));
        }
    }
    errors
}
fn summarize_cell(raw: &Value) -> Value {
    let pairs = raw["pairs"].as_array().unwrap();
    let mut engines = Vec::new();
    for name in ["llama", "decisive"] {
        let blocks: Vec<_> = pairs
            .iter()
            .map(|p| p.get(name).or_else(|| p.get("disco")).unwrap_or(&p["beam"]))
            .collect();
        let latencies: Vec<_> = blocks
            .iter()
            .flat_map(|b| b["samples"].as_array().unwrap())
            .filter(|s| s["status"] == 200)
            .map(|s| s["latency_ms"].as_f64().unwrap())
            .collect();
        let elapsed: f64 = blocks
            .iter()
            .map(|b| b["elapsed_ms"].as_f64().unwrap())
            .sum();
        let requests: u64 = blocks
            .iter()
            .map(|b| b["successful"].as_u64().unwrap())
            .sum();
        let errors: u64 = blocks.iter().map(|b| b["errors"].as_u64().unwrap()).sum();
        engines.push(json!({"name":name,"timing":latency_summary(&latencies),"block_wall_ms":elapsed,"successful_requests":requests,"errors":errors,"successful_requests_per_second":1000.*requests as f64/elapsed,"max_server_cpu_over_wall":blocks.iter().filter_map(|b|b["server_cpu_over_wall"].as_f64()).fold(0.0,f64::max)}));
    }
    let ratios: Vec<_> = pairs
        .iter()
        .map(|p| {
            p.get("decisive")
                .or_else(|| p.get("disco"))
                .unwrap_or(&p["beam"])["elapsed_ms"]
                .as_f64()
                .unwrap()
                / p["llama"]["elapsed_ms"].as_f64().unwrap()
        })
        .collect();
    json!({"cell":raw["cell"],"rounds":pairs.len(),"llama":engines[0],"decisive":engines[1],"paired_block_time_decisive_over_llama":if ratios.is_empty(){Value::Null}else{benchmark::paired_interval(&ratios)},"violations":raw["violations"]})
}
fn run(args: &Args, schedule: &[Cell], out: &Path) -> Result<()> {
    let llama = args.llama.as_ref().context("--llama required")?;
    ensure!(args.decisive_port != args.llama_port, "ports must differ");
    let records = read_jsonl(&args.input)?;
    let fixtures: Vec<Fixture> = records
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()?;
    let expected = read_jsonl(&args.reference)?;
    ensure!(
        fixtures.len() == expected.len(),
        "fixture/reference length mismatch"
    );
    for (fixture, reference) in fixtures.iter().zip(&expected) {
        tokens_match(fixture, reference)?;
    }
    let slots = if args.llama_slots == 0 {
        *args.concurrency.iter().max().unwrap()
    } else {
        args.llama_slots
    };
    ensure!((1..=32).contains(&slots), "llama slots must be 1..32");
    let version = Command::new(llama).arg("--version").output()?;
    ensure!(version.status.success(), "llama --version failed");
    let mut manifest = json!({
        "protocol_version":1,"scope":"Warm CPU-only finite closed-loop HTTP blocks; engines measured sequentially. Total worker budget is independent of client count. Shared host; no physical-core isolation claim.",
        "matrix":args.matrix,"cells":schedule,"rounds":args.rounds,"warmup":args.warmup,"minimum_requests":args.minimum_requests,"storage":args.storage,"backend_label":args.backend_label,
        "decisive_sha256":framework::digest(&args.decisive)?,"llama_sha256":framework::digest(llama)?,"input_sha256":framework::digest(&args.input)?,"reference_sha256":framework::digest(&args.reference)?,
        "llama_version_stdout":String::from_utf8_lossy(&version.stdout),"llama_version_stderr":String::from_utf8_lossy(&version.stderr),
        "hardware":decisive::laya::cpu::hardware::detect(),"host_before":host(),"references":serde_json::from_str::<Value>(&fs::read_to_string("references.json")?)?,
        "decisive_execution_lanes":1,"decisive_avx2_long":true,"token_cache":false,"state_cache":false,"llama_slots":slots,"llama_context":4096*slots,"llama_batch":2048,"llama_ubatch":2048,"llama_flash_attention":"auto","llama_gpu_layers":0,"llama_poll":0,"llama_poll_batch":0,"omp_wait_policy":"PASSIVE",
        "sampling":"Four-round balanced ABBA/BAAB engine order per cell, rotating cell order. Each client sends max(2,ceil(minimum_requests/concurrency)) requests per measured block. Equal workload schedules for both engines. Different refund/support inputs are interleaved. Warmups are separate. No cache reuse optimization. Every output retained and checked after timing.",
        "statistics":"Aggregate successful RPS uses completed requests divided by block wall time, never summed overlapping latencies. Latencies include server queueing. Paired intervals resample whole blocks, not correlated individual requests. Small-sample p95 is descriptive, not a production SLO estimate. Closed-loop clients are not an open-loop saturation test.",
        "correctness":"Decisive checked against pinned SDK and each engine checked against its own serial responses at 1e-4. Equal full input-token counts required. llama.cpp omits act/answer_confidence and uses different confidence/head evaluation/arithmetic. Common answer diagnostics are reported separately; numerical equivalence is not claimed.",
        "model_assets":{},"source_assets":{},"groups":[]
    });
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "references.json",
        "benches/serving_matrix.rs",
        "benches/serving_inputs.rs",
        "benches/framework_probe.rs",
        "src/models/laya/benchmark.rs",
        "benches/serving_matrix_report.html",
    ] {
        manifest["source_assets"][path] = json!(framework::digest(Path::new(path))?);
    }
    manifest["runner_sha256"] = json!(framework::digest(&std::env::current_exe()?)?);
    for checkpoint in &args.checkpoints {
        let directory = if checkpoint == "english" {
            "laya"
        } else {
            "laya-multilingual"
        };
        let model = args.models.join(directory);
        let stamp: Value =
            serde_json::from_str(&fs::read_to_string(model.join("beam-revision.json"))?)?;
        ensure!(
            stamp == manifest["references"]["models"][checkpoint],
            "checkpoint revision stamp mismatch: {checkpoint}"
        );
        let gguf = args
            .models
            .join(format!("{directory}-{}.gguf", args.storage));
        let mut assets = serde_json::Map::new();
        for name in [
            "beam-revision.json",
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "model.safetensors",
        ] {
            assets.insert(name.into(), json!(framework::digest(&model.join(name))?));
        }
        assets.insert("gguf_sha256".into(), json!(framework::digest(&gguf)?));
        manifest["model_assets"][checkpoint] = json!(assets);
    }
    save(&out.join("manifest.json"), &manifest)?;
    let private = out.join("private-logs");
    fs::create_dir(&private)?;
    fs::write(
        private.join("README.txt"),
        "Local diagnostics can contain filesystem paths. Do not publish this directory.\n",
    )?;
    let mut all_summaries = Vec::new();
    for checkpoint in &args.checkpoints {
        for &threads in &args.threads {
            let group: Vec<_> = schedule
                .iter()
                .filter(|c| c.checkpoint == *checkpoint && c.threads == threads)
                .cloned()
                .collect();
            if group.is_empty() {
                continue;
            }
            let reservations = [
                TcpListener::bind(("127.0.0.1", args.llama_port))?,
                TcpListener::bind(("127.0.0.1", args.decisive_port))?,
            ];
            let directory = if checkpoint == "english" {
                "laya"
            } else {
                "laya-multilingual"
            };
            let model = args.models.join(directory);
            let gguf = args
                .models
                .join(format!("{directory}-{}.gguf", args.storage));
            let group_name = format!("{checkpoint}-{threads}");
            drop(reservations);
            let mut command = Command::new(llama);
            command.arg("-m").arg(&gguf).args([
                "-t",
                &threads.to_string(),
                "-tb",
                &threads.to_string(),
                "-c",
                &(4096 * slots).to_string(),
                "-b",
                "2048",
                "-ub",
                "2048",
                "-np",
                &slots.to_string(),
                "-ngl",
                "0",
                "--poll",
                "0",
                "--poll-batch",
                "0",
                "--flash-attn",
                "auto",
                "--host",
                "127.0.0.1",
                "--port",
                &args.llama_port.to_string(),
                "--no-warmup",
            ]);
            for name in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "MKL_NUM_THREADS",
            ] {
                command.env(name, threads.to_string());
            }
            command.env("OMP_WAIT_POLICY", "PASSIVE");
            let mut cpp =
                framework::spawn(command, &private.join(format!("{group_name}-llama.log")))?;
            framework::ready(&mut cpp, args.llama_port)?;
            let mut command = Command::new(&args.decisive);
            command.arg("laya");
            command
                .arg("--model")
                .arg(&model)
                .args([
                    "--threads",
                    &threads.to_string(),
                    "--storage",
                    &args.storage,
                    "--kernel",
                    "packed",
                    "--avx2-long",
                    "serve",
                    "--listen",
                    &format!("127.0.0.1:{}", args.decisive_port),
                ])
                .env("TOKENIZERS_PARALLELISM", "false");
            let mut decisive =
                framework::spawn(command, &private.join(format!("{group_name}-decisive.log")))?;
            framework::ready(&mut decisive, args.decisive_port)?;
            let urls = [
                format!("http://127.0.0.1:{}/v1/systemone", args.llama_port),
                format!("http://127.0.0.1:{}/v1/systemone", args.decisive_port),
            ];
            let used: Vec<_> = (0..fixtures.len())
                .filter(|&i| {
                    group.iter().any(|c| {
                        c.checkpoint == fixtures[i].checkpoint
                            && c.language == fixtures[i].language
                            && c.tokens == fixtures[i].tokens
                    })
                })
                .collect();
            let mut own = [BTreeMap::new(), BTreeMap::new()];
            let mut preflight = Vec::new();
            for &index in &used {
                let bodies = [fixtures[index].request.to_string()];
                let mut outputs = Vec::new();
                for engine in 0..2 {
                    let output = burst(&urls[engine], &bodies, 1, 1, 0)?;
                    let sample = &output["samples"][0];
                    save(
                        &out.join(format!("{group_name}-preflight-current.json")),
                        &json!({"fixture":index,"engine":engine,"block":output}),
                    )?;
                    ensure!(sample["status"] == 200, "serial preflight HTTP error");
                    tokens_match(&fixtures[index], &sample["output"])?;
                    if engine == 1 {
                        benchmark::compare_outputs(&expected[index], &sample["output"], 1e-4)
                            .with_context(|| {
                                format!(
                                    "SDK preflight {checkpoint}/{}/{}/{}",
                                    fixtures[index].language,
                                    fixtures[index].tokens,
                                    fixtures[index].scenario
                                )
                            })?;
                    }
                    own[engine].insert(index, sample["output"].clone());
                    outputs.push(sample["output"].clone());
                }
                let common = benchmark::compare_outputs(
                    &common_answer(&expected[index]),
                    &common_answer(&outputs[0]),
                    1e-4,
                );
                preflight.push(json!({"fixture":index,"llama":outputs[0],"decisive":outputs[1],"llama_common_sdk_error":common.as_ref().ok(),"llama_common_sdk_violation":common.err().map(|e|e.to_string())}));
            }
            save(
                &out.join(format!("{group_name}-preflight.json")),
                &json!(preflight),
            )?;
            let indices: Vec<Vec<usize>> = group
                .iter()
                .map(|c| {
                    (0..fixtures.len())
                        .filter(|&i| {
                            fixtures[i].checkpoint == c.checkpoint
                                && fixtures[i].language == c.language
                                && fixtures[i].tokens == c.tokens
                        })
                        .collect()
                })
                .collect();
            ensure!(
                indices.iter().all(|i| i.len() >= 2),
                "each cell needs at least two distinct scenarios"
            );
            let bodies: Vec<Vec<String>> = indices
                .iter()
                .map(|ids| {
                    ids.iter()
                        .map(|&i| fixtures[i].request.to_string())
                        .collect()
                })
                .collect();
            let mut raw: Vec<Value> = group
                .iter()
                .map(|c| json!({"cell":c,"warmups":[],"pairs":[],"violations":[]}))
                .collect();
            let group_path = out.join(format!("{group_name}.json"));
            for warmup in 0..args.warmup {
                for (i, cell) in group.iter().enumerate() {
                    for (engine, server) in [&cpp, &decisive].iter().enumerate() {
                        let result = sampled_burst(
                            server,
                            &urls[engine],
                            &bodies[i],
                            cell.concurrency,
                            1,
                            warmup,
                            threads,
                        )?;
                        let errors = validate(
                            &result,
                            &indices[i],
                            &fixtures,
                            &expected,
                            &own[engine],
                            engine == 1,
                        );
                        raw[i]["warmups"]
                            .as_array_mut()
                            .unwrap()
                            .push(json!({"engine":engine,"block":result,"violations":errors}));
                        save(&group_path, &json!(raw))?;
                        ensure!(
                            errors.is_empty(),
                            "warmup correctness/CPU gate failed; raw evidence retained"
                        );
                    }
                }
            }
            for round in 0..args.rounds {
                for i in (0..group.len()).map(|j| (j + round) % group.len()) {
                    let cell = &group[i];
                    let per_client = args.minimum_requests.div_ceil(cell.concurrency).max(2);
                    let left_first = (round % 4 == 0 || round % 4 == 3) != (i % 2 == 1);
                    let mut results = [Value::Null, Value::Null];
                    let before = host();
                    for engine in if left_first { [0, 1] } else { [1, 0] } {
                        results[engine] = sampled_burst(
                            if engine == 0 { &cpp } else { &decisive },
                            &urls[engine],
                            &bodies[i],
                            cell.concurrency,
                            per_client,
                            round,
                            threads,
                        )?;
                    }
                    let mut violations = Vec::new();
                    for engine in 0..2 {
                        for error in validate(
                            &results[engine],
                            &indices[i],
                            &fixtures,
                            &expected,
                            &own[engine],
                            engine == 1,
                        ) {
                            violations.push(json!({"round":round,"engine":engine,"detail":error}));
                        }
                    }
                    raw[i]["pairs"].as_array_mut().unwrap().push(json!({"round":round,"llama_first":left_first,"per_client":per_client,"llama":results[0],"decisive":results[1],"host_before":before,"host_after":host()}));
                    raw[i]["violations"]
                        .as_array_mut()
                        .unwrap()
                        .extend(violations.clone());
                    save(&group_path, &json!(raw))?;
                    ensure!(
                        violations.is_empty(),
                        "measured correctness/CPU gate failed; raw evidence retained"
                    );
                    eprintln!(
                        "{group_name} {}/{} language={} tokens={} concurrency={} round={}/{}",
                        i + 1,
                        group.len(),
                        cell.language,
                        cell.tokens,
                        cell.concurrency,
                        round + 1,
                        args.rounds
                    );
                }
            }
            all_summaries.extend(raw.iter().map(summarize_cell));
            manifest["groups"].as_array_mut().unwrap().push(json!({"checkpoint":checkpoint,"threads":threads,"cells":group.len(),"llama_loaded_libraries":framework::loaded_library_hashes(cpp.0.id())?,"decisive_loaded_libraries":framework::loaded_library_hashes(decisive.0.id())?}));
            save(&out.join("manifest.json"), &manifest)?;
            save(
                &out.join("summary.json"),
                &json!({"complete":false,"cells":all_summaries}),
            )?;
        }
    }
    manifest["host_after"] = host();
    save(&out.join("manifest.json"), &manifest)?;
    ensure!(all_summaries.len() == schedule.len(), "incomplete schedule");
    save(
        &out.join("summary.json"),
        &json!({"complete":true,"cells":all_summaries}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    ensure!(
        !args.threads.is_empty() && args.threads.iter().all(|&v| v > 0),
        "invalid worker counts"
    );
    ensure!(
        !args.concurrency.is_empty() && args.concurrency.iter().all(|&v| (1..=32).contains(&v)),
        "invalid concurrency"
    );
    ensure!(
        args.rounds > 0 && args.rounds % 4 == 0 && args.warmup > 0 && args.minimum_requests > 0,
        "rounds must be a positive multiple of four; warmup and request counts positive"
    );
    ensure!(
        args.checkpoints
            .iter()
            .all(|s| ["english", "multilingual"].contains(&s.as_str())),
        "unsupported checkpoint"
    );
    ensure!(
        args.lengths.iter().all(|&n| (64..=2048).contains(&n)),
        "this server protocol supports 64..2048 full tokens"
    );
    let schedule = cells(&args);
    ensure!(!schedule.is_empty(), "empty schedule");
    if args.plan {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &json!({"matrix":args.matrix,"cells":schedule,"cell_count":schedule.len(),"rounds":args.rounds,"minimum_requests":args.minimum_requests,"scope":"CPU worker budgets, not physical-core isolation. Focused sweeps cover each axis but not the full cross-product. No inference or downloads in plan mode."})
            )?
        );
        return Ok(());
    }
    let out = args.out.as_ref().context("--out required")?;
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(out).context("output directory must be new")?;
    save(
        &out.join("status.json"),
        &json!({"status":"running","planned_cells":schedule.len()}),
    )?;
    let result = run(&args, &schedule, out);
    save(
        &out.join("status.json"),
        &match &result {
            Ok(()) => json!({"status":"complete","cells":schedule.len()}),
            Err(error) => json!({"status":"failed","error":format!("{error:#}")}),
        },
    )?;
    result?;
    report(out)?;
    println!("{}", json!({"status":"complete","cells":schedule.len()}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plans_distinguish_focused_coverage_from_full_cross_product() {
        let focused = Args::parse_from(["serving_matrix", "--plan"]);
        assert_eq!(cells(&focused).len(), 69);
        let full = Args::parse_from(["serving_matrix", "--plan", "--matrix", "full"]);
        assert_eq!(cells(&full).len(), 512);
        assert!(cells(&focused).iter().all(|c| cells(&full).contains(c)));
    }
    #[test]
    fn clients_overlap_and_keep_responses_associated_with_their_requests() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            let active = &active;
            let peak = &peak;
            scope.spawn(move || {
                std::thread::scope(|handlers| {
                    for mut request in server.incoming_requests().take(8) {
                        handlers.spawn(move || {
                            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(current, Ordering::SeqCst);
                            let mut body = String::new();
                            request.as_reader().read_to_string(&mut body).unwrap();
                            std::thread::sleep(Duration::from_millis(40));
                            active.fetch_sub(1, Ordering::SeqCst);
                            request
                                .respond(tiny_http::Response::from_string(body))
                                .unwrap();
                        });
                    }
                });
            });
            let bodies = ["{\"case\":0}".to_owned(), "{\"case\":1}".to_owned()];
            let result = burst(&format!("http://{address}/test"), &bodies, 4, 2, 0).unwrap();
            assert_eq!(result["successful"], 8);
            assert_eq!(result["errors"], 0);
            assert!(peak.load(Ordering::SeqCst) >= 2 && peak.load(Ordering::SeqCst) <= 4);
            for sample in result["samples"].as_array().unwrap() {
                assert_eq!(sample["fixture"], sample["output"]["case"]);
            }
            let sum: f64 = result["samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s["latency_ms"].as_f64().unwrap())
                .sum();
            assert!(result["successful_requests_per_second"].as_f64().unwrap() > 8000. / sum);
        });
    }
}
