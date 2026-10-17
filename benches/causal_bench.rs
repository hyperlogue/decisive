use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(about = "Compare causal model runtimes and prefix caches")]
pub(crate) struct Args {
    #[command(subcommand)]
    command: Mode,
}
#[derive(Subcommand)]
enum Mode {
    #[command(about = "Benchmark-only server with cache reset and unrounded Kev probabilities")]
    Serve {
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 1)]
        threads: usize,
        #[arg(long, default_value_t = 18991)]
        port: u16,
        #[arg(long, default_value_t = 128)]
        cache_mib: usize,
        #[arg(help = "Model-owned public length rules, applied before serving requests")]
        #[arg(long)]
        length_rules: Option<PathBuf>,
    },
    Run {
        #[arg(long, default_value = "benches/configs/causal-frameworks.json")]
        config: PathBuf,
        #[arg(help = "Override the default per-model corpus with explicit JSONL files")]
        #[arg(long)]
        corpus: Vec<PathBuf>,
        #[arg(long)]
        sources: PathBuf,
        #[arg(long)]
        llama: PathBuf,
        #[arg(long)]
        python: PathBuf,
        #[arg(long, default_value = "benchmark-runs/causal")]
        output: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "1,6")]
        threads: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "128,512,1024,2048")]
        lengths: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "en,zh,fr,ar")]
        languages: Vec<String>,
        #[arg(long, value_delimiter = ',', default_value = "kev,jev")]
        models: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        engines: Vec<String>,
        #[arg(long, default_value_t = 5)]
        rounds: usize,
        #[arg(long, default_value_t = 2)]
        blocks: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 18991)]
        port: u16,
        #[arg(long, default_value_t = 1e-4)]
        max_error: f64,
    },
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Engine {
    id: String,
    model: String,
    command: Vec<String>,
    reset: Vec<String>,
    precision: String,
    notes: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    engines: Vec<Engine>,
    assets: Vec<String>,
}
#[derive(Deserialize, Serialize)]
struct Case {
    id: String,
    model: String,
    language: String,
    length: usize,
    requests: Vec<Trial>,
    reference: Value,
}
#[derive(Deserialize, Serialize)]
struct Trial {
    kind: String,
    request: Value,
    expected: Value,
    input_tokens: usize,
    state_tokens: usize,
}
struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let group = format!("-{}", self.0.id());
        let _ = Command::new("kill")
            .args(["-TERM", "--", &group])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if self.0.try_wait().ok().flatten().is_some() {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        let _ = Command::new("kill")
            .args(["-KILL", "--", &group])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = self.0.wait();
    }
}
fn http(port: u16, path: &str, body: Option<&Value>) -> Result<Value> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(600)))
        .build()
        .into();
    let url = format!("http://127.0.0.1:{port}{path}");
    let mut r = if let Some(b) = body {
        agent
            .post(&url)
            .header("Content-Type", "application/json")
            .send(b.to_string())?
    } else {
        agent.get(&url).call()?
    };
    Ok(serde_json::from_str(&r.body_mut().read_to_string()?)?)
}
fn snapshot() -> Value {
    let read = |p| fs::read_to_string(p).unwrap_or_default();
    json!({"loadavg":read("/proc/loadavg"),"cpu_pressure":read("/proc/pressure/cpu"),"memory_pressure":read("/proc/pressure/memory"),"cpu_stat":read("/proc/stat"),"memory":read("/proc/meminfo")})
}
fn memory(pid: u32) -> Value {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    json!(
        status
            .lines()
            .filter(|l| l.starts_with("VmRSS:")
                || l.starts_with("VmHWM:")
                || l.starts_with("Threads:"))
            .collect::<Vec<_>>()
    )
}
fn hash(path: &Path) -> Result<String> {
    let o = Command::new("sha256sum").arg(path).output()?;
    ensure!(o.status.success(), "cannot hash {}", path.display());
    Ok(String::from_utf8(o.stdout)?
        .split_whitespace()
        .next()
        .context("empty hash")?
        .to_owned())
}
fn probabilities(v: &Value, id: &str) -> Result<BTreeMap<String, f64>> {
    if let Some(raw) = v["probabilities"][id].as_array() {
        let ans = &v["answers"][id];
        let keys: Vec<String> = if ans["type"] == "noul" {
            vec!["false".into(), "true".into()]
        } else {
            ans["probabilities"]
                .as_object()
                .context("missing distribution")?
                .keys()
                .cloned()
                .collect()
        };
        ensure!(keys.len() == raw.len(), "raw probability shape mismatch");
        return keys
            .into_iter()
            .zip(raw)
            .map(|(k, x)| Ok((k, x.as_f64().context("invalid probability")?)))
            .collect();
    }
    let ans = &v["answers"][id];
    if ans["type"] == "noul" {
        let p = ans["noul"].as_f64().context("missing noul")?;
        Ok(BTreeMap::from([
            ("false".into(), 1. - p),
            ("true".into(), p),
        ]))
    } else {
        ans["probabilities"]
            .as_object()
            .context("missing probabilities")?
            .iter()
            .map(|(k, x)| Ok((k.clone(), x.as_f64().context("invalid probability")?)))
            .collect()
    }
}
fn compare(actual: &Value, expected: &Value) -> Result<Value> {
    let a = actual["answers"].as_object().context("missing answers")?;
    let e = expected["answers"]
        .as_object()
        .context("missing oracle answers")?;
    ensure!(a.len() == e.len(), "answer count mismatch");
    let mut error = 0f64;
    let mut changed = 0;
    let mut api_error = 0f64;
    for (id, answer) in e {
        ensure!(
            a.contains_key(id) && a[id]["type"] == answer["type"],
            "missing/wrong question type"
        );
        let (ap, ep) = (probabilities(actual, id)?, probabilities(expected, id)?);
        ensure!(ap.keys().eq(ep.keys()), "option mismatch");
        for (k, p) in &ep {
            let x = ap[k];
            ensure!(
                x.is_finite() && (0.0..=1.0).contains(&x),
                "invalid probability"
            );
            error = error.max((x - p).abs());
        }
        if answer["type"] == "choice" {
            changed += usize::from(a[id]["choice"] != answer["choice"]);
        } else if answer["type"] == "noul" {
            changed += usize::from((ap["true"] >= 0.5) != (ep["true"] >= 0.5));
        } else {
            let top = |p: &BTreeMap<String, f64>| {
                p.iter()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|x| x.0.clone())
            };
            changed += usize::from(top(&ap) != top(&ep));
        }
        for key in ["confidence", "score", "noul"] {
            if let Some(p) = answer[key].as_f64() {
                api_error =
                    api_error.max((a[id][key].as_f64().context("missing scalar")? - p).abs());
            }
        }
    }
    Ok(
        json!({"max_probability_error":error,"changed_decisions":changed,"max_api_scalar_error":api_error,
        "input_usage_matches":actual["usage"]["input_tokens"]==expected["usage"]["input_tokens"]}),
    )
}
fn serve(
    model: &str,
    threads: usize,
    port: u16,
    cache_mib: usize,
    length_rules: Option<&Path>,
) -> Result<()> {
    ensure!(threads > 0, "threads must be positive");
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    enum Loaded {
        Kev(decisive::kev::Model, decisive::kev::Session),
        Jev(decisive::jev_style::Model, decisive::jev_style::Session),
    }
    let start = Instant::now();
    let mut loaded = match model {
        "kev" => Loaded::Kev(
            decisive::kev::Model::load("models/kev-0.8b", "models/qwen3.5-0.8b-base", false)?,
            Default::default(),
        ),
        "jev" => Loaded::Jev(
            decisive::jev_style::Model::load("models/jev-style-0.8b-v3", false)?,
            Default::default(),
        ),
        _ => bail!("unknown model"),
    };
    let cache_bytes = cache_mib
        .checked_mul(1024 * 1024)
        .context("cache budget overflow")?;
    match &mut loaded {
        Loaded::Kev(_, s) => s.set_cache_bytes(cache_bytes),
        Loaded::Jev(_, s) => s.set_cache_bytes(cache_bytes),
    }
    let rules = length_rules
        .map(|p| -> Result<Value> { Ok(serde_json::from_slice(&fs::read(p)?)?) })
        .transpose()?;
    if let Some(rules) = &rules {
        match &mut loaded {
            Loaded::Kev(_, s) => {
                s.set_length_rules(Some(serde_json::from_value(rules.clone())?))?
            }
            Loaded::Jev(_, s) => {
                s.set_length_rules(Some(serde_json::from_value(rules.clone())?))?
            }
        }
    }
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let server =
        tiny_http::Server::http(format!("127.0.0.1:{port}")).map_err(|e| anyhow::anyhow!("{e}"))?;
    for mut req in server.incoming_requests() {
        let result = (|| -> Result<Value> {
            if req.url() == "/health" {
                return Ok(
                    json!({"engine":"decisive","model":model,"threads":threads,"load_ms":load_ms,"cache_budget_mib":cache_mib,"length_rules":rules}),
                );
            }
            ensure!(req.method() == &tiny_http::Method::Post, "POST required");
            if req.url() == "/benchmark/reset" {
                match &mut loaded {
                    Loaded::Kev(_, s) => s.clear_cache(),
                    Loaded::Jev(_, s) => s.clear_cache(),
                }
                return Ok(json!({"reset":true}));
            }
            ensure!(req.url() == "/v1/systemone", "unknown route");
            let body: Value = serde_json::from_reader(req.as_reader())?;
            let (mut out, before, after) = match &mut loaded {
                Loaded::Kev(m, s) => {
                    let b = serde_json::to_value(&s.stats)?;
                    let out = m.decide_json(&body, s)?;
                    (out, b, serde_json::to_value(&s.stats)?)
                }
                Loaded::Jev(m, s) => {
                    let b = serde_json::to_value(&s.stats)?;
                    let out = m.predict_json(&body, s)?;
                    (out, b, serde_json::to_value(&s.stats)?)
                }
            };
            let retained_bytes = match &loaded {
                Loaded::Kev(_, s) => s.cached_bytes(),
                Loaded::Jev(_, s) => s.cached_bytes(),
            };
            out["benchmark_cache"] =
                json!({"before":before,"after":after,"retained_bytes":retained_bytes});
            Ok(out)
        })();
        let (status, body) = match result {
            Ok(v) => (200, v),
            Err(e) => (422, json!({"error":e.to_string()})),
        };
        req.respond(
            tiny_http::Response::from_string(body.to_string())
                .with_status_code(status)
                .with_header(
                    tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap(),
                ),
        )?;
    }
    Ok(())
}
fn percentile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    if q == 0.5 && v.len() % 2 == 0 {
        return (v[v.len() / 2 - 1] + v[v.len() / 2]) * 0.5;
    }
    v[((v.len() as f64 * q).ceil() as usize)
        .saturating_sub(1)
        .min(v.len() - 1)]
}
pub(crate) fn run(args: Args) -> Result<()> {
    let Mode::Run {
        config,
        corpus,
        sources,
        llama,
        python,
        output,
        threads,
        lengths,
        languages,
        models,
        engines,
        rounds,
        blocks,
        warmup,
        port,
        max_error,
    } = args.command
    else {
        if let Mode::Serve {
            model,
            threads,
            port,
            cache_mib,
            length_rules,
        } = args.command
        {
            return serve(&model, threads, port, cache_mib, length_rules.as_deref());
        }
        unreachable!()
    };
    ensure!(
        rounds > 0
            && blocks > 0
            && threads.iter().all(|n| *n > 0)
            && max_error.is_finite()
            && max_error >= 0.,
        "invalid benchmark counts/bound"
    );
    ensure!(
        !output.exists(),
        "output already exists; use a new directory"
    );
    let cfg: Config = serde_json::from_slice(&fs::read(&config)?)?;
    let corpus = if corpus.is_empty() {
        models
            .iter()
            .map(|model| PathBuf::from(format!("benches/causal-{model}.jsonl")))
            .collect()
    } else {
        corpus
    };
    let mut cases = Vec::<Case>::new();
    for path in &corpus {
        for line in fs::read_to_string(path)?.lines().filter(|x| !x.is_empty()) {
            let c: Case = serde_json::from_str(line)?;
            if models.contains(&c.model)
                && lengths.contains(&c.length)
                && languages.contains(&c.language)
            {
                ensure!(
                    !c.requests.is_empty() && c.requests[0].kind == "initial",
                    "request sequence must start with initial"
                );
                let mut kinds = std::collections::BTreeSet::new();
                ensure!(
                    c.requests.iter().all(|r| kinds.insert(&r.kind)),
                    "duplicate phase in request sequence"
                );
                cases.push(c);
            }
        }
    }
    ensure!(!cases.is_empty(), "empty case selection");
    fs::create_dir_all(output.join("private-logs"))?;
    let mut hashes = BTreeMap::new();
    for asset in &cfg.assets {
        hashes.insert(asset.clone(), hash(Path::new(asset))?);
    }
    for entry in fs::read_dir(llama.join("build/bin"))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type()?.is_file() && (name.contains(".so") || name == "llama-server") {
            hashes.insert(format!("baseline/llama/{name}"), hash(&entry.path())?);
        }
    }
    for name in [
        "jev/jev_style_decision.py",
        "jev-gguf/jev_style_decision_gguf.py",
        "jev-gguf/jev_score.cpp",
        "jev-gguf/build/jev-score",
    ] {
        hashes.insert(format!("baseline/{name}"), hash(&sources.join(name))?);
    }
    for name in [
        "jev-gguf-f32kv/jev_score.cpp",
        "jev-gguf-f32kv/build/jev-score",
    ] {
        let path = sources.join(name);
        if path.is_file() {
            hashes.insert(format!("baseline/{name}"), hash(&path)?);
        }
    }
    for path in &corpus {
        hashes.insert(path.to_string_lossy().into_owned(), hash(path)?);
    }
    let info = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu: Vec<_> = info
        .lines()
        .filter(|l| l.starts_with("model name") || l.starts_with("flags"))
        .take(2)
        .collect();
    let mut manifest = json!({"schema":1,"config":cfg,"corpus":corpus,"hashes":hashes,"threads":threads,"lengths":lengths,"languages":languages,
        "models":models,"engines":engines,"rounds":rounds,"blocks":blocks,"warmup_sequences_per_case":warmup,"max_error":max_error,
        "concurrency":1,"cpu":cpu,"available_parallelism":thread::available_parallelism()?.get(),"host_before":snapshot(),
        "method":"Warm weights. Reset all request state before each request sequence; reset outside timed region. Reverse engine and case order in alternating blocks. End-to-end loopback HTTP latency. No process loading, quality checks or serialization of results in timed region.",
        "limitations":["Shared host, no affinity or exclusive cores", "Synthetic SDK compatibility fixtures, not labeled task quality", "p95 suppressed below 20 samples", "Python baselines use upstream APIs through benchmark HTTP glue, not production servers", "Concurrency is one; no throughput or server scaling claim"],
        "versions":{},"starts":[]});
    for (name, dir) in [
        ("llama", llama.clone()),
        ("kev-sdk", sources.join("kev-source")),
        ("jev-sdk", sources.join("jev-source")),
    ] {
        let o = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["rev-parse", "HEAD"])
            .output()?;
        ensure!(o.status.success(), "cannot identify {name}");
        manifest["versions"][name] = json!(String::from_utf8(o.stdout)?.trim());
    }
    let cache = fs::read_to_string(llama.join("build/CMakeCache.txt"))?;
    manifest["llama_build_options"] = json!(
        cache
            .lines()
            .filter(|l| (l.starts_with("GGML_") && l.contains(":BOOL="))
                || l.starts_with("CMAKE_BUILD_TYPE:STRING="))
            .collect::<Vec<_>>()
    );
    ensure!(
        cfg.engines
            .iter()
            .any(|e| models.contains(&e.model) && (engines.is_empty() || engines.contains(&e.id))),
        "empty engine selection"
    );
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let mut raw = fs::File::create(output.join("samples.jsonl"))?;
    let mut samples = Vec::<Value>::new();
    for block in 0..blocks {
        for &nt in &threads {
            let mut selected: Vec<_> = cfg
                .engines
                .iter()
                .filter(|e| {
                    models.contains(&e.model) && (engines.is_empty() || engines.contains(&e.id))
                })
                .collect();
            if block % 2 == 1 {
                selected.reverse();
            }
            for engine in selected {
                let replace = |s: &str| {
                    s.replace("{threads}", &nt.to_string())
                        .replace("{port}", &port.to_string())
                        .replace("{output}", &output.to_string_lossy())
                        .replace("{sources}", &sources.to_string_lossy())
                        .replace("{llama}", &llama.to_string_lossy())
                        .replace("{python}", &python.to_string_lossy())
                };
                let argv: Vec<_> = engine.command.iter().map(|s| replace(s)).collect();
                ensure!(!argv.is_empty(), "empty engine command");
                // Refuse stale listeners so another process cannot be measured.
                ensure!(
                    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok(),
                    "port already in use"
                );
                let log = fs::File::create(
                    output
                        .join("private-logs")
                        .join(format!("{}-t{nt}-b{block}.log", engine.id)),
                )?;
                let start = Instant::now();
                let mut cmd = Command::new("setpriv");
                cmd.args(["--pdeathsig", "TERM", "--", "setsid"])
                    .args(&argv)
                    .stdout(log.try_clone()?)
                    .stderr(log);
                for key in [
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "RAYON_NUM_THREADS",
                ] {
                    cmd.env(key, nt.to_string());
                }
                cmd.env("TOKENIZERS_PARALLELISM", "false");
                let mut child = Process(cmd.spawn()?);
                let health = loop {
                    if let Ok(v) = http(port, "/health", None) {
                        break v;
                    }
                    ensure!(
                        child.0.try_wait()?.is_none(),
                        "{} exited; see private log",
                        engine.id
                    );
                    ensure!(
                        start.elapsed() < Duration::from_secs(600),
                        "server readiness timeout"
                    );
                    thread::sleep(Duration::from_millis(200));
                };
                manifest["starts"].as_array_mut().unwrap().push(json!({"engine":engine.id,"threads":nt,"block":block,"ready_ms":start.elapsed().as_secs_f64()*1000.,"health":health,"host_before":snapshot()}));
                let mut selected_cases: Vec<_> =
                    cases.iter().filter(|c| c.model == engine.model).collect();
                if block % 2 == 1 {
                    selected_cases.reverse();
                }
                for case in selected_cases {
                    eprintln!("{} t{nt} block{block} {}", engine.id, case.id);
                    for round in 0..warmup + rounds {
                        for reset in &engine.reset {
                            http(port, reset, Some(&json!({})))?;
                        }
                        for trial in &case.requests {
                            let before = snapshot();
                            let timer = Instant::now();
                            let result = http(port, "/v1/systemone", Some(&trial.request));
                            let elapsed = timer.elapsed().as_secs_f64() * 1000.;
                            let after = snapshot();
                            let response = result
                                .as_ref()
                                .map(|v| v.clone())
                                .unwrap_or_else(|e| json!({"error":e.to_string()}));
                            let parity = compare(&response, &trial.expected)
                                .unwrap_or_else(|e| json!({"error":e.to_string()}));
                            let pass = parity["max_probability_error"]
                                .as_f64()
                                .is_some_and(|v| v <= max_error)
                                && parity["changed_decisions"] == 0
                                && parity["input_usage_matches"] == true
                                && parity["max_api_scalar_error"]
                                    .as_f64()
                                    .is_some_and(|v| v <= max_error + 4. * f64::EPSILON);
                            let record = json!({"engine":engine.id,"model":case.model,"case":case.id,"language":case.language,"length":case.length,
                            "input_tokens":trial.input_tokens,"state_tokens":trial.state_tokens,"kind":trial.kind,"threads":nt,"block":block,
                            "round":round,"warmup":round<warmup,"latency_ms":elapsed,"parity":parity,"pass":pass,"response":response,
                            "server_memory":memory(child.0.id()),"host_before":before,"host_after":after});
                            writeln!(raw, "{record}")?;
                            raw.flush()?;
                            samples.push(record);
                        }
                    }
                }
                drop(child);
                fs::write(
                    output.join("manifest.json"),
                    serde_json::to_vec_pretty(&manifest)?,
                )?;
            }
        }
    }
    let mut groups: BTreeMap<(String, String, String, u64), Vec<&Value>> = BTreeMap::new();
    for row in &samples {
        if row["warmup"] == false {
            groups
                .entry((
                    row["engine"].as_str().unwrap().into(),
                    row["case"].as_str().unwrap().into(),
                    row["kind"].as_str().unwrap().into(),
                    row["threads"].as_u64().unwrap(),
                ))
                .or_default()
                .push(row);
        }
    }
    let summary:Vec<_>=groups.into_iter().map(|((engine,case,kind,nt),rows)|{
        let mut times:Vec<_>=rows.iter().map(|r|r["latency_ms"].as_f64().unwrap()).collect();
        let max=rows.iter().filter_map(|r|r["parity"]["max_probability_error"].as_f64()).reduce(f64::max);
        let all_pass= samples.iter().filter(|r|r["engine"]==engine&&r["case"]==case&&r["kind"]==kind&&r["threads"]==nt).all(|r|r["pass"]==true);
        json!({"engine":engine,"case":case,"kind":kind,"threads":nt,"n":times.len(),"median_ms":percentile(&mut times,0.5),
            "min_ms":times[0],"max_ms":times[times.len()-1],"p95_ms":if times.len()>=20 {Some(percentile(&mut times,0.95))}else{None},
            "max_probability_error":max,"all_pass_including_warmup":all_pass})
    }).collect();
    manifest["host_after"] = snapshot();
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    let failures = samples.iter().filter(|r| r["pass"] != true).count();
    println!(
        "{} samples; {failures} failed comparisons; {}",
        samples.len(),
        output.display()
    );
    ensure!(
        failures == 0,
        "numerical gate failed; results retained, failing cells are not equivalent-performance comparisons"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_missing_changed_and_bad_probabilities() {
        let e = json!({"answers":{"q":{"type":"choice","choice":"a","probabilities":{"a":0.7,"b":0.3},"confidence":0.4}}});
        assert!(compare(&json!({"answers":{}}), &e).is_err());
        let mut a = e.clone();
        a["answers"]["q"]["probabilities"]["c"] = json!(0.);
        assert!(compare(&a, &e).is_err());
        let mut a = e.clone();
        a["answers"]["q"]["choice"] = json!("b");
        assert_eq!(compare(&a, &e).unwrap()["changed_decisions"], 1);
        let mut a = e.clone();
        a["answers"]["q"]["probabilities"]["a"] = Value::Null;
        assert!(compare(&a, &e).is_err());
    }
    #[test]
    fn raw_probabilities_take_precedence_over_rounding() {
        let e = json!({"answers":{"q":{"type":"choice","choice":"a","probabilities":{"a":0.5,"b":0.5}}},"probabilities":{"q":[0.50004,0.49996]}});
        let mut a = e.clone();
        a["probabilities"]["q"] = json!([0.50005, 0.49995]);
        let out = compare(&a, &e).unwrap();
        assert!((out["max_probability_error"].as_f64().unwrap() - 1e-5).abs() < 1e-12);
    }
}

fn main() -> Result<()> {
    run(Args::parse_from(
        std::env::args().filter(|arg| arg != "--bench"),
    ))
}
