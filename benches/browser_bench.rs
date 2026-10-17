use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    fs,
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tungstenite::{Message, WebSocket, stream::MaybeTlsStream};
#[derive(Clone, Parser)]
#[command(about = "Run browser inference checks and benchmark matrices")]
pub(crate) struct Args {
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[arg(long, default_value = "chromium")]
    browser: PathBuf,
    #[arg(help = "Linux CPU list for Chromium and its children, for example 0 or 0-3")]
    #[arg(long)]
    cpu_affinity: Option<String>,
    #[arg(help = "A local test page, including query parameters that define the run")]
    #[arg(
        long,
        default_value = "/examples/browser/platform.html?model=decima&backend=wasm"
    )]
    page: String,
    #[arg(help = "JSON matrix of named pages. Rotates order and uses a fresh process per run")]
    #[arg(long, conflicts_with_all = ["cpu_profile", "cpu_accounting"])]
    matrix: Option<PathBuf>,
    #[arg(help = "Result JSON file, or a result directory when --matrix is supplied")]
    #[arg(long)]
    output: PathBuf,
    #[arg(help = "Chromium sampling profile, starting after model loading and warmup")]
    #[arg(long)]
    cpu_profile: Option<PathBuf>,
    #[arg(help = "Include dedicated workers in a profile collection (JSON profiles array)")]
    #[arg(long, requires = "cpu_profile")]
    cpu_profile_workers: bool,
    #[arg(help = "Collect Linux Chromium CPU time/context switches without V8 sampling")]
    #[arg(long)]
    cpu_accounting: bool,
    #[arg(long, default_value_t = 1800)]
    timeout_seconds: u64,
    #[arg(help = "Required in some containers; ordinary desktop runs retain the sandbox")]
    #[arg(long)]
    no_sandbox: bool,
    #[arg(help = "Software WebGPU is for correctness only, never GPU performance claims")]
    #[arg(long)]
    software_gpu: bool,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Matrix {
    repetitions: usize,
    warmup: usize,
    rounds: usize,
    cases: Vec<Case>,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    page: String,
}

fn matrix(args: &Args, path: &Path) -> Result<()> {
    let matrix: Matrix = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        matrix.repetitions > 0 && matrix.rounds > 0 && !matrix.cases.is_empty(),
        "matrix needs cases, repetitions and timed rounds"
    );
    let mut names = std::collections::HashSet::new();
    for case in &matrix.cases {
        ensure!(
            !case.name.is_empty()
                && case
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
            "case names may contain letters, digits, hyphens and underscores"
        );
        ensure!(
            names.insert(&case.name),
            "duplicate case name: {}",
            case.name
        );
    }
    fs::create_dir_all(&args.output)?;
    ensure!(
        fs::read_dir(&args.output)?.next().is_none(),
        "matrix output directory must be empty; choose a new output directory"
    );
    fs::write(args.output.join("matrix.json"), fs::read(path)?)?;
    let mut results = Vec::new();
    let mut failures = 0;
    for repetition in 0..matrix.repetitions {
        for offset in 0..matrix.cases.len() {
            let index = if repetition % 2 == 0 {
                offset
            } else {
                matrix.cases.len() - 1 - offset
            };
            let case = &matrix.cases[index];
            let (page, query) = case.page.split_once('?').unwrap_or((&case.page, ""));
            let mut query: Vec<String> = query
                .split('&')
                .filter(|p| !p.is_empty() && !p.starts_with("warmup=") && !p.starts_with("rounds="))
                .map(str::to_string)
                .collect();
            query.push(format!("warmup={}", matrix.warmup));
            query.push(format!("rounds={}", matrix.rounds));
            let mut run_args = args.clone();
            run_args.page = format!("{page}?{}", query.join("&"));
            eprintln!("matrix repetition {repetition}: {}", case.name);
            let result =
                run_browser(&run_args).unwrap_or_else(|e| json!({"error":format!("{e:#}")}));
            let file = format!("rep-{repetition:02}-{}.json", case.name);
            ensure!(
                !args.output.join(&file).exists(),
                "result already exists: {file}; choose a new output directory"
            );
            fs::write(args.output.join(&file), serde_json::to_vec_pretty(&result)?)?;
            if result.get("error").is_some() {
                failures += 1;
            }
            results.push(
                json!({"case":case.name,"repetition":repetition,"file":file,"result":result}),
            );
            fs::write(
                args.output.join("runs.json"),
                serde_json::to_vec_pretty(&results)?,
            )?;
        }
    }
    // Keep paired baseline/candidate samples separate.
    let mut samples: std::collections::BTreeMap<(String, Option<String>, usize), Vec<f64>> =
        Default::default();
    for run in &results {
        if let Some(records) = run["result"]["records"].as_array() {
            // Contract pages have no timings; retain their records without latency statistics.
            if run["result"]["all_exact"] == true
                && run["result"]["compared"].as_u64() == Some(records.len() as u64)
                && records.iter().all(|r| r.get("latencies_ms").is_none())
            {
                continue;
            }
            for record in records {
                let key = (
                    run["case"].as_str().unwrap().to_owned(),
                    record["variant"].as_str().map(str::to_owned),
                    record["fixture"].as_u64().context("record fixture index")? as usize,
                );
                samples.entry(key).or_default().extend(
                    record["latencies_ms"]
                        .as_array()
                        .context("latency samples")?
                        .iter()
                        .map(|v| v.as_f64().unwrap()),
                );
            }
        }
    }
    let mut summary = Vec::new();
    for ((case, variant, fixture), mut times) in samples {
        times.sort_by(f64::total_cmp);
        ensure!(
            !times.is_empty() && times.iter().all(|t| t.is_finite() && *t >= 0.),
            "invalid timing samples"
        );
        let n = times.len();
        let median = (times[(n - 1) / 2] + times[n / 2]) / 2.;
        let p95 = times[(n * 95).div_ceil(100) - 1];
        summary.push(
            json!({"case":case,"variant":variant,"fixture":fixture,"samples":n,"median_ms":median,"p95_ms":p95}),
        );
    }
    fs::write(
        args.output.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"failures":failures,"repetitions":matrix.repetitions,"warmup":matrix.warmup,"rounds":matrix.rounds,"order":"forward on even repetitions, reverse on odd repetitions; one fresh browser per case","cases":summary}),
        )?,
    )?;
    ensure!(
        failures == 0,
        "{failures} matrix runs failed; individual results were retained"
    );
    Ok(())
}
struct Browser {
    child: Child,
    directory: PathBuf,
}
impl Drop for Browser {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.directory);
    }
}
fn serve(root: PathBuf) -> Result<(u16, Arc<AtomicBool>)> {
    let server = tiny_http::Server::http("127.0.0.1:0").map_err(|e| anyhow::anyhow!("{e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .context("loopback HTTP address")?
        .port();
    let stop = Arc::new(AtomicBool::new(false));
    let stopping = stop.clone();
    std::thread::spawn(move || {
        while !stopping.load(Ordering::Relaxed) {
            let Ok(Some(request)) = server.recv_timeout(Duration::from_millis(100)) else {
                continue;
            };
            let name = request.url().split('?').next().unwrap_or("");
            let allowed = [
                "/examples/browser/",
                "/examples/pplx-flash/",
                "/tests/fixtures/",
                "/benches/",
                "/models/",
                "/artifacts/web/",
            ]
            .iter()
            .any(|p| name.starts_with(p))
                && !name.contains('%')
                && !name.split('/').any(|p| p.starts_with('.'));
            let path = root.join(name.trim_start_matches('/'));
            let file = if allowed {
                path.canonicalize()
                    .ok()
                    .filter(|p| p.starts_with(&root))
                    .and_then(|p| fs::File::open(p).ok())
            } else {
                None
            };
            if let Some(mut file) = file {
                let mime = match path.extension().and_then(|s| s.to_str()) {
                    Some("html") => "text/html",
                    Some("js" | "mjs") => "text/javascript",
                    Some("wasm") => "application/wasm",
                    Some("json" | "jsonl") => "application/json",
                    _ => "application/octet-stream",
                };
                use std::io::{Read, Seek, SeekFrom};
                let size = file.metadata().map(|m| m.len()).unwrap_or(0);
                let range = request
                    .headers()
                    .iter()
                    .find(|h| h.field.equiv("Range"))
                    .map(|h| h.value.as_str());
                let (start, end, partial) = if let Some(range) = range {
                    let bounds = range
                        .strip_prefix("bytes=")
                        .and_then(|s| s.split_once('-'))
                        .and_then(|(a, b)| Some((a.parse::<u64>().ok()?, b.parse::<u64>().ok()?)));
                    let Some((start, end)) =
                        bounds.filter(|&(start, end)| start <= end && end < size)
                    else {
                        let _ = request.respond(tiny_http::Response::empty(416));
                        continue;
                    };
                    (start, end + 1, true)
                } else {
                    (0, size, false)
                };
                if file.seek(SeekFrom::Start(start)).is_err() {
                    let _ = request.respond(tiny_http::Response::empty(500));
                    continue;
                }
                let mut response = tiny_http::Response::new(
                    tiny_http::StatusCode(if partial { 206 } else { 200 }),
                    Vec::new(),
                    file.take(end - start),
                    usize::try_from(end - start).ok(),
                    None,
                );
                response
                    .add_header(tiny_http::Header::from_bytes("Accept-Ranges", "bytes").unwrap());
                if partial {
                    response.add_header(
                        tiny_http::Header::from_bytes(
                            "Content-Range",
                            format!("bytes {start}-{}/{size}", end - 1),
                        )
                        .unwrap(),
                    );
                }
                for (key, value) in [
                    ("Content-Type", mime),
                    ("Cache-Control", "no-store"),
                    ("Cross-Origin-Opener-Policy", "same-origin"),
                    ("Cross-Origin-Embedder-Policy", "require-corp"),
                ] {
                    response.add_header(tiny_http::Header::from_bytes(key, value).unwrap());
                }
                let _ = request.respond(response);
            } else {
                let _ = request.respond(tiny_http::Response::empty(404));
            }
        }
    });
    Ok((port, stop))
}
struct Cdp {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    id: u64,
    deadline: Instant,
    page_session: Option<String>,
    workers: Vec<Value>,
}
impl Cdp {
    fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let session = self.page_session.clone();
        self.call_session(method, params, session.as_deref())
    }
    fn call_session(
        &mut self,
        method: &str,
        params: Value,
        session: Option<&str>,
    ) -> Result<Value> {
        self.id += 1;
        let id = self.id;
        let mut request = json!({"id":id,"method":method,"params":params});
        if let Some(session) = session {
            request["sessionId"] = json!(session);
        }
        self.socket
            .send(Message::Text(request.to_string().into()))?;
        loop {
            ensure!(
                Instant::now() < self.deadline,
                "browser deadline reached during {method}"
            );
            let message = match self.socket.read() {
                Err(tungstenite::Error::Io(e))
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                    ) =>
                {
                    continue;
                }
                result => result?,
            };
            let Message::Text(text) = message else {
                continue;
            };
            let response: Value = serde_json::from_str(text.as_str())?;
            if session.is_some() && std::env::var_os("DECISIVE_CDP_TRACE").is_some() {
                eprintln!(
                    "CDP response: {}",
                    text.chars().take(600).collect::<String>()
                );
            }
            if response["method"] == "Target.attachedToTarget"
                && response["params"]["targetInfo"]["type"] == "worker"
            {
                let params = &response["params"];
                self.workers.push(json!({"type":"worker","url":params["targetInfo"]["url"],"session":params["sessionId"],"children":false}));
            }
            if response["id"] == id {
                ensure!(
                    response.get("error").is_none(),
                    "CDP {method}: {}",
                    response["error"]
                );
                return Ok(response["result"].clone());
            }
        }
    }
    fn attach_worker_children(&mut self) -> Result<()> {
        let mut index = 0;
        while index < self.workers.len() {
            if self.workers[index]["children"] != true {
                self.workers[index]["children"] = json!(true);
                let session = self.workers[index]["session"]
                    .as_str()
                    .context("worker session")?
                    .to_owned();
                self.call_session(
                    "Target.setAutoAttach",
                    json!({"autoAttach":true,"waitForDebuggerOnStart":false,"flatten":true}),
                    Some(&session),
                )?;
            }
            index += 1;
        }
        Ok(())
    }
}
fn start_profiles(cdp: &mut Cdp, workers: bool) -> Result<Vec<Value>> {
    let mut targets = vec![json!({"type":"page", "session":cdp.page_session})];
    if workers {
        targets.extend(cdp.workers.iter().cloned());
        ensure!(
            targets.len() > 1,
            "no worker targets available for sampling"
        );
    }
    for target in &targets {
        eprintln!("Starting profiler: {} {}", target["type"], target["url"]);
        let session = target["session"].as_str();
        if target["type"] == "worker" {
            cdp.call_session("Runtime.enable", json!({}), session)?;
        }
        cdp.call_session("Profiler.enable", json!({}), session)?;
        cdp.call_session(
            "Profiler.setSamplingInterval",
            json!({"interval":1000}),
            session,
        )?;
        cdp.call_session("Profiler.start", json!({}), session)?;
    }
    Ok(targets)
}
fn stop_profiles(cdp: &mut Cdp, targets: &[Value], path: &Path, workers: bool) -> Result<()> {
    let mut profiles = vec![];
    for target in targets {
        let profile = cdp.call_session("Profiler.stop", json!({}), target["session"].as_str())?;
        profiles
            .push(json!({"type":target["type"],"url":target["url"],"profile":profile["profile"]}));
    }
    if let Some(parent) = path.parent().filter(|p| *p != Path::new("")) {
        fs::create_dir_all(parent)?;
    }
    let result = if workers {
        json!({"profiles":profiles})
    } else {
        profiles[0]["profile"].clone()
    };
    fs::write(path, serde_json::to_vec(&result)?)?;
    Ok(())
}

#[derive(Default)]
struct CpuSnapshot {
    threads: std::collections::BTreeMap<(u32, u64), [u64; 4]>,
}
fn cpu_snapshot(root: u32) -> CpuSnapshot {
    let mut result = CpuSnapshot::default();
    if !cfg!(target_os = "linux") {
        return result;
    }
    let mut pending = vec![root];
    let mut seen = std::collections::HashSet::new();
    while let Some(pid) = pending.pop() {
        if !seen.insert(pid) {
            continue;
        }
        let Ok(tasks) = fs::read_dir(format!("/proc/{pid}/task")) else {
            continue;
        };
        for task in tasks.flatten() {
            let path = task.path();
            if let Ok(children) = fs::read_to_string(path.join("children")) {
                pending.extend(
                    children
                        .split_whitespace()
                        .filter_map(|s| s.parse::<u32>().ok()),
                );
            }
            let Ok(tid) = task.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            let Ok(stat) = fs::read_to_string(path.join("stat")) else {
                continue;
            };
            let Some((_, fields)) = stat.rsplit_once(')') else {
                continue;
            };
            let fields: Vec<_> = fields.split_whitespace().collect();
            let number = |i| fields.get(i).and_then(|s: &&str| s.parse::<u64>().ok());
            let (Some(user), Some(system), Some(start)) = (number(11), number(12), number(19))
            else {
                continue;
            };
            let Ok(status) = fs::read_to_string(path.join("status")) else {
                continue;
            };
            let switches = |key: &str| {
                status
                    .lines()
                    .find_map(|line| line.strip_prefix(key)?.trim().parse::<u64>().ok())
            };
            let (Some(voluntary), Some(involuntary)) = (
                switches("voluntary_ctxt_switches:"),
                switches("nonvoluntary_ctxt_switches:"),
            ) else {
                continue;
            };
            result
                .threads
                .insert((tid, start), [user, system, voluntary, involuntary]);
        }
    }
    result
}
fn cpu_delta(
    before: &CpuSnapshot,
    after: &CpuSnapshot,
    wall: Duration,
    ticks: Option<u64>,
) -> Value {
    let mut totals = [0_u64; 4];
    let mut matched = 0;
    for (key, values) in &after.threads {
        if let Some(previous) = before.threads.get(key) {
            for i in 0..4 {
                totals[i] += values[i].saturating_sub(previous[i]);
            }
            matched += 1;
        }
    }
    let seconds = ticks
        .filter(|n| *n > 0)
        .map(|n| (totals[0] + totals[1]) as f64 / n as f64);
    json!({"available":!before.threads.is_empty() && ticks.is_some(),
        "scope":"Chromium process tree during the profiled window; matched live threads only; includes browser/profiler overhead, excludes benchmark runner",
        "wall_seconds":wall.as_secs_f64(),"clock_ticks_per_second":ticks,
        "user_ticks":totals[0],"system_ticks":totals[1],"cpu_seconds":seconds,
        "average_cpu_cores":seconds.map(|s|s/wall.as_secs_f64()),
        "voluntary_context_switches":totals[2],"involuntary_context_switches":totals[3],
        "matched_threads":matched,"new_threads_excluded":after.threads.len()-matched,
        "exited_threads_excluded":before.threads.len()-matched})
}
fn run_browser(args: &Args) -> Result<Value> {
    ensure!(
        cfg!(target_os = "linux") || !args.cpu_accounting,
        "--cpu-accounting requires Linux /proc"
    );
    let profile_window = args.cpu_profile.is_some() || args.cpu_accounting;
    ensure!(
        cfg!(target_os = "linux") || args.cpu_affinity.is_none(),
        "--cpu-affinity requires Linux taskset"
    );
    ensure!(
        args.page.starts_with("/examples/browser/") && !args.page.contains('#'),
        "page must be a local browser test"
    );
    let deadline = Instant::now() + Duration::from_secs(args.timeout_seconds);
    let (port, stop) = serve(args.root.canonicalize()?)?;
    struct Stop(Arc<AtomicBool>);
    impl Drop for Stop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    let _stop = Stop(stop);
    let directory = std::env::temp_dir().join(format!(
        "decisive-browser-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    fs::create_dir(&directory)?;
    let log = fs::File::create(directory.join("browser.log"))?;
    let mut command = if let Some(cpus) = &args.cpu_affinity {
        let mut command = Command::new("taskset");
        command.args(["-c", cpus]).arg(&args.browser);
        command
    } else {
        Command::new(&args.browser)
    };
    command
        .args([
            "--headless=new",
            "--disable-dev-shm-usage",
            "--remote-debugging-port=0",
            "--enable-unsafe-webgpu",
        ])
        .arg(format!("--user-data-dir={}", directory.display()))
        .arg("about:blank")
        .stdout(Stdio::null())
        .stderr(Stdio::from(log));
    if args.no_sandbox {
        command.arg("--no-sandbox");
    }
    if args.software_gpu {
        command.arg("--use-angle=swiftshader");
    }
    let mut browser = Browser {
        child: command
            .spawn()
            .context("start Chromium; supply --browser if it is not on PATH")?,
        directory,
    };
    let active = browser.directory.join("DevToolsActivePort");
    let cdp_port = loop {
        if let Ok(text) = fs::read_to_string(&active) {
            break text
                .lines()
                .next()
                .context("missing DevTools port")?
                .parse::<u16>()?;
        }
        if let Some(status) = browser.child.try_wait()? {
            bail!(
                "Chromium exited: {status}; {}",
                fs::read_to_string(browser.directory.join("browser.log"))?
            );
        }
        ensure!(Instant::now() < deadline, "Chromium startup timeout");
        std::thread::sleep(Duration::from_millis(100));
    };
    let target: Value = serde_json::from_str(
        &ureq::put(&format!("http://127.0.0.1:{cdp_port}/json/new?about:blank"))
            .send_empty()?
            .body_mut()
            .read_to_string()?,
    )?;
    let endpoint = if args.cpu_profile_workers {
        serde_json::from_str::<Value>(
            &ureq::get(&format!("http://127.0.0.1:{cdp_port}/json/version"))
                .call()?
                .body_mut()
                .read_to_string()?,
        )?
    } else {
        target.clone()
    };
    let (mut socket, _) = tungstenite::connect(
        endpoint["webSocketDebuggerUrl"]
            .as_str()
            .context("DevTools WebSocket URL")?,
    )?;
    if let MaybeTlsStream::Plain(stream) = socket.get_mut() {
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    }
    let mut cdp = Cdp {
        socket,
        id: 0,
        deadline,
        page_session: None,
        workers: vec![],
    };
    if args.cpu_profile_workers {
        let attached = cdp.call_session(
            "Target.attachToTarget",
            json!({"targetId":target["id"],"flatten":true}),
            None,
        )?;
        cdp.page_session = Some(
            attached["sessionId"]
                .as_str()
                .context("page session")?
                .into(),
        );
        cdp.call(
            "Target.setAutoAttach",
            json!({"autoAttach":true,"waitForDebuggerOnStart":false,"flatten":true}),
        )?;
    }
    cdp.call("Network.enable", json!({}))?;
    cdp.call("Network.setCacheDisabled", json!({"cacheDisabled":true}))?;
    cdp.call("Page.enable", json!({}))?;
    let page = if profile_window {
        format!(
            "{}{}profile=1",
            args.page,
            if args.page.contains('?') { "&" } else { "?" }
        )
    } else {
        args.page.clone()
    };
    cdp.call(
        "Page.navigate",
        json!({"url":format!("http://127.0.0.1:{port}{page}")}),
    )?;
    let mut previous = String::new();
    let mut profiling = false;
    let mut profile_recorded = false;
    let mut profile_targets = vec![];
    let clock_ticks = if profile_window && cfg!(target_os = "linux") {
        Command::new("getconf")
            .arg("CLK_TCK")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
    } else {
        None
    };
    let mut cpu_start = None;
    let mut cpu_accounting = Value::Null;
    loop {
        ensure!(Instant::now() < deadline, "browser run timeout");
        let r=cdp.call("Runtime.evaluate",json!({"expression":"({progress:window.progress,result:window.result,profileReady:window.profileReady,profileDone:window.profileDone})","returnByValue":true}))?;
        ensure!(
            r.get("exceptionDetails").is_none(),
            "browser exception: {}",
            r["exceptionDetails"]
        );
        let state = &r["result"]["value"];
        if args.cpu_profile_workers {
            cdp.attach_worker_children()?;
        }
        if profile_window && !profiling && !profile_recorded && state["profileReady"] == true {
            if args.cpu_profile.is_some() {
                profile_targets = start_profiles(&mut cdp, args.cpu_profile_workers)?;
            }
            cpu_start = Some((cpu_snapshot(browser.child.id()), Instant::now()));
            profiling = true;
            cdp.call(
                "Runtime.evaluate",
                json!({"expression":"window.profileStarted=true"}),
            )?;
        }
        if profiling && state["profileDone"] == true {
            if let Some((before, start)) = cpu_start.take() {
                cpu_accounting = cpu_delta(
                    &before,
                    &cpu_snapshot(browser.child.id()),
                    start.elapsed(),
                    clock_ticks,
                );
            }
            if let Some(path) = &args.cpu_profile {
                stop_profiles(&mut cdp, &profile_targets, path, args.cpu_profile_workers)?;
            }
            profiling = false;
            profile_recorded = true;
            cdp.call(
                "Runtime.evaluate",
                json!({"expression":"window.profileStopped=true"}),
            )?;
        }
        if let Some(progress) = state["progress"].as_str() {
            if progress != previous {
                eprintln!("{progress}");
                previous = progress.into();
            }
        }
        if let Some(result) = state.get("result") {
            if profiling && let Some(path) = &args.cpu_profile {
                stop_profiles(&mut cdp, &profile_targets, path, args.cpu_profile_workers)?;
            }
            let mut result = result.clone();
            result["software_gpu_requested"] = json!(args.software_gpu);
            result["cpu_affinity"] = json!(args.cpu_affinity);
            if profile_window {
                result["profile_cpu_accounting"] = cpu_accounting;
                result["v8_sampling"] = json!(args.cpu_profile.is_some());
            }
            return Ok(result);
        }
        std::thread::sleep(Duration::from_millis(if profiling { 20 } else { 500 }));
    }
}
pub(crate) fn run(args: Args) -> Result<()> {
    if let Some(path) = &args.matrix {
        return matrix(&args, path);
    }
    let result = run_browser(&args)?;
    if let Some(parent) = args.output.parent().filter(|p| *p != Path::new("")) {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, serde_json::to_vec_pretty(&result)?)?;
    ensure!(
        result.get("error").is_none(),
        "browser check failed: {}",
        result["error"]
    );
    println!(
        "saved {} records",
        result["records"].as_array().map_or(0, Vec::len)
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse_from(
        std::env::args().filter(|arg| arg != "--bench"),
    ))
}
