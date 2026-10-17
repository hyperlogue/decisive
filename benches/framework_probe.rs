use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
#[allow(dead_code)]
#[path = "../src/models/laya/benchmark.rs"]
mod benchmark;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "target/release/decisive")]
    decisive: PathBuf,
    #[arg(long)]
    llama: PathBuf,
    #[arg(long, default_value = "models/laya")]
    model: PathBuf,
    #[arg(long)]
    gguf: PathBuf,
    #[arg(long, default_value = "benches/long-inputs.jsonl")]
    input: PathBuf,
    #[arg(long, default_value = "tests/fixtures/outputs/long-inputs.jsonl")]
    reference: PathBuf,
    #[arg(help = "Fresh directory; contains raw results, metadata and private server logs")]
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = 8)]
    rounds: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value = "f16", value_parser = ["f16", "f32"])]
    decisive_storage: String,
    #[arg(long, default_value_t = 8092)]
    llama_port: u16,
    #[arg(long, default_value_t = 8094)]
    decisive_port: u16,
    #[arg(long, default_value = "auto", value_parser = ["auto", "on", "off"])]
    flash_attention: String,
}
pub(crate) struct Server(pub(crate) Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
pub(crate) fn spawn(mut command: Command, log: &Path) -> Result<Server> {
    let stderr = fs::File::create(log)?;
    command.stdout(Stdio::null()).stderr(stderr);
    Ok(Server(command.spawn()?))
}
pub(crate) fn digest(path: &Path) -> Result<String> {
    let result = Command::new("sha256sum").arg(path).output()?;
    ensure!(result.status.success(), "sha256sum failed");
    let hash = String::from_utf8(result.stdout)?
        .split_whitespace()
        .next()
        .context("empty hash")?
        .to_owned();
    ensure!(
        hash.len() == 64 && hash.bytes().all(|c| c.is_ascii_hexdigit()),
        "invalid hash"
    );
    Ok(hash)
}
pub(crate) fn ready(server: &mut Server, port: u16) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(240);
    loop {
        ensure!(
            server.0.try_wait()?.is_none(),
            "server exited; inspect its log"
        );
        if ureq::get(&format!("http://127.0.0.1:{port}/health"))
            .call()
            .is_ok()
        {
            return Ok(());
        }
        ensure!(Instant::now() < deadline, "server readiness timeout");
        std::thread::sleep(Duration::from_millis(100));
    }
}
#[cfg(target_os = "linux")]
pub(crate) fn process_cpu_seconds(pid: u32) -> Option<f64> {
    use std::os::raw::{c_int, c_long};
    #[repr(C)]
    struct Timespec {
        seconds: c_long,
        nanoseconds: c_long,
    }
    unsafe extern "C" {
        fn clock_getcpuclockid(pid: c_int, clock: *mut c_int) -> c_int;
        fn clock_gettime(clock: c_int, time: *mut Timespec) -> c_int;
    }
    let mut clock = 0;
    let mut time = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    if unsafe { clock_getcpuclockid(pid as c_int, &mut clock) } != 0
        || unsafe { clock_gettime(clock, &mut time) } != 0
    {
        return None;
    }
    Some(time.seconds as f64 + time.nanoseconds as f64 * 1e-9)
}
#[cfg(not(target_os = "linux"))]
pub(crate) fn process_cpu_seconds(_: u32) -> Option<f64> {
    None
}

#[cfg(target_os = "linux")]
pub(crate) fn loaded_library_hashes(pid: u32) -> Result<Value> {
    use std::collections::BTreeSet;
    let maps = fs::read_to_string(format!("/proc/{pid}/maps"))?;
    let mut paths = BTreeSet::new();
    for line in maps.lines() {
        // Skip five metadata fields; never publish local paths or process IDs.
        let mut tail = line;
        for _ in 0..5 {
            tail = tail.trim_start();
            tail = tail.find(char::is_whitespace).map_or("", |i| &tail[i..]);
        }
        let path = tail.trim_start();
        if path.starts_with('/')
            && Path::new(path)
                .file_name()
                .is_some_and(|n| n.to_string_lossy().contains(".so"))
        {
            ensure!(
                !path.ends_with(" (deleted)"),
                "loaded library was deleted during measurement"
            );
            paths.insert(PathBuf::from(path));
        }
    }
    let mut libraries = Vec::new();
    for path in paths {
        libraries.push(json!({"name":path.file_name().context("missing library name")?.to_string_lossy(),"sha256":digest(&path)?}));
    }
    Ok(
        json!({"scope":"Linux mapped shared libraries after measurement, including dynamically loaded backends. Basenames and content hashes only; no local paths or process identifiers.","libraries":libraries}),
    )
}
#[cfg(not(target_os = "linux"))]
pub(crate) fn loaded_library_hashes(_: u32) -> Result<Value> {
    Ok(Value::Null)
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    ensure!(
        args.threads > 0 && args.rounds > 0 && args.rounds % 4 == 0 && args.warmup > 0,
        "positive counts; measured rounds must be a multiple of four"
    );
    ensure!(args.decisive_port != args.llama_port, "ports must differ");
    let reservations = [
        TcpListener::bind(("127.0.0.1", args.llama_port))?,
        TcpListener::bind(("127.0.0.1", args.decisive_port))?,
    ];
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&args.out).context("output directory must be new")?;
    let requests: Vec<Value> = fs::read_to_string(&args.input)?
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let expected: Vec<Value> = fs::read_to_string(&args.reference)?
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    ensure!(
        !requests.is_empty() && requests.len() == expected.len(),
        "empty corpus or reference count mismatch"
    );
    let mut model_hashes = serde_json::Map::new();
    for name in [
        "beam-revision.json",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "model.safetensors",
    ] {
        model_hashes.insert(name.into(), json!(digest(&args.model.join(name))?));
    }
    let version = Command::new(&args.llama).arg("--version").output()?;
    let mut manifest = json!({
        "scope":"Warm sequential CPU-only HTTP comparison, one server slot, explicit thread budget, native-pool polling disabled and OpenMP passive waiting. Output-contract differences remain visible. Matching token counts do not prove numerical equivalence. Server logs are private diagnostics until sanitized.",
        "threads":args.threads,"rounds":args.rounds,"warmup":args.warmup,"decisive_storage":args.decisive_storage,
        "decisive_avx2_long":true,"token_cache":false,"state_cache":false,
        "llama_context":4096,"llama_batch":2048,"llama_ubatch":2048,"llama_slots":1,"llama_gpu_layers":0,
        "llama_flash_attention":args.flash_attention,"llama_poll":0,"llama_poll_batch":0,
        "omp_wait_policy":"PASSIVE","omp_num_threads":args.threads,
        "openblas_num_threads":args.threads,"blis_num_threads":args.threads,"mkl_num_threads":args.threads,
        "decisive_sha256":digest(&args.decisive)?,"llama_sha256":digest(&args.llama)?,"gguf_sha256":digest(&args.gguf)?,
        "model_assets_sha256":model_hashes,"reference_sha256":digest(&args.reference)?,
        "llama_version_stdout":String::from_utf8_lossy(&version.stdout),"llama_version_stderr":String::from_utf8_lossy(&version.stderr),
        "input_sha256":digest(&args.input)?,"references":serde_json::from_str::<Value>(&fs::read_to_string("references.json")?)?
    });
    fs::write(
        args.out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    drop(reservations);
    let mut cpp = Command::new(&args.llama);
    cpp.arg("-m")
        .arg(&args.gguf)
        .args([
            "-t",
            &args.threads.to_string(),
            "-tb",
            &args.threads.to_string(),
            "-c",
            "4096",
            "-b",
            "2048",
            "-ub",
            "2048",
            "-np",
            "1",
            "-ngl",
            "0",
            "--poll",
            "0",
            "--poll-batch",
            "0",
            "--flash-attn",
            &args.flash_attention,
            "--host",
            "127.0.0.1",
            "--port",
            &args.llama_port.to_string(),
            "--no-warmup",
        ])
        .env("OMP_NUM_THREADS", args.threads.to_string())
        .env("OPENBLAS_NUM_THREADS", args.threads.to_string())
        .env("BLIS_NUM_THREADS", args.threads.to_string())
        .env("MKL_NUM_THREADS", args.threads.to_string())
        .env("OMP_WAIT_POLICY", "PASSIVE");
    let mut cpp = spawn(cpp, &args.out.join("llama.log"))?;
    ready(&mut cpp, args.llama_port)?;
    let mut decisive = Command::new(&args.decisive);
    decisive
        .arg("laya")
        .arg("--model")
        .arg(&args.model)
        .args([
            "--threads",
            &args.threads.to_string(),
            "--storage",
            &args.decisive_storage,
            "--kernel",
            "packed",
            "--avx2-long",
            "serve",
            "--listen",
            &format!("127.0.0.1:{}", args.decisive_port),
        ])
        .env("TOKENIZERS_PARALLELISM", "false");
    let mut decisive = spawn(decisive, &args.out.join("decisive.log"))?;
    ready(&mut decisive, args.decisive_port)?;
    let left = format!("http://127.0.0.1:{}/v1/systemone", args.llama_port);
    let right = format!("http://127.0.0.1:{}/v1/systemone", args.decisive_port);
    let cpu_before = [
        process_cpu_seconds(cpp.0.id()),
        process_cpu_seconds(decisive.0.id()),
    ];
    #[cfg(target_os = "linux")]
    ensure!(
        cpu_before.iter().all(Option::is_some),
        "Linux server CPU clocks unavailable; cannot audit the worker budget"
    );
    let comparison_start = Instant::now();
    let mut result =
        benchmark::http_compare(&requests, &left, &right, args.warmup, args.rounds, true)?;
    let comparison_seconds = comparison_start.elapsed().as_secs_f64();
    let cpu_after = [
        process_cpu_seconds(cpp.0.id()),
        process_cpu_seconds(decisive.0.id()),
    ];
    let cpu_delta: Vec<_> = cpu_before
        .iter()
        .zip(cpu_after)
        .map(|(a, b)| a.zip(b).map(|(a, b)| b - a))
        .collect();
    let combined_cpu_over_wall = cpu_delta[0]
        .zip(cpu_delta[1])
        .map(|(a, b)| (a + b) / comparison_seconds);
    let request_wall = &result["request_wall_seconds_including_warmup"];
    let cpu_over_request_wall = ["left", "right"]
        .iter()
        .enumerate()
        .map(|(i, key)| {
            cpu_delta[i]
                .zip(request_wall[*key].as_f64())
                .map(|(cpu, wall)| cpu / wall)
        })
        .collect::<Vec<_>>();
    result["process_cpu_accounting"] = json!({"scope":"Linux POSIX process CPU clocks, summed over all server threads. CPU interval covers both sequential servers including warmups and client gaps. Each server's CPU is also divided by its own summed request wall time, including warmups; idle spinning is deliberately included in its CPU numerator. Aggregate process audit, not per-request CPU attribution. No hardware PMU or process identifiers. Other operating systems may report null.","wall_seconds":comparison_seconds,"left_cpu_seconds":cpu_delta[0],"right_cpu_seconds":cpu_delta[1],"combined_cpu_over_wall":combined_cpu_over_wall,"left_cpu_over_request_wall":cpu_over_request_wall[0],"right_cpu_over_request_wall":cpu_over_request_wall[1]});
    let mut sdk_error = 0f64;
    let mut sdk_violations = Vec::new();
    let pairs = result["pairs"].as_array().context("missing timed pairs")?;
    ensure!(
        pairs.len() == args.rounds * requests.len(),
        "incomplete comparison"
    );
    for pair in pairs {
        let i = pair["request"].as_u64().context("missing request index")? as usize;
        let reference = expected.get(i).context("invalid request index")?;
        match benchmark::compare_outputs(reference, &pair["right_output"], 1e-4) {
            Ok(error) => sdk_error = sdk_error.max(error),
            Err(error) => sdk_violations
                .push(json!({"round":pair["round"],"request":i,"error":error.to_string()})),
        }
    }
    let checked_predictions = pairs.len();
    result["decisive_sdk_checked_predictions"] = json!(checked_predictions);
    result["decisive_sdk_max_output_error"] = json!(sdk_error);
    result["decisive_sdk_violations"] = json!(sdk_violations);
    fs::write(
        args.out.join("results.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    manifest["llama_loaded_libraries"] = loaded_library_hashes(cpp.0.id())?;
    manifest["decisive_loaded_libraries"] = loaded_library_hashes(decisive.0.id())?;
    fs::write(
        args.out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    ensure!(
        sdk_violations.is_empty(),
        "Decisive SDK accuracy gate failed; results retained"
    );
    ensure!(
        combined_cpu_over_wall.is_none_or(|v| v <= args.threads as f64 * 1.05),
        "combined server CPU exceeds the declared thread budget; results retained"
    );
    #[cfg(target_os = "linux")]
    ensure!(
        cpu_over_request_wall.iter().all(Option::is_some),
        "Linux server CPU clock missing at completion; results retained"
    );
    ensure!(
        cpu_over_request_wall
            .iter()
            .all(|r| r.is_none_or(|v| v.is_finite() && v >= 0. && v <= args.threads as f64 * 1.05)),
        "a server exceeds the declared CPU budget for its own request intervals; results retained"
    );
    println!(
        "{}",
        json!({"status":"complete","threads":args.threads,"per_request":result["per_request"],"contract_mismatches":result["mismatches"].as_array().unwrap().len()})
    );
    Ok(())
}
