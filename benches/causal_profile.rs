use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "kev")]
    model: String,
    #[arg(long, default_value_t = 512)]
    length: usize,
    #[arg(long, default_value = "en")]
    language: String,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long)]
    followup: bool,
    #[arg(long)]
    portable: bool,
    #[arg(long)]
    length_rules: Option<PathBuf>,
    #[arg(long)]
    control: Option<PathBuf>,
    #[arg(long, requires = "control")]
    ack: Option<PathBuf>,
}

fn check(a: &Value, b: &Value, worst: &mut f64) -> Result<()> {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                check(a.get(k).with_context(|| format!("missing {k}"))?, v, worst)?;
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            ensure!(a.len() == b.len(), "array length mismatch");
            for (a, b) in a.iter().zip(b) {
                check(a, b, worst)?;
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let error = (a.as_f64().unwrap() - b.as_f64().unwrap()).abs();
            *worst = worst.max(error);
            ensure!(error <= 1e-4 + 4. * f64::EPSILON, "output error {error}");
        }
        _ => ensure!(a == b, "output mismatch: {a} != {b}"),
    }
    Ok(())
}

fn check_output(a: &Value, b: &Value, worst: &mut f64) -> Result<()> {
    for field in ["answers", "probabilities", "usage"] {
        if !b[field].is_null() {
            check(&a[field], &b[field], worst)?;
        }
    }
    Ok(())
}

fn process_seconds() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        #[repr(C)]
        struct Time {
            seconds: std::os::raw::c_long,
            nanos: std::os::raw::c_long,
        }
        unsafe extern "C" {
            fn clock_gettime(clock: i32, time: *mut Time) -> i32;
        }
        let mut time = Time {
            seconds: 0,
            nanos: 0,
        };
        if unsafe { clock_gettime(2, &mut time) } == 0 {
            return Some(time.seconds as f64 + time.nanos as f64 * 1e-9);
        }
    }
    None
}

fn main() -> Result<()> {
    let a = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    ensure!(
        a.rounds > 0 && a.warmup > 0 && a.threads > 0,
        "positive counts required"
    );
    ensure!(
        a.model == "kev" || a.model == "jev",
        "model must be kev or jev"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads)
        .build_global()?;
    let row: Value = fs::read_to_string(format!("benches/causal-{}.jsonl", a.model))?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .find(|r| r["length"] == a.length && r["language"] == a.language)
        .context("no matching oracle case")?;
    let rules = a.length_rules.as_ref().map(fs::read).transpose()?;
    let mut run: Box<dyn FnMut(bool) -> Result<Value>> = if a.model == "kev" {
        let model =
            decisive::kev::Model::load("models/kev-0.8b", "models/qwen3.5-0.8b-base", a.portable)?;
        let mut session = decisive::kev::Session::default();
        if let Some(r) = &rules {
            session.set_length_rules(Some(serde_json::from_slice(r)?))?;
        }
        let initial = row["requests"][0]["request"].clone();
        let followup = row["requests"][1]["request"].clone();
        Box::new(move |follow| {
            if !follow {
                session.clear_cache();
            }
            model.decide_json(if follow { &followup } else { &initial }, &mut session)
        })
    } else {
        let model = decisive::jev_style::Model::load("models/jev-style-0.8b-v3", a.portable)?;
        let mut session = decisive::jev_style::Session::default();
        if let Some(r) = &rules {
            session.set_length_rules(Some(serde_json::from_slice(r)?))?;
        }
        let initial = row["requests"][0]["request"].clone();
        let followup = row["requests"][1]["request"].clone();
        Box::new(move |follow| {
            if !follow {
                session.clear_cache();
            }
            model.predict_json(if follow { &followup } else { &initial }, &mut session)
        })
    };
    let expected = &row["requests"][usize::from(a.followup)]["expected"];
    let mut error = 0.;
    if a.followup {
        run(false)?;
    }
    for _ in 0..a.warmup {
        check_output(&run(a.followup)?, expected, &mut error)?;
    }
    let mut ctl = a
        .control
        .as_ref()
        .map(|p| fs::OpenOptions::new().write(true).open(p))
        .transpose()?;
    let mut ack = a
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
                    "perf acknowledgement {line:?}"
                );
            }
        }
        Ok(())
    };
    let mut outputs = Vec::with_capacity(a.rounds);
    let mut times = Vec::with_capacity(a.rounds);
    let mut cpu_times = Vec::with_capacity(a.rounds);
    gate("enable")?;
    for _ in 0..a.rounds {
        let cpu_start = process_seconds();
        let start = Instant::now();
        outputs.push(run(a.followup)?);
        times.push(start.elapsed().as_secs_f64() * 1000.);
        cpu_times.push(
            process_seconds()
                .zip(cpu_start)
                .map(|(end, start)| (end - start) * 1000.),
        );
    }
    gate("disable")?;
    for output in &outputs {
        check_output(output, expected, &mut error)?;
    }
    println!(
        "{}",
        json!({"model":a.model,"case":row["id"],"threads":a.threads,"followup":a.followup,"portable":a.portable,"latency_ms":times,"process_cpu_ms":cpu_times,"maximum_output_error":error,"outputs":outputs})
    );
    Ok(())
}
