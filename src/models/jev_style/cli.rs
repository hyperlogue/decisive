use anyhow::{Context, Result, ensure};
use clap::{Args as ClapArgs, Subcommand};
use decisive::jev_style::{Model, Session};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead},
    path::PathBuf,
    time::Instant,
};
#[derive(ClapArgs)]
#[command(about = "Jev-Style 0.8B Decision v3 inference")]
pub(crate) struct Args {
    #[arg(long, default_value = "models/jev-style-0.8b-v3")]
    model: PathBuf,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long)]
    portable: bool,
    #[arg(long, default_value_t = 128)]
    cache_mib: usize,
    #[arg(long)]
    length_rules: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Fetch,
    Predict {
        #[arg(long)]
        input: Option<PathBuf>,
    },
    #[command(
        about = "Canonical Jev-Style questions: {state, question:{t,ins,crit}, category?, temperature?}"
    )]
    Decide {
        #[arg(long)]
        input: Option<PathBuf>,
    },
    Inspect {
        #[arg(long)]
        input: PathBuf,
    },
    Evaluate {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 1)]
        rounds: usize,
        #[arg(long, default_value_t = 0)]
        warmup: usize,
        #[arg(long, default_value_t = 1e-4)]
        max_error: f64,
    },
    Serve {
        #[arg(long, default_value = "127.0.0.1:8092")]
        listen: String,
    },
}
fn fetch(root: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(root)?;
    let stamp = root.join("beam-revision.json");
    let revision = json!({"repository":decisive::jev_style::REPOSITORY,"revision":decisive::jev_style::REVISION});
    if stamp.exists() {
        ensure!(
            serde_json::from_slice::<Value>(&std::fs::read(&stamp)?)? == revision,
            "checkpoint revision mismatch"
        );
    }
    for name in [
        "config.json",
        "tokenizer.json",
        "readout_config.json",
        "model.safetensors",
        "LICENSE",
        "NOTICE",
    ] {
        let dest = root.join(name);
        if dest.exists() && stamp.exists() {
            continue;
        }
        eprintln!("fetching {name}");
        let mut response = ureq::get(format!(
            "https://huggingface.co/{}/resolve/{}/{name}",
            decisive::jev_style::REPOSITORY,
            decisive::jev_style::REVISION
        ))
        .call()?;
        let expected = response
            .headers()
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let temp = dest.with_extension("partial");
        let mut file = std::fs::File::create(&temp)?;
        let count = io::copy(&mut response.body_mut().as_reader(), &mut file)?;
        ensure!(
            expected.is_none_or(|n| n == count),
            "incomplete download {name}"
        );
        file.sync_all()?;
        std::fs::rename(temp, dest)?;
    }
    std::fs::write(stamp, serde_json::to_vec_pretty(&revision)?)?;
    Ok(())
}
fn rows(input: Option<&PathBuf>) -> Result<Vec<Value>> {
    let reader: Box<dyn BufRead> = match input {
        Some(p) => Box::new(io::BufReader::new(std::fs::File::open(p)?)),
        None => Box::new(io::BufReader::new(io::stdin())),
    };
    reader
        .lines()
        .filter_map(|l| match l {
            Ok(l) if l.trim().is_empty() => None,
            other => Some(other),
        })
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}
fn decide(model: &Model, row: &Value, session: &mut Session) -> Result<Value> {
    model.decide(
        row.get("state").context("missing state")?,
        row.get("question").context("missing question")?,
        row.get("category").and_then(Value::as_str),
        row.get("temperature").and_then(Value::as_f64),
        session,
    )
}
pub(crate) fn run(args: Args) -> Result<()> {
    if matches!(args.command, Command::Fetch) {
        return fetch(&args.model);
    }
    ensure!(args.threads > 0, "threads must be positive");
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let start = Instant::now();
    let model = Model::load(&args.model, args.portable)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    eprintln!("loaded in {load_ms:.2} ms");
    let mut session = Session::default();
    session.set_cache_bytes(
        args.cache_mib
            .checked_mul(1024 * 1024)
            .context("cache size overflow")?,
    );
    if let Some(p) = args.length_rules {
        session.set_length_rules(Some(serde_json::from_slice(&std::fs::read(p)?)?))?;
    }
    match args.command {
        Command::Fetch => unreachable!(),
        Command::Predict { input } => {
            for row in rows(input.as_ref())? {
                println!("{}", model.predict_json(&row, &mut session)?);
            }
        }
        Command::Decide { input } => {
            for row in rows(input.as_ref())? {
                println!("{}", decide(&model, &row, &mut session)?);
            }
        }
        Command::Inspect { input } => {
            for row in rows(Some(&input))? {
                println!("{}", model.encode(&row["state"], &row["question"])?);
            }
        }
        Command::Evaluate {
            input,
            rounds,
            warmup,
            max_error,
        } => {
            ensure!(
                rounds > 0 && max_error.is_finite() && max_error >= 0.,
                "invalid evaluation counts/bound"
            );
            let mut cases = Vec::new();
            let mut worst = 0f64;
            let mut changed = 0;
            for row in rows(Some(&input))? {
                for _ in 0..warmup {
                    decide(&model, &row, &mut session)?;
                }
                let mut samples = Vec::new();
                let mut output = Value::Null;
                let mut error = 0f64;
                for _ in 0..rounds {
                    let begin = Instant::now();
                    output = decide(&model, &row, &mut session)?;
                    samples.push(begin.elapsed().as_secs_f64() * 1000.);
                    let expected = &row["expected"];
                    let probs = expected["probabilities"]
                        .as_object()
                        .context("expected.probabilities")?;
                    ensure!(
                        probs.len() == output["probabilities"].as_object().unwrap().len(),
                        "probability count mismatch"
                    );

                    for (name, p) in probs {
                        error = error.max(
                            (p.as_f64().context("expected probability")?
                                - output["probabilities"][name]
                                    .as_f64()
                                    .context("missing probability")?)
                            .abs(),
                        );
                    }
                    worst = worst.max(error);
                    changed += usize::from(output["answer"] != expected["answer"]);
                }
                cases.push(json!({"id":row["id"],"output":output,"max_probability_error":error,"latency_ms":samples}));
            }
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"repository":decisive::jev_style::REPOSITORY,"revision":decisive::jev_style::REVISION,"threads":args.threads,"portable":args.portable,"load_ms":load_ms,"max_probability_error":worst,"changed_decisions":changed,"cache":session.stats,"cached_bytes":session.cached_bytes(),"cases":cases})
                )?
            );
            ensure!(
                worst <= max_error && changed == 0,
                "output parity failed: error={worst}, changed={changed}"
            );
        }
        Command::Serve { listen } => {
            let server = tiny_http::Server::http(&listen).map_err(|e| anyhow::anyhow!("{e}"))?;
            eprintln!("listening on {listen}");
            for mut request in server.incoming_requests() {
                let result = if request.url() == "/health" {
                    Ok(json!({"model":"jev-style-0.8b-v3"}))
                } else if request.url() == "/v1/systemone"
                    && request.method() == &tiny_http::Method::Post
                {
                    (|| -> Result<Value> {
                        use std::io::Read;
                        let mut bytes = Vec::new();
                        request
                            .as_reader()
                            .take(4 * 1024 * 1024 + 1)
                            .read_to_end(&mut bytes)?;
                        ensure!(bytes.len() <= 4 * 1024 * 1024, "request exceeds 4 MiB");
                        model.predict_json(&serde_json::from_slice(&bytes)?, &mut session)
                    })()
                } else {
                    request.respond(
                        tiny_http::Response::from_string("not found").with_status_code(404),
                    )?;
                    continue;
                };
                let (status, body) = match result {
                    Ok(v) => (200, v),
                    Err(e) => (422, json!({"error":e.to_string()})),
                };
                request.respond(
                    tiny_http::Response::from_string(body.to_string())
                        .with_status_code(status)
                        .with_header(
                            tiny_http::Header::from_bytes("Content-Type", "application/json")
                                .unwrap(),
                        ),
                )?;
            }
        }
    }
    Ok(())
}
