use anyhow::{Context, Result, ensure};
use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use decisive::decima::{Kernel, Kind, Model, Options, Precision, Question, Session};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead},
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(ClapArgs)]
#[command(about = "Decima-small inference")]
pub(crate) struct Args {
    #[arg(long, default_value = "models/decima-small")]
    model: PathBuf,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, value_enum, default_value = "f32")]
    precision: Format,
    #[arg(long)]
    portable: bool,
    #[arg(long, default_value_t = 32)]
    cache_mib: usize,
    #[arg(help = "Decima-local CPU length rules as JSON")]
    #[arg(long)]
    length_rules: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
enum Format {
    F32,
    #[cfg(feature = "int8")]
    Int8,
    #[cfg(feature = "int8")]
    Int8Weights,
    #[cfg(feature = "int8")]
    Int8Full,
    #[cfg(feature = "int8")]
    Int8Compact,
}
#[derive(Subcommand)]
enum Command {
    #[command(about = "Download the pinned upstream checkpoint (no Python dependency)")]
    Fetch,
    #[command(about = "Write a compact portable INT8 checkpoint, including the vocabulary")]
    #[cfg(feature = "int8")]
    PackInt8 {
        #[arg(long)]
        output: PathBuf,
        #[arg(help = "Smaller groups spend more scale bytes to reduce quantization error")]
        #[arg(long, default_value_t = 32)]
        group_size: usize,
        #[arg(help = "Retain projection storage as FP16; vocabulary remains INT8")]
        #[cfg_attr(feature = "fp16", arg(long))]
        #[cfg_attr(not(feature = "fp16"), arg(skip))]
        fp16_projections: bool,
        #[arg(help = "Store vocabulary scales as FP16 too; requires separate quality checks")]
        #[cfg_attr(feature = "fp16", arg(long))]
        #[cfg_attr(not(feature = "fp16"), arg(skip))]
        fp16_vocabulary_scales: bool,
    },
    #[command(about = "Run isolated processes across precision, worker and choice-cache settings")]
    Matrix {
        #[arg(long, default_value = "benches/decima-reference.jsonl")]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,6")]
        cores: Vec<usize>,
        #[arg(long, default_value_t = 5)]
        rounds: usize,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = 2)]
        repetitions: usize,
        #[arg(help = "Select supported precisions explicitly when testing a packed checkpoint")]
        #[arg(
            long,
            value_enum,
            value_delimiter = ',',
            default_value = if cfg!(feature = "int8") { "f32,int8,int8-weights" } else { "f32" }
        )]
        precisions: Vec<Format>,
    },
    #[command(about = "JSONL requests on stdin or a file; JSONL responses on stdout")]
    Predict {
        #[arg(long)]
        input: Option<PathBuf>,
    },
    #[command(about = "Serve the System One JSON boundary. Requests are serialized per process")]
    Serve {
        #[arg(long, default_value = "127.0.0.1:8090")]
        listen: String,
    },
    #[command(about = "Raw-probability parity, gold-label quality and reproducible timings")]
    Evaluate {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 1)]
        rounds: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long)]
        cold_choices: bool,
        #[arg(long)]
        max_error: Option<f64>,
        #[arg(long)]
        require_gold_parity: bool,
    },
}
fn matrix(
    args: &Args,
    input: &Path,
    output: &Path,
    cores: &[usize],
    rounds: usize,
    warmup: usize,
    repetitions: usize,
    precisions: &[Format],
) -> Result<()> {
    ensure!(
        rounds > 0
            && repetitions > 0
            && !precisions.is_empty()
            && !cores.is_empty()
            && cores.iter().all(|n| *n > 0),
        "invalid matrix dimensions"
    );
    ensure!(!output.exists(), "choose a fresh output directory");
    std::fs::create_dir_all(output)?;
    std::fs::copy(input, output.join("input.jsonl"))?;
    let mut labeled_cases = 0;
    for line in std::fs::read_to_string(input)?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let row: Value = serde_json::from_str(line)?;
        labeled_cases += usize::from(row["gold"].as_u64().is_some());
    }
    let executable = std::env::current_exe()?;
    let mut runs = vec![];
    let formats: Vec<String> = precisions
        .iter()
        .map(|p| p.to_possible_value().unwrap().get_name().to_owned())
        .collect();
    for repeat in 0..repetitions {
        for &threads in cores {
            for cold in [true, false] {
                for offset in 0..formats.len() {
                    let format = formats[(offset + repeat) % formats.len()].as_str();
                    let name = format!(
                        "r{repeat}-t{threads}-{format}-{}.json",
                        if cold { "cold" } else { "cached" }
                    );
                    eprintln!("{name}");
                    let mut command = std::process::Command::new(&executable);
                    command.arg("decima").arg("--model").arg(&args.model).args([
                        "--threads",
                        &threads.to_string(),
                        "--precision",
                        format,
                        "--cache-mib",
                        &args.cache_mib.to_string(),
                    ]);
                    if let Some(path) = &args.length_rules {
                        command.arg("--length-rules").arg(path);
                    }
                    if args.portable {
                        command.arg("--portable");
                    }
                    command.arg("evaluate").arg("--input").arg(input).args([
                        "--rounds",
                        &rounds.to_string(),
                        "--warmup",
                        &warmup.to_string(),
                    ]);
                    if labeled_cases > 0 {
                        command.arg("--require-gold-parity");
                    }
                    if format == "f32" {
                        command.args(["--max-error", "0.0001"]);
                    } else {
                        command.args(["--max-error", "0.02"]);
                    }
                    if cold {
                        command.arg("--cold-choices");
                    }
                    let start = Instant::now();
                    let result = command.output()?;
                    std::fs::write(output.join(&name), &result.stdout)?;
                    std::fs::write(output.join(name.replace(".json", ".log")), &result.stderr)?;
                    runs.push(json!({"file":name,"exit_code":result.status.code(),"elapsed_s":start.elapsed().as_secs_f64()}));
                    let cpu = std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
                        s.lines()
                            .find(|l| l.starts_with("model name"))
                            .and_then(|l| l.split_once(':'))
                            .map(|(_, v)| v.trim().to_owned())
                    });
                    std::fs::write(
                        output.join("manifest.json"),
                        serde_json::to_vec_pretty(
                            &json!({"model_revision":decisive::decima::REVISION,"sdk_revision":decisive::decima::SDK_REVISION,"cpu":cpu,"architecture":std::env::consts::ARCH,"cores":cores,"precisions":formats,"labeled_cases":labeled_cases,"rounds":rounds,"warmup":warmup,"repetitions":repetitions,"cache_mib":args.cache_mib,"runs":runs,"notes":"Warm resident model. Cold means cleared choice embeddings before each timed request, not cold weights. Processes run sequentially; no throughput/concurrency claim. Use an otherwise idle VM and OS CPU affinity for release measurements."}),
                        )?,
                    )?;
                }
            }
        }
    }
    ensure!(
        runs.iter().all(|r| r["exit_code"] == 0),
        "one or more matrix runs failed a quality gate; see individual logs"
    );
    Ok(())
}
fn fetch(root: &Path) -> Result<()> {
    let stamp = root.join("beam-revision.json");
    let revision =
        json!({"repository":decisive::decima::REPOSITORY,"revision":decisive::decima::REVISION});
    if stamp.exists() {
        ensure!(
            serde_json::from_slice::<Value>(&std::fs::read(&stamp)?)? == revision,
            "destination contains a different revision"
        );
    }
    for name in [
        "pytorch/decima.json",
        "pytorch/encoder/config.json",
        "pytorch/encoder/tokenizer.json",
        "pytorch/encoder/tokenizer_config.json",
        "pytorch/encoder/model.safetensors",
        "pytorch/head.safetensors",
        "LICENSE",
    ] {
        let path = root.join(name);
        if path.exists() && stamp.exists() {
            continue;
        }
        std::fs::create_dir_all(path.parent().unwrap())?;
        let tmp = path.with_extension("partial");
        eprintln!("fetching {name}");
        let mut response = ureq::get(format!(
            "https://huggingface.co/{}/resolve/{}/{name}",
            decisive::decima::REPOSITORY,
            decisive::decima::REVISION
        ))
        .call()?;
        let expected = response
            .headers()
            .get("Content-Length")
            .and_then(|s| s.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        let mut file = std::fs::File::create(&tmp)?;
        let bytes = io::copy(&mut response.body_mut().as_reader(), &mut file)?;
        ensure!(
            expected.is_none_or(|n| n == bytes),
            "incomplete download {name}"
        );
        file.sync_all()?;
        std::fs::rename(tmp, path)?;
    }
    std::fs::write(stamp, serde_json::to_vec_pretty(&revision)?)?;
    Ok(())
}
fn question(v: &Value) -> Result<Question> {
    Ok(Question {
        text: v["text"].as_str().context("question.text")?.into(),
        choices: serde_json::from_value(v["choices"].clone())?,
        lang: v["lang"].as_str().unwrap_or("en").into(),
        kind: match v["kind"].as_str().unwrap_or("choice") {
            "choice" | "choose" => Kind::Choice,
            "noul" | "verify" => Kind::Noul,
            "score" => Kind::Score,
            "rank" => Kind::Rank,
            _ => anyhow::bail!("unknown question kind"),
        },
    })
}
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .unwrap()
        .0
}
pub(crate) fn run(args: Args) -> Result<()> {
    #[cfg(feature = "int8")]
    if let Command::PackInt8 {
        output,
        group_size,
        fp16_projections,
        fp16_vocabulary_scales,
    } = &args.command
    {
        if *fp16_vocabulary_scales {
            return decisive::decima::pack::int8_checkpoint_half_scales(
                &args.model,
                output,
                *group_size,
                *fp16_projections,
            );
        }
        return decisive::decima::pack::int8_checkpoint(
            &args.model,
            output,
            *group_size,
            *fp16_projections,
        );
    }
    if matches!(args.command, Command::Fetch) {
        return fetch(&args.model);
    }
    if let Command::Matrix {
        input,
        output,
        cores,
        rounds,
        warmup,
        repetitions,
        precisions,
    } = &args.command
    {
        return matrix(
            &args,
            input,
            output,
            cores,
            *rounds,
            *warmup,
            *repetitions,
            precisions,
        );
    }
    ensure!(args.threads > 0, "threads must be positive");
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let options = Options {
        precision: match args.precision {
            Format::F32 => Precision::F32,
            #[cfg(feature = "int8")]
            Format::Int8 => Precision::Int8,
            #[cfg(feature = "int8")]
            Format::Int8Weights => Precision::Int8Weights,
            #[cfg(feature = "int8")]
            Format::Int8Full => Precision::Int8Full,
            #[cfg(feature = "int8")]
            Format::Int8Compact => Precision::Int8Compact,
        },
        kernel: if args.portable {
            Kernel::Portable
        } else {
            Kernel::Auto
        },
    };
    let start = Instant::now();
    let model = Model::load(&args.model, options)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    eprintln!(
        "loaded Decima-small in {load_ms:.1} ms, {:?}, {} workers",
        options.precision, args.threads
    );
    let mut session = Session::default();
    if let Some(path) = &args.length_rules {
        session.set_length_rules(Some(serde_json::from_slice(&std::fs::read(path)?)?))?;
    }
    session.set_cache_bytes(
        args.cache_mib
            .checked_mul(1024 * 1024)
            .context("cache size overflow")?,
    );
    match args.command {
        Command::Fetch | Command::Matrix { .. } => unreachable!(),
        #[cfg(feature = "int8")]
        Command::PackInt8 { .. } => unreachable!(),
        Command::Predict { input } => {
            let reader: Box<dyn BufRead> = match input {
                Some(p) => Box::new(io::BufReader::new(std::fs::File::open(p)?)),
                None => Box::new(io::stdin().lock()),
            };
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let r: Value = serde_json::from_str(&line)?;
                println!("{}", model.predict_json(&r, &mut session)?);
            }
        }
        Command::Serve { listen } => {
            let server = tiny_http::Server::http(&listen).map_err(|e| anyhow::anyhow!("{e}"))?;
            eprintln!("listening on {listen}");
            for mut request in server.incoming_requests() {
                let response =
                    if request.url() == "/health" && request.method() == &tiny_http::Method::Get {
                        Ok(json!({"model":"decima-small","precision":options.precision}))
                    } else if request.url() == "/v1/systemone"
                        && request.method() == &tiny_http::Method::Post
                    {
                        (|| -> Result<Value> {
                            use std::io::Read;
                            let mut bytes = vec![];
                            request
                                .as_reader()
                                .take(4 * 1024 * 1024 + 1)
                                .read_to_end(&mut bytes)?;
                            ensure!(bytes.len() <= 4 * 1024 * 1024, "request exceeds 4 MiB");
                            model.predict_json(&serde_json::from_slice(&bytes)?, &mut session)
                        })()
                    } else {
                        let _ = request.respond(
                            tiny_http::Response::from_string("not found").with_status_code(404),
                        );
                        continue;
                    };
                let (status, body) = match response {
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
        Command::Evaluate {
            input,
            rounds,
            warmup,
            cold_choices,
            max_error,
            require_gold_parity,
        } => {
            ensure!(rounds > 0, "rounds must be positive");
            let mut cases = vec![];
            let mut worst = 0_f64;
            let mut correct = 0;
            let mut reference_correct = 0;
            let mut labeled = 0;
            let mut changed = 0;
            let mut brier = [0_f64; 2];
            let mut nll = [0_f64; 2];
            for line in std::fs::read_to_string(input)?
                .lines()
                .filter(|l| !l.trim().is_empty())
            {
                let row: Value = serde_json::from_str(line)?;
                let q = question(&row["question"])?;
                let state = row["state"].as_str().context("state")?;
                let expected: Vec<f32> = serde_json::from_value(row["expected"].clone())?;
                ensure!(
                    expected.len() == q.choices.len() && expected.iter().all(|v| v.is_finite()),
                    "invalid expected probabilities"
                );
                for _ in 0..warmup {
                    model.decide(state, &q, &mut session)?;
                }
                let mut samples = vec![];
                let mut p = vec![];
                for _ in 0..rounds {
                    if cold_choices {
                        session.clear_cache();
                    }
                    let start = Instant::now();
                    p = model.decide(state, &q, &mut session)?;
                    samples.push(start.elapsed().as_secs_f64() * 1000.);
                    ensure!(
                        p.len() == expected.len() && p.iter().all(|v| v.is_finite()),
                        "non-finite or malformed output in evaluation case {}",
                        row["id"]
                    );
                }
                let error = p
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (*a as f64 - *b as f64).abs())
                    .fold(0., f64::max);
                worst = worst.max(error);
                let did_change = argmax(&p) != argmax(&expected);
                changed += usize::from(did_change);
                if let Some(gold) = row["gold"].as_u64() {
                    ensure!((gold as usize) < p.len(), "gold index outside choices");
                    labeled += 1;
                    correct += usize::from(argmax(&p) == gold as usize);
                    reference_correct += usize::from(argmax(&expected) == gold as usize);
                    for (i, probabilities) in [&p, &expected].into_iter().enumerate() {
                        brier[i] += probabilities
                            .iter()
                            .enumerate()
                            .map(|(k, p)| {
                                (*p as f64 - if k == gold as usize { 1. } else { 0. }).powi(2)
                            })
                            .sum::<f64>();
                        nll[i] -= (probabilities[gold as usize] as f64).max(1e-12).ln();
                    }
                }
                cases.push(json!({"id":row["id"],"language":q.lang,"pair_id":row["pair_id"],"probabilities":p,"expected":expected,"max_error":error,"changed_decision":did_change,"gold":row["gold"],"latency_ms":samples}));
                if cases.len() % 100 == 0 {
                    eprintln!(
                        "evaluated {} cases; max probability error {:.6}",
                        cases.len(),
                        worst
                    );
                }
            }
            ensure!(!cases.is_empty(), "empty evaluation");
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"repository":decisive::decima::REPOSITORY,"revision":decisive::decima::REVISION,"precision":options.precision,"threads":args.threads,"portable":args.portable,"cold_choices":cold_choices,"rounds":rounds,"warmup":warmup,"load_ms":load_ms,"max_probability_error":worst,"changed_decisions":changed,"gold":{"labeled":labeled,"correct":correct,"reference_correct":reference_correct,"brier":brier[0]/labeled.max(1) as f64,"reference_brier":brier[1]/labeled.max(1) as f64,"nll":nll[0]/labeled.max(1) as f64,"reference_nll":nll[1]/labeled.max(1) as f64},"stats":session.stats,"cache_bytes":session.cache_bytes(),"cases":cases})
                )?
            );
            ensure!(
                max_error.is_none_or(|bound| worst <= bound),
                "probability error {worst} exceeds bound"
            );
            ensure!(
                !require_gold_parity || (labeled > 0 && correct >= reference_correct),
                "task quality regressed (or no gold labels supplied)"
            );
        }
    }
    Ok(())
}
