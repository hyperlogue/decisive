use anyhow::{Result, ensure};
use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use decisive::laya::{Model, Scratch, input::Request, weights::Storage};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead},
    path::PathBuf,
    time::Instant,
};
#[path = "artifact.rs"]
mod artifact;
#[path = "benchmark.rs"]
mod benchmark;
#[path = "tuning.rs"]
mod tuning;

#[derive(ClapArgs)]
#[command(about = "Laya inference")]
pub(crate) struct Args {
    #[arg(long, default_value = "models/laya")]
    model: PathBuf,
    #[arg(help = "Select a checkpoint per request, with a bounded resident model cache")]
    #[arg(long)]
    auto: bool,
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    resident_models: usize,
    #[arg(long, default_value = "multilingual")]
    default_variant: String,
    #[arg(long)]
    auto_task_detection: bool,
    #[arg(help = "Worker budget; defaults to the OS estimate of available parallelism")]
    #[arg(long, default_value_t = decisive::laya::cpu::hardware::available_threads())]
    threads: usize,
    #[arg(long, default_value_t = 64)]
    attention_tile: usize,
    #[arg(help = "JSON length rules for this CPU session. Overrides length-sensitive tiles")]
    #[arg(long)]
    length_rules: Option<PathBuf>,
    #[arg(help = "Experimental reuse of RoPE tables and native attention workspaces")]
    #[arg(long)]
    reuse_attention: bool,
    #[arg(help = "Experimental tile-local packed GEGLU; portable kernels keep their fallback")]
    #[arg(long)]
    fuse_gelu: bool,
    #[arg(help = "Minimum packed output panels per Rayon split (experimental)")]
    #[arg(long, default_value_t = 1)]
    panel_group: usize,
    #[arg(help = "Experimental packed GEMM K blocking; zero keeps the full reduction kernel")]
    #[arg(long, default_value_t = 0)]
    projection_block: usize,
    #[arg(help = "Experimental AVX2 kernel for 2–5 leftover projection rows")]
    #[arg(long, conflicts_with = "projection_block")]
    tiled_projection_tails: bool,
    #[arg(
        help = "Experimental AVX2 producer layout, SIMD GEGLU, tiled attention and worker execution"
    )]
    #[arg(long, conflicts_with_all=["projection_block","tiled_projection_tails","fuse_gelu"])]
    avx2_pipeline: bool,
    #[arg(
        help = "Experimental long attention, vector exp, selective last head and wide-row blocking"
    )]
    #[arg(long, conflicts_with_all=["avx2_pipeline","projection_block","tiled_projection_tails","fuse_gelu","no_packed_attention","long_attention"])]
    avx2_long: bool,
    #[arg(
        help = "Keep the existing attention kernel while testing the AVX2 projection/worker pipeline"
    )]
    #[arg(long, requires = "avx2_pipeline")]
    no_packed_attention: bool,
    #[arg(help = "Experimental long packed attention and omission of masked softmax work")]
    #[arg(
        long,
        requires = "avx2_pipeline",
        conflicts_with = "no_packed_attention"
    )]
    long_attention: bool,
    #[arg(
        help = "Retain this many token embeddings / first-layer unrotated QKV rows (zero disables)"
    )]
    #[arg(long, default_value_t = 0)]
    token_cache: usize,
    #[arg(
        help = "Experimental first-layer unchanged-state attention cache; token bound includes final SEP"
    )]
    #[arg(long, default_value_t = 0)]
    state_cache: usize,
    #[arg(long, value_enum, default_value = "f32")]
    storage: Precision,
    #[arg(help = "Opt-in blockwise encoder quantization; decision heads remain float")]
    #[arg(long, value_enum, default_value = "none")]
    quantization: Quantization,
    #[arg(long, default_value_t = 32)]
    quant_block: usize,
    #[arg(help = "Optional path to a native CBLAS library; omitted uses the Rust fallback")]
    #[arg(long)]
    blas: Option<PathBuf>,
    #[arg(help = "Compatibility alias for --kernel packed")]
    #[arg(long)]
    packed: bool,
    #[arg(long, value_enum, default_value = "auto")]
    kernel: Kernel,
    #[arg(help = "Accumulate wall time by operation and print it to stderr after execution")]
    #[arg(long)]
    profile: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    F32,
    #[cfg(feature = "fp16")]
    F16,
}
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum Quantization {
    None,
    #[cfg(feature = "int8")]
    Int8,
    #[cfg(feature = "int8")]
    Int8Weights,
}
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum Kernel {
    Auto,
    Portable,
    Packed,
}
#[derive(Subcommand)]
enum Command {
    #[command(about = "Inspect available workers, SIMD features and reported cache geometry")]
    Hardware,
    #[command(about = "Measure worker/tile choices against official SDK reference outputs")]
    Tune(tuning::Tune),
    #[command(about = "Inspect routing without loading weights")]
    Route {
        #[arg(long)]
        input: PathBuf,
    },
    #[command(about = "Fetch a checkpoint at the revision recorded in references.json")]
    Fetch {
        #[arg(long, default_value = "english")]
        variant: String,
        #[arg(long)]
        directory: PathBuf,
    },
    #[command(about = "Enforce the final-output accuracy gate against official reference JSONL")]
    Verify {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        expected: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    #[command(about = "Serve the complete decision outputs over HTTP")]
    Serve {
        #[arg(long, default_value = "127.0.0.1:8088")]
        listen: String,
    },
    #[command(about = "Alternate identical HTTP requests between two running servers")]
    Compare {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        left: String,
        #[arg(long)]
        right: String,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        rounds: usize,
        #[arg(help = "Fail if the two servers report unequal or missing input token counts")]
        #[arg(long)]
        require_matched_tokens: bool,
    },
    #[command(about = "One request per JSON line, responses on stdout")]
    Predict {
        #[arg(long)]
        input: Option<PathBuf>,
    },
    #[command(about = "Report warm end-to-end latency for a JSONL request corpus")]
    Bench {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        #[arg(long, default_value_t = 10)]
        rounds: usize,
    },
    #[command(
        about = "Dump token ids, raw logits and optional intermediate activations for parity checks"
    )]
    Inspect {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        trace: Option<PathBuf>,
    },
}
pub(crate) fn run(args: Args) -> Result<()> {
    if matches!(&args.command, Command::Hardware) {
        println!(
            "{}",
            serde_json::to_string_pretty(&decisive::laya::cpu::hardware::detect())?
        );
        return Ok(());
    }
    let default_variant = decisive::laya::router::Variant::parse(&args.default_variant)?;
    if let Command::Route { input } = &args.command {
        for line in std::fs::read_to_string(input)?
            .lines()
            .filter(|s| !s.trim().is_empty())
        {
            let request: Value = serde_json::from_str(line)?;
            println!(
                "{}",
                serde_json::to_value(decisive::laya::router::route(
                    &request,
                    default_variant,
                    args.auto_task_detection
                )?)?
            );
        }
        return Ok(());
    }
    if let Command::Fetch { variant, directory } = &args.command {
        return artifact::fetch(variant, directory);
    }
    if let Command::Compare {
        input,
        left,
        right,
        warmup,
        rounds,
        require_matched_tokens,
    } = &args.command
    {
        let requests: Vec<Value> = std::fs::read_to_string(input)?
            .lines()
            .filter(|s| !s.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        let result = benchmark::http_compare(
            &requests,
            left,
            right,
            *warmup,
            *rounds,
            *require_matched_tokens,
        )?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    ensure!(args.threads > 0, "threads must be positive");
    ensure!(
        args.quantization == Quantization::None
            || (args.blas.is_none() && !matches!(args.command, Command::Tune(_))),
        "quantization uses Rust projection kernels; BLAS and float tuning are separate paths"
    );
    #[cfg(feature = "int8")]
    let quantization = match args.quantization {
        Quantization::None => None,
        Quantization::Int8 => Some(decisive::laya::cpu::int8::Mode::WeightsActivations),
        Quantization::Int8Weights => Some(decisive::laya::cpu::int8::Mode::Weights),
    };
    Scratch::default().set_attention_tile(args.attention_tile)?;
    Scratch::default().set_optimizations(args.reuse_attention, args.fuse_gelu, args.panel_group)?;
    Scratch::default().set_projection_block(args.projection_block)?;
    ensure!(
        !matches!(&args.command, Command::Tune(_))
            || (!args.reuse_attention
                && !args.fuse_gelu
                && args.panel_group == 1
                && args.projection_block == 0
                && !args.tiled_projection_tails
                && !args.avx2_pipeline
                && !args.avx2_long
                && args.token_cache == 0
                && args.state_cache == 0),
        "tune uses the established kernel path; use the strata_probe example for optimization experiments"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    ensure!(
        !args.packed || args.kernel != Kernel::Portable,
        "--packed conflicts with --kernel portable"
    );
    let packed = args.packed
        || match args.kernel {
            Kernel::Packed => true,
            Kernel::Portable => false,
            Kernel::Auto => args.blas.is_none() && decisive::laya::cpu::packed_supported(),
        };
    ensure!(
        !packed || args.blas.is_none(),
        "choose packed Rust or BLAS, not both"
    );
    ensure!(
        !matches!(&args.command, Command::Tune(_)) || args.blas.is_none(),
        "tune currently supports Rust kernels only; native BLAS may use a separate worker pool"
    );
    let cpu_kernel = if packed {
        decisive::laya::cpu::Kernel::X86Avx2
    } else {
        decisive::laya::cpu::Kernel::Portable
    };
    cpu_kernel.validate()?;
    let blas = args
        .blas
        .as_ref()
        .map(|path| decisive::laya::cpu::Blas::load(path, args.threads).map(std::sync::Arc::new))
        .transpose()?;
    let kernel = if packed { "packed" } else { "portable" };
    eprintln!(
        "kernel={kernel} threads={} attention_tile={}",
        args.threads, args.attention_tile
    );
    let start = Instant::now();
    let storage = match args.storage {
        Precision::F32 => Storage::F32,
        #[cfg(feature = "fp16")]
        Precision::F16 => Storage::F16,
    };
    ensure!(
        !args.auto
            || !matches!(
                &args.command,
                Command::Bench { .. } | Command::Inspect { .. } | Command::Tune(_)
            ),
        "bench, inspect and tune require a fixed model; use compare for routed HTTP measurements"
    );
    let mut router = if args.auto {
        let mut router = decisive::laya::router::Router::new_with_kernel(
            &args.models_dir,
            storage,
            args.resident_models,
            cpu_kernel,
        )?;
        router.set_blas(blas.clone())?;
        #[cfg(feature = "int8")]
        router.set_quantization(quantization.map(|mode| (mode, args.quant_block)))?;
        router.default = default_variant;
        router.auto_task_detection = args.auto_task_detection;
        Some(router)
    } else {
        None
    };
    let model = if args.auto {
        None
    } else {
        let mut model = Model::load_with_kernel(&args.model, storage, cpu_kernel)?;
        if let Some(blas) = &blas {
            model.set_blas(Some(blas.clone()))?;
        }
        #[cfg(feature = "int8")]
        if let Some(mode) = quantization {
            let bytes = model.quantize_encoder(mode, args.quant_block)?;
            eprintln!(
                "quantized encoder projections: {bytes} bytes, block {}",
                args.quant_block
            );
        }
        Some(model)
    };
    let mut predict = |request: Value, scratch: &mut Scratch| {
        if let Some(router) = &mut router {
            router.predict(request, scratch)
        } else {
            model.as_ref().unwrap().predict_json(request, scratch)
        }
    };
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("loaded in {load_ms:.2} ms");
    let mut scratch = Scratch::default();
    scratch.set_attention_tile(args.attention_tile)?;
    scratch.set_optimizations(args.reuse_attention, args.fuse_gelu, args.panel_group)?;
    scratch.set_projection_block(args.projection_block)?;
    if args.tiled_projection_tails {
        scratch.set_tiled_projection_tails(true);
    }
    if args.avx2_pipeline {
        scratch.enable_avx2_pipeline();
        scratch.set_packed_attention(!args.no_packed_attention);
        scratch.set_long_attention(args.long_attention);
    }
    if args.avx2_long {
        scratch.enable_avx2_long_pipeline();
    }
    scratch.set_token_cache_capacity(args.token_cache);
    scratch.set_state_cache_capacity(args.state_cache);
    if let Some(path) = &args.length_rules {
        scratch.set_length_rules(Some(serde_json::from_slice(&std::fs::read(path)?)?))?;
    }
    if args.profile {
        scratch.profile = Some(Default::default());
    }
    match args.command {
        Command::Hardware => unreachable!(),
        Command::Tune(tune) => {
            let result = tune.run(
                model.as_ref().unwrap(),
                args.threads,
                kernel,
                match args.storage {
                    Precision::F32 => "f32",
                    #[cfg(feature = "fp16")]
                    Precision::F16 => "f16",
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            ensure!(
                !result["recommendation"].is_null(),
                "no tuning candidate passed the accuracy gate"
            );
        }
        Command::Route { .. } => unreachable!(),
        Command::Fetch { .. } => unreachable!(),
        Command::Verify {
            input,
            expected,
            output,
        } => {
            let raw_requests: Vec<Value> = std::fs::read_to_string(&input)?
                .lines()
                .filter(|s| !s.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<Result<_, _>>()?;
            let requests = &raw_requests;
            let expected: Vec<Value> = std::fs::read_to_string(expected)?
                .lines()
                .filter(|s| !s.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<Result<_, _>>()?;
            ensure!(
                expected.len() == requests.len(),
                "reference length differs from request corpus"
            );
            let mut max_error: f64 = 0.0;
            let mut mismatches = Vec::new();
            let mut answers = Vec::new();
            for (i, (request, reference)) in requests.iter().zip(&expected).enumerate() {
                let actual = predict(request.clone(), &mut scratch)?;
                match benchmark::compare_outputs(reference, &actual, 1e-4) {
                    Ok(e) => max_error = max_error.max(e),
                    Err(e) => mismatches.push(json!({"request":i,"error":e.to_string()})),
                }
                answers.push(actual);
                eprintln!("verified {}/{}", i + 1, requests.len());
            }
            if let Some(path) = output {
                use std::io::Write;
                let mut file = std::fs::File::create(path)?;
                for answer in &answers {
                    writeln!(file, "{answer}")?;
                }
            }
            let reference_quality = benchmark::quality(&raw_requests, &expected)?;
            let actual_quality = benchmark::quality(&raw_requests, &answers)?;
            println!(
                "{}",
                json!({"requests":requests.len(),"max_output_error_on_matching_requests":max_error,"mismatches":mismatches,"reference_quality":reference_quality,"actual_quality":actual_quality})
            );
            ensure!(mismatches.is_empty(), "accuracy gate failed");
            ensure!(
                actual_quality["correct"] == reference_quality["correct"],
                "task decisions changed"
            );
        }
        Command::Compare { .. } => unreachable!(),
        Command::Serve { listen } => {
            let server =
                tiny_http::Server::http(&listen).map_err(|e| anyhow::anyhow!("listen: {e}"))?;
            eprintln!("listening on {listen}");
            for mut request in server.incoming_requests() {
                if request.url() == "/health" {
                    request.respond(tiny_http::Response::from_string("{\"status\":\"ok\"}"))?;
                    continue;
                }
                if request.url() != "/v1/systemone" || request.method() != &tiny_http::Method::Post
                {
                    request.respond(
                        tiny_http::Response::from_string("not found").with_status_code(404),
                    )?;
                    continue;
                }
                let result = (|| -> Result<Value> {
                    let parsed: Value = serde_json::from_reader(request.as_reader())?;
                    predict(parsed, &mut scratch)
                })();
                let (body, status) = match result {
                    Ok(v) => (v.to_string(), 200),
                    Err(e) => (json!({"error":e.to_string()}).to_string(), 400),
                };
                let header =
                    tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap();
                request.respond(
                    tiny_http::Response::from_string(body)
                        .with_status_code(status)
                        .with_header(header),
                )?;
            }
        }
        Command::Predict { input } => {
            let reader: Box<dyn BufRead> = if let Some(p) = input {
                Box::new(io::BufReader::new(std::fs::File::open(p)?))
            } else {
                Box::new(io::BufReader::new(io::stdin()))
            };
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let request: Value = serde_json::from_str(&line)?;
                println!("{}", predict(request, &mut scratch)?);
            }
        }
        Command::Bench {
            input,
            warmup,
            rounds,
        } => {
            let model = model.as_ref().unwrap();
            ensure!(rounds > 0, "rounds must be positive");
            let requests = read_requests(input)?;
            ensure!(!requests.is_empty(), "empty corpus");
            for _ in 0..warmup {
                for request in &requests {
                    std::hint::black_box(model.predict(request, &mut scratch)?);
                }
            }
            let mut times = Vec::new();
            let mut outputs = Vec::new();
            for round in 0..rounds {
                for request in &requests {
                    let t = Instant::now();
                    let output = model.predict(request, &mut scratch)?;
                    times.push(t.elapsed().as_secs_f64() * 1000.0);
                    if round == 0 {
                        outputs.push(output);
                    }
                }
            }
            let sum: f64 = times.iter().sum();
            let mut sorted = times.clone();
            sorted.sort_by(f64::total_cmp);
            let percentile = |p: f64| {
                sorted[((sorted.len() as f64 * p).ceil() as usize)
                    .saturating_sub(1)
                    .min(sorted.len() - 1)]
            };
            println!(
                "{}",
                json!({"load_ms":load_ms,"threads":args.threads,"kernel":kernel,"attention_tile":args.attention_tile,"requests":times.len(),"p50_ms":percentile(0.5),"p95_ms":percentile(0.95),"requests_per_second":1000.0*times.len() as f64/sum,"latencies_ms":times,"outputs":outputs})
            );
        }
        Command::Inspect { input, trace } => {
            let model = model.as_ref().unwrap();
            let requests = read_requests(input)?;
            if let Some(dir) = &trace {
                std::fs::create_dir_all(dir)?;
                scratch.trace = Some(Vec::new());
            }
            for (r, request) in requests.iter().enumerate() {
                let encoded = model.encode(request)?;
                let mut raw = Vec::new();
                for (i, q) in encoded.iter().enumerate() {
                    if let Some(t) = &mut scratch.trace {
                        t.clear();
                    }
                    let (logits, act) = model.forward(q, &mut scratch)?;
                    raw.push(json!({"encoded":q,"logits":logits,"act_probability":act}));
                    if let Some(dir) = &trace {
                        for (name, values) in scratch.trace.as_ref().unwrap() {
                            std::fs::write(
                                dir.join(format!("{r}.{i}.{name}.f32")),
                                bytemuck::cast_slice(values),
                            )?;
                        }
                    }
                }
                println!(
                    "{}",
                    json!({"raw":raw,"output":model.predict(request,&mut scratch)?})
                );
            }
        }
    }
    if let Some(timings) = &scratch.profile {
        eprintln!("{}", json!({"operation_nanoseconds":timings}));
    }
    Ok(())
}
fn read_requests(p: PathBuf) -> Result<Vec<Request>> {
    let text = std::fs::read_to_string(p)?;
    let mut result = Vec::new();
    for line in text.lines().filter(|s| !s.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)?;
        if let Some(a) = v.as_array() {
            for v in a {
                result.push(serde_json::from_value(v.clone())?);
            }
        } else {
            result.push(serde_json::from_value(v)?);
        }
    }
    Ok(result)
}
