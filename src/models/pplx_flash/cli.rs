use anyhow::{Result, ensure};
use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use decisive::pplx_flash::{Model, Precision, REPOSITORY, REVISION, SHARDS};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Read},
    path::PathBuf,
};
#[derive(ClapArgs)]
#[command(about = "Perplexity Decider Flash v1.2 text inference (experimental)")]
pub(crate) struct Args {
    #[arg(long, default_value = "models/pplx-decider-flash-v1.2")]
    model: PathBuf,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = 262144)]
    max_length: usize,
    #[arg(long)]
    portable: bool,
    #[arg(
        long,
        value_enum,
        default_value = "bf16",
        help = "bf16: original weights; q8: W8A8/E8; q4: W4A8/E8; q4-e4: W4A8/E4; -mse: fitted integer Q4; q4-e4-nf4: fitted weights and NF4 embeddings; q4-e4-affine: fitted scale and offset; q4-all: W4A4/E4 (readout follows weights)"
    )]
    quantization: Quantization,
    #[arg(
        long,
        help = "Include projection versus remaining computation timing in evaluation metrics"
    )]
    profile: bool,
    #[arg(
        long,
        help = "Keep packed BF16 decoder weights resident; quantized modes are always resident"
    )]
    resident: bool,
    #[arg(long)]
    full_last_layer: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
enum Quantization {
    Bf16,
    #[cfg(feature = "int8")]
    Int8,
    #[cfg(feature = "int8")]
    Q8,
    #[cfg(feature = "int8")]
    Q4,
    #[cfg(feature = "int8")]
    #[value(name = "q4-e4")]
    Q4E4,
    #[cfg(feature = "int8")]
    #[value(name = "q4-mse")]
    Q4Mse,
    #[cfg(feature = "int8")]
    #[value(name = "q4-e4-mse")]
    Q4E4Mse,
    #[cfg(feature = "int8")]
    #[value(name = "q4-e4-nf4")]
    Q4E4Nf4,
    #[cfg(feature = "int8")]
    #[value(name = "q4-e4-affine")]
    Q4E4Affine,
    #[cfg(feature = "int8")]
    Q4All,
}
impl From<Quantization> for Precision {
    fn from(value: Quantization) -> Self {
        match value {
            Quantization::Bf16 => Self::Bf16,
            #[cfg(feature = "int8")]
            Quantization::Int8 => Self::Int8,
            #[cfg(feature = "int8")]
            Quantization::Q8 => Self::Q8,
            #[cfg(feature = "int8")]
            Quantization::Q4 => Self::Q4,
            #[cfg(feature = "int8")]
            Quantization::Q4E4 => Self::Q4E4,
            #[cfg(feature = "int8")]
            Quantization::Q4Mse => Self::Q4Mse,
            #[cfg(feature = "int8")]
            Quantization::Q4E4Mse => Self::Q4E4Mse,
            #[cfg(feature = "int8")]
            Quantization::Q4E4Nf4 => Self::Q4E4Nf4,
            #[cfg(feature = "int8")]
            Quantization::Q4E4Affine => Self::Q4E4Affine,
            #[cfg(feature = "int8")]
            Quantization::Q4All => Self::Q4All,
        }
    }
}
#[derive(Subcommand)]
enum Command {
    Fetch,
    ExportWebgpuQ4 { #[arg(long)] output: PathBuf },
    Predict {
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
        #[arg(
            long,
            help = "Read one request per line; load weights once for the whole suite"
        )]
        jsonl: bool,
        #[arg(long, default_value_t = 1)]
        rounds: usize,
    },
    Serve {
        #[arg(long, default_value = "127.0.0.1:8094")]
        listen: String,
    },
}
fn fetch(root: &std::path::Path) -> Result<()> {
    use sha2::{Digest, Sha256};
    std::fs::create_dir_all(root)?;
    let base = format!("https://huggingface.co/{REPOSITORY}/resolve/{REVISION}");
    let manifest: Value = serde_json::from_str(
        &ureq::get(format!("{base}/release-manifest.json"))
            .call()?
            .body_mut()
            .read_to_string()?,
    )?;
    let names = [
        "config.json",
        "decision_config.json",
        "joint_config.json",
        "tokenizer.json",
        "readout.safetensors",
        "LICENSE",
        "NOTICE",
        "chat_template.jinja",
        "tokenizer_config.json",
        "source/src/pplx_decider/protocol.py",
        "source/src/pplx_decider/types.py",
        SHARDS[0],
        SHARDS[1],
    ];
    for name in names {
        let expected = manifest["files"][name]["sha256"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing digest: {name}"))?;
        let path = root.join(name);
        if path.exists() {
            let mut f = std::fs::File::open(&path)?;
            let mut h = Sha256::new();
            io::copy(&mut f, &mut h)?;
            if format!("{:x}", h.finalize()) == expected {
                continue;
            }
        }
        eprintln!("fetching {name}");
        let mut response = ureq::get(format!("{base}/{name}")).call()?;
        std::fs::create_dir_all(path.parent().unwrap())?;
        let temp = path.with_extension("partial");
        let mut f = std::fs::File::create(&temp)?;
        io::copy(&mut response.body_mut().as_reader(), &mut f)?;
        f.sync_all()?;
        drop(f);
        let mut h = Sha256::new();
        io::copy(&mut std::fs::File::open(&temp)?, &mut h)?;
        ensure!(
            format!("{:x}", h.finalize()) == expected,
            "checksum mismatch: {name}"
        );
        std::fs::rename(temp, path)?;
    }
    std::fs::write(
        root.join("decisive-revision.json"),
        serde_json::to_vec_pretty(&json!({"repository":REPOSITORY,"revision":REVISION}))?,
    )?;
    Ok(())
}
pub(crate) fn run(args: Args) -> Result<()> {
    ensure!(args.threads > 0, "threads must be positive");
    if matches!(args.command, Command::Fetch) {
        return fetch(&args.model);
    }
    if let Command::ExportWebgpuQ4 { output } = &args.command {
        return decisive::pplx_flash::export_webgpu_q4(&args.model, output);
    }
    if let Command::Inspect { input } = &args.command {
        let model = Model::load(&args.model, args.portable)?;
        let mut info = model.engine.inspect(
            &serde_json::from_slice(&std::fs::read(input)?)?,
            args.max_length,
        )?;
        info["precision"] = json!(Precision::from(args.quantization).name());
        println!("{info}");
        return Ok(());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    pool.install(||{
        let loading=std::time::Instant::now();
        let mut model=Model::load_with_precision(&args.model,args.portable,args.quantization.into())?;model.engine.prune_last=!args.full_last_layer;model.profile=args.profile;
        if args.resident{model.preload_weights()?;}
        let model_load_ms=loading.elapsed().as_secs_f64()*1000.;
        match args.command{
            Command::Fetch|Command::Inspect{..}|Command::ExportWebgpuQ4{..}=>unreachable!(),
            Command::Evaluate{input,rounds,jsonl}=>{
                ensure!(rounds>0,"rounds must be positive");
                let text=std::fs::read_to_string(input)?;
                let requests:Vec<Value>=if jsonl {text.lines().filter(|s|!s.trim().is_empty()).map(serde_json::from_str).collect::<std::result::Result<_,_>>()?}else{vec![serde_json::from_str(&text)?]};
                ensure!(!requests.is_empty(),"no evaluation requests");
                for round in 0..rounds{for(case,request)in requests.iter().enumerate(){let(response,metrics)=model.evaluate(request,args.max_length)?;println!("{}",json!({"round":round,"case":case,"threads":args.threads,"model_load_ms":model_load_ms,"response":response,"metrics":metrics}));}}
            }
            Command::Predict{input}=>{
                let reader:Box<dyn BufRead>=match input{Some(path)=>Box::new(io::BufReader::new(std::fs::File::open(path)?)),None=>Box::new(io::stdin().lock())};
                for line in reader.lines(){let line=line?;if !line.trim().is_empty(){println!("{}",model.predict(&serde_json::from_str(&line)?,args.max_length)?);}}
            }
            Command::Serve{listen}=>{
                let server=tiny_http::Server::http(&listen).map_err(|e|anyhow::anyhow!("{e}"))?;eprintln!("listening on {listen}");
                for mut req in server.incoming_requests(){
                    let (status,body)=if req.method()==&tiny_http::Method::Get&&req.url()=="/health"{(200,json!({"status":"ok","model":REPOSITORY,"revision":REVISION,"text_only":true}))}
                    else if req.method()==&tiny_http::Method::Post&&req.url()=="/v1/systemone"{
                        let result=(||->Result<Value>{let mut bytes=Vec::new();req.as_reader().take(16*1024*1024+1).read_to_end(&mut bytes)?;ensure!(bytes.len()<=16*1024*1024,"request exceeds 16 MiB");model.predict(&serde_json::from_slice(&bytes)?,args.max_length)})();
                        match result{Ok(v)=>(200,v),Err(e)=>(422,json!({"error":e.to_string()}))}
                    }else{(404,json!({"error":"not found"}))};
                    let _=req.respond(tiny_http::Response::from_string(body.to_string()).with_status_code(status).with_header(tiny_http::Header::from_bytes("Content-Type","application/json").unwrap()));
                }
            }
        }Ok(())
    })
}
