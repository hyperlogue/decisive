use anyhow::Result;
use clap::{Args as ClapArgs, ValueEnum};
use serde_json::{Value, json};
use std::io::IsTerminal;

#[derive(ClapArgs)]
pub(crate) struct Args {
    #[arg(long, help = "Print a machine-readable capability manifest")]
    json: bool,
    #[arg(long, value_enum, default_value = "auto")]
    color: Color,
}
#[derive(Clone, Copy, ValueEnum)]
enum Color {
    Auto,
    Always,
    Never,
}

fn manifest() -> Value {
    let mut models = Vec::new();
    for (id, enabled) in [
        ("pplx-flash", cfg!(feature = "pplx-flash")),
        ("laya", cfg!(feature = "laya")),
        ("decima", cfg!(feature = "decima")),
        ("jev-style", cfg!(feature = "jev-style")),
        ("kev", cfg!(feature = "kev")),
    ] {
        if !enabled {
            continue;
        }
        let mut formats = if id == "pplx-flash" {
            vec!["bf16-storage-f32-compute"]
        } else {
            vec!["f32"]
        };
        if id == "laya" && cfg!(feature = "fp16") {
            formats.push("f16-storage");
        }
        if id == "pplx-flash" && cfg!(feature = "int8") {
            formats.extend([
                "int8",
                "q8",
                "q4",
                "q4-e4",
                "q4-mse",
                "q4-e4-mse",
                "q4-e4-nf4",
                "q4-e4-affine",
                "q4-all",
            ]);
        }
        if matches!(id, "laya" | "decima") && cfg!(feature = "int8") {
            formats.extend(["int8", "int8-weights"]);
            if id == "decima" {
                formats.extend(["int8-full", "int8-compact"]);
            }
        }
        let mut backends = vec![json!({
            "id":"cpu", "precisions":formats, "interfaces":["cli","http","rust"],
            "implementation":"portable Rust with runtime SIMD dispatch"
        })];
        if matches!(id, "laya" | "decima") && cfg!(feature = "gpu-native") {
            backends.push(json!({
                "id":"webgpu", "precisions":["f32"], "interfaces":["rust"],
                "implementation":"wgpu Vulkan; adapter availability is not probed"
            }));
        }
        models.push(json!({"id":id,"backends":backends}));
    }
    let features: Vec<_> = [
        ("cpu", cfg!(feature = "cpu")),
        ("wasm", cfg!(feature = "wasm")),
        ("webgpu", cfg!(feature = "webgpu")),
        ("gpu-native", cfg!(feature = "gpu-native")),
        ("fp16", cfg!(feature = "fp16")),
        ("int8", cfg!(feature = "int8")),
        ("bench", cfg!(feature = "bench")),
        ("browser-bench", cfg!(feature = "browser-bench")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect();
    json!({
        "schema_version":1, "name":"decisive", "version":env!("CARGO_PKG_VERSION"),
        "target":{"arch":std::env::consts::ARCH,"os":std::env::consts::OS},
        "features":features, "models":models,
        "http":{"available":!models.is_empty(),"endpoint":"/v1/systemone","method":"POST","backend":"cpu","concurrency":"serial per process","command":"decisive <model> serve"},
        "notes":[
            "Capabilities describe this compiled native binary, not installed checkpoints or model quality.",
            "Floating-point accumulation is retained with quantized weights. PPLX q8/q4/q4-all also quantize embeddings and readout; legacy int8 retains floating-point heads.",
            "Decima mixed FP16/INT8 checkpoints require both fp16 and int8 features.",
            "Browser Wasm/WebGPU packages are separate target builds; enabling wasm does not make the native CLI a Wasm runner.",
            "WebGPU inference is not exposed by the CLI or HTTP server."
        ]
    })
}

pub(crate) fn run(args: Args) -> Result<()> {
    let report = manifest();
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let color = match args.color {
        Color::Always => true,
        Color::Never => false,
        Color::Auto => std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
    };
    let paint = |code, text: &str| {
        if color {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    };
    println!(
        "{}  {} · {} / {}\n",
        paint("1;36", "Decisive"),
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!(
        "{}",
        paint(
            "1",
            &format!(
                "{:<12} {:<10} {:<56} {}",
                "MODEL", "BACKEND", "PRECISION / QUANTIZATION", "ACCESS"
            )
        )
    );
    println!("{}", paint("2", &"─".repeat(101)));
    for model in report["models"].as_array().unwrap() {
        for backend in model["backends"].as_array().unwrap() {
            let strings = |key: &str| {
                backend[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            println!(
                "{} {:<10} {:<56} {}",
                paint("32", &format!("{:<12}", model["id"].as_str().unwrap())),
                backend["id"].as_str().unwrap(),
                strings("precisions"),
                strings("interfaces")
            );
        }
    }
    if report["models"].as_array().unwrap().is_empty() {
        println!("No model features enabled.");
    }
    if report["http"]["available"] == true {
        println!(
            "\n{} POST /v1/systemone · CPU · serial requests",
            paint("1;36", "HTTP server")
        );
        println!("  decisive <model> serve --listen 127.0.0.1:8090\n");
    }
    for note in report["notes"].as_array().unwrap() {
        println!("{}", paint("2", note.as_str().unwrap()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn manifest_does_not_claim_uncompiled_quantization_or_http_gpu() {
        let data = manifest();
        for model in data["models"].as_array().unwrap() {
            for backend in model["backends"].as_array().unwrap() {
                let formats = backend["precisions"].as_array().unwrap();
                assert!(
                    !formats
                        .iter()
                        .any(|v| v.as_str().unwrap().starts_with("int8"))
                        || cfg!(feature = "int8")
                );
                assert!(!formats.contains(&json!("f16-storage")) || cfg!(feature = "fp16"));
                if model["id"] == "pplx-flash" {
                    for mode in [
                        "q8",
                        "q4",
                        "q4-e4",
                        "q4-mse",
                        "q4-e4-mse",
                        "q4-e4-nf4",
                        "q4-e4-affine",
                        "q4-all",
                    ] {
                        assert_eq!(formats.contains(&json!(mode)), cfg!(feature = "int8"));
                    }
                }
                if backend["id"] == "webgpu" {
                    assert_eq!(backend["interfaces"], json!(["rust"]));
                }
            }
        }
    }
}
