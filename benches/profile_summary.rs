use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf, process::Command};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    run: PathBuf,
    #[arg(long)]
    folded: PathBuf,
    #[arg(long)]
    stack_quality: PathBuf,
    #[arg(long)]
    binary: PathBuf,
    #[arg(long)]
    arithmetic: PathBuf,
    #[arg(
        help = "Record the actual profiling build flags; complete stacks alone do not prove them"
    )]
    #[arg(long, default_value = "unspecified")]
    rustflags: String,
    #[arg(help = "Frequency used by perf; leave absent if the capture configuration is unknown")]
    #[arg(long)]
    sample_hz: Option<u32>,
    #[arg(help = "Lost-event count verified from perf metadata; absence is not reported as zero")]
    #[arg(long)]
    lost_samples: Option<u64>,
    #[arg(help = "Optional historical register-only control; never an achievable model ceiling")]
    #[arg(long)]
    register_gflops: Option<f64>,
}

fn group(stack: &str) -> &'static str {
    for frame in stack.rsplit(';') {
        if frame.contains("x86_avx2::inline_panel")
            || frame.contains("x86_avx2::cache_tiles::block")
            || frame.contains("x86_avx2::kernel")
        {
            return "projection kernels";
        }
        if frame.contains("packed_attention::product") {
            return "attention products";
        }
        if frame.contains("x86_avx2::softmax::")
            || frame.contains("x86_avx2::packed_softmax::")
            || frame.contains("packed_attention::normalize")
        {
            return "softmax and exponentials";
        }
        if frame.contains("x86_avx2::activation::") {
            return "activation";
        }
        if frame.contains("packed_attention::pack_") {
            return "attention packing";
        }
        if frame.contains("x86_avx2::input::") {
            return "projection input preparation";
        }
    }
    "other"
}

fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    let read = |path: &PathBuf| -> Result<Value> { Ok(serde_json::from_slice(&fs::read(path)?)?) };
    let run = read(&args.run)?;
    let quality = read(&args.stack_quality)?;
    ensure!(
        run["threads"] == 1 && run["long_pipeline"] == true,
        "requires a one-worker long-pipeline profile"
    );
    let expected_layers: Vec<_> = (0..28)
        .map(|i| {
            if i % 3 == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    ensure!(
        run["model_shape"]
            == json!({"hidden_size":1024,"intermediate_size":2624,"num_hidden_layers":28,"num_attention_heads":16,"local_attention":128,"global_attn_every_n_layers":3,"layer_types":expected_layers,"head_layers":2}),
        "arithmetic requires the recorded English checkpoint geometry and attention schedule"
    );
    ensure!(
        quality["unavailable_caller_stacks"] == 0,
        "complete caller chains required for ancestral attribution"
    );
    let wall = run["wall_seconds"]
        .as_f64()
        .filter(|x| *x > 0.)
        .context("missing wall interval")?;
    let workers: Vec<_> = run["scheduler"]
        .as_array()
        .context("missing scheduler data")?
        .iter()
        .filter(|r| r["role"] == "worker")
        .collect();
    ensure!(
        workers.len() == 1,
        "requires exactly one worker scheduler record"
    );
    let scheduled = workers[0]["scheduled_ns"]
        .as_f64()
        .context("missing scheduled time")?;
    let runqueue = workers[0]["runqueue_wait_ns"]
        .as_f64()
        .context("missing runqueue time")?;
    let mut counts = BTreeMap::<&str, u64>::new();
    let mut roots = BTreeMap::<String, u64>::new();
    let mut depths = BTreeMap::<usize, u64>::new();
    let mut entry_samples = 0_u64;
    let mut unresolved_samples = 0_u64;
    let mut samples = 0_u64;
    for line in fs::read_to_string(&args.folded)?.lines() {
        let (stack, count) = line.rsplit_once(' ').context("invalid folded row")?;
        let count = count.parse::<u64>()?;
        let frames: Vec<_> = stack.split(';').collect();
        let root = frames.first().context("empty stack")?;
        *roots.entry(root.to_string()).or_default() += count;
        *depths.entry(frames.len()).or_default() += count;
        if ["_start", "__GI___clone3", "__clone3", "clone3", "__clone"].contains(root) {
            entry_samples += count;
        }
        if frames
            .iter()
            .any(|s| s.contains("[unknown]") || s.contains("[caller stack unavailable]"))
        {
            unresolved_samples += count;
        }
        samples += count;
        *counts.entry(group(stack)).or_default() += count;
    }
    ensure!(
        samples > 0 && quality["samples"] == samples,
        "sample count mismatch"
    );
    let groups: BTreeMap<_, _> = counts
        .iter()
        .map(|(&k, &v)| (k, v as f64 * 100. / samples as f64))
        .collect();
    let operations: BTreeMap<String, f64> = run["operation_nanoseconds_three_requests"]
        .as_object()
        .context("run with --operations")?
        .iter()
        .map(|(k, v)| {
            Ok((
                k.clone(),
                v.as_f64().context("invalid operator time")? / 3e6,
            ))
        })
        .collect::<Result<_>>()?;
    let layers = run["layer_operation_nanoseconds_three_requests"]
        .as_object()
        .context("run with --layer-operations")?;
    let layer_ms = |key: &str| -> Result<f64> {
        Ok(layers
            .get(key)
            .and_then(Value::as_f64)
            .with_context(|| format!("missing {key}"))?
            / 3e6)
    };
    let operator_total: f64 = operations.values().sum();
    let first_attention = layer_ms("encoder.0.attention")?;
    let first_qkv = layer_ms("encoder.0.qkv")?;
    let embedding = layer_ms("embedding")?;
    let token_count = run["output"]["usage"]["input_tokens"]
        .as_u64()
        .context("missing token count")?;
    ensure!(
        run["output"]["answers"]
            .as_object()
            .is_some_and(|x| x.len() == 1),
        "arithmetic expects one question"
    );
    ensure!(
        run["output"]["answers"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap()["probabilities"]
            .as_object()
            .is_some_and(|x| x.len() == 3),
        "arithmetic expects three option markers plus CLS"
    );
    let arithmetic = read(&args.arithmetic)?;
    let shape = arithmetic["shapes"]
        .as_array()
        .context("missing arithmetic shapes")?
        .iter()
        .find(|x| x["tokens"] == token_count)
        .context("arithmetic shape absent")?;
    let gflops = shape
        .get("decisive_total_flops")
        .or_else(|| shape.get("disco_total_flops"))
        .or_else(|| shape.get("beam_total_flops"))
        .context("missing arithmetic count")?
        .as_f64()
        .context("missing arithmetic count")?
        / 1e9;
    if let Some(rate) = args.register_gflops {
        ensure!(rate.is_finite() && rate > 0., "invalid register throughput");
    }
    let hash = Command::new("sha256sum").arg(&args.binary).output()?;
    ensure!(hash.status.success(), "sha256sum failed");
    let hash = String::from_utf8(hash.stdout)?
        .split_whitespace()
        .next()
        .context("empty binary hash")?
        .to_owned();
    ensure!(
        hash.len() == 64 && hash.bytes().all(|c| c.is_ascii_hexdigit()),
        "invalid binary hash"
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "scope":"One-worker software CPU-clock profile; operator wall timings come from a separate three-request pass. Profiling build differs from normal release. Repeat consistency is not an independent SDK quality gate. No hardware PMU interpretation.",
            "binary_sha256":hash,"rustflags":args.rustflags,"samples":samples,
            "sample_hz":args.sample_hz,"lost_samples":args.lost_samples,
            "caller_chain_audit":{"scope":"Additional Linux frame-pointer audit of sanitized stacks. A nonempty stack alone does not prove complete unwinding. Recorded roots, depth distribution and unresolved symbols remain visible.","root_symbols":roots,"depth_samples":depths,"samples_reaching_process_or_thread_entry":entry_samples,"unresolved_symbol_samples":unresolved_samples},
            "attribution":"Each complete sample belongs to the nearest recognized kernel family while walking leaf-to-caller. Shared libm helpers inherit their actual caller family; broad Rayon ancestors are not counted as waiting. Unclassified samples remain other.",
            "groups_samples":counts,"groups_percent":groups,
            "worker_scheduled_percent":scheduled / (wall*1e9)*100.,
            "worker_runqueue_percent":runqueue / (wall*1e9)*100.,
            "operator_total_ms":operator_total,"operator_ms":operations,
            "cache_work_ceiling":{"scope":"Optimistic removal of embedding, first QKV and the entire first attention layer. Real hits still compare state, compute new-prefix attention and merge; later contextual layers execute.","first_attention_ms":first_attention,"first_qkv_ms":first_qkv,"embedding_ms":embedding,"fraction":(first_attention+first_qkv+embedding)/operator_total},
            "arithmetic":{"scope":arithmetic["scope"],"tokens":token_count,"total_gflops":gflops,"shape":shape,"register_control_gflops_per_second":args.register_gflops,"historical_register_only_floor_ms":args.register_gflops.map(|rate|gflops/rate*1000.),"warning":"Historical register-only throughput is frequency/host sensitive and excludes non-matmul work; this floor is not an achievable inference forecast."}
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_math_helpers_follow_their_actual_callers() {
        assert_eq!(
            group("rayon::wait;decisive::laya::cpu::x86_avx2::activation::gated;libm::math::expf"),
            "activation"
        );
        assert_eq!(
            group(
                "rayon::wait;decisive::laya::cpu::x86_avx2::softmax::exponentials;libm::math::expf"
            ),
            "softmax and exponentials"
        );
        assert_eq!(
            group("rayon::wait;decisive::laya::cpu::x86_avx2::inline_panel::<true,true>"),
            "projection kernels"
        );
        assert_eq!(group("rayon::wait;unknown"), "other");
    }
}
