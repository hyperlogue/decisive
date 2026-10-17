mod backbone;
mod cpu;
mod input;
#[cfg(all(feature = "cpu", feature = "int8"))]
mod int8;
#[cfg(feature = "cpu")]
mod native;
pub mod weights;
#[cfg(feature = "cpu")]
pub use native::Model;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd;
#[cfg(feature = "wasm")]
mod web;
#[cfg(all(feature = "wasm", feature = "webgpu"))]
mod web_gpu;
#[cfg(all(feature = "wasm", feature = "webgpu"))]
mod webgpu;
use anyhow::{Result, ensure};
use backbone::{D, Layer, V, Work};
use serde_json::{Value, json};
use weights::Weights;
pub const REPOSITORY: &str = "perplexity-ai/pplx-decider-flash-v1.2";
pub const REVISION: &str = "6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d";
pub const SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];
#[derive(Clone, Copy)]
struct Plan {
    projection_rows: usize,
    attention_queries: usize,
}
impl Default for Plan {
    fn default() -> Self {
        Self {
            projection_rows: if cfg!(target_arch = "wasm32") { 4 } else { 6 },
            attention_queries: 16,
        }
    }
}
pub struct Engine {
    config: Value,
    input: input::Encoder,
    readout: Vec<f32>,
    #[cfg(feature = "cpu")]
    avx2: bool,
    pub prune_last: bool,
}
impl Engine {
    pub fn new(
        config: &[u8],
        decision: &[u8],
        joint: &[u8],
        tokenizer: &[u8],
        readout: &[u8],
        portable: bool,
    ) -> Result<Self> {
        #[cfg(not(feature = "cpu"))]
        let _ = portable;
        let config: Value = serde_json::from_slice(config)?;
        let c = &config["text_config"];
        for (name, value) in [
            ("hidden_size", D),
            ("intermediate_size", 9216),
            ("vocab_size", V),
            ("num_hidden_layers", 32),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 4),
            ("head_dim", 256),
            ("linear_num_key_heads", 16),
            ("linear_num_value_heads", 32),
            ("linear_key_head_dim", 128),
            ("linear_value_head_dim", 128),
            ("linear_conv_kernel_dim", 4),
        ] {
            ensure!(
                c[name].as_u64() == Some(value as u64),
                "unsupported config: {name}"
            );
        }
        ensure!(
            c["model_type"] == "qwen3_5_text"
                && c["hidden_act"] == "silu"
                && c["attention_bias"] == false
                && c["attn_output_gate"] == true
                && c["rms_norm_eps"].as_f64() == Some(1e-6),
            "unsupported Qwen variant"
        );
        ensure!(
            c["rope_parameters"]["rope_theta"].as_f64() == Some(10_000_000.)
                && c["rope_parameters"]["partial_rotary_factor"].as_f64() == Some(0.25)
                && c["rope_parameters"]["rope_type"] == "default",
            "unsupported rotary configuration"
        );
        ensure!(
            c["layer_types"]
                .as_array()
                .is_some_and(|types| types.len() == 32
                    && types.iter().enumerate().all(|(i, v)| v
                        == if i % 4 == 3 {
                            "full_attention"
                        } else {
                            "linear_attention"
                        })),
            "unsupported layer order"
        );
        ensure!(
            c["max_position_embeddings"] == 262144,
            "unsupported position limit"
        );
        let input = input::Encoder::new(decision, joint, tokenizer)?;
        let w = Weights::from_bytes(readout)?;
        let readout = w.get("weight", &[255, D])?;
        Ok(Self {
            config,
            input,
            readout,
            #[cfg(feature = "cpu")]
            avx2: !portable && cpu::avx2_available(),
            prune_last: true,
        })
    }
    fn logits(
        &self,
        work: &mut Work,
        norm: &[f32],
        chunk: &input::Encoded,
    ) -> Result<Vec<Vec<f32>>> {
        cpu::rms(&mut work.hidden, norm, 1e-6, true);
        let mut logits = Vec::new();
        for (i, (&marker, &count)) in chunk.markers.iter().zip(&chunk.counts).enumerate() {
            let at = if self.prune_last { i } else { marker };
            let x = &work.hidden[at * D..(at + 1) * D];
            let values: Vec<f32> = self
                .readout
                .chunks_exact(D)
                .take(count)
                .map(|w| w.iter().zip(x).map(|(w, x)| w * x).sum())
                .collect();
            ensure!(values.iter().all(|v| v.is_finite()), "non-finite logits");
            logits.push(values);
        }
        Ok(logits)
    }
    pub fn inspect(&self, request: &Value, max_length: usize) -> Result<Value> {
        let chunks = self.input.plan(request, max_length)?;
        Ok(
            json!({"chunks":chunks,"revision":REVISION,"attention":"noncausal_full_attention","precision":"bf16-storage-f32-compute","vision":false,"prefix_cache":false}),
        )
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the pinned 8.45 GiB checkpoint; runs full-model FP32 reference parity"]
    fn checkpoint_protocol_and_output_match_reference() -> Result<()> {
        let mut model = Model::load(
            std::path::Path::new("models/pplx-decider-flash-v1.2"),
            false,
        )?;
        let data: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/pplx_flash/reference.json"
        ))?;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build()?;
        for case in data["cases"].as_array().unwrap() {
            let inspect = model.engine.inspect(&case["request"], 262144)?;
            assert_eq!(inspect["chunks"][0]["ids"], case["ids"]);
            assert_eq!(inspect["chunks"][0]["markers"], case["markers"]);
            let actual = pool.install(|| model.predict(&case["request"], 262144))?;
            compare(&actual, &case["response"]);
            model.engine.prune_last = false;
            let full = pool.install(|| model.predict(&case["request"], 262144))?;
            compare(&actual, &full);
            model.engine.prune_last = true;
        }
        Ok(())
    }
    fn compare(a: &Value, b: &Value) {
        match (a, b) {
            (Value::Number(a), Value::Number(b)) => assert!(
                (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 1e-4,
                "{a} != {b}"
            ),
            (Value::Object(a), Value::Object(b)) => {
                assert_eq!(a.len(), b.len());
                for (k, v) in a {
                    compare(v, &b[k]);
                }
            }
            _ => assert_eq!(a, b),
        }
    }
}

#[cfg(all(
    feature = "kernel-checks",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
pub(crate) fn kernel_checks() -> Result<Value> {
    wasm_simd::checks()
}
