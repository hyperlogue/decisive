//! Small real-GPU/real-Wasm integration fixture; no checkpoint download needed.
use super::{Gpu, Tensor, streaming::WeightBuffers};
use crate::pplx_flash::weights::Index;
use anyhow::{Result, ensure};
use serde_json::json;
use std::{
    collections::{HashMap, VecDeque},
    rc::Rc,
};
use wasm_bindgen::prelude::*;

fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}
fn outputs(layer: usize) -> usize {
    if layer % 4 == 3 { 9 } else { 7 }
}
fn weight(layer: usize, row: usize, col: usize) -> f32 {
    ((row * 5 + col * 3 + layer * 7) % 31) as f32 / 8. - 15. / 8.
}

#[wasm_bindgen(js_name=PplxFlashStreamingCheck)]
pub struct StreamingCheck {
    gpu: Gpu,
    buffers: WeightBuffers,
    layer: usize,
    weight_bits: Option<u32>,
    max_error: f32,
    pending: VecDeque<PendingProbe>,
}
struct PendingProbe {
    layer: usize,
    work: super::QueuedLayer,
    tensors: HashMap<String, Tensor>,
    output: wgpu::Buffer,
    input: Vec<f32>,
}
#[wasm_bindgen(js_class=PplxFlashStreamingCheck)]
impl StreamingCheck {
    pub async fn create(mode: &str) -> Result<StreamingCheck, JsValue> {
        let gpu = Gpu::checkpoint(mode, false).await.map_err(error)?;
        let mut header = serde_json::Map::new();
        for layer in 0..32 {
            for (suffix, shape) in [
                ("probe.weight", vec![outputs(layer), 16]),
                ("probe.norm", vec![16]),
            ] {
                let bytes = shape.iter().product::<usize>() * 2;
                header.insert(
                    format!("language_model.layers.{layer}.{suffix}"),
                    json!({"dtype":"BF16","shape":shape,"data_offsets":[0,bytes]}),
                );
            }
        }
        let index =
            Rc::new(Index::new(&[serde_json::Value::Object(header).to_string()]).map_err(error)?);
        let buffers = gpu.weight_uploader(index);
        let weight_bits = if gpu.q4.is_some() {
            Some(4)
        } else if gpu.int8.is_some() {
            Some(8)
        } else {
            None
        };
        Ok(Self {
            gpu,
            buffers,
            layer: 0,
            weight_bits,
            max_error: 0.,
            pending: VecDeque::new(),
        })
    }
    #[wasm_bindgen(js_name=weightBuffers)]
    pub fn weight_buffers(&self) -> WeightBuffers {
        self.buffers.clone()
    }
    pub fn begin(&mut self) -> Result<(), JsValue> {
        self.buffers.reset()?;
        self.gpu.clear_checkpoint_weights();
        self.gpu.reset();
        self.buffers.begin_chunk().map_err(error)?;
        self.layer = 0;
        self.max_error = 0.;
        Ok(())
    }
    pub fn ranges(&self, layer: usize) -> Result<String, JsValue> {
        serde_json::to_string(&self.buffers.index.layer(layer).map_err(error)?).map_err(error)
    }
    #[wasm_bindgen(js_name=runLayer)]
    pub async fn run_layer(&mut self) -> Result<f32, JsValue> {
        let result = self.step().await;
        if result.is_err() {
            self.buffers.fail();
        }
        result.map_err(error)
    }
    #[wasm_bindgen(js_name=queueLayer)]
    pub fn queue_layer(&mut self) -> Result<(), JsValue> {
        self.enqueue().map_err(error)
    }
    #[wasm_bindgen(js_name=waitLayer)]
    pub async fn wait_layer(&mut self) -> Result<f32, JsValue> {
        let pending = self
            .pending
            .pop_front()
            .ok_or_else(|| error("no queued probe"))?;
        self.gpu.collect_layer(pending.work).await.map_err(error)?;
        let actual = self
            .gpu
            .device
            .read(
                self.gpu.device.encoder(),
                &pending.output,
                5 * outputs(pending.layer),
            )
            .await
            .map_err(error)?;
        let result = self.compare(pending.layer, &pending.input, &actual);
        for t in pending.tensors.values() {
            if !self.buffers.owns(&t.buffer) {
                t.buffer.destroy();
            }
        }
        self.buffers.collected(pending.layer);
        result.map_err(error)
    }
    pub fn finish(&self) -> Result<String, JsValue> {
        self.buffers.end_chunk().map_err(error)?;
        let memory: serde_json::Value =
            serde_json::from_str(&self.buffers.diagnostics()).map_err(error)?;
        if memory["peak_raw_weight_bytes"].as_u64().unwrap() > 640
            || memory["reused_buffers"].as_u64().unwrap() < 30
        {
            return Err(error("weight banks grew or failed to reuse buffers"));
        }
        Ok(
            json!({"passed":true,"layers":self.layer,"max_absolute_error":self.max_error,
            "bound":1e-4,"weight_buffers":memory,"adapter":self.gpu.info()})
            .to_string(),
        )
    }
}
impl StreamingCheck {
    fn enqueue(&mut self) -> Result<()> {
        ensure!(self.pending.len() < 2, "queued probe is full");
        let layer = self.layer;
        let tensors = self
            .buffers
            .start_layer(layer)?
            .into_iter()
            .map(|(n, t)| Ok((n, self.gpu.prepare_uploaded(&t)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        self.gpu.prepare_queued_weights(&tensors)?;
        let input: Vec<_> = (0..80).map(|i| (i % 9) as f32 / 8. - 0.5).collect();
        let x = self.gpu.device.buffer(&input, "queued probe input")?;
        let mut encoder = self.gpu.device.encoder();
        let output = self.gpu.linear(
            &mut encoder,
            &x,
            &tensors[&format!("language_model.layers.{layer}.probe.weight")].buffer,
            5,
            16,
            outputs(layer),
        )?;
        self.gpu.scratch.borrow_mut().push(x);
        let work = self.gpu.queue_encoded(encoder, output.clone(), 5);
        self.buffers.queued(layer);
        self.pending.push_back(PendingProbe {
            layer,
            work,
            tensors,
            output,
            input,
        });
        self.layer += 1;
        Ok(())
    }
    async fn step(&mut self) -> Result<f32> {
        let layer = self.layer;
        let tensors = self
            .buffers
            .start_layer(layer)?
            .into_iter()
            .map(|(n, t)| Ok((n, self.gpu.prepare_uploaded(&t)?)))
            .collect::<Result<HashMap<String, Tensor>>>()?;
        self.gpu.prepare_checkpoint_weights(&tensors).await?;
        let (m, k, n) = (5, 16, outputs(layer));
        let input: Vec<_> = (0..m * k).map(|i| (i % 9) as f32 / 8. - 0.5).collect();
        let x = self.gpu.device.buffer(&input, "streaming check input")?;
        let mut encoder = self.gpu.device.encoder();
        let out = self.gpu.linear(
            &mut encoder,
            &x,
            &tensors[&format!("language_model.layers.{layer}.probe.weight")].buffer,
            m,
            k,
            n,
        )?;
        let buffers = self.buffers.clone();
        let actual = self
            .gpu
            .device
            .read_submitted(encoder, &out, m * n, move || buffers.submitted())
            .await?;
        let max_error = self.compare(layer, &input, &actual)?;
        self.gpu.clear_checkpoint_weights();
        for t in tensors.values() {
            if !self.buffers.owns(&t.buffer) {
                t.buffer.destroy();
            }
        }
        x.destroy();
        self.gpu.discard_scratch();
        self.buffers.finish_layer(layer);
        self.layer += 1;
        Ok(max_error)
    }
    fn compare(&mut self, layer: usize, input: &[f32], actual: &[f32]) -> Result<f32> {
        let (m, k, n) = (5, 16, outputs(layer));
        let mut max_error = 0f32;
        for row in 0..m {
            for col in 0..n {
                let weights: Vec<_> = (0..k).map(|i| weight(layer, col, i)).collect();
                let values = &input[row * k..(row + 1) * k];
                let expected = if let Some(bits) = self.weight_bits {
                    let sx = values.iter().map(|x| x.abs()).fold(0f32, f32::max) / 127.;
                    let limit = ((1u32 << (bits - 1)) - 1) as f32;
                    let sw = weights.iter().map(|x| x.abs()).fold(0f32, f32::max) / limit;
                    let dot: i32 = values
                        .iter()
                        .zip(&weights)
                        .map(|(&x, &w)| {
                            (x / sx).round_ties_even().clamp(-127., 127.) as i32
                                * (w / sw).round_ties_even().clamp(-limit, limit) as i32
                        })
                        .sum();
                    dot as f32 * (sx * sw)
                } else {
                    values.iter().zip(&weights).map(|(x, w)| x * w).sum()
                };
                max_error = max_error.max((actual[row * n + col] - expected).abs());
            }
        }
        ensure!(
            max_error <= 1e-4,
            "streamed layer {layer}: error {max_error}"
        );
        self.max_error = self.max_error.max(max_error);
        Ok(max_error)
    }
}
impl Drop for StreamingCheck {
    fn drop(&mut self) {
        self.buffers.close();
    }
}
