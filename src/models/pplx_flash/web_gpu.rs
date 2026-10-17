use super::*;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn monotonic_now() -> f64;
}
fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}
#[wasm_bindgen(js_name=PplxFlashWebGpu)]
pub struct BrowserGpu {
    engine: Engine,
    index: weights::Index,
    request: Value,
    chunks: Vec<input::Encoded>,
    chunk: usize,
    layer: usize,
    tensors: HashMap<String, webgpu::Tensor>,
    gpu: webgpu::Gpu,
    work: Work,
    answers: serde_json::Map<String, Value>,
    started: bool,
}
#[wasm_bindgen(js_class=PplxFlashWebGpu)]
impl BrowserGpu {
    pub async fn create(
        config: &[u8],
        decision: &[u8],
        joint: &[u8],
        tokenizer: &[u8],
        readout: &[u8],
        headers_json: &str,
    ) -> Result<BrowserGpu, JsValue> {
        let headers: Vec<String> = serde_json::from_str(headers_json).map_err(error)?;
        Ok(Self {
            gpu: webgpu::Gpu::new().await.map_err(error)?,
            engine: Engine::new(config, decision, joint, tokenizer, readout, false)
                .map_err(error)?,
            index: weights::Index::new(&headers).map_err(error)?,
            request: Value::Null,
            chunks: Vec::new(),
            chunk: 0,
            layer: 0,
            tensors: HashMap::new(),
            work: Work::default(),
            answers: serde_json::Map::new(),
            started: false,
        })
    }
    pub fn prepare(&mut self, request: &str, max_length: usize) -> Result<String, JsValue> {
        let request: Value = serde_json::from_str(request).map_err(error)?;
        let chunks = self
            .engine
            .input
            .plan(&request, max_length)
            .map_err(error)?;
        if chunks.iter().any(|c| c.ids.len() > 2048) {
            return Err(error(
                "Pplx WebGPU currently supports at most 2048 tokens per sequence",
            ));
        }
        self.request = request;
        self.chunks = chunks;
        self.chunk = 0;
        self.layer = 0;
        self.started = false;
        self.work = Work::default();
        self.gpu.reset();
        for t in self.tensors.values() {
            t.buffer.destroy();
        }
        self.tensors.clear();
        self.answers.clear();
        serde_json::to_string(&self.chunks).map_err(error)
    }
    #[wasm_bindgen(js_name=tensorRange)]
    pub fn tensor_range(&self, name: &str) -> Result<String, JsValue> {
        serde_json::to_string(&self.index.tensor(name).map_err(error)?).map_err(error)
    }
    pub fn ranges(&self, index: usize) -> Result<String, JsValue> {
        serde_json::to_string(&self.index.layer(index).map_err(error)?).map_err(error)
    }
    pub fn begin(&mut self, embedding_rows: &[u8]) -> Result<(), JsValue> {
        if self.started || self.chunk >= self.chunks.len() {
            return Err(error("no prepared chunk, or previous chunk unfinished"));
        }
        if embedding_rows.len() != self.chunks[self.chunk].ids.len() * D * 2 {
            return Err(error("embedding row count mismatch"));
        }
        self.work.hidden.clear();
        self.work.hidden.extend(
            embedding_rows
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)),
        );
        self.gpu.begin(&self.work.hidden).map_err(error)?;
        self.work.hidden.clear();
        self.started = true;
        self.layer = 0;
        Ok(())
    }
    pub fn tensor(&mut self, name: &str, bytes: Vec<u8>) -> Result<(), JsValue> {
        if !self.started || self.layer >= 32 {
            return Err(error("no active layer"));
        }
        let prefix = format!("language_model.layers.{}.", self.layer);
        if !name.starts_with(&prefix) {
            return Err(error("tensor belongs to another layer"));
        }
        let range = self.index.tensor(name).map_err(error)?;
        if bytes.len() as u64 != range.end - range.start {
            return Err(error("tensor byte length mismatch"));
        }
        if let Some(old) = self.tensors.insert(
            name.into(),
            self.gpu.upload(&bytes, range.shape).map_err(error)?,
        ) {
            old.buffer.destroy();
        }
        Ok(())
    }
    pub fn adapter(&self) -> String {
        self.gpu.info().to_string()
    }
    #[wasm_bindgen(js_name=runLayer)]
    pub async fn run_layer(&mut self) -> Result<String, JsValue> {
        if !self.started || self.layer >= 32 {
            return Err(error("no active layer"));
        }
        let start = monotonic_now();
        let result = self
            .gpu
            .layer(
                &self.tensors,
                self.layer,
                if self.layer == 31 {
                    Some(&self.chunks[self.chunk].markers)
                } else {
                    None
                },
            )
            .await;
        for t in self.tensors.values() {
            t.buffer.destroy();
        }
        self.tensors.clear();
        self.gpu.discard_scratch();
        result.map_err(error)?;
        self.layer += 1;
        Ok(json!({"pack_ms":0.,"compute_ms":monotonic_now()-start,"timing":"dispatch + synchronized GPU completion"}).to_string())
    }
    #[wasm_bindgen(js_name=finishChunk)]
    pub async fn finish_chunk(&mut self, norm: &[u8]) -> Result<(), JsValue> {
        if !self.started || self.layer != 32 || norm.len() != D * 2 {
            return Err(error("unfinished chunk or invalid norm"));
        }
        let norm: Vec<_> = norm
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect();
        self.work.hidden = self.gpu.read_hidden().await.map_err(error)?;
        let chunk = &self.chunks[self.chunk];
        let logits = self
            .engine
            .logits(&mut self.work, &norm, chunk)
            .map_err(error)?;
        for (key, values) in chunk.keys.iter().zip(logits) {
            self.answers.insert(
                key.clone(),
                self.engine
                    .input
                    .answer(&self.request["questions"][key], &values)
                    .map_err(error)?,
            );
        }
        self.chunk += 1;
        self.started = false;
        Ok(())
    }
    pub fn result(&self) -> Result<String, JsValue> {
        if self.chunks.is_empty() || self.chunk != self.chunks.len() {
            return Err(error("request is unfinished"));
        }
        Ok(json!({"answers":self.answers,"usage":{"input_tokens":self.chunks.iter().map(|c|c.ids.len()).sum::<usize>(),"output_tokens":0,"sequences":self.chunks.len()}}).to_string())
    }
}

#[cfg(feature = "kernel-checks")]
#[wasm_bindgen(js_name=pplxFlashWebGpuCheck)]
pub async fn kernel_check() -> Result<String, JsValue> {
    Ok(webgpu::checks::run().await.map_err(error)?.to_string())
}
