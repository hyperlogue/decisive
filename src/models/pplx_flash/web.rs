use super::*;
use safetensors::{Dtype, tensor::TensorView};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn monotonic_now() -> f64;
}

#[wasm_bindgen(js_name = PplxFlashHasher)]
pub struct DownloadHasher(sha2::Sha256);

#[wasm_bindgen(js_class = PplxFlashHasher)]
impl DownloadHasher {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        use sha2::Digest;
        Self(sha2::Sha256::new())
    }

    pub fn update(&mut self, bytes: &[u8]) {
        use sha2::Digest;
        self.0.update(bytes);
    }

    pub fn finish(self) -> String {
        use sha2::Digest;
        format!("{:x}", self.0.finalize())
    }
}
fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}
#[wasm_bindgen(js_name=PplxFlashModel)]
pub struct BrowserModel {
    engine: Engine,
    index: weights::Index,
    request: Value,
    chunks: Vec<input::Encoded>,
    chunk: usize,
    layer: usize,
    tensors: HashMap<String, (Vec<usize>, Vec<u8>)>,
    work: Work,
    answers: serde_json::Map<String, Value>,
    started: bool,
}
#[wasm_bindgen(js_class=PplxFlashModel)]
impl BrowserModel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: &[u8],
        decision: &[u8],
        joint: &[u8],
        tokenizer: &[u8],
        readout: &[u8],
        headers_json: &str,
    ) -> Result<BrowserModel, JsValue> {
        let headers: Vec<String> = serde_json::from_str(headers_json).map_err(error)?;
        Ok(Self {
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
        self.request = request;
        self.chunks = chunks;
        self.chunk = 0;
        self.layer = 0;
        self.started = false;
        self.work = Work::default();
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
        self.tensors.insert(name.into(), (range.shape, bytes));
        Ok(())
    }
    #[wasm_bindgen(js_name=runLayer)]
    pub fn run_layer(&mut self) -> Result<String, JsValue> {
        let start = monotonic_now();
        if !self.started || self.layer >= 32 {
            return Err(error("no active layer"));
        }
        let views = self
            .tensors
            .iter()
            .map(|(name, (shape, data))| {
                Ok((
                    name.clone(),
                    TensorView::new(Dtype::BF16, shape.clone(), data)?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()
            .map_err(error)?;
        let layer =
            Layer::load(&Weights(views), &self.engine.config, self.layer, false).map_err(error)?;
        self.tensors.clear();
        let compute = monotonic_now();
        layer
            .forward(
                &mut self.work,
                Plan::default(),
                false,
                if self.layer == 31 {
                    Some(&self.chunks[self.chunk].markers)
                } else {
                    None
                },
            )
            .map_err(error)?;
        self.layer += 1;
        Ok(json!({"pack_ms":compute-start,"compute_ms":monotonic_now()-compute}).to_string())
    }
    #[wasm_bindgen(js_name=finishChunk)]
    pub fn finish_chunk(&mut self, norm: &[u8]) -> Result<(), JsValue> {
        if !self.started || self.layer != 32 || norm.len() != D * 2 {
            return Err(error("unfinished chunk or invalid norm"));
        }
        let norm: Vec<_> = norm
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect();
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
