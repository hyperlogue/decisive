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
    #[cfg(feature = "kernel-checks")]
    benchmark: bool,
    #[cfg(feature = "kernel-checks")]
    raw_logits: Vec<Value>,
    #[cfg(feature = "kernel-checks")]
    resident_layers: Vec<HashMap<String, webgpu::Tensor>>,
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
        Self::from_parts(
            webgpu::Gpu::new().await.map_err(error)?,
            config,
            decision,
            joint,
            tokenizer,
            readout,
            headers_json,
        )
    }
    #[cfg(feature = "kernel-checks")]
    #[wasm_bindgen(js_name=createConfigured)]
    pub async fn create_configured(
        config: &[u8],
        decision: &[u8],
        joint: &[u8],
        tokenizer: &[u8],
        readout: &[u8],
        headers_json: &str,
        mode: &str,
        profile: bool,
    ) -> Result<BrowserGpu, JsValue> {
        let gpu = webgpu::Gpu::checkpoint(mode, profile)
            .await
            .map_err(error)?;
        let mut model = Self::from_parts(
            gpu,
            config,
            decision,
            joint,
            tokenizer,
            readout,
            headers_json,
        )?;
        model.benchmark = true;
        Ok(model)
    }
    #[cfg(feature = "kernel-checks")]
    pub fn diagnostics(&self) -> String {
        json!({"logits":self.raw_logits}).to_string()
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
        #[cfg(feature = "kernel-checks")]
        {
            self.raw_logits.clear();
            if self.resident_layers.is_empty() {
                self.gpu.clear_checkpoint_weights();
            } else if self.resident_layers.len() != 32 {
                return Err(error("resident weight loading is incomplete"));
            }
        }
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
    #[cfg(feature = "kernel-checks")]
    #[wasm_bindgen(js_name=stageTensor)]
    pub fn stage_tensor(&mut self, layer: usize, name: &str, bytes: &[u8]) -> Result<(), JsValue> {
        if !self.benchmark || !self.chunks.is_empty() || layer != self.resident_layers.len() || layer >= 32 {
            return Err(error("resident weights must be loaded in order before preparing a request"));
        }
        if !name.starts_with(&format!("language_model.layers.{layer}.")) || self.tensors.contains_key(name) {
            return Err(error("duplicate tensor or wrong resident layer"));
        }
        let range = self.index.tensor(name).map_err(error)?;
        if bytes.len() as u64 != range.end - range.start {
            return Err(error("tensor byte length mismatch"));
        }
        self.tensors.insert(name.into(), self.gpu.upload(bytes, range.shape).map_err(error)?);
        Ok(())
    }
    #[cfg(feature = "kernel-checks")]
    #[wasm_bindgen(js_name=cacheLayer)]
    pub async fn cache_layer(&mut self, layer: usize) -> Result<(), JsValue> {
        if !self.benchmark || !self.chunks.is_empty() || layer != self.resident_layers.len() || layer >= 32 {
            return Err(error("invalid resident layer load"));
        }
        let ranges = self.index.layer(layer).map_err(error)?;
        if ranges.len() != self.tensors.len() || ranges.iter().any(|r| !self.tensors.contains_key(&r.name)) {
            return Err(error("resident layer is missing tensors"));
        }
        self.gpu.prepare_resident_weights(&mut self.tensors).await.map_err(error)?;
        self.resident_layers.push(std::mem::take(&mut self.tensors));
        Ok(())
    }
    pub fn adapter(&self) -> String {
        #[allow(unused_mut)]
        let mut info = self.gpu.info();
        #[cfg(feature = "kernel-checks")]
        if self.resident_layers.len() == 32 {
            info["weight_residency"] = json!("all-decoder-layers");
            info["resident_decoder_weight_bytes"] = json!(self.resident_layers.iter()
                .flat_map(|layer| layer.values()).map(|t| t.buffer.size()).sum::<u64>());
            info["quantization"] = self.gpu.q4_info();
        }
        info.to_string()
    }
    #[wasm_bindgen(js_name=runLayer)]
    pub async fn run_layer(&mut self) -> Result<String, JsValue> {
        if !self.started || self.layer >= 32 {
            return Err(error("no active layer"));
        }
        let resident = false;
        #[cfg(feature = "kernel-checks")]
        let resident = resident || self.resident_layers.len() == 32;
        let pack_start = monotonic_now();
        #[cfg(feature = "kernel-checks")]
        if self.benchmark {
            if !resident {
                self.gpu.prepare_checkpoint_weights(&self.tensors).await.map_err(error)?;
            }
            self.gpu.start_profile();
        }
        let pack_ms = if resident { 0.0 } else { monotonic_now() - pack_start };
        let start = monotonic_now();
        let tensors = &self.tensors;
        #[cfg(feature = "kernel-checks")]
        let tensors = if resident { &self.resident_layers[self.layer] } else { tensors };
        let result = self
            .gpu
            .layer(
                tensors,
                self.layer,
                if self.layer == 31 {
                    Some(&self.chunks[self.chunk].markers)
                } else {
                    None
                },
            )
            .await;
        #[cfg(feature = "kernel-checks")]
        if !resident {
            self.gpu.clear_checkpoint_weights();
        }
        for (_, t) in self.tensors.drain() {
            #[cfg(feature = "kernel-checks")]
            self.gpu.retain_weight_buffer(t.buffer);
            #[cfg(not(feature = "kernel-checks"))]
            t.buffer.destroy();
        }
        self.gpu.discard_scratch();
        result.map_err(error)?;
        let compute_ms = monotonic_now() - start;
        #[allow(unused_mut)]
        let mut profile = Value::Null;
        #[cfg(feature = "kernel-checks")]
        if self.benchmark {
            profile = self.gpu.finish_profile().await.map_err(error)?;
        }
        self.layer += 1;
        Ok(json!({"pack_ms":pack_ms,"compute_ms":compute_ms,"profile":profile,"timing":"dispatch + synchronized GPU completion; optional weight preparation is reported in pack_ms"}).to_string())
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
        #[cfg(feature = "kernel-checks")]
        if self.benchmark {
            if !logits.iter().flatten().all(|v| v.is_finite()) {
                return Err(error("nonfinite checkpoint logits"));
            }
            self.raw_logits
                .push(json!({"keys":chunk.keys,"values":logits}));
        }
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

#[cfg(feature = "kernel-checks")]
#[wasm_bindgen(js_name=pplxFlashWebGpuQ4Check)]
pub async fn q4_check() -> Result<String, JsValue> {
    Ok(webgpu::q4::check().await.map_err(error)?.to_string())
}

impl BrowserGpu {
    fn from_parts(
        gpu: webgpu::Gpu,
        config: &[u8],
        decision: &[u8],
        joint: &[u8],
        tokenizer: &[u8],
        readout: &[u8],
        headers_json: &str,
    ) -> Result<Self, JsValue> {
        let headers: Vec<String> = serde_json::from_str(headers_json).map_err(error)?;
        Ok(Self {
            gpu,
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
            #[cfg(feature = "kernel-checks")]
            benchmark: false,
            #[cfg(feature = "kernel-checks")]
            raw_logits: Vec::new(),
            #[cfg(feature = "kernel-checks")]
            resident_layers: Vec::new(),
        })
    }
}
