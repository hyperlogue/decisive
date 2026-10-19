use super::*;
use std::{collections::{HashMap, VecDeque}, rc::Rc};
use super::webgpu::streaming::WeightBuffers;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn monotonic_now() -> f64;
}
fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}
struct PendingLayer {
    layer: usize,
    submitted_ms: f64,
    work: webgpu::QueuedLayer,
    tensors: HashMap<String, webgpu::Tensor>,
}
#[wasm_bindgen(js_name=PplxFlashWebGpu)]
pub struct BrowserGpu {
    engine: Engine,
    index: Rc<weights::Index>,
    buffers: Option<WeightBuffers>,
    pending_layers: VecDeque<PendingLayer>,
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
    #[cfg(feature = "kernel-checks")]
    auxiliary_bits: u32,
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
        let (decoder_mode, auxiliary_bits) = match mode {
            "int8-all" => ("int8-tuned", 8),
            "q4-all" => ("q4-fused", 4),
            "q4-e4" | "q4-e4-serial" => ("q4-e4-reuse", 0),
            "q4-e4-signed" => ("q4-e4-signed-reuse", 0),
            "q4-e4-baseline" => ("q4-e4-baseline-reuse", 0),
            "q4-e4-int8" => ("q4-e4-int8-reuse", 0),
            _ => (mode, 0),
        };
        let gpu = webgpu::Gpu::checkpoint(decoder_mode, profile)
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
        if model.gpu.grouped_q4() && model.engine.packed_readout.is_empty() {
            return Err(error("grouped Q4 requires a packed Q4 readout"));
        }
        model.auxiliary_bits = auxiliary_bits;
        if auxiliary_bits != 0 {
            reconstruct_rows(&mut model.engine.readout, D, auxiliary_bits).map_err(error)?;
        }
        Ok(model)
    }
    #[cfg(feature = "kernel-checks")]
    pub fn diagnostics(&self) -> String {
        json!({"logits":self.raw_logits}).to_string()
    }
    pub fn prepare(&mut self, request: &str, max_length: usize) -> Result<String, JsValue> {
        if !self.pending_layers.is_empty() { return Err(error("queued layers must finish before preparing another request")); }
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
        if let Some(buffers) = &self.buffers { buffers.reset()?; }
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
        let row_bytes = if self.gpu.grouped_q4() {
            super::packed_q4::size(1, D).map_err(error)?
        } else {
            D * 2
        };
        if embedding_rows.len() != self.chunks[self.chunk].ids.len() * row_bytes {
            return Err(error("embedding row count mismatch"));
        }
        self.work.hidden.clear();
        if self.gpu.grouped_q4() {
            super::packed_q4::unpack(embedding_rows, &mut self.work.hidden).map_err(error)?;
        } else {
            self.work.hidden.extend(
                embedding_rows
                    .chunks_exact(2)
                    .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)),
            );
        }
        #[cfg(feature = "kernel-checks")]
        if self.auxiliary_bits != 0 {
            reconstruct_rows(&mut self.work.hidden, D, self.auxiliary_bits).map_err(error)?;
        }
        self.gpu.begin(&self.work.hidden).map_err(error)?;
        if let Some(buffers) = &self.buffers { buffers.begin_chunk().map_err(error)?; }
        self.work.hidden.clear();
        self.started = true;
        self.layer = 0;
        Ok(())
    }
    pub fn tensor(&mut self, name: &str, bytes: Vec<u8>) -> Result<(), JsValue> {
        if self.buffers.is_some() { return Err(error("use the separate weight uploader for buffered inference")); }
        if !self.started || self.layer >= 32 {
            return Err(error("no active layer"));
        }
        let prefix = format!("language_model.layers.{}.", self.layer);
        if !name.starts_with(&prefix) {
            return Err(error("tensor belongs to another layer"));
        }
        let range = self.index.tensor(name).map_err(error)?;
        let expected = if self.gpu.grouped_q4() && range.shape.len() == 2 {
            super::packed_q4::size(range.shape[0], range.shape[1]).map_err(error)? as u64
        } else {
            range.end - range.start
        };
        if bytes.len() as u64 != expected {
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
    pub async fn stage_tensor(&mut self, layer: usize, name: &str, bytes: &[u8]) -> Result<(), JsValue> {
        if self.buffers.is_some() { return Err(error("resident and double-buffered weights are mutually exclusive")); }
        if !self.benchmark || !self.chunks.is_empty() || layer != self.resident_layers.len() || layer >= 32 {
            return Err(error("resident weights must be loaded in order before preparing a request"));
        }
        if !name.starts_with(&format!("language_model.layers.{layer}.")) || self.tensors.contains_key(name) {
            return Err(error("duplicate tensor or wrong resident layer"));
        }
        let range = self.index.tensor(name).map_err(error)?;
        let expected = if self.gpu.grouped_q4() && range.shape.len() == 2 {
            super::packed_q4::size(range.shape[0], range.shape[1]).map_err(error)? as u64
        } else {
            range.end - range.start
        };
        if bytes.len() as u64 != expected {
            return Err(error("tensor byte length mismatch"));
        }
        self.tensors.insert(name.into(), self.gpu.upload_resident(bytes, range.shape).await.map_err(error)?);
        Ok(())
    }
    #[cfg(feature = "kernel-checks")]
    #[wasm_bindgen(js_name=cacheLayer)]
    pub async fn cache_layer(&mut self, layer: usize) -> Result<(), JsValue> {
        if self.buffers.is_some() { return Err(error("resident and double-buffered weights are mutually exclusive")); }
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
        if self.gpu.grouped_q4() {
            info["quantization"] = self.gpu.q4_info();
        }
        if self.buffers.is_some() {
            info["weight_residency"] = json!("two-layer-banks");
            info["weight_buffer_banks"] = json!(2);
            info["queued_streaming_supported"] = json!(self.gpu.supports_queued());
        }
        #[cfg(feature = "kernel-checks")]
        if self.resident_layers.len() == 32 {
            info["weight_residency"] = json!("all-decoder-layers");
            info["resident_decoder_weight_bytes"] = json!(self.resident_layers.iter()
                .flat_map(|layer| layer.values()).map(|t| t.buffer.size()).sum::<u64>());
            info["resident_decoder_layers"] = json!(self.resident_layers.len());
            info["embedding"] = json!("wasm-bounded-row-cache");
            info["queued_resident_supported"] = json!(self.gpu.grouped_q4() && self.gpu.supports_queued());
            info["quantization"] = self.gpu.q4_info();
        }
        #[cfg(feature = "kernel-checks")]
        {
            info["auxiliary_quantization"] = json!({
                "weight_bits": self.auxiliary_bits,
                "tensors": if self.auxiliary_bits == 0 { Vec::<&str>::new() } else { vec!["language_model.embed_tokens.weight", "readout.weight"] },
                "scheme": "symmetric-per-row-absmax-round-even-fp32-scale",
                "execution": "quality-only-quantize-dequantize-to-fp32",
                "floating_point_exceptions": ["normalization", "convolution", "recurrent parameters"],
                "performance_eligible": self.auxiliary_bits == 0
            });
        }
        info.to_string()
    }
    #[wasm_bindgen(js_name=weightBuffers)]
    pub fn weight_buffers(&mut self) -> Result<WeightBuffers, JsValue> {
        if !self.chunks.is_empty() || self.started { return Err(error("configure buffering before preparing a request")); }
        if !self.tensors.is_empty() { return Err(error("weights are already being staged through the model")); }
        #[cfg(feature = "kernel-checks")]
        if !self.resident_layers.is_empty() { return Err(error("resident and double-buffered weights are mutually exclusive")); }
        let buffers = self.buffers.get_or_insert_with(|| self.gpu.weight_uploader(self.index.clone()));
        Ok(buffers.clone())
    }
    #[wasm_bindgen(js_name=runLayer)]
    pub async fn run_layer(&mut self) -> Result<String, JsValue> {
        if !self.pending_layers.is_empty() { return Err(error("collect queued layers before synchronous execution")); }
        let result = self.run_layer_inner().await;
        if result.is_err() {
            #[cfg(feature = "kernel-checks")]
            if self.resident_layers.is_empty() { self.gpu.clear_checkpoint_weights(); }
            for (_, tensor) in self.tensors.drain() { tensor.buffer.destroy(); }
            self.gpu.reset();
            self.started = false;
            if let Some(buffers) = &self.buffers { buffers.fail(); }
        }
        result
    }
    #[wasm_bindgen(js_name=supportsQueued)]
    pub fn supports_queued(&self) -> bool {
        let weights = self.buffers.is_some();
        #[cfg(feature = "kernel-checks")]
        let weights = weights || self.resident_layers.len() == 32 && self.gpu.grouped_q4();
        weights && self.gpu.supports_queued()
    }

    #[wasm_bindgen(js_name=queueLayer)]
    pub fn queue_layer(&mut self) -> Result<String, JsValue> {
        if !self.started || self.layer >= 32 || self.pending_layers.len() >= 2 || !self.supports_queued() {
            return Err(error("queued layer unavailable; collect the oldest layer before reusing its bank"));
        }
        let start = monotonic_now();
        if let Some(buffers) = &self.buffers {
            for (name, tensor) in buffers.start_layer(self.layer).map_err(error)? {
                self.tensors.insert(name, self.gpu.prepare_uploaded(&tensor).map_err(error)?);
            }
        }
        #[cfg(feature = "kernel-checks")]
        if self.buffers.is_some() { self.gpu.prepare_queued_weights(&self.tensors).map_err(error)?; }
        let tensors = &self.tensors;
        #[cfg(feature = "kernel-checks")]
        let tensors = if self.resident_layers.len() == 32 { &self.resident_layers[self.layer] } else { tensors };
        let work = self.gpu.queue_layer(tensors, self.layer,
            (self.layer == 31).then(|| self.chunks[self.chunk].markers.as_slice())).map_err(error)?;
        if let Some(buffers) = &self.buffers { buffers.queued(self.layer); }
        self.pending_layers.push_back(PendingLayer { layer:self.layer, submitted_ms:monotonic_now(),
            work, tensors:std::mem::take(&mut self.tensors) });
        self.layer += 1;
        Ok(json!({"layer":self.layer-1,"encode_enqueue_ms":monotonic_now()-start,
            "pack_ms":null,"compute_ms":null,"profile":null,"in_flight_layers":self.pending_layers.len(),
            "timing":"host encoding and submission only; completion is collected under backpressure; no per-layer GPU readback"}).to_string())
    }
    #[wasm_bindgen(js_name=waitLayer)]
    pub async fn wait_layer(&mut self) -> Result<String, JsValue> {
        let pending = self.pending_layers.pop_front().ok_or_else(|| error("no queued layer"))?;
        let start = monotonic_now();
        let result = self.gpu.collect_layer(pending.work).await;
        for tensor in pending.tensors.into_values() {
            if self.buffers.as_ref().is_some_and(|buffers| buffers.owns(&tensor.buffer)) { continue; }
            #[cfg(feature = "kernel-checks")]
            self.gpu.retain_weight_buffer(tensor.buffer);
            #[cfg(not(feature = "kernel-checks"))]
            tensor.buffer.destroy();
        }
        if let Some(buffers) = &self.buffers { buffers.collected(pending.layer); }
        result.map_err(error)?;
        Ok(json!({"layer":pending.layer,"queue_wait_ms":monotonic_now()-start,
            "submission_to_collection_ms":monotonic_now()-pending.submitted_ms}).to_string())
    }
    #[wasm_bindgen(js_name=abortQueued)]
    pub async fn abort_queued(&mut self) -> Result<(), JsValue> {
        let mut failure = None;
        while !self.pending_layers.is_empty() {
            if let Err(error) = self.wait_layer().await { failure = Some(error); }
        }
        // Preparation or encoding may have failed after submitting conversions.
        // Drain these too before destroying their uniforms or weight buffers.
        if let Err(error) = self.gpu.wait_idle().await { failure = Some(JsValue::from_str(&error.to_string())); }
        #[cfg(feature = "kernel-checks")]
        if self.resident_layers.is_empty() { self.gpu.clear_checkpoint_weights(); }
        for tensor in self.tensors.drain().map(|(_, t)| t) { tensor.buffer.destroy(); }
        self.gpu.reset();
        self.started = false;
        if let Some(buffers) = &self.buffers { buffers.fail(); }
        failure.map_or(Ok(()), Err)
    }
    #[wasm_bindgen(js_name=finishChunk)]
    pub async fn finish_chunk(&mut self, norm: &[u8]) -> Result<(), JsValue> {
        if !self.started || self.layer != 32 || !self.pending_layers.is_empty() || norm.len() != D * 2 {
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
        if let Some(buffers) = &self.buffers { buffers.end_chunk().map_err(error)?; }
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

#[cfg(feature = "kernel-checks")]
#[wasm_bindgen(js_name=pplxFlashWebGpuGroupedQ4Check)]
pub async fn grouped_q4_check() -> Result<String, JsValue> {
    Ok(webgpu::grouped_q4::check().await.map_err(error)?.to_string())
}

impl BrowserGpu {
    async fn run_layer_inner(&mut self) -> Result<String, JsValue> {
        if !self.started || self.layer >= 32 {
            return Err(error("no active layer"));
        }
        let resident = false;
        #[cfg(feature = "kernel-checks")]
        let resident = resident || self.resident_layers.len() == 32;
        let pack_start = monotonic_now();
        if let Some(buffers) = &self.buffers {
            self.tensors = buffers.start_layer(self.layer).map_err(error)?.into_iter()
                .map(|(name, tensor)| Ok((name, self.gpu.prepare_uploaded(&tensor)?)))
                .collect::<Result<HashMap<_, _>>>().map_err(error)?;
        }
        #[cfg(feature = "kernel-checks")]
        if self.benchmark {
            if !resident {
                self.gpu.prepare_checkpoint_weights(&self.tensors).await.map_err(error)?;
            }
            self.gpu.start_profile();
        }
        let pack_ms = if resident || self.gpu.grouped_q4() {
            0.0
        } else {
            monotonic_now() - pack_start
        };
        let start = monotonic_now();
        let tensors = &self.tensors;
        #[cfg(feature = "kernel-checks")]
        let tensors = if resident { &self.resident_layers[self.layer] } else { tensors };
        let upload = self.buffers.clone();
        let result = self
            .gpu
            .layer_submitted(
                tensors,
                self.layer,
                if self.layer == 31 {
                    Some(&self.chunks[self.chunk].markers)
                } else {
                    None
                },
                move || { if let Some(buffers) = upload { buffers.submitted(); } },
            )
            .await;
        #[cfg(feature = "kernel-checks")]
        if !resident {
            self.gpu.clear_checkpoint_weights();
        }
        for (_, t) in self.tensors.drain() {
            if self.buffers.as_ref().is_some_and(|b| b.owns(&t.buffer)) { continue; }
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
        if let Some(buffers) = &self.buffers { buffers.finish_layer(self.layer); }
        self.layer += 1;
        Ok(json!({"pack_ms":pack_ms,"compute_ms":compute_ms,"profile":profile,"timing":"dispatch + synchronized GPU completion; optional weight preparation is reported in pack_ms"}).to_string())
    }

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
            index: Rc::new(weights::Index::new(&headers).map_err(error)?),
            buffers: None,
            pending_layers: VecDeque::new(),
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
            #[cfg(feature = "kernel-checks")]
            auxiliary_bits: 0,
        })
    }
}

#[cfg(feature = "kernel-checks")]
fn reconstruct_rows(values: &mut [f32], columns: usize, bits: u32) -> Result<()> {
    ensure!([4, 8].contains(&bits) && columns > 0 && values.len() % columns == 0, "invalid row quantization shape or precision");
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite row quantization input");
    let limit = ((1u32 << (bits - 1)) - 1) as f32;
    for row in values.chunks_exact_mut(columns) {
        let maximum = row.iter().map(|v| v.abs()).fold(0f32, f32::max);
        if maximum == 0. { row.fill(0.); continue; }
        let scale = maximum / limit;
        ensure!(scale.is_normal(), "row quantization scale underflow");
        for value in row {
            let code = (*value / scale).round_ties_even().clamp(-limit, limit) as i32;
            *value = code as f32 * scale;
        }
    }
    Ok(())
}

#[cfg(feature = "kernel-checks")]
#[wasm_bindgen(js_name=pplxFlashQuantizeRows)]
pub fn quantize_rows(values: &[f32], columns: usize, bits: u32) -> Result<Vec<f32>, JsValue> {
    let mut output = values.to_vec();
    reconstruct_rows(&mut output, columns, bits).map_err(error)?;
    Ok(output)
}

impl Drop for BrowserGpu {
    fn drop(&mut self) {
        if let Some(buffers) = &self.buffers { buffers.close(); }
        for tensor in self.tensors.values() { tensor.buffer.destroy(); }
        #[cfg(feature = "kernel-checks")]
        for tensor in self.resident_layers.iter().flat_map(|layer| layer.values()) { tensor.buffer.destroy(); }
    }
}
