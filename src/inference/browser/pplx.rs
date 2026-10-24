use super::*;
use crate::pplx_flash::{browser_gpu::BrowserGpu, webgpu::streaming::WeightBuffers};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Deserialize)]
struct Range {
    name: String,
    start: u64,
    end: u64,
    #[serde(default)]
    file: usize,
}
#[derive(Deserialize)]
struct Layer {
    file: String,
    tensors: Vec<Range>,
}
#[derive(Deserialize)]
struct Packed {
    schema_version: usize,
    scheme: String,
    headers: Vec<String>,
    layers: Vec<Layer>,
    embedding: String,
    embedding_row_bytes: usize,
    readout: String,
    norm: String,
}
struct Group {
    file: String,
    start: u64,
    end: u64,
    ranges: Vec<Range>,
}
struct Reading {
    promises: js_sys::Array,
}
impl Reading {
    async fn drain(self) {
        let _ = JsFuture::from(Promise::all_settled(&self.promises)).await;
    }
}
pub(super) struct EmbeddingCache {
    capacity: usize,
    entries: HashMap<u32, Vec<u8>>,
    order: VecDeque<u32>,
}
impl EmbeddingCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }
    fn keep(&mut self, id: u32, bytes: Vec<u8>) {
        if bytes.len() > self.capacity {
            return;
        }
        let rows = self.capacity / bytes.len();
        self.order.retain(|existing| *existing != id);
        self.order.push_back(id);
        self.entries.insert(id, bytes);
        while self.order.len() > rows {
            if let Some(id) = self.order.pop_front() {
                self.entries.remove(&id);
            }
        }
    }
    async fn load(
        &mut self,
        ids: &[u32],
        source: &Source,
        file: &str,
        offset: u64,
        row_bytes: usize,
        control: &Control,
    ) -> Result<Vec<u8>> {
        let mut selected = HashMap::new();
        let mut missing = Vec::new();
        for &id in ids {
            if selected.contains_key(&id) || missing.contains(&id) {
                continue;
            }
            if let Some(bytes) = self.entries.get(&id).cloned() {
                selected.insert(id, bytes.clone());
                self.keep(id, bytes);
            } else {
                missing.push(id);
            }
        }
        for chunk in missing.chunks(8) {
            control.check()?;
            let reads = js_sys::Array::new();
            for &id in chunk {
                let start = u64::from(id)
                    .checked_mul(row_bytes as u64)
                    .and_then(|n| n.checked_add(offset))
                    .ok_or_else(|| Error::new("INVALID_CHECKPOINT", "Embedding offset overflow"))?;
                reads.push(
                    &source
                        .start(
                            file,
                            start,
                            start.checked_add(row_bytes as u64).ok_or_else(|| {
                                Error::new("INVALID_CHECKPOINT", "Embedding range overflow")
                            })?,
                        )?
                        .into(),
                );
            }
            let result = js_sys::Array::from(
                &JsFuture::from(Promise::all(&reads))
                    .await
                    .map_err(from_js)?,
            );
            for (i, &id) in chunk.iter().enumerate() {
                let bytes = js_sys::Uint8Array::new(&result.get(i as u32)).to_vec();
                if bytes.len() != row_bytes {
                    return Err(Error::new("INVALID_CHECKPOINT", "Incomplete embedding row"));
                }
                selected.insert(id, bytes.clone());
                self.keep(id, bytes);
            }
        }
        let mut rows = Vec::with_capacity(ids.len() * row_bytes);
        for id in ids {
            rows.extend_from_slice(&selected[id]);
        }
        Ok(rows)
    }
}

pub(super) struct Runtime {
    model: Executor,
    buffers: Option<WeightBuffers>,
    source: Source,
    layers: Vec<Vec<Group>>,
    embedding: String,
    embedding_offset: u64,
    row_bytes: usize,
    norm: Vec<u8>,
    reading: Option<Reading>,
    tuning: bool,
}
impl Runtime {
    pub async fn load(source: Source, config: &Configuration, control: &Control) -> Result<Self> {
        let format = match config.format.as_deref() {
            Some(format @ ("q4-e4" | "q8-e8")) => Some(format),
            Some("bf16") | None => None,
            _ => {
                return Err(Error::new(
                    "UNSUPPORTED_FORMAT",
                    "Pplx supports bf16, q4-e4 and q8-e8",
                ));
            }
        };
        let packed = format.is_some();
        let mut manifest = None;
        let names = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        let mut headers = Vec::new();
        if let Some(format) = format {
            let (scheme, row_bytes) = if format == "q4-e4" {
                ("q4-w4a8-e4-r4-g64", 1440)
            } else {
                ("q8-w8a8-e8-r8-g64", 2720)
            };
            let value: Packed =
                serde_json::from_slice(&source.read(&format!("packed-{format}.json")).await?)
                    .map_err(Error::engine)?;
            if value.schema_version != 1
                || value.scheme != scheme
                || value.layers.len() != 32
                || value.headers.len() != 2
                || value.embedding_row_bytes != row_bytes
            {
                return Err(Error::new(
                    "INVALID_CHECKPOINT",
                    format!("Invalid packed {} manifest", format.to_uppercase()),
                ));
            }
            headers = value.headers.clone();
            manifest = Some(value);
        } else {
            for name in names {
                let bytes = Source::finish(source.start(name, 0, 8)?, 8).await?;
                let length = u64::from_le_bytes(bytes.try_into().unwrap());
                if length > 16 * 1024 * 1024 {
                    return Err(Error::new(
                        "INVALID_CHECKPOINT",
                        "Safetensors header is too large",
                    ));
                }
                let bytes = Source::finish(source.start(name, 8, 8 + length)?, length).await?;
                headers.push(String::from_utf8(bytes).map_err(Error::engine)?);
            }
        }
        let configuration = source.read("config.json").await?;
        let decision = source.read("decision_config.json").await?;
        let joint = source.read("joint_config.json").await?;
        let tokenizer = source.read("tokenizer.json").await?;
        let readout = source
            .read(
                manifest
                    .as_ref()
                    .map_or("readout.safetensors", |m| m.readout.as_str()),
            )
            .await?;
        control.progress("tuning", 0, 1)?;
        let tuning = packed && config.backend == "webgpu" && config.tuning.mode == TuningMode::Auto;
        let mut model = if config.backend == "wasm" {
            if packed {
                return Err(Error::new(
                    "UNSUPPORTED_FORMAT",
                    "Pplx WASM uses bf16 source weights",
                ));
            }
            Executor::Cpu(
                crate::pplx_flash::web::BrowserModel::new(
                    &configuration,
                    &decision,
                    &joint,
                    &tokenizer,
                    &readout,
                    &serde_json::to_string(&headers).unwrap(),
                )
                .map_err(from_js)?,
            )
        } else if packed {
            #[cfg(any(feature = "kernel-checks", feature = "pplx-webgpu"))]
            {
                BrowserGpu::configured(
                    &configuration,
                    &decision,
                    &joint,
                    &tokenizer,
                    &readout,
                    &serde_json::to_string(&headers).unwrap(),
                    format.unwrap(),
                    false,
                    tuning,
                    Some(config.tuning.budget_ms),
                )
                .await
                .map(Executor::Gpu)
                .map_err(from_js)?
            }
            #[cfg(not(any(feature = "kernel-checks", feature = "pplx-webgpu")))]
            {
                return Err(Error::new(
                    "UNSUPPORTED_FORMAT",
                    "Build with pplx-webgpu for packed weights",
                ));
            }
        } else {
            BrowserGpu::create(
                &configuration,
                &decision,
                &joint,
                &tokenizer,
                &readout,
                &serde_json::to_string(&headers).unwrap(),
            )
            .await
            .map(Executor::Gpu)
            .map_err(from_js)?
        };
        control.check()?;
        let buffers = match &mut model {
            Executor::Gpu(model) => Some(model.weight_buffers().map_err(from_js)?),
            Executor::Cpu(_) => None,
        };
        let embedding: Range = parse(
            &model
                .tensor_range("language_model.embed_tokens.weight")
                .map_err(from_js)?,
        )?;
        let norm: Range = parse(
            &model
                .tensor_range("language_model.norm.weight")
                .map_err(from_js)?,
        )?;
        let (embedding, embedding_offset, row_bytes, norm) = if let Some(manifest) = &manifest {
            (
                manifest.embedding.clone(),
                0,
                manifest.embedding_row_bytes,
                source.read(&manifest.norm).await?,
            )
        } else {
            (
                names[embedding.file].into(),
                embedding.start,
                5120,
                Source::finish(
                    source.start(names[norm.file], norm.start, norm.end)?,
                    norm.end - norm.start,
                )
                .await?,
            )
        };
        let mut layers = Vec::new();
        for layer in 0..32 {
            let (mut ranges, layer_file): (Vec<Range>, Option<String>) =
                if let Some(manifest) = &manifest {
                    let item = &manifest.layers[layer];
                    if item.file != format!("{}-layer-{layer:02}.bin", format.unwrap()) {
                        return Err(Error::new(
                            "INVALID_CHECKPOINT",
                            "Unexpected layer filename",
                        ));
                    }
                    (item.tensors.clone(), Some(item.file.clone()))
                } else {
                    (parse(&model.ranges(layer).map_err(from_js)?)?, None)
                };
            ranges.sort_by_key(|r| (r.file, r.start));
            let mut groups: Vec<Group> = Vec::new();
            for range in ranges {
                if range.start >= range.end {
                    return Err(Error::new("INVALID_CHECKPOINT", "Invalid tensor range"));
                }
                let file = match &layer_file {
                    Some(file) => file.clone(),
                    None => names
                        .get(range.file)
                        .ok_or_else(|| Error::new("INVALID_CHECKPOINT", "Invalid shard"))?
                        .to_string(),
                };
                if range.end > source.size(&file)? {
                    return Err(Error::new("INVALID_CHECKPOINT", "Truncated layer"));
                }
                if let Some(group) = groups.last_mut().filter(|g| {
                    g.file == file && range.start >= g.start && g.end - g.start < 4 * 1024 * 1024
                }) {
                    group.end = group.end.max(range.end);
                    group.ranges.push(range);
                } else {
                    groups.push(Group {
                        file,
                        start: range.start,
                        end: range.end,
                        ranges: vec![range],
                    });
                }
            }
            layers.push(groups);
        }
        Ok(Self {
            model,
            buffers,
            source,
            layers,
            embedding,
            embedding_offset,
            row_bytes,
            norm,
            reading: None,
            tuning,
        })
    }
    pub fn info(&self) -> Value {
        serde_json::json!({"mode":if self.tuning { "auto" } else { "off" },"selection":serde_json::from_str::<Value>(&self.model.adapter()).unwrap_or(Value::Null)})
    }
    fn prefetch(&self, layer: usize) -> Result<Reading> {
        let promises = js_sys::Array::new();
        for group in &self.layers[layer] {
            promises.push(
                &self
                    .source
                    .start(&group.file, group.start, group.end)?
                    .into(),
            );
        }
        Ok(Reading { promises })
    }
    async fn upload(&mut self, layer: usize, reading: Reading) -> Result<()> {
        let result = js_sys::Array::from(
            &JsFuture::from(Promise::all(&reading.promises))
                .await
                .map_err(from_js)?,
        );
        for (i, group) in self.layers[layer].iter().enumerate() {
            let bytes = js_sys::Uint8Array::new(&result.get(i as u32)).to_vec();
            if bytes.len() as u64 != group.end - group.start {
                return Err(Error::new("INVALID_CHECKPOINT", "Incomplete layer read"));
            }
            for range in &group.ranges {
                let bytes = &bytes
                    [(range.start - group.start) as usize..(range.end - group.start) as usize];
                if let Some(buffers) = &self.buffers {
                    buffers
                        .stage_tensor(layer, &range.name, bytes)
                        .map_err(from_js)?;
                } else if let Executor::Cpu(model) = &mut self.model {
                    model.tensor(&range.name, bytes.to_vec()).map_err(from_js)?;
                }
            }
        }
        if let Some(buffers) = &self.buffers {
            buffers.finish_upload(layer).map_err(from_js)?;
        }
        Ok(())
    }
    pub async fn predict(
        &mut self,
        request: &Request,
        parameters: &PplxParameters,
        cache: &mut EmbeddingCache,
        control: &Control,
    ) -> Result<Value> {
        parameters.validate()?;
        let result = self.run(request, parameters, cache, control).await;
        if result.is_err() {
            if let Some(reading) = self.reading.take() {
                reading.drain().await;
            }
            self.model.abort_queued().await.map_err(from_js)?;
        }
        result
    }
    async fn run(
        &mut self,
        request: &Request,
        parameters: &PplxParameters,
        cache: &mut EmbeddingCache,
        control: &Control,
    ) -> Result<Value> {
        #[derive(Deserialize)]
        struct Chunk {
            ids: Vec<u32>,
        }
        let chunks: Vec<Chunk> = parse(
            &self
                .model
                .prepare(&request.value().to_string(), parameters.max_sequence_tokens)
                .map_err(from_js)?,
        )?;
        let queued = self.model.supports_queued();
        for (chunk_index, chunk) in chunks.iter().enumerate() {
            control.check()?;
            let rows = cache
                .load(
                    &chunk.ids,
                    &self.source,
                    &self.embedding,
                    self.embedding_offset,
                    self.row_bytes,
                    control,
                )
                .await?;
            self.model.begin(&rows).map_err(from_js)?;
            self.reading = Some(self.prefetch(0)?);
            let mut pending = 0;
            for layer in 0..32 {
                control.progress("inference", chunk_index * 32 + layer, chunks.len() * 32)?;
                let reading = self.reading.take().unwrap();
                if layer < 31 {
                    self.reading = Some(self.prefetch(layer + 1)?);
                }
                if pending == 2 {
                    self.model.wait_layer().await.map_err(from_js)?;
                    pending -= 1;
                }
                self.upload(layer, reading).await?;
                if queued {
                    self.model.queue_layer().map_err(from_js)?;
                    pending += 1;
                } else {
                    self.model.run_layer().await.map_err(from_js)?;
                }
                yield_browser().await;
            }
            while pending > 0 {
                self.model.wait_layer().await.map_err(from_js)?;
                pending -= 1;
            }
            control.check()?;
            self.model.finish_chunk(&self.norm).await.map_err(from_js)?;
        }
        parse(&self.model.result().map_err(from_js)?)
    }
}

enum Executor {
    Gpu(BrowserGpu),
    Cpu(crate::pplx_flash::web::BrowserModel),
}
impl Executor {
    fn tensor_range(&self, name: &str) -> std::result::Result<String, JsValue> {
        match self {
            Self::Gpu(m) => m.tensor_range(name),
            Self::Cpu(m) => m.tensor_range(name),
        }
    }
    fn ranges(&self, layer: usize) -> std::result::Result<String, JsValue> {
        match self {
            Self::Gpu(m) => m.ranges(layer),
            Self::Cpu(m) => m.ranges(layer),
        }
    }
    fn prepare(&mut self, request: &str, max: usize) -> std::result::Result<String, JsValue> {
        match self {
            Self::Gpu(m) => m.prepare(request, max),
            Self::Cpu(m) => m.prepare(request, max),
        }
    }
    fn begin(&mut self, rows: &[u8]) -> std::result::Result<(), JsValue> {
        match self {
            Self::Gpu(m) => m.begin(rows),
            Self::Cpu(m) => m.begin(rows),
        }
    }
    fn result(&self) -> std::result::Result<String, JsValue> {
        match self {
            Self::Gpu(m) => m.result(),
            Self::Cpu(m) => m.result(),
        }
    }
    fn supports_queued(&self) -> bool {
        matches!(self, Self::Gpu(m) if m.supports_queued())
    }
    fn adapter(&self) -> String {
        match self {
            Self::Gpu(m) => m.adapter(),
            Self::Cpu(_) => "{}".into(),
        }
    }
    fn queue_layer(&mut self) -> std::result::Result<String, JsValue> {
        match self {
            Self::Gpu(m) => m.queue_layer(),
            _ => unreachable!(),
        }
    }
    async fn wait_layer(&mut self) -> std::result::Result<String, JsValue> {
        match self {
            Self::Gpu(m) => m.wait_layer().await,
            _ => unreachable!(),
        }
    }
    async fn run_layer(&mut self) -> std::result::Result<String, JsValue> {
        match self {
            Self::Gpu(m) => m.run_layer().await,
            Self::Cpu(m) => m.run_layer(),
        }
    }
    async fn finish_chunk(&mut self, norm: &[u8]) -> std::result::Result<(), JsValue> {
        match self {
            Self::Gpu(m) => m.finish_chunk(norm).await,
            Self::Cpu(m) => m.finish_chunk(norm),
        }
    }
    async fn abort_queued(&mut self) -> std::result::Result<(), JsValue> {
        match self {
            Self::Gpu(m) => m.abort_queued().await,
            Self::Cpu(_) => Ok(()),
        }
    }
}
