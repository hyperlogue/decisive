#[cfg(all(feature = "pplx-flash", feature = "webgpu"))]
mod pplx;
mod source;

use super::*;
use js_sys::{Function, Promise, Reflect};
use source::Source;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn now() -> f64;
    #[wasm_bindgen(js_namespace = globalThis, js_name = setTimeout)]
    fn timeout(callback: &Function, delay: u32);
}
pub(crate) async fn yield_browser() {
    let promise = Promise::new(&mut |resolve, _| timeout(&resolve, 0));
    let _ = JsFuture::from(promise).await;
}

#[wasm_bindgen(js_name = CheckpointHasher)]
pub struct CheckpointHasher(sha2::Sha256);
#[wasm_bindgen(js_class = CheckpointHasher)]
impl CheckpointHasher {
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
fn js_error(error: Error) -> JsValue {
    let value = js_sys::Error::new(&error.message);
    let _ = Reflect::set(&value, &"code".into(), &error.code.into());
    value.into()
}
fn from_js(error: JsValue) -> Error {
    Error::engine(
        error
            .as_string()
            .or_else(|| Reflect::get(&error, &"message".into()).ok()?.as_string())
            .unwrap_or_else(|| format!("{error:?}")),
    )
}
fn parse<T: serde::de::DeserializeOwned>(json: &str) -> Result<T> {
    serde_json::from_str(json).map_err(|e| Error::new("INVALID_PARAMETERS", e.to_string()))
}

#[wasm_bindgen(js_name = InferenceControl)]
pub struct Control {
    cancelled: Cancellation,
    progress: Option<Function>,
}
#[wasm_bindgen(js_class = InferenceControl)]
impl Control {
    #[wasm_bindgen(constructor)]
    pub fn new(progress: Option<Function>) -> Self {
        Self {
            cancelled: Cancellation::default(),
            progress,
        }
    }
    pub fn cancel(&self) {
        self.cancelled.cancel();
    }
}
impl Control {
    fn check(&self) -> Result<()> {
        self.cancelled.check()
    }
    fn progress(&self, phase: &str, completed: usize, total: usize) -> Result<()> {
        self.check()?;
        if let Some(callback) = &self.progress {
            let event = serde_json::json!({"phase":phase,"completed":completed,"total":total});
            callback
                .call1(&JsValue::NULL, &event.to_string().into())
                .map_err(from_js)?;
        }
        self.check()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Configuration {
    model: String,
    backend: String,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    parameters: Value,
    #[serde(default)]
    tuning: TuningOptions,
}

enum Engine {
    #[cfg(feature = "decima")]
    Decima(crate::decima::web::BrowserModel),
    #[cfg(all(feature = "decima", feature = "webgpu"))]
    DecimaGpu(crate::decima::webgpu::browser::BrowserModel),
    #[cfg(feature = "laya")]
    Laya(crate::laya::web::BrowserModel),
    #[cfg(all(feature = "laya", feature = "webgpu"))]
    LayaGpu(crate::laya::webgpu::browser::BrowserModel),
    #[cfg(feature = "kev")]
    Kev(crate::kev::Model),
    #[cfg(feature = "jev-style")]
    Jev(crate::jev_style::Model),
    #[cfg(all(feature = "pplx-flash", feature = "webgpu"))]
    Pplx(pplx::Runtime),
}
enum CacheState {
    #[cfg(feature = "decima")]
    Decima(crate::decima::Session),
    #[cfg(all(feature = "decima", feature = "webgpu"))]
    DecimaGpu(crate::decima::webgpu::Session),
    #[cfg(feature = "laya")]
    Laya(crate::laya::Scratch),
    #[cfg(feature = "kev")]
    Kev(crate::kev::Session),
    #[cfg(feature = "jev-style")]
    Jev(crate::jev_style::Session),
    #[cfg(all(feature = "pplx-flash", feature = "webgpu"))]
    Pplx(pplx::EmbeddingCache),
    Empty,
}
impl CacheState {
    fn clear(&mut self) {
        match self {
            #[cfg(feature = "decima")]
            Self::Decima(cache) => cache.clear_cache(),
            #[cfg(all(feature = "decima", feature = "webgpu"))]
            Self::DecimaGpu(cache) => cache.clear_cache(),
            #[cfg(feature = "laya")]
            Self::Laya(cache) => {
                cache.clear_state_cache();
                cache.clear_token_cache();
            }
            #[cfg(feature = "kev")]
            Self::Kev(cache) => cache.clear_cache(),
            #[cfg(feature = "jev-style")]
            Self::Jev(cache) => cache.clear_cache(),
            #[cfg(all(feature = "pplx-flash", feature = "webgpu"))]
            Self::Pplx(cache) => cache.clear(),
            Self::Empty => {}
        }
    }
}

#[wasm_bindgen(js_name = InferenceCache)]
pub struct BrowserCache {
    owner: Rc<Cell<bool>>,
    state: RefCell<Option<CacheState>>,
    closed: Cell<bool>,
}
#[wasm_bindgen(js_class = InferenceCache)]
impl BrowserCache {
    pub fn clear(&self) -> std::result::Result<(), JsValue> {
        if self.closed.get() || !self.owner.get() {
            return Err(js_error(Error::new("DISPOSED", "Cache is disposed")));
        }
        self.state
            .borrow_mut()
            .as_mut()
            .ok_or_else(|| js_error(Error::new("CACHE_BUSY", "Cache is in use")))?
            .clear();
        Ok(())
    }
    pub fn dispose(&self) -> std::result::Result<(), JsValue> {
        if self.closed.get() {
            return Ok(());
        }
        if self.state.borrow().is_none() {
            return Err(js_error(Error::new("CACHE_BUSY", "Cache is in use")));
        }
        self.state.borrow_mut().take();
        self.closed.set(true);
        Ok(())
    }
}

struct Lease<'a, T> {
    slot: &'a RefCell<Option<T>>,
    value: Option<T>,
}
impl<'a, T> Lease<'a, T> {
    fn take(slot: &'a RefCell<Option<T>>, code: &str) -> Result<Self> {
        Ok(Self {
            slot,
            value: Some(
                slot.borrow_mut()
                    .take()
                    .ok_or_else(|| Error::new(code, "Object is already in use"))?,
            ),
        })
    }
    fn get(&mut self) -> &mut T {
        self.value.as_mut().unwrap()
    }
}
impl<T> Drop for Lease<'_, T> {
    fn drop(&mut self) {
        *self.slot.borrow_mut() = self.value.take();
    }
}

#[wasm_bindgen(js_name = DecisiveModel)]
pub struct BrowserModel {
    engine: RefCell<Option<Engine>>,
    owner: Rc<Cell<bool>>,
    configuration: Configuration,
}

#[wasm_bindgen(js_class = DecisiveModel)]
impl BrowserModel {
    pub async fn load(
        configuration: &str,
        files: JsValue,
        control: &Control,
    ) -> std::result::Result<BrowserModel, JsValue> {
        let configuration: Configuration = parse(configuration).map_err(js_error)?;
        support::browser_selection(
            &configuration.model,
            &configuration.backend,
            configuration.format.as_deref(),
        )
        .map_err(js_error)?;
        validate_parameters(
            &configuration.model,
            &if configuration.parameters.is_null() {
                serde_json::json!({})
            } else {
                configuration.parameters.clone()
            },
        )
        .map_err(js_error)?;
        control.progress("loading", 0, 1).map_err(js_error)?;
        let source = Source::new(files);
        let engine = Engine::load(&configuration, &source, control)
            .await
            .map_err(|error| {
                js_error(support::load_error(
                    error,
                    &configuration.model,
                    &configuration.backend,
                    configuration.format.as_deref(),
                ))
            })?;
        control.check().map_err(js_error)?;
        let model = Self {
            engine: RefCell::new(Some(engine)),
            owner: Rc::new(Cell::new(true)),
            configuration,
        };
        model
            .validate_parameters(&model.defaults())
            .map_err(js_error)?;
        Ok(model)
    }
    #[wasm_bindgen(js_name = createCache)]
    pub fn create_cache(&self, options: &str) -> std::result::Result<BrowserCache, JsValue> {
        self.check().map_err(js_error)?;
        let options: CacheOptions = parse(options).map_err(js_error)?;
        let state = self
            .engine
            .borrow()
            .as_ref()
            .ok_or_else(|| js_error(Error::new("MODEL_BUSY", "Model is in use")))?
            .cache(&options)
            .map_err(js_error)?;
        Ok(BrowserCache {
            owner: self.owner.clone(),
            state: RefCell::new(Some(state)),
            closed: Cell::new(false),
        })
    }
    pub fn info(&self) -> std::result::Result<String, JsValue> {
        self.check().map_err(js_error)?;
        let engine = self.engine.borrow();
        let engine = engine
            .as_ref()
            .ok_or_else(|| js_error(Error::new("MODEL_BUSY", "Model is in use")))?;
        Ok(serde_json::json!({"model":self.configuration.model,"backend":self.configuration.backend,
            "caches":engine.caches(),"tuning":engine.tuning(),"parameters":self.defaults()}).to_string())
    }
    pub async fn predict(
        &self,
        requests: &str,
        parameters: &str,
        cache: &BrowserCache,
        control: &Control,
        batch: bool,
    ) -> std::result::Result<String, JsValue> {
        self.run(requests, parameters, cache, control, batch)
            .await
            .map_err(js_error)
    }
    pub fn dispose(&self) -> std::result::Result<(), JsValue> {
        if !self.owner.get() {
            return Ok(());
        }
        if self.engine.borrow().is_none() {
            return Err(js_error(Error::new("MODEL_BUSY", "Model is in use")));
        }
        self.engine.borrow_mut().take();
        self.owner.set(false);
        Ok(())
    }
}

impl Engine {
    async fn load(config: &Configuration, source: &Source, control: &Control) -> Result<Self> {
        control.check()?;
        match (config.model.as_str(), config.backend.as_str()) {
            #[cfg(feature = "decima")]
            ("decima", backend @ ("wasm" | "webgpu")) => {
                if backend == "webgpu" && !matches!(config.format.as_deref(), None | Some("f32")) {
                    return Err(Error::new("UNSUPPORTED_FORMAT", "Decima WebGPU uses f32"));
                }
                let prefix = if source.file("pytorch/decima.json").is_ok() {
                    "pytorch/"
                } else {
                    ""
                };
                let configuration = source.read(&format!("{prefix}decima.json")).await?;
                let encoder = source.read(&format!("{prefix}encoder/config.json")).await?;
                let tokenizer = source
                    .read(&format!("{prefix}encoder/tokenizer.json"))
                    .await?;
                let head = source.read(&format!("{prefix}head.safetensors")).await?;
                control.check()?;
                if backend == "webgpu" {
                    #[cfg(feature = "webgpu")]
                    return crate::decima::webgpu::browser::BrowserModel::create(
                        configuration,
                        encoder,
                        tokenizer,
                        source.file(&format!("{prefix}encoder/model.safetensors"))?,
                        head,
                    )
                    .await
                    .map(Self::DecimaGpu)
                    .map_err(from_js);
                    #[cfg(not(feature = "webgpu"))]
                    return Err(Error::new("UNSUPPORTED_BACKEND", "WebGPU was not compiled"));
                }
                let weights = source
                    .read(&format!("{prefix}encoder/model.safetensors"))
                    .await?;
                crate::decima::web::BrowserModel::new(
                    &configuration,
                    &encoder,
                    &tokenizer,
                    &weights,
                    &head,
                    config.format.as_deref().unwrap_or("f32"),
                )
                .map(Self::Decima)
                .map_err(from_js)
            }
            #[cfg(feature = "laya")]
            ("laya", backend @ ("wasm" | "webgpu")) => {
                let encoder = source.read("encoder/config.json").await?;
                let agent = source.read("rl_agent_config.json").await?;
                let tokenizer = source.read("tokenizer/tokenizer.json").await?;
                let tokenizer_config = source.read("tokenizer/tokenizer_config.json").await?;
                let weights = source.read("model.safetensors").await?;
                let fp16 = match config.format.as_deref() {
                    None | Some("f32") => false,
                    Some("f16") => true,
                    _ => return Err(Error::new("UNSUPPORTED_FORMAT", "Laya expects f32 or f16")),
                };
                control.check()?;
                if backend == "webgpu" {
                    #[cfg(feature = "webgpu")]
                    return crate::laya::webgpu::browser::BrowserModel::create(
                        encoder,
                        agent,
                        tokenizer,
                        tokenizer_config,
                        weights,
                        fp16,
                    )
                    .await
                    .map(Self::LayaGpu)
                    .map_err(from_js);
                    #[cfg(not(feature = "webgpu"))]
                    return Err(Error::new("UNSUPPORTED_BACKEND", "WebGPU was not compiled"));
                }
                crate::laya::web::BrowserModel::new(
                    &encoder,
                    &agent,
                    &tokenizer,
                    &tokenizer_config,
                    &weights,
                    fp16,
                )
                .map(Self::Laya)
                .map_err(from_js)
            }
            #[cfg(feature = "kev")]
            ("kev", "wasm") => {
                if !matches!(config.format.as_deref(), None | Some("f32")) {
                    return Err(Error::new("UNSUPPORTED_FORMAT", "Kev uses f32"));
                }
                source.resident_budget(
                    &[
                        "base/model.safetensors-00001-of-00001.safetensors",
                        "adapter_model.safetensors",
                    ],
                    3,
                )?;
                let base = source.read("base/config.json").await?;
                let tokenizer = source.read("tokenizer.json").await?;
                let adapter_config = source.read("adapter_config.json").await?;
                let adapter = source.read("adapter_model.safetensors").await?;
                let head = source.read("head.pt").await?;
                let weights = source
                    .read("base/model.safetensors-00001-of-00001.safetensors")
                    .await?;
                control.check()?;
                crate::kev::Model::from_bytes(
                    &base,
                    &tokenizer,
                    &adapter_config,
                    &adapter,
                    &head,
                    &weights,
                    true,
                )
                .map(Self::Kev)
                .map_err(Error::engine)
            }
            #[cfg(feature = "jev-style")]
            ("jev-style", "wasm") => {
                if !matches!(config.format.as_deref(), None | Some("f32")) {
                    return Err(Error::new("UNSUPPORTED_FORMAT", "Jev-style uses f32"));
                }
                source.resident_budget(&["model.safetensors"], 3)?;
                let config = source.read("config.json").await?;
                let tokenizer = source.read("tokenizer.json").await?;
                let readout = source.read("readout_config.json").await?;
                let weights = source.read("model.safetensors").await?;
                control.check()?;
                crate::jev_style::Model::from_bytes(&config, &tokenizer, &readout, &weights, true)
                    .map(Self::Jev)
                    .map_err(Error::engine)
            }
            #[cfg(all(feature = "pplx-flash", feature = "webgpu"))]
            ("pplx-flash", "webgpu" | "wasm") => {
                pplx::Runtime::load(source.clone(), config, control)
                    .await
                    .map(Self::Pplx)
            }
            _ => Err(Error::new(
                "UNSUPPORTED_BACKEND",
                "Model/backend is unavailable in this build",
            )),
        }
    }
    fn caches(&self) -> Vec<&'static str> {
        match self {
            #[cfg(feature = "decima")]
            Self::Decima(_) => vec!["choice"],
            #[cfg(all(feature = "decima", feature = "webgpu"))]
            Self::DecimaGpu(_) => vec!["choice"],
            #[cfg(feature = "kev")]
            Self::Kev(_) => vec!["prefix"],
            #[cfg(feature = "jev-style")]
            Self::Jev(_) => vec!["prefix"],
            #[cfg(all(feature = "pplx-flash", feature = "webgpu"))]
            Self::Pplx(_) => vec!["embedding"],
            _ => vec![],
        }
    }
    fn tuning(&self) -> Value {
        match self {
            #[cfg(all(feature = "pplx-flash", feature = "webgpu"))]
            Self::Pplx(runtime) => runtime.info(),
            _ => serde_json::json!({"mode":"heuristic"}),
        }
    }
    fn cache(&self, options: &CacheOptions) -> Result<CacheState> {
        if options.require_prefix && !self.caches().contains(&"prefix") {
            return Err(Error::new(
                "UNSUPPORTED_CACHE",
                "Prefix caching is unavailable",
            ));
        }
        Ok(match self {
            #[cfg(feature = "decima")]
            Self::Decima(_) => {
                let mut c = crate::decima::Session::default();
                c.set_cache_bytes(options.max_bytes);
                CacheState::Decima(c)
            }
            #[cfg(all(feature = "decima", feature = "webgpu"))]
            Self::DecimaGpu(_) => {
                let mut c = crate::decima::webgpu::Session::default();
                c.set_cache_bytes(options.max_bytes);
                CacheState::DecimaGpu(c)
            }
            #[cfg(feature = "laya")]
            Self::Laya(_) => {
                CacheState::Laya(crate::laya::web::inference_scratch().map_err(Error::engine)?)
            }
            #[cfg(feature = "kev")]
            Self::Kev(_) => {
                let mut c = crate::kev::Session::default();
                c.set_cache_bytes(options.max_bytes);
                CacheState::Kev(c)
            }
            #[cfg(feature = "jev-style")]
            Self::Jev(_) => {
                let mut c = crate::jev_style::Session::default();
                c.set_cache_bytes(options.max_bytes);
                CacheState::Jev(c)
            }
            #[cfg(all(feature = "pplx-flash", feature = "webgpu"))]
            Self::Pplx(_) => CacheState::Pplx(pplx::EmbeddingCache::new(options.max_bytes)),
            _ => CacheState::Empty,
        })
    }
    async fn predict(
        &mut self,
        request: &Request,
        parameters: &Value,
        cache: &mut CacheState,
        control: &Control,
    ) -> Result<Value> {
        let json = parameters.to_string();
        control.progress("analyzing", 0, 1)?;
        yield_browser().await;
        control.check()?;
        match (self, cache) {
            #[cfg(feature = "decima")]
            (Self::Decima(model), CacheState::Decima(cache)) => model
                .model
                .predict_json(&parse::<DecimaParameters>(&json)?.request(request), cache)
                .map_err(Error::engine),
            #[cfg(all(feature = "decima", feature = "webgpu"))]
            (Self::DecimaGpu(model), CacheState::DecimaGpu(cache)) => model
                .model
                .predict_json(&parse::<DecimaParameters>(&json)?.request(request), cache)
                .await
                .map_err(Error::engine),
            #[cfg(feature = "laya")]
            (Self::Laya(model), CacheState::Laya(cache)) => model
                .model
                .predict_json(parse::<LayaParameters>(&json)?.request(request), cache)
                .map_err(Error::engine),
            #[cfg(all(feature = "laya", feature = "webgpu"))]
            (Self::LayaGpu(model), CacheState::Empty) => model
                .model
                .predict_json(parse::<LayaParameters>(&json)?.request(request))
                .await
                .map_err(Error::engine),
            #[cfg(feature = "kev")]
            (Self::Kev(model), CacheState::Kev(cache)) => model
                .predict_json(&request.value(), cache)
                .map_err(Error::engine),
            #[cfg(feature = "jev-style")]
            (Self::Jev(model), CacheState::Jev(cache)) => model
                .predict_json(&request.value(), cache)
                .map_err(Error::engine),
            #[cfg(all(feature = "pplx-flash", feature = "webgpu"))]
            (Self::Pplx(runtime), CacheState::Pplx(cache)) => {
                runtime
                    .predict(request, &parse::<PplxParameters>(&json)?, cache, control)
                    .await
            }
            _ => Err(Error::new(
                "CACHE_OWNER",
                "Cache implementation does not match the model",
            )),
        }
    }
}
impl Drop for BrowserModel {
    fn drop(&mut self) {
        self.owner.set(false);
    }
}

impl BrowserModel {
    fn check(&self) -> Result<()> {
        if self.owner.get() {
            Ok(())
        } else {
            Err(Error::new("DISPOSED", "Model is disposed"))
        }
    }
    fn defaults(&self) -> Value {
        if self.configuration.parameters.is_null() {
            serde_json::json!({})
        } else {
            self.configuration.parameters.clone()
        }
    }
    fn validate_parameters(&self, parameters: &Value) -> Result<()> {
        validate_parameters(&self.configuration.model, parameters)
    }
    async fn run(
        &self,
        requests: &str,
        parameters: &str,
        cache: &BrowserCache,
        control: &Control,
        batch: bool,
    ) -> Result<String> {
        self.check()?;
        if !Rc::ptr_eq(&self.owner, &cache.owner) {
            return Err(Error::new("CACHE_OWNER", "Cache belongs to another model"));
        }
        if cache.closed.get() {
            return Err(Error::new("DISPOSED", "Cache is disposed"));
        }
        let mut state = Lease::take(&cache.state, "CACHE_BUSY")?;
        let mut engine = Lease::take(&self.engine, "MODEL_BUSY")?;
        let supplied: Value = parse(parameters)?;
        let mut parameters = self.defaults();
        if !supplied.is_object() {
            return Err(Error::new(
                "INVALID_PARAMETERS",
                "Parameters must be an object",
            ));
        }
        let defaults = parameters
            .as_object_mut()
            .ok_or_else(|| Error::new("INVALID_PARAMETERS", "Parameters must be an object"))?;
        defaults.extend(supplied.as_object().unwrap().clone());
        self.validate_parameters(&parameters)?;
        let requests: Vec<Value> = if batch {
            parse(requests)?
        } else {
            vec![parse(requests)?]
        };
        let mut outcomes = Vec::with_capacity(requests.len());
        for value in requests {
            let start = now();
            let output = async {
                control.check()?;
                let request: Request = serde_json::from_value(value)
                    .map_err(|e| Error::new("INVALID_REQUEST", e.to_string()))?;
                request.validate()?;
                engine
                    .get()
                    .predict(&request, &parameters, state.get(), control)
                    .await
            }
            .await
            .and_then(|response| {
                control.check()?;
                Ok(response)
            });
            match output {
                Ok(response) => {
                    let result = serde_json::json!({"response":response,"model":self.configuration.model,"backend":self.configuration.backend,"elapsed_ms":now()-start});
                    if !batch {
                        return Ok(result.to_string());
                    }
                    outcomes.push(serde_json::json!({"status":"success","result":result}));
                }
                Err(error) => {
                    state.get().clear();
                    if !batch {
                        return Err(error);
                    }
                    outcomes.push(serde_json::json!({"status":"error","error":error}));
                }
            }
            yield_browser().await;
        }
        Ok(Value::Array(outcomes).to_string())
    }
}

fn validate_parameters(model: &str, parameters: &Value) -> Result<()> {
    let json = parameters.to_string();
    match model {
        "decima" => {
            let _: DecimaParameters = parse(&json)?;
        }
        "laya" => {
            let _: LayaParameters = parse(&json)?;
        }
        "pplx-flash" => {
            parse::<PplxParameters>(&json)?.validate()?;
        }
        "kev" => {
            let _: KevParameters = parse(&json)?;
        }
        "jev-style" => {
            let _: JevParameters = parse(&json)?;
        }
        _ => return Err(Error::new("UNSUPPORTED_MODEL", "Unknown model")),
    }
    Ok(())
}
