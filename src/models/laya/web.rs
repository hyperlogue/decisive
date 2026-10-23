use crate::laya::{Model, Scratch, weights::Storage};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[wasm_bindgen]
pub struct BrowserModel {
    pub(crate) model: Model,
    pub(crate) scratch: Scratch,
}

#[wasm_bindgen]
#[derive(Default)]
pub struct BrowserRuntime {
    resident: Option<BrowserModel>,
    source: Option<(String, bool)>,
}

#[wasm_bindgen]
impl BrowserRuntime {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn load(&mut self, base_url: String, fp16: bool) -> Result<(), JsValue> {
        self.resident = None;
        self.source = None;
        let window =
            web_sys::window().ok_or_else(|| JsValue::from_str("browser window unavailable"))?;
        let cache = match window.caches() {
            Ok(storage) => JsFuture::from(storage.open("beam-checkpoints-v1"))
                .await
                .ok()
                .and_then(|v| v.dyn_into::<web_sys::Cache>().ok()),
            Err(_) => None,
        };
        let base = base_url.trim_end_matches('/');
        let encoder = download(
            &window,
            cache.as_ref(),
            &format!("{base}/encoder/config.json"),
        )
        .await?;
        let config = download(
            &window,
            cache.as_ref(),
            &format!("{base}/rl_agent_config.json"),
        )
        .await?;
        let tokenizer = download(
            &window,
            cache.as_ref(),
            &format!("{base}/tokenizer/tokenizer.json"),
        )
        .await?;
        let tokenizer_config = download(
            &window,
            cache.as_ref(),
            &format!("{base}/tokenizer/tokenizer_config.json"),
        )
        .await?;
        let weights = download(
            &window,
            cache.as_ref(),
            &format!("{base}/model.safetensors"),
        )
        .await?;
        self.resident = Some(BrowserModel::new(
            &encoder,
            &config,
            &tokenizer,
            &tokenizer_config,
            &weights,
            fp16,
        )?);
        self.source = Some((base_url, fp16));
        Ok(())
    }

    pub fn unload(&mut self) {
        self.resident = None;
        self.source = None;
    }
    pub fn set_token_cache_capacity(&mut self, tokens: usize) -> Result<(), JsValue> {
        self.resident
            .as_mut()
            .ok_or_else(|| JsValue::from_str("load a checkpoint before configuring its cache"))?
            .set_token_cache_capacity(tokens);
        Ok(())
    }
    pub fn set_state_cache_capacity(&mut self, tokens: usize) -> Result<(), JsValue> {
        self.resident
            .as_mut()
            .ok_or_else(|| JsValue::from_str("load a checkpoint before configuring its cache"))?
            .set_state_cache_capacity(tokens);
        Ok(())
    }
    pub fn token_cache_stats(&self) -> Result<String, JsValue> {
        self.resident
            .as_ref()
            .ok_or_else(|| JsValue::from_str("load a checkpoint before inspecting its cache"))
            .map(BrowserModel::token_cache_stats)
    }
    pub fn state_cache_stats(&self) -> Result<String, JsValue> {
        self.resident
            .as_ref()
            .ok_or_else(|| JsValue::from_str("load a checkpoint before inspecting its cache"))
            .map(BrowserModel::state_cache_stats)
    }

    pub async fn predict_auto(
        &mut self,
        request: String,
        base_urls: String,
        fp16: bool,
        auto_task_detection: bool,
    ) -> Result<String, JsValue> {
        let request = serde_json::from_str(&request).map_err(js_error)?;
        let urls = serde_json::from_str(&base_urls).map_err(js_error)?;
        self.predict_routed(request, &urls, fp16, auto_task_detection)
            .await
            .map(|v| v.to_string())
    }

    pub fn predict(&mut self, request: &str) -> Result<String, JsValue> {
        self.resident
            .as_mut()
            .ok_or_else(|| JsValue::from_str("load a checkpoint before predicting"))?
            .predict(request)
    }
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

impl BrowserRuntime {
    async fn predict_routed(
        &mut self,
        mut request: serde_json::Value,
        urls: &serde_json::Value,
        fp16: bool,
        auto_task: bool,
    ) -> Result<serde_json::Value, JsValue> {
        use serde_json::Value;
        if let Some(requests) = request.as_array() {
            let mut outputs = Vec::with_capacity(requests.len());
            for r in requests {
                outputs
                    .push(Box::pin(self.predict_routed(r.clone(), urls, fp16, auto_task)).await?);
            }
            return Ok(Value::Array(outputs));
        }
        if let Some(states) = request.get("states").and_then(Value::as_array).cloned() {
            request.as_object_mut().unwrap().remove("states");
            let mut outputs = Vec::with_capacity(states.len());
            for state in states {
                let mut r = request.clone();
                r["state"] = state;
                outputs.push(Box::pin(self.predict_routed(r, urls, fp16, auto_task)).await?);
            }
            return Ok(Value::Array(outputs));
        }
        let decision = crate::laya::router::route(
            &request,
            crate::laya::router::Variant::Multilingual,
            auto_task,
        )
        .map_err(js_error)?;
        let url = urls[decision.model.name()]
            .as_str()
            .ok_or_else(|| JsValue::from_str("selected checkpoint is missing from base_urls"))?;
        if self
            .source
            .as_ref()
            .is_none_or(|(source, half)| source != url || *half != fp16)
        {
            self.load(url.to_owned(), fp16).await?;
        }
        let resident = self.resident.as_mut().unwrap();
        let mut result = resident
            .model
            .predict_json(request.clone(), &mut resident.scratch)
            .map_err(js_error)?;
        crate::laya::router::attach(&mut result, &request, &decision).map_err(js_error)?;
        Ok(result)
    }
}

async fn download(
    window: &web_sys::Window,
    cache: Option<&web_sys::Cache>,
    url: &str,
) -> Result<Vec<u8>, JsValue> {
    let cached = if let Some(cache) = cache {
        JsFuture::from(cache.match_with_str(url))
            .await
            .ok()
            .and_then(|v| v.dyn_into::<web_sys::Response>().ok())
    } else {
        None
    };
    let response = if let Some(response) = cached {
        response
    } else {
        let response: web_sys::Response = JsFuture::from(window.fetch_with_str(url))
            .await?
            .dyn_into()?;
        if !response.ok() {
            return Err(JsValue::from_str(&format!(
                "download {url}: HTTP {}",
                response.status()
            )));
        }
        if let Some(cache) = cache {
            if let Ok(copy) = response.clone() {
                let _ = JsFuture::from(cache.put_with_str(url, &copy)).await;
            }
        }
        response
    };
    let buffer = JsFuture::from(response.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

#[wasm_bindgen]
impl BrowserModel {
    #[cfg(feature = "int8")]
    pub fn quantize_encoder(&mut self, activations: bool, block: usize) -> Result<usize, JsValue> {
        self.model
            .quantize_encoder(
                if activations {
                    crate::laya::cpu::int8::Mode::WeightsActivations
                } else {
                    crate::laya::cpu::int8::Mode::Weights
                },
                block,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(constructor)]
    pub fn new(
        encoder_config: &[u8],
        agent_config: &[u8],
        tokenizer: &[u8],
        tokenizer_config: &[u8],
        weights: &[u8],
        fp16: bool,
    ) -> Result<BrowserModel, JsValue> {
        let kernel = crate::laya::cpu::Kernel::detect();
        let model = Model::from_bytes_with_kernel(
            encoder_config,
            agent_config,
            tokenizer,
            tokenizer_config,
            weights,
            if fp16 { Storage::F16 } else { Storage::F32 },
            kernel,
        )
        .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
        let scratch = inference_scratch().map_err(js_error)?;
        Ok(Self { model, scratch })
    }
    #[wasm_bindgen(js_name=setLengthRules)]
    pub fn set_length_rules(&mut self, json: &str) -> Result<(), JsValue> {
        self.scratch
            .set_length_rules(serde_json::from_str(json).map_err(js_error)?)
            .map_err(js_error)
    }
    pub fn predict(&mut self, request: &str) -> Result<String, JsValue> {
        let request: serde_json::Value =
            serde_json::from_str(request).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.model
            .predict_json(request, &mut self.scratch)
            .map(|v| v.to_string())
            .map_err(|e| JsValue::from_str(&format!("{e:#}")))
    }
    pub fn set_token_cache_capacity(&mut self, tokens: usize) {
        self.scratch.set_token_cache_capacity(tokens);
    }
    pub fn token_cache_stats(&self) -> String {
        serde_json::to_string(&self.scratch.token_cache_stats()).unwrap()
    }
    pub fn set_state_cache_capacity(&mut self, tokens: usize) {
        self.scratch.set_state_cache_capacity(tokens);
    }
    pub fn state_cache_stats(&self) -> String {
        serde_json::to_string(&self.scratch.state_cache_stats()).unwrap()
    }
}

pub(crate) fn inference_scratch() -> anyhow::Result<Scratch> {
    let kernel = crate::laya::cpu::Kernel::detect();
    let mut scratch = Scratch::default();
    if kernel == crate::laya::cpu::Kernel::WasmSimd {
        scratch.set_prune_head(true);
        scratch
        .set_length_rules(Some(crate::laya::specialization::wasm()))
        ?;
        scratch
        .set_optimizations(true, false, 1)
        ?;
    }
    Ok(scratch)
}
