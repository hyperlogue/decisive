use super::{Kernel, Model, Options, Precision, Session};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name=DecimaModel)]
pub struct BrowserModel {
    model: Model,
    session: Session,
}
fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}
#[wasm_bindgen(js_class=DecimaModel)]
impl BrowserModel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: &[u8],
        encoder_config: &[u8],
        tokenizer: &[u8],
        encoder_weights: &[u8],
        head_weights: &[u8],
        precision: &str,
    ) -> Result<BrowserModel, JsValue> {
        let precision = match precision {
            "f32" => Precision::F32,
            "int8" => Precision::Int8,
            "int8-weights" => Precision::Int8Weights,
            "int8-full" => Precision::Int8Full,
            "int8-compact" => Precision::Int8Compact,
            _ => {
                return Err(error(
                    "precision must be f32, int8, int8-weights, int8-full or int8-compact",
                ));
            }
        };
        Ok(Self {
            model: Model::from_bytes(
                config,
                encoder_config,
                tokenizer,
                encoder_weights,
                head_weights,
                Options {
                    precision,
                    kernel: Kernel::Auto,
                },
            )
            .map_err(error)?,
            session: Session::default(),
        })
    }
    pub fn predict(&mut self, request_json: &str) -> Result<String, JsValue> {
        let request = serde_json::from_str(request_json).map_err(error)?;
        let result = self
            .model
            .predict_json(&request, &mut self.session)
            .map_err(error)?;
        serde_json::to_string(&result).map_err(error)
    }
    pub fn decide(&mut self, state: &str, question_json: &str) -> Result<String, JsValue> {
        let question = serde_json::from_str(question_json).map_err(error)?;
        serde_json::to_string(
            &self
                .model
                .decide(state, &question, &mut self.session)
                .map_err(error)?,
        )
        .map_err(error)
    }
    #[wasm_bindgen(js_name=setLengthRules)]
    pub fn set_length_rules(&mut self, json: &str) -> Result<(), JsValue> {
        let rules = serde_json::from_str(json).map_err(error)?;
        self.session.set_length_rules(rules).map_err(error)
    }
    #[wasm_bindgen(js_name=clearCache)]
    pub fn clear_cache(&mut self) {
        self.session.clear_cache();
    }
    #[wasm_bindgen(js_name=setCacheBytes)]
    pub fn set_cache_bytes(&mut self, bytes: usize) {
        self.session.set_cache_bytes(bytes);
    }
    #[cfg(all(
        feature = "kernel-checks",
        target_arch = "wasm32",
        target_feature = "simd128"
    ))]
    #[wasm_bindgen(js_name=checkEncoderSegments)]
    pub fn check_encoder_segments(&self) -> Result<String, JsValue> {
        let result = self.model.check_encoder_segments().map_err(error)?;
        serde_json::to_string(&result).map_err(error)
    }
}
