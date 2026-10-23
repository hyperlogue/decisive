mod parameters;
pub use parameters::*;
mod adapters;
#[cfg(feature = "wasm")]
pub(crate) mod browser;
#[cfg(any(feature = "wasm", test))]
mod support;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub state: Value,
    pub questions: Map<String, Value>,
}

impl Request {
    pub fn validate(&self) -> Result<()> {
        if !matches!(
            self.state,
            Value::String(_) | Value::Object(_) | Value::Array(_)
        ) || self.questions.is_empty()
        {
            return Err(Error::new(
                "INVALID_REQUEST",
                "state and nonempty questions are required",
            ));
        }
        Ok(())
    }
    pub(crate) fn value(&self) -> Value {
        serde_json::json!({"state":self.state,"questions":self.questions})
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Error {
    pub code: String,
    pub message: String,
}
impl Error {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
    pub(crate) fn engine(error: impl std::fmt::Display) -> Self {
        let message = error.to_string();
        let code = if message.contains("STATE_TRUNCATED")
            || message.contains("exceeds token limit")
            || message.contains("input budget exceeded")
        {
            "INPUT_TOO_LONG"
        } else if message.contains("required WebGPU precision is unavailable") {
            "MISSING_GPU_FEATURE"
        } else if message.contains("no WebGPU") || message.contains("request WebGPU device") {
            "GPU_UNAVAILABLE"
        } else if message.contains("exceeds device limits") {
            "RESOURCE_LIMIT"
        } else if message.contains("the fp16 Cargo feature")
            || message.contains("the int8 Cargo feature")
        {
            "FORMAT_NOT_COMPILED"
        } else {
            "INFERENCE_FAILED"
        };
        Self::new(code, message)
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn check(&self) -> Result<()> {
        if self.0.load(Ordering::Relaxed) {
            Err(Error::new("CANCELLED", "Inference was cancelled"))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct CacheOptions {
    pub max_bytes: usize,
    pub require_prefix: bool,
}
impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024 * 1024,
            require_prefix: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    pub model: &'static str,
    pub backend: &'static str,
    pub caches: Vec<&'static str>,
    pub tuning: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct Response {
    pub response: Value,
    pub model: String,
    pub backend: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum Outcome {
    Success { result: Response },
    Error { error: Error },
}

#[allow(async_fn_in_trait)]
pub trait Backend {
    type Parameters: Default;
    type Cache;
    fn capabilities(&self) -> Capabilities;
    fn create_cache(&self, options: &CacheOptions) -> Result<Self::Cache>;
    fn clear_cache(&self, cache: &mut Self::Cache);
    async fn predict(
        &mut self,
        request: &Request,
        parameters: &Self::Parameters,
        cache: &mut Self::Cache,
        cancellation: &Cancellation,
    ) -> Result<Value>;
}

pub struct Model<B: Backend> {
    backend: B,
    identity: Arc<()>,
    defaults: B::Parameters,
}
pub struct Cache<B: Backend> {
    identity: Arc<()>,
    state: B::Cache,
}

impl<B: Backend> Model<B> {
    pub fn new(backend: B) -> Self {
        Self::with_parameters(backend, B::Parameters::default())
    }
    pub fn with_parameters(backend: B, defaults: B::Parameters) -> Self {
        Self {
            backend,
            identity: Arc::new(()),
            defaults,
        }
    }
    pub fn capabilities(&self) -> Capabilities {
        self.backend.capabilities()
    }
    pub fn create_cache(&self, options: CacheOptions) -> Result<Cache<B>> {
        if options.require_prefix && !self.capabilities().caches.contains(&"prefix") {
            return Err(Error::new(
                "UNSUPPORTED_CACHE",
                "This model/backend does not support prefix caching",
            ));
        }
        Ok(Cache {
            identity: self.identity.clone(),
            state: self.backend.create_cache(&options)?,
        })
    }
    pub fn clear_cache(&self, cache: &mut Cache<B>) -> Result<()> {
        self.check_cache(cache)?;
        self.backend.clear_cache(&mut cache.state);
        Ok(())
    }
    fn check_cache(&self, cache: &Cache<B>) -> Result<()> {
        if !Arc::ptr_eq(&self.identity, &cache.identity) {
            return Err(Error::new(
                "CACHE_OWNER",
                "Cache belongs to another model instance",
            ));
        }
        Ok(())
    }
    pub async fn predict(
        &mut self,
        request: &Request,
        parameters: Option<&B::Parameters>,
        cache: Option<&mut Cache<B>>,
        cancellation: &Cancellation,
    ) -> Result<Response> {
        let mut temporary;
        let cache = match cache {
            Some(cache) => cache,
            None => {
                temporary = self.create_cache(CacheOptions {
                    max_bytes: 0,
                    require_prefix: false,
                })?;
                &mut temporary
            }
        };
        self.check_cache(cache)?;
        cancellation.check()?;
        request.validate()?;
        let result = self
            .backend
            .predict(
                request,
                parameters.unwrap_or(&self.defaults),
                &mut cache.state,
                cancellation,
            )
            .await;
        let result = result.and_then(|value| {
            cancellation.check()?;
            Ok(value)
        });
        if result.is_err() {
            self.backend.clear_cache(&mut cache.state);
        }
        let capabilities = self.capabilities();
        Ok(Response {
            response: result?,
            model: capabilities.model.into(),
            backend: capabilities.backend.into(),
        })
    }
    pub async fn predict_batch(
        &mut self,
        requests: &[Request],
        parameters: Option<&B::Parameters>,
        mut cache: Option<&mut Cache<B>>,
        cancellation: &Cancellation,
    ) -> Result<Vec<Outcome>> {
        if let Some(cache) = &cache {
            self.check_cache(cache)?;
        }
        let mut outcomes = Vec::with_capacity(requests.len());
        for request in requests {
            outcomes.push(
                match self
                    .predict(request, parameters, cache.as_deref_mut(), cancellation)
                    .await
                {
                    Ok(result) => Outcome::Success { result },
                    Err(error) => Outcome::Error { error },
                },
            );
        }
        Ok(outcomes)
    }
}
