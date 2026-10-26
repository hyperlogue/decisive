use super::{Error, Request, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TuningMode {
    #[default]
    Auto,
    Off,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct TuningOptions {
    pub mode: TuningMode,
    pub budget_ms: u32,
    // `info.tuning.cache` from an earlier load; used only while its key still matches.
    pub cache: Option<Value>,
}
impl Default for TuningOptions {
    fn default() -> Self {
        Self {
            mode: TuningMode::Auto,
            budget_ms: 1000,
            cache: None,
        }
    }
}

pub(crate) struct TuningBudget {
    enabled: bool,
    deadline: Option<f64>,
}
impl TuningBudget {
    pub fn new(enabled: bool, budget_ms: Option<u32>, start: f64) -> Self {
        Self {
            enabled,
            deadline: budget_ms.map(|budget| start + f64::from(budget)),
        }
    }
    pub fn deadline(&self) -> Option<f64> {
        self.deadline
    }
    pub fn permits(&self, now: f64) -> bool {
        self.enabled && self.deadline.is_none_or(|deadline| now < deadline)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct DecimaParameters {
    pub long: bool,
    pub window: Option<usize>,
    pub overlap: Option<usize>,
    pub lang: Option<String>,
    pub min_confidence: Option<f64>,
}
impl DecimaParameters {
    pub(crate) fn request(&self, request: &Request) -> Value {
        let mut value = request.value();
        if self.long {
            value["long"] = json!(true);
        }
        if let Some(window) = self.window {
            value["window"] = json!(window);
        }
        if let Some(overlap) = self.overlap {
            value["overlap"] = json!(overlap);
        }
        if let Some(lang) = &self.lang {
            value["lang"] = json!(lang);
        }
        if let Some(min) = self.min_confidence {
            value["min_confidence"] = json!(min);
        }
        value
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct PplxParameters {
    pub max_sequence_tokens: usize,
}
impl Default for PplxParameters {
    fn default() -> Self {
        Self {
            max_sequence_tokens: 1024,
        }
    }
}
impl PplxParameters {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.max_sequence_tokens == 0 {
            return Err(Error::new(
                "INVALID_PARAMETERS",
                "maxSequenceTokens must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct LayaParameters {
    pub long: bool,
    pub window: Option<usize>,
    pub stride: Option<usize>,
    pub lang: Option<String>,
    pub min_confidence: Option<f64>,
}
impl LayaParameters {
    pub(crate) fn request(&self, request: &Request) -> Value {
        let mut value = request.value();
        if self.long {
            value["long"] = json!(true);
        }
        if let Some(window) = self.window {
            value["window"] = json!(window);
        }
        if let Some(stride) = self.stride {
            value["stride"] = json!(stride);
        }
        if let Some(lang) = &self.lang {
            value["lang"] = json!(lang);
        }
        if let Some(min) = self.min_confidence {
            value["min_confidence"] = json!(min);
        }
        value
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct KevParameters {}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct JevParameters {}
