use crate::laya::language::{self, Detection};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Variant {
    English,
    Multilingual,
    TypedDecisions,
}

impl Variant {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name.trim().to_lowercase().as_str() {
            "english" | "en" | "laya" | "default" => Self::English,
            "multilingual" | "multi" | "ml" | "laya-multilingual" => Self::Multilingual,
            "typed-decisions"
            | "typed"
            | "typed_decisions"
            | "laya-typed-decisions"
            | "decisions" => Self::TypedDecisions,
            _ => bail!("unknown checkpoint: {name}"),
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Multilingual => "multilingual",
            Self::TypedDecisions => "typed-decisions",
        }
    }
    pub fn directory(self) -> &'static str {
        match self {
            Self::English => "laya",
            Self::Multilingual => "laya-multilingual",
            Self::TypedDecisions => "laya-typed-decisions",
        }
    }
    pub fn repository(self) -> &'static str {
        match self {
            Self::English => "convaiinnovations/laya",
            Self::Multilingual => "convaiinnovations/laya-multilingual",
            Self::TypedDecisions => "convaiinnovations/laya-typed-decisions",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Decision {
    pub model: Variant,
    pub repo: &'static str,
    pub reason: String,
    pub detection: Option<Detection>,
    pub workflow: Option<&'static str>,
}

fn language_code(code: &str) -> Option<bool> {
    let code = code.trim().to_lowercase();
    let primary = code.split(['.', '_', '-']).next().unwrap_or("");
    match primary {
        "" | "c" | "posix" | "und" | "zxx" | "mul" => None,
        "en" | "eng" | "english" => Some(true),
        _ => Some(false),
    }
}

fn workflow(questions: &Value) -> Option<&'static str> {
    let questions = questions.as_object()?;
    for (name, fields) in [
        (
            "agent_trace_observability",
            ["action", "needs_review", "outcome", "risk", "urgency"],
        ),
        (
            "customer_service",
            ["action", "category", "churn_risk", "needs_human", "urgency"],
        ),
        (
            "invoice_processing",
            [
                "discrepancy_severity",
                "disposition",
                "duplicate",
                "matches_order",
                "urgency",
            ],
        ),
        (
            "security_incidents",
            [
                "credential_compromise",
                "disposition",
                "severity",
                "true_positive",
                "urgency",
            ],
        ),
    ] {
        if questions.len() == fields.len() && fields.iter().all(|f| questions.contains_key(*f)) {
            return Some(name);
        }
    }
    None
}

pub fn route(request: &Value, default: Variant, auto_task_detection: bool) -> Result<Decision> {
    let mut decision = Decision {
        model: default,
        repo: default.repository(),
        reason: String::new(),
        detection: None,
        workflow: None,
    };
    for field in ["model", "task"] {
        if let Some(name) = request.get(field).filter(|v| !v.is_null()) {
            let name = name.as_str().context("model and task must be strings")?;
            decision.model = Variant::parse(name)?;
            decision.repo = decision.model.repository();
            decision.reason = format!("explicit {field}={name:?}");
            return Ok(decision);
        }
    }
    let schema;
    let questions = if let Some(value) = request.get("schema") {
        schema = Value::Object(crate::laya::schema::Schema::parse(value)?.questions);
        &schema
    } else {
        &request["questions"]
    };
    decision.workflow = workflow(questions);
    if auto_task_detection && decision.workflow.is_some() {
        decision.model = Variant::TypedDecisions;
        decision.reason = format!(
            "question ids match the {:?} typed-decisions workflow",
            decision.workflow.unwrap()
        );
    } else if let Some((source, english)) = ["lang", "lang_guess"].into_iter().find_map(|key| {
        request[key]
            .as_str()
            .and_then(language_code)
            .map(|en| (key, en))
    }) {
        decision.model = if english {
            Variant::English
        } else {
            Variant::Multilingual
        };
        decision.reason = format!(
            "{source}: caller identified {} text",
            if english { "English" } else { "non-English" }
        );
    } else {
        let detected = language::analyse(&request["state"]);
        if detected.script == "unknown" {
            decision.reason = format!(
                "no letters detected in state; using default ({})",
                default.name()
            );
        } else if detected.script != "latin" {
            decision.model = Variant::Multilingual;
            decision.reason = format!("non-Latin script ({})", detected.script);
        } else if !detected.is_english {
            decision.model = Variant::Multilingual;
            decision.reason = format!(
                "non-English Latin text ({})",
                detected.language.as_deref().unwrap_or("unidentified")
            );
        } else if detected.language_undecided {
            decision.reason = format!(
                "Latin script, language not identified; using default ({})",
                default.name()
            );
        } else {
            decision.model = Variant::English;
            decision.reason = "English Latin text".into();
        }
        decision.detection = Some(detected);
    }
    decision.repo = decision.model.repository();
    Ok(decision)
}

pub fn attach(output: &mut Value, request: &Value, decision: &Decision) -> Result<()> {
    if request.get("schema").is_none() || request["return_details"].as_bool().unwrap_or(false) {
        output["routing"] = serde_json::to_value(decision)?;
    }
    Ok(())
}

#[cfg(feature = "cpu")]
pub struct Router {
    directory: std::path::PathBuf,
    storage: crate::laya::weights::Storage,
    kernel: crate::laya::cpu::Kernel,
    blas: Option<std::sync::Arc<crate::laya::cpu::Blas>>,
    capacity: usize,
    #[cfg(feature = "int8")]
    quantization: Option<(crate::laya::cpu::int8::Mode, usize)>,
    resident: std::collections::VecDeque<(Variant, crate::laya::Model)>,
    pub default: Variant,
    pub auto_task_detection: bool,
}

#[cfg(feature = "cpu")]
impl Router {
    pub fn new(
        directory: impl Into<std::path::PathBuf>,
        storage: crate::laya::weights::Storage,
        capacity: usize,
    ) -> Result<Self> {
        Self::new_with_kernel(
            directory,
            storage,
            capacity,
            crate::laya::cpu::Kernel::Portable,
        )
    }
    pub fn new_with_kernel(
        directory: impl Into<std::path::PathBuf>,
        storage: crate::laya::weights::Storage,
        capacity: usize,
        kernel: crate::laya::cpu::Kernel,
    ) -> Result<Self> {
        anyhow::ensure!(capacity > 0, "resident model limit must be positive");
        kernel.validate()?;
        Ok(Self {
            directory: directory.into(),
            storage,
            kernel,
            blas: None,
            capacity,
            #[cfg(feature = "int8")]
            quantization: None,
            resident: Default::default(),
            default: Variant::Multilingual,
            auto_task_detection: false,
        })
    }
    pub fn set_blas(&mut self, blas: Option<std::sync::Arc<crate::laya::cpu::Blas>>) -> Result<()> {
        anyhow::ensure!(
            blas.is_none()
                || (self.kernel == crate::laya::cpu::Kernel::Portable && {
                    #[cfg(feature = "int8")]
                    {
                        self.quantization.is_none()
                    }
                    #[cfg(not(feature = "int8"))]
                    {
                        true
                    }
                }),
            "BLAS requires portable float projections"
        );
        self.resident.clear();
        self.blas = blas;
        Ok(())
    }
    #[cfg(feature = "int8")]
    pub fn set_quantization(
        &mut self,
        value: Option<(crate::laya::cpu::int8::Mode, usize)>,
    ) -> Result<()> {
        anyhow::ensure!(
            value.is_none() || self.blas.is_none(),
            "disable BLAS before quantizing"
        );
        anyhow::ensure!(
            value.is_none_or(|(_, block)| matches!(block, 16 | 32 | 64 | 128)),
            "invalid INT8 block size"
        );
        if self.quantization != value {
            self.resident.clear();
            self.quantization = value;
        }
        Ok(())
    }
    pub fn predict(
        &mut self,
        mut request: Value,
        scratch: &mut crate::laya::Scratch,
    ) -> Result<Value> {
        if let Some(requests) = request.as_array() {
            return requests
                .iter()
                .map(|r| self.predict(r.clone(), scratch))
                .collect::<Result<Vec<_>>>()
                .map(Value::Array);
        }
        if let Some(states) = request.get("states").and_then(Value::as_array).cloned() {
            request.as_object_mut().unwrap().remove("states");
            return states
                .into_iter()
                .map(|state| {
                    let mut r = request.clone();
                    r["state"] = state;
                    self.predict(r, scratch)
                })
                .collect::<Result<Vec<_>>>()
                .map(Value::Array);
        }
        let decision = route(&request, self.default, self.auto_task_detection)?;
        if let Some(i) = self.resident.iter().position(|(v, _)| *v == decision.model) {
            let entry = self.resident.remove(i).unwrap();
            self.resident.push_back(entry);
        } else {
            while self.resident.len() >= self.capacity {
                self.resident.pop_front();
            }
            let mut model = crate::laya::Model::load_with_kernel(
                self.directory.join(decision.model.directory()),
                self.storage,
                self.kernel,
            )?;
            if let Some(blas) = &self.blas {
                model.set_blas(Some(blas.clone()))?;
            }
            #[cfg(feature = "int8")]
            if let Some((mode, block)) = self.quantization {
                model.quantize_encoder(mode, block)?;
            }
            self.resident.push_back((decision.model, model));
        }
        let mut output = self
            .resident
            .back()
            .unwrap()
            .1
            .predict_json(request.clone(), scratch)?;
        attach(&mut output, &request, &decision)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn precedence_and_uncertainty() {
        let select = |r| route(&r, Variant::Multilingual, false).unwrap().model;
        assert_eq!(select(json!({"state":"hello"})), Variant::Multilingual);
        assert_eq!(
            select(json!({"state":"This is the request that I would like you to check."})),
            Variant::English
        );
        assert_eq!(
            select(
                json!({"state":"This is the request that I would like you to check.","lang":"C.UTF-8"})
            ),
            Variant::English
        );
        assert_eq!(
            select(
                json!({"state":"This is the request that I would like you to check.","lang":"de"})
            ),
            Variant::Multilingual
        );
        assert_eq!(
            select(json!({"state":"hello","lang":"en_US.UTF-8","task":"typed_decisions"})),
            Variant::TypedDecisions
        );
        assert_eq!(
            select(json!({"state":"hello","task":"typed","model":"EN"})),
            Variant::English
        );
        let r = json!({"questions":{"action":{},"category":{},"churn_risk":{},"needs_human":{},"urgency":{}},"lang":"en"});
        assert_eq!(select(r.clone()), Variant::English);
        assert_eq!(
            route(&r, Variant::Multilingual, true).unwrap().model,
            Variant::TypedDecisions
        );
    }
}
