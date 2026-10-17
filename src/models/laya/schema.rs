use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value, json};

enum Kind {
    Choice(Vec<(String, Value)>),
    Boolean,
    Score(i64),
}
pub struct Schema {
    pub questions: Map<String, Value>,
    fields: Vec<(String, Kind)>,
}

impl Schema {
    pub fn parse(root: &Value) -> Result<Self> {
        ensure!(
            root["type"].is_null() || root["type"] == "object",
            "schema root must be an object"
        );
        let properties = root["properties"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("schema needs properties"))?;
        ensure!(
            !properties.is_empty() && properties.len() <= 32,
            "schema must contain 1 to 32 properties"
        );
        let mut questions = Map::new();
        let mut fields = Vec::new();
        for (name, property) in properties {
            let prop = resolve(property, root, &mut Vec::new())?;
            let description = prop["description"].as_str().filter(|s| !s.is_empty());
            let enumeration = prop
                .get("const")
                .map(|v| vec![v.clone()])
                .or_else(|| prop["enum"].as_array().cloned());
            let (kind, question) = if let Some(values) = enumeration {
                ensure!(
                    !values.is_empty() && values.len() <= 32,
                    "{name}: enum must contain 1 to 32 values"
                );
                if values.iter().all(Value::is_boolean) {
                    (Kind::Boolean, boolean_question(name, description))
                } else {
                    let mut options = Vec::new();
                    let mut criteria = Map::new();
                    for value in values {
                        let label = match &value {
                            Value::Null => "null".to_owned(),
                            Value::Bool(true) => "True".into(),
                            Value::Bool(false) => "False".into(),
                            Value::String(s) => s.clone(),
                            Value::Number(_) => value.to_string(),
                            _ => bail!("{name}: enum values must be scalars"),
                        };
                        ensure!(
                            !criteria.contains_key(&label),
                            "{name}: duplicate enum label"
                        );
                        criteria.insert(label.clone(), Value::Null);
                        options.push((label, value));
                    }
                    (
                        Kind::Choice(options),
                        json!({"type":"choice", "instructions":description.map(str::to_owned).unwrap_or_else(|| format!("What is `{name}`?")), "criteria":criteria}),
                    )
                }
            } else {
                let kinds: Vec<_> = if let Some(types) = prop["type"].as_array() {
                    types
                        .iter()
                        .filter_map(Value::as_str)
                        .filter(|&v| v != "null")
                        .collect()
                } else {
                    prop["type"].as_str().into_iter().collect()
                };
                ensure!(
                    kinds.len() == 1,
                    "{name}: schema must name one non-null type"
                );
                match kinds[0] {
                    "boolean" => (Kind::Boolean, boolean_question(name, description)),
                    "integer" | "number" => {
                        let lo = prop["minimum"]
                            .as_i64()
                            .ok_or_else(|| anyhow::anyhow!("{name}: integer minimum required"))?;
                        let hi = prop["maximum"]
                            .as_i64()
                            .ok_or_else(|| anyhow::anyhow!("{name}: integer maximum required"))?;
                        let count = hi
                            .checked_sub(lo)
                            .and_then(|v| v.checked_add(1))
                            .unwrap_or(0);
                        ensure!(
                            (1..=10).contains(&count),
                            "{name}: score needs 1 to 10 levels"
                        );
                        (
                            Kind::Score(lo),
                            json!({"type":"score", "instructions":description.map(str::to_owned).unwrap_or_else(|| format!("Score `{name}` from {lo} to {hi}")), "criteria":(lo..=hi).map(|v| v.to_string()).collect::<Vec<_>>()}),
                        )
                    }
                    _ => bail!("{name}: use an enum, boolean, or bounded integer"),
                }
            };
            questions.insert(name.clone(), question);
            fields.push((name.clone(), kind));
        }
        Ok(Self { questions, fields })
    }

    pub fn project(&self, response: &Value, details: bool) -> Result<Value> {
        let answers = response["answers"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("missing answers"))?;
        let mut values = Map::new();
        let mut confidence = Map::new();
        let mut answer_confidence = Map::new();
        let mut probabilities = Map::new();
        for (name, kind) in &self.fields {
            let Some(answer) = answers.get(name) else {
                continue;
            };
            let value = if answer["low_confidence"] == true {
                Value::Null
            } else {
                match kind {
                    Kind::Boolean => json!(answer["noul"].as_f64().unwrap_or(0.0) >= 0.5),
                    Kind::Score(minimum) => {
                        let probs = answer["probabilities"].as_object().ok_or_else(|| {
                            anyhow::anyhow!("{name}: missing score probabilities")
                        })?;
                        let mut best = 0;
                        for i in 1..probs.len() {
                            if probs[&i.to_string()].as_f64() > probs[&best.to_string()].as_f64() {
                                best = i;
                            }
                        }
                        json!(minimum + best as i64)
                    }
                    Kind::Choice(options) => {
                        let label = answer["choice"]
                            .as_str()
                            .ok_or_else(|| anyhow::anyhow!("{name}: missing enum choice"))?;
                        options
                            .iter()
                            .find(|(key, _)| key == label)
                            .map(|(_, v)| v.clone())
                            .unwrap_or_else(|| json!(label))
                    }
                }
            };
            values.insert(name.clone(), value);
            confidence.insert(name.clone(), answer["confidence"].clone());
            answer_confidence.insert(name.clone(), answer["answer_confidence"].clone());
            probabilities.insert(
                name.clone(),
                if matches!(kind, Kind::Boolean) {
                    let p = answer["noul"].as_f64().unwrap_or(0.0);
                    json!({"false":crate::laya::input::round4(1.0-p), "true":p})
                } else {
                    answer["probabilities"].clone()
                },
            );
        }
        Ok(if details {
            json!({"values":values,"confidence":confidence,"answer_confidence":answer_confidence,"probabilities":probabilities,"answers":answers,"usage":response["usage"],"routing":response["routing"]})
        } else {
            Value::Object(values)
        })
    }
}

fn boolean_question(name: &str, description: Option<&str>) -> Value {
    json!({"type":"noul","instructions":description.map(str::to_owned).unwrap_or_else(|| format!("Is `{name}` true?"))})
}

fn resolve(value: &Value, root: &Value, seen: &mut Vec<String>) -> Result<Value> {
    let mut prop = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("schema property must be an object"))?
        .clone();
    while let Some(reference) = prop.get("$ref") {
        let reference = reference
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("$ref must be a string"))?
            .to_owned();
        ensure!(
            !seen.contains(&reference),
            "recursive schema reference {reference}"
        );
        seen.push(reference.clone());
        let (container, name) = if let Some(name) = reference.strip_prefix("#/$defs/") {
            ("$defs", name)
        } else if let Some(name) = reference.strip_prefix("#/definitions/") {
            ("definitions", name)
        } else {
            bail!("only local schema definitions are supported")
        };
        let mut merged = root[container][name]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("unresolved schema reference {reference}"))?
            .clone();
        merged.remove("description");
        prop.remove("$ref");
        merged.extend(prop);
        prop = merged;
    }
    if !["const", "enum", "type"]
        .iter()
        .any(|key| prop.contains_key(*key))
    {
        if let Some(branches) = prop
            .get("allOf")
            .and_then(Value::as_array)
            .filter(|v| v.len() == 1)
        {
            let mut branch = branches[0]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("allOf branch must be an object"))?
                .clone();
            prop.remove("allOf");
            branch.extend(prop);
            return resolve(&Value::Object(branch), root, seen);
        }
        if let Some(branches) = prop
            .get("anyOf")
            .or_else(|| prop.get("oneOf"))
            .and_then(Value::as_array)
        {
            let branches: Vec<_> = branches
                .iter()
                .filter(|v| v.is_object() && v["type"] != "null")
                .collect();
            ensure!(
                branches.len() == 1,
                "only a single non-null union branch is supported"
            );
            let mut branch = branches[0].as_object().unwrap().clone();
            if !branch.contains_key("description") {
                branch.insert(
                    "description".into(),
                    prop.get("description").cloned().unwrap_or(Value::Null),
                );
            }
            return resolve(&Value::Object(branch), root, seen);
        }
    }
    Ok(Value::Object(prop))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nullable_references_and_nonzero_scales() {
        let schema = Schema::parse(&json!({"type":"object", "$defs":{"Mode":{"enum":["off","on"],"description":"type documentation"}},"properties":{"mode":{"anyOf":[{"$ref":"#/$defs/Mode"},{"type":"null"}]},"rating":{"type":"integer","minimum":3,"maximum":5}}})).unwrap();
        assert_eq!(schema.questions["mode"]["instructions"], "What is `mode`?");
        let output = schema.project(&json!({"answers":{"mode":{"choice":"on"},"rating":{"probabilities":{"0":0.4,"1":0.1,"2":0.5},"score":1.1}}}), false).unwrap();
        assert_eq!(output, json!({"mode":"on","rating":5}));
    }
    #[test]
    fn recursive_references_are_rejected() {
        assert!(
            Schema::parse(
                &json!({"properties":{"x":{"$ref":"#/$defs/A"}},"$defs":{"A":{"$ref":"#/$defs/A"}}})
            )
            .is_err()
        );
    }
}
