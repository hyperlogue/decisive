use crate::laya::{
    Model, Scratch,
    input::{Request, render},
};
use anyhow::{Result, ensure};
use serde_json::{Map, Value, json};

pub(crate) fn budget(
    requested: usize,
    room: usize,
    stride: Option<usize>,
) -> Result<(usize, usize)> {
    ensure!(room > 0, "questions leave no room for the state");
    let size = requested.min(room);
    let mut step = stride.filter(|&s| s > 0).unwrap_or((size / 2).max(1));
    if step > size && size < requested && step <= requested {
        step = (size / 2).max(1);
    }
    ensure!(
        step <= size,
        "stride exceeds the effective window and would skip input tokens"
    );
    Ok((size, step))
}

impl Model {
    pub fn predict_long(
        &self,
        request: &Request,
        window: Option<usize>,
        stride: Option<usize>,
        scratch: &mut Scratch,
    ) -> Result<Value> {
        ensure!(
            !request.state.is_null(),
            "long input state must not be null"
        );
        ensure!(
            request.min_confidence.is_none(),
            "long-document aggregation does not accept min_confidence"
        );
        let max_len = request
            .max_len
            .unwrap_or(self.agent_config["max_len"].as_u64().unwrap_or(512) as usize);
        let head_max_len = request
            .head_max_len
            .unwrap_or(self.agent_config["head_max_len"].as_u64().unwrap_or(192) as usize);
        let requested = window
            .filter(|&w| w > 0)
            .unwrap_or(max_len.saturating_sub(head_max_len + 8).max(64));
        let mut empty = request.clone();
        empty.state = json!("");
        let heads = self.encode(&empty)?;
        let room = heads
            .iter()
            .map(|q| max_len.saturating_sub(q.ids.len()))
            .min()
            .unwrap_or(requested);
        let (size, step) = budget(requested, room, stride)?;
        let tokens = self.input.tokens(&render(&request.state))?;
        let single_limit = if window.is_some_and(|w| w > 0) {
            size
        } else {
            room
        };
        if tokens.len() <= single_limit {
            let mut result = self.predict(request, scratch)?;
            result["usage"]["windows"] = json!(1);
            return Ok(result);
        }
        let mut starts = Vec::new();
        let mut results = Vec::new();
        for start in (0..tokens.len()).step_by(step) {
            let end = (start + size).min(tokens.len());
            let mut item = request.clone();
            item.state = json!(self.input.text(&tokens[start..end])?);
            starts.push(start);
            results.push(self.predict(&item, scratch)?);
            if end == tokens.len() {
                break;
            }
        }
        let mut answers = Map::new();
        for (id, question) in &request.questions {
            let key = if question["type"] == "noul" {
                "noul"
            } else {
                "answer_confidence"
            };
            let mut best = 0;
            for i in 1..results.len() {
                if results[i]["answers"][id][key].as_f64()
                    > results[best]["answers"][id][key].as_f64()
                {
                    best = i;
                }
            }
            let mut answer = results[best]["answers"][id].clone();
            answer["window"] = json!({"index":best,"token_start":starts[best],"token_end":(starts[best]+size).min(tokens.len()),"count":results.len()});
            answers.insert(id.clone(), answer);
        }
        let mut usage = Map::new();
        for result in &results {
            for (key, value) in result["usage"].as_object().unwrap() {
                match value {
                    Value::Bool(v) => {
                        let prev = usage.get(key).and_then(Value::as_u64).unwrap_or(0);
                        usage.insert(key.clone(), json!(prev + u64::from(*v)));
                    }
                    Value::Number(v) => {
                        let prev = usage.get(key).and_then(Value::as_u64).unwrap_or(0);
                        usage.insert(key.clone(), json!(prev + v.as_u64().unwrap_or(0)));
                    }
                    Value::Object(values) => {
                        let entry = usage.entry(key.clone()).or_insert_with(|| json!({}));
                        entry.as_object_mut().unwrap().extend(values.clone());
                    }
                    _ => {
                        usage.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        usage.insert("windows".into(), json!(results.len()));
        Ok(json!({"model":"laya-rl-agent","answers":answers,"usage":usage}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reduced_windows_do_not_leave_gaps() {
        assert_eq!(budget(312, 100, None).unwrap(), (100, 50));
        assert_eq!(budget(312, 100, Some(156)).unwrap(), (100, 50));
        assert!(budget(312, 100, Some(313)).is_err());
        assert!(budget(312, 0, None).is_err());
    }
}
