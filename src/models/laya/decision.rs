use crate::laya::input::{Encoded, Request};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub type RawDecision = (Vec<f32>, f32);

pub fn decode(
    agent_config: &Value,
    request: &Request,
    encoded: &[Encoded],
    results: Vec<RawDecision>,
) -> Result<Value> {
    ensure!(
        encoded.len() == request.questions.len() && results.len() == encoded.len(),
        "backend result count differs from prepared questions"
    );
    for (q, (logits, act)) in encoded.iter().zip(&results) {
        ensure!(
            logits.len() == q.markers.len()
                && logits.iter().all(|v| v.is_finite())
                && act.is_finite()
                && (0.0..=1.0).contains(act),
            "invalid backend decision result"
        );
    }
    let mut answers = serde_json::Map::new();
    let mut tokens = 0;
    for (((qid, _), q), (logits, act)) in request.questions.iter().zip(encoded).zip(results) {
        let bucket = crate::laya::input::bucket(q.qtype, q.markers.len());
        let temperature = agent_config["temperature_by_options"][bucket]
            .as_f64()
            .or_else(|| agent_config["temperature"][q.qtype].as_f64())
            .unwrap_or(1.0)
            .clamp(0.5, 5.0) as f32;
        let mut answer = crate::laya::input::decode(q, &logits, act, temperature);
        if let Some(value) = &request.min_confidence {
            crate::laya::input::gate(&mut answer, value, q.qtype, q.markers.len());
        }
        answers.insert(qid.clone(), answer);
        tokens += q.ids.len();
    }
    let state_tokens = encoded.first().map_or(0, |q| q.state_tokens);
    let dropped = encoded
        .iter()
        .map(|q| q.state_tokens - q.state_tokens_used)
        .max()
        .unwrap_or(0);
    let truncated: Vec<_> = request
        .questions
        .keys()
        .zip(encoded)
        .filter(|(_, q)| q.state_tokens_used < q.state_tokens)
        .map(|(k, _)| k)
        .collect();
    let mut usage = json!({"input_tokens":tokens,"output_tokens":0,
            "state_tokens":state_tokens,"state_tokens_dropped":dropped,"truncated":dropped>0,"truncated_questions":truncated});
    let collapsed: serde_json::Map<_,_> = request.questions.keys().zip(encoded).filter(|(_,q)| q.options_distinct < q.markers.len()).map(|(id,q)| (id.clone(),json!({"total":q.markers.len(),"distinct":q.options_distinct,"tokens_per_option":q.tokens_per_option}))).collect();
    if !collapsed.is_empty() {
        usage["options"] = Value::Object(collapsed);
    }
    Ok(json!({"model":"laya-rl-agent","answers":answers,"usage":usage}))
}

pub fn prepare(
    input: &crate::laya::input::Input,
    agent_config: &Value,
    max_positions: usize,
    request: &Request,
) -> Result<Vec<Encoded>> {
    if let Some(value) = &request.min_confidence {
        crate::laya::input::validate_threshold(value)?;
    }
    let max_len = request
        .max_len
        .unwrap_or(agent_config["max_len"].as_u64().unwrap_or(512) as usize);
    ensure!(max_len <= max_positions, "max_len exceeds model positions");
    input.encode(
        &request.state,
        &request.questions,
        max_len,
        request
            .head_max_len
            .unwrap_or(agent_config["head_max_len"].as_u64().unwrap_or(192) as usize),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_missing_or_nonfinite_backend_results() {
        let request: Request = serde_json::from_value(
            json!({"state":"test","questions":{"q":{"type":"choice","criteria":["a","b"]}}}),
        )
        .unwrap();
        let encoded = Encoded {
            ids: vec![1, 2],
            markers: vec![0, 1],
            qtype: 0,
            labels: vec![json!("a"), json!("b")],
            descriptions: vec!["a".into(), "b".into()],
            order: vec![0, 1],
            state_tokens: 0,
            state_tokens_used: 0,
            options_distinct: 2,
            tokens_per_option: None,
        };
        assert!(decode(&json!({}), &request, std::slice::from_ref(&encoded), vec![]).is_err());
        assert!(
            decode(
                &json!({}),
                &request,
                std::slice::from_ref(&encoded),
                vec![(vec![0.], 0.5)]
            )
            .is_err()
        );
        assert!(
            decode(
                &json!({}),
                &request,
                std::slice::from_ref(&encoded),
                vec![(vec![0., f32::NAN], 0.5)]
            )
            .is_err()
        );
        assert!(
            decode(
                &json!({}),
                &request,
                std::slice::from_ref(&encoded),
                vec![(vec![0., 1.], 1.5)]
            )
            .is_err()
        );
    }
}
