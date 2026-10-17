use anyhow::{Result, anyhow, ensure};
use serde_json::{Value, json};
use std::time::Instant;

pub fn compare_outputs(reference: &Value, actual: &Value, bound: f64) -> Result<f64> {
    if let Some(items) = reference.as_array() {
        let actual = actual
            .as_array()
            .ok_or_else(|| anyhow!("batch output is not an array"))?;
        ensure!(items.len() == actual.len(), "batch output length differs");
        return items.iter().zip(actual).try_fold(0.0_f64, |max, (r, a)| {
            Ok(max.max(compare_outputs(r, a, bound)?))
        });
    }
    if reference.get("answers").is_none() {
        ensure!(reference == actual, "structured decision values changed");
        return Ok(0.0);
    }
    if let Some(values) = reference.get("values") {
        ensure!(
            Some(values) == actual.get("values"),
            "structured decision values changed"
        );
    }
    if let Some(usage) = reference["usage"].as_object() {
        for (key, value) in usage {
            ensure!(actual["usage"][key] == *value, "usage differs at {key}");
        }
    }
    fn compare(
        r: &Value,
        a: &Value,
        path: &str,
        bound: f64,
        max: &mut f64,
        score_range: f64,
    ) -> Result<()> {
        match r {
            Value::Object(o) => {
                ensure!(
                    a.as_object().is_some_and(|v| v.len() == o.len()),
                    "output fields differ at {path}"
                );
                if path.ends_with(".probabilities") {
                    ensure!(
                        argmax(r) == argmax(a),
                        "probability argmax changed at {path}"
                    );
                }
                let score_range = o
                    .get("legend")
                    .and_then(Value::as_object)
                    .map_or(score_range, |v| v.len().saturating_sub(1).max(1) as f64);
                for (k, v) in o {
                    let a = a
                        .get(k)
                        .ok_or_else(|| anyhow!("missing output {path}.{k}"))?;
                    if k == "choice" || k == "type" || k == "legend" {
                        ensure!(v == a, "discrete output mismatch at {path}.{k}: {v} != {a}");
                    } else {
                        compare(v, a, &format!("{path}.{k}"), bound, max, score_range)?;
                    }
                }
            }
            Value::Number(n) => {
                let expected = n.as_f64().unwrap();
                let got = a
                    .as_f64()
                    .ok_or_else(|| anyhow!("non-numeric output at {path}"))?;
                ensure!(
                    expected.is_finite() && got.is_finite(),
                    "non-finite output at {path}"
                );
                let divisor = if path.ends_with(".score") {
                    score_range
                } else {
                    1.0
                };
                let error = (got - expected).abs() / divisor;
                *max = max.max(error);
                ensure!(
                    error <= bound + 1e-12,
                    "{path}: error {error} exceeds {bound} ({expected} vs {got})"
                );
                if path.ends_with(".noul") || path.ends_with(".act_probability") {
                    ensure!(
                        (expected >= 0.5) == (got >= 0.5),
                        "binary decision changed at {path}"
                    );
                }
            }
            _ => ensure!(r == a, "output differs at {path}"),
        }
        Ok(())
    }
    let mut max = 0.0;
    ensure!(
        reference["answers"].is_object() && actual["answers"].is_object(),
        "missing answers"
    );
    compare(
        &reference["answers"],
        &actual["answers"],
        "answers",
        bound,
        &mut max,
        1.0,
    )?;
    Ok(max)
}

fn argmax(probabilities: &Value) -> Option<&str> {
    let mut best: Option<(&str, f64)> = None;
    for (key, value) in probabilities.as_object()? {
        let value = value.as_f64()?;
        if best.is_none_or(|(_, score)| value > score) {
            best = Some((key, value));
        }
    }
    best.map(|(key, _)| key)
}

pub fn quality(requests: &[Value], outputs: &[Value]) -> Result<Value> {
    let mut count = 0;
    let mut correct = 0;
    let mut brier = 0.0;
    let mut nll = 0.0;
    for (request, output) in requests.iter().zip(outputs) {
        let Some(gold) = request["gold"].as_object() else {
            continue;
        };
        for (id, target) in gold {
            let answer = &output["answers"][id];
            let probabilities = if answer["type"] == "noul" {
                let p = answer["noul"]
                    .as_f64()
                    .ok_or_else(|| anyhow!("missing noul {id}"))?;
                json!({"false":1.0-p,"true":p})
            } else {
                answer["probabilities"].clone()
            };
            let label = target["label"]
                .as_str()
                .ok_or_else(|| anyhow!("missing gold label {id}"))?;
            ensure!(
                probabilities.get(label).is_some(),
                "gold label {label} absent from {id}"
            );
            count += 1;
            correct += usize::from(argmax(&probabilities) == Some(label));
            for (key, value) in probabilities.as_object().unwrap() {
                brier += (value.as_f64().unwrap() - if key == label { 1.0 } else { 0.0 }).powi(2);
            }
            nll -= probabilities[label].as_f64().unwrap().max(1e-12).ln();
        }
    }
    if count == 0 {
        return Ok(Value::Null);
    }
    Ok(
        json!({"decisions":count,"correct":correct,"accuracy":correct as f64/count as f64,"brier":brier/count as f64,"nll":nll/count as f64}),
    )
}

pub fn summarize(times: &[f64]) -> Value {
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    let p = |q: f64| {
        sorted[((sorted.len() as f64 * q).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1)]
    };
    json!({"requests":times.len(),"p50_ms":p(0.5),"p95_ms":p(0.95),"requests_per_second":1000.0*times.len() as f64/times.iter().sum::<f64>(),"latencies_ms":times})
}

#[allow(dead_code)]
pub fn paired_interval(ratios: &[f64]) -> Value {
    assert!(!ratios.is_empty() && ratios.iter().all(|r| r.is_finite() && *r > 0.));
    let mean = ratios.iter().map(|x| x.ln()).sum::<f64>() / ratios.len() as f64;
    let mut state = 4531_u64;
    let mut boots = Vec::new();
    for _ in 0..5000 {
        let mut sum = 0.;
        for _ in ratios {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            sum += ratios[state as usize % ratios.len()].ln();
        }
        boots.push((sum / ratios.len() as f64).exp());
    }
    boots.sort_by(f64::total_cmp);
    json!({"geometric_ratio":mean.exp(),"lower_95":boots[124],"upper_95":boots[4874],"method":"5000 paired log-ratio bootstrap resamples; xorshift seed 4531; descriptive, not immune to host drift or multiple comparisons"})
}

pub fn http_compare(
    requests: &[Value],
    left: &str,
    right: &str,
    warmup: usize,
    rounds: usize,
    require_matched_tokens: bool,
) -> Result<Value> {
    ensure!(!requests.is_empty() && rounds > 0, "empty benchmark");
    let client = ureq::Agent::new_with_defaults();
    let call = |url: &str, request: &Value| -> Result<(Value, f64)> {
        let start = Instant::now();
        let body = client
            .post(url)
            .header("Content-Type", "application/json")
            .send(request.to_string().as_bytes())?
            .body_mut()
            .read_to_string()?;
        let output: Value = serde_json::from_str(&body)?;
        Ok((output, start.elapsed().as_secs_f64() * 1000.0))
    };
    let (mut left_wall_ms, mut right_wall_ms) = (0., 0.);
    for _ in 0..warmup {
        for request in requests {
            left_wall_ms += call(left, request)?.1;
            right_wall_ms += call(right, request)?.1;
        }
    }
    let (mut ltimes, mut rtimes, mut loutputs, mut routputs) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut pairs = Vec::new();
    let mut max_error: f64 = 0.0;
    let mut mismatches = Vec::new();
    for round in 0..rounds {
        for i in (0..requests.len()).map(|i| (i + round) % requests.len()) {
            let request = &requests[i];
            let left_first = (round % 4 == 0 || round % 4 == 3) != (i % 2 == 1);
            let (l, r) = if left_first {
                (call(left, request)?, call(right, request)?)
            } else {
                let r = call(right, request)?;
                (call(left, request)?, r)
            };
            left_wall_ms += l.1;
            right_wall_ms += r.1;
            if require_matched_tokens {
                let left_tokens = l.0["usage"]["input_tokens"].as_u64();
                let right_tokens = r.0["usage"]["input_tokens"].as_u64();
                ensure!(
                    left_tokens.is_some() && left_tokens == right_tokens,
                    "request {i}: unequal/missing input token counts ({left_tokens:?} versus {right_tokens:?}); comparison is invalid"
                );
            }
            pairs.push(json!({"round":round,"request":i,"left_first":left_first,"left_ms":l.1,"right_ms":r.1,"left_tokens":l.0["usage"]["input_tokens"],"right_tokens":r.0["usage"]["input_tokens"],"left_output":l.0.clone(),"right_output":r.0.clone()}));
            ltimes.push(l.1);
            rtimes.push(r.1);
            match compare_outputs(&l.0, &r.0, 1e-4) {
                Ok(e) => max_error = max_error.max(e),
                Err(e) => mismatches.push(json!({"round":round,"request":i,"error":e.to_string()})),
            }
            if round == 0 {
                loutputs.push(l.0);
                routputs.push(r.0);
            }
        }
    }
    let left_stats = summarize(&ltimes);
    let right_stats = summarize(&rtimes);
    let per_request: Vec<_> = (0..requests.len()).map(|i| {
        let samples: Vec<_> = pairs.iter().filter(|p| p["request"] == i).collect();
        let left: Vec<_> = samples.iter().map(|p| p["left_ms"].as_f64().unwrap()).collect();
        let right: Vec<_> = samples.iter().map(|p| p["right_ms"].as_f64().unwrap()).collect();
        let ratios: Vec<_> = left.iter().zip(&right).map(|(l,r)| r/l).collect();
        json!({"request":i,"left_tokens":samples[0]["left_tokens"],"right_tokens":samples[0]["right_tokens"],"left":summarize(&left),"right":summarize(&right),"paired_right_over_left":paired_interval(&ratios)})
    }).collect();
    Ok(
        json!({"left":left,"right":right,"left_stats":left_stats,"right_stats":right_stats,
        "scope":"Sequential warm HTTP requests; request serialization, transfer and response parsing included. Balanced ABBA/BAAB engine order per workload, rotating workload order. Contract differences are retained; matching token counts alone do not prove matching model semantics.",
        "warmup":warmup,"rounds":rounds,"require_matched_tokens":require_matched_tokens,"pairs":pairs,"per_request":per_request,
        "request_wall_seconds_including_warmup":{"left":left_wall_ms/1000.,"right":right_wall_ms/1000.},
        "p50_speedup":left_stats["p50_ms"].as_f64().unwrap()/right_stats["p50_ms"].as_f64().unwrap(),
        "p95_speedup":left_stats["p95_ms"].as_f64().unwrap()/right_stats["p95_ms"].as_f64().unwrap(),
        "max_output_error_on_matching_requests":max_error,"mismatches":mismatches,"left_outputs":loutputs,"right_outputs":routputs}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accuracy_gate_checks_outliers_and_decisions() {
        let r = json!({"answers":{"a":{"type":"choice","choice":"x","probabilities":{"x":0.6,"y":0.4}}}});
        let mut a = r.clone();
        a["answers"]["a"]["probabilities"]["x"] = json!(0.6002);
        assert!(compare_outputs(&r, &a, 1e-4).is_err());
        a = r.clone();
        a["answers"]["a"]["choice"] = json!("y");
        assert!(compare_outputs(&r, &a, 1e-4).is_err());
    }
}
