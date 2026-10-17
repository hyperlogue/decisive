use super::*;
fn probabilities(actual: &Value, expected: &Value) {
    for (k, v) in expected.as_object().unwrap() {
        let a = actual[k].as_array().unwrap();
        let e = v.as_array().unwrap();
        assert_eq!(a.len(), e.len());
        let best = |p: &Vec<Value>| {
            (0..p.len())
                .max_by(|&i, &j| {
                    p[i].as_f64()
                        .unwrap()
                        .total_cmp(&p[j].as_f64().unwrap())
                        .then_with(|| j.cmp(&i))
                })
                .unwrap()
        };
        assert_eq!(best(a), best(e));
        for (a, e) in a.iter().zip(e) {
            assert!(
                (a.as_f64().unwrap() - e.as_f64().unwrap()).abs() <= 1e-4,
                "{k}: {a} != {e}"
            );
        }
    }
}
#[test]
#[ignore = "requires pinned Kev and base checkpoints; run in release mode"]
fn official_checkpoint_contract() -> Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(6).build()?.install(||->Result<()>{
        let rows:Vec<Value>=include_str!("../../../benches/kev-reference.jsonl").lines().map(serde_json::from_str).collect::<std::result::Result<_,_>>()?;
        let mut s=Session::default();
        for portable in [false,true]{
            let model=Model::load("models/kev-0.8b","models/qwen3.5-0.8b-base",portable)?;let prior_hits=s.stats.hits;
            for(i,row)in rows.iter().enumerate(){
                if portable&&i>2&&row["id"]!="long"{continue;}
                assert_eq!(model.encode(row)?,row["token_plan"],"token plan {}",row["id"]);
                let a=model.decide_json(row,&mut s)?;if i==0{assert_eq!(s.stats.hits,prior_hits,"state leaked across models");}
                probabilities(&a["probabilities"],&row["expected"]["probabilities"]);assert_eq!(a["answers"],row["expected"]["answers"]);for field in ["input_tokens","output_tokens"]{assert_eq!(a["usage"][field],row["expected"]["usage"][field]);}
                let hits=s.stats.hits;assert_eq!(a,model.decide_json(row,&mut s)?);assert_eq!(s.stats.hits,hits+1);
                if i==0{for(id,q)in row["questions"].as_object().unwrap(){let mut fresh=Session::default();let single=json!({"state":row["state"],"questions":{id:q}});let independent=model.decide_json(&single,&mut fresh)?;assert_eq!(a["answers"][id],independent["answers"][id]);assert_eq!(a["probabilities"][id],independent["probabilities"][id]);}}
            }
            let row=&rows[0];let before=model.decide_json(row,&mut s)?;let limit=s.cache_bytes;
            s.set_length_rules(Some(serde_json::from_value(json!({"bands":[],"fallback":{"projection_rows":3,"attention_queries":7,"prefill_tokens":7}}))?))?;assert_eq!(s.cached_bytes(),0);assert_eq!(s.cache_bytes,limit);
            let after=model.decide_json(row,&mut s)?;probabilities(&after["probabilities"],&before["probabilities"]);
            let bytes=s.cached_bytes();assert!(s.set_length_rules(Some(serde_json::from_value(json!({"bands":[],"fallback":{"projection_rows":0,"attention_queries":7,"prefill_tokens":7}}))?)).is_err());assert_eq!(s.cached_bytes(),bytes);
            s.set_length_rules(None)?;s.set_cache_bytes(0);assert_eq!(s.cached_bytes(),0);let hits=s.stats.hits;probabilities(&model.decide_json(row,&mut s)?["probabilities"],&before["probabilities"]);assert_eq!(s.stats.hits,hits);
            let mut bad=row.clone();bad["state"]=json!("record ".repeat(66000));assert!(model.decide_json(&bad,&mut s).is_err());bad=row.clone();bad["questions"]=json!({});assert!(model.decide_json(&bad,&mut s).is_err());
            s.set_cache_bytes(limit);model.decide_json(row,&mut s)?;
        }Ok(())
    })
}

fn cache_close(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 1e-4 + 4. * f64::EPSILON)
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                cache_close(a, b);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (k, b) in b {
                cache_close(&a[k], b);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
#[ignore = "requires pinned checkpoint; run in release mode"]
fn growing_and_edited_prefix_contract() -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()?
        .install(|| -> Result<()> {
            let model = Model::load("models/kev-0.8b", "models/qwen3.5-0.8b-base", false)?;
            let corpus: Vec<Value> = include_str!("../../../benches/causal-kev.jsonl")
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            let base = corpus
                .iter()
                .find(|r| r["language"] == "en" && r["length"] == 512)
                .unwrap()["requests"][0]["request"]
                .clone();
            let text = base["state"].as_str().unwrap();
            for (label, changed) in [
                (
                    "append",
                    format!("{text} The customer also requested a status update."),
                ),
                ("token boundary", format!("{text}s")),
                ("unicode append", format!("{text} 客户还需要查询订单状态。")),
                ("early edit", format!("Different record: {text}")),
            ] {
                let mut cached = Session::default();
                model.decide_json(&base, &mut cached)?;
                assert!(cached.prefix.as_ref().unwrap().checkpoint.is_some());
                let mut request = base.clone();
                request["state"] = json!(changed);
                let ids = model.input.encode(&request)?.prefix;
                let entry = cached.prefix.as_ref().unwrap();
                let expected_reused = if ids.starts_with(&entry.ids) {
                    entry.state.position
                } else {
                    entry
                        .checkpoint
                        .as_ref()
                        .filter(|s| {
                            ids.len() >= s.position && ids[..s.position] == entry.ids[..s.position]
                        })
                        .map_or(0, |s| s.position)
                };
                if label == "early edit" {
                    assert_eq!(expected_reused, 0);
                } else {
                    assert!(expected_reused > 0);
                }
                let reused = cached.stats.prefix_tokens_reused;
                let computed = cached.stats.prefix_tokens_computed;
                let actual = model.decide_json(&request, &mut cached)?;
                assert_eq!(
                    cached.stats.prefix_tokens_reused - reused,
                    expected_reused,
                    "{label}"
                );
                assert_eq!(
                    cached.stats.prefix_tokens_computed - computed,
                    ids.len() - expected_reused,
                    "{label}"
                );
                assert!(cached.cached_bytes() <= cached.cache_bytes);
                let mut fresh = Session::default();
                fresh.set_cache_bytes(0);
                let expected = model.decide_json(&request, &mut fresh)?;
                cache_close(&actual, &expected);
                let hits = cached.stats.hits;
                let again = model.decide_json(&request, &mut cached)?;
                assert_eq!(
                    actual, again,
                    "identical request must reuse the complete state"
                );
                assert_eq!(cached.stats.hits, hits + 1);
            }
            let mut cached = Session::default();
            model.decide_json(&base, &mut cached)?;
            let entry = cached.prefix.as_ref().unwrap();
            let limit = entry.ids.len() * 4 + entry.state.bytes();
            cached.set_cache_bytes(limit);
            model.decide_json(&base, &mut cached)?;
            assert!(cached.prefix.as_ref().unwrap().checkpoint.is_none());
            assert!(cached.cached_bytes() <= limit);
            cached.set_cache_bytes(0);
            assert_eq!(cached.cached_bytes(), 0);
            let rules: crate::length::Rules<Plan> = serde_json::from_value(json!({
                "bands":[{"through":512,"settings":{"projection_rows":3,"attention_queries":7,"prefill_tokens":64}}],
                "fallback":{"projection_rows":6,"attention_queries":32,"prefill_tokens":256}
            }))?;
            let mut cached=Session::default();
            cached.set_length_rules(Some(rules.clone()))?;
            model.decide_json(&base,&mut cached)?;
            let mut longer=base.clone();
            longer["state"]=json!(format!("{text} {}", "Additional context. ".repeat(32)));
            let actual=model.decide_json(&longer,&mut cached)?;
            assert!(cached.stats.prefix_tokens_reused > 0);
            let mut fresh=Session::default();
            fresh.set_length_rules(Some(rules))?;
            fresh.set_cache_bytes(0);
            let expected=model.decide_json(&longer,&mut fresh)?;
            cache_close(&actual,&expected);
            Ok(())
        })
}

#[test]
#[ignore = "requires pinned checkpoint; verifies complete in-chunk hybrid snapshots"]
fn interior_snapshots_resume() -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()?
        .install(|| -> Result<()> {
            let model = Model::load("models/kev-0.8b", "models/qwen3.5-0.8b-base", false)?;
            let net = &model.backbone;
            let ids: Vec<u32> = (0..37).map(|i| 1000 + i * 17).collect();
            let mut before = net.empty_state();
            let mut work = Work::default();
            net.forward(&ids[..5], &mut before, &mut work, Plan::default())?;
            for cuts in [[1, 2], [3, 17], [17, 32], [32, 17], [17, 17]] {
                let mut whole = before.clone();
                let saved = net.forward_saving(
                    &ids[5..],
                    &mut whole,
                    &mut work,
                    Plan::default(),
                    cuts.map(|c| Some(5 + c)),
                )?;
                let hidden = work.hidden.clone();
                for (cut, saved) in cuts.into_iter().zip(saved) {
                    let mut expected = before.clone();
                    net.forward(&ids[5..5 + cut], &mut expected, &mut work, Plan::default())?;
                    let mut saved = saved.unwrap();
                    backbone::assert_same_state(&saved, &expected);
                    if 5 + cut < ids.len() {
                        net.forward(&ids[5 + cut..], &mut saved, &mut work, Plan::default())?;
                        backbone::assert_same_state(&saved, &whole);
                        let error = work
                            .hidden
                            .iter()
                            .zip(&hidden[cut * 1024..])
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        assert!(error <= 1e-4, "resumed hidden error {error}");
                    }
                }
            }
            Ok(())
        })
}
