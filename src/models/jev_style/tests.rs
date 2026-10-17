use super::*;
fn close(actual: &Value, expected: &Value) {
    match (expected, actual) {
        (Value::Number(e), Value::Number(a)) => assert!(
            (e.as_f64().unwrap() - a.as_f64().unwrap()).abs() <= 1e-4,
            "{a} != {e}"
        ),
        (Value::Object(e), Value::Object(a)) => {
            for (k, v) in e {
                close(a.get(k).unwrap_or_else(|| panic!("missing {k}")), v);
            }
        }
        (Value::Array(e), Value::Array(a)) => {
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e) {
                close(a, e);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}
fn run(m: &Model, row: &Value, s: &mut Session) -> Result<Value> {
    m.decide(
        &row["state"],
        &row["question"],
        row["category"].as_str(),
        row["temperature"].as_f64(),
        s,
    )
}
#[test]
#[ignore = "requires pinned Jev checkpoint; run in release mode"]
fn official_checkpoint_contract() -> Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(6).build()?.install(||->Result<()>{
        let rows:Vec<Value>=include_str!("../../../benches/jev-reference.jsonl").lines().map(serde_json::from_str).collect::<std::result::Result<_,_>>()?;
        let mut s=Session::default();
        for portable in [false,true]{
            let model=Model::load("models/jev-style-0.8b-v3",portable)?;
            let prior_hits=s.stats.hits;
            for (i,row) in rows.iter().enumerate(){
                if portable&&i>2&&row["id"]!="long"{continue;}
                assert_eq!(model.encode(&row["state"],&row["question"])?,row["token_plan"]);
                let a=run(&model,row,&mut s)?;
                if i==0 {assert_eq!(s.stats.hits,prior_hits,"state leaked across model identities");}
                close(&a["probabilities"],&row["expected"]["probabilities"]);assert_eq!(a["answer"],row["expected"]["answer"]);
                let hits=s.stats.hits;assert_eq!(a,run(&model,row,&mut s)?);assert_eq!(s.stats.hits,hits+1);
            }
            for row in include_str!("../../../benches/jev-systemone.jsonl").lines(){let row:Value=serde_json::from_str(row)?;let output=model.predict_json(&row["request"],&mut s)?;close(&output,&row["expected"]);
                for(id,q)in row["request"]["questions"].as_object().unwrap(){let mut separate=Session::default();let single=json!({"state":row["request"]["state"],"questions":{id:q}});close(&model.predict_json(&single,&mut separate)?["answers"][id],&output["answers"][id]);}
            }
            let row=&rows[0];let before=run(&model,row,&mut s)?;
            let limit=s.cache_bytes;
            s.set_length_rules(Some(serde_json::from_value(json!({"bands":[],"fallback":{"projection_rows":3,"attention_queries":7,"prefill_tokens":7}}))?))?;
            assert_eq!(s.cached_bytes(),0);assert_eq!(s.cache_bytes,limit);
            let after=run(&model,row,&mut s)?;close(&after["probabilities"],&before["probabilities"]);assert_eq!(after["answer"],before["answer"]);
            let cached=s.cached_bytes();assert!(s.set_length_rules(Some(serde_json::from_value(json!({"bands":[],"fallback":{"projection_rows":0,"attention_queries":7,"prefill_tokens":7}}))?)).is_err());assert_eq!(s.cached_bytes(),cached);
            s.set_length_rules(None)?;s.set_cache_bytes(0);assert_eq!(s.cached_bytes(),0);let hits=s.stats.hits;close(&run(&model,row,&mut s)?["probabilities"],&before["probabilities"]);assert_eq!(s.stats.hits,hits);
            assert!(model.decide(&json!("record ".repeat(26000)),&row["question"],None,None,&mut s).is_err());
            assert!(model.predict_json(&json!({"state":"x","questions":{"q":{"type":"score","instructions":"x","criteria":["only"]}}}),&mut s).is_err());
            s.set_cache_bytes(limit);run(&model,row,&mut s)?;
        }Ok(())
    })
}

#[test]
#[ignore = "requires pinned checkpoint; run in release mode"]
fn growing_and_edited_prefix_contract() -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()?
        .install(|| -> Result<()> {
            let model = Model::load("models/jev-style-0.8b-v3", false)?;
            let corpus: Vec<Value> = include_str!("../../../benches/causal-jev.jsonl")
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
                model.predict_json(&base, &mut cached)?;
                assert!(cached.prefix.as_ref().unwrap().checkpoint.is_some());
                let mut request = base.clone();
                request["state"] = json!(changed);
                let ids = model.input.prefix(&request["state"])?;
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
                let actual = model.predict_json(&request, &mut cached)?;
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
                let expected = model.predict_json(&request, &mut fresh)?;
                close(&actual, &expected);
                let hits = cached.stats.hits;
                let again = model.predict_json(&request, &mut cached)?;
                assert_eq!(
                    actual, again,
                    "identical request must reuse the complete state"
                );
                assert_eq!(cached.stats.hits, hits + 1);
            }
            let mut cached = Session::default();
            model.predict_json(&base, &mut cached)?;
            let entry = cached.prefix.as_ref().unwrap();
            let limit = entry.ids.len() * 4 + entry.state.bytes();
            cached.set_cache_bytes(limit);
            model.predict_json(&base, &mut cached)?;
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
            model.predict_json(&base,&mut cached)?;
            let mut longer=base.clone();
            longer["state"]=json!(format!("{text} {}", "Additional context. ".repeat(32)));
            let actual=model.predict_json(&longer,&mut cached)?;
            assert!(cached.stats.prefix_tokens_reused > 0);
            let mut fresh=Session::default();
            fresh.set_length_rules(Some(rules))?;
            fresh.set_cache_bytes(0);
            let expected=model.predict_json(&longer,&mut fresh)?;
            close(&actual,&expected);
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
            let model = Model::load("models/jev-style-0.8b-v3", false)?;
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
