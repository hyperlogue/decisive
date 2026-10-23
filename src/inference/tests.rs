use super::*;
use std::{
    future::Future,
    pin::pin,
    task::{Context, Poll, Waker},
};

fn run<T>(future: impl Future<Output = T>) -> T {
    match pin!(future)
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("test backend should complete synchronously"),
    }
}
struct TestBackend;
impl Backend for TestBackend {
    type Parameters = usize;
    type Cache = Vec<String>;
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            model: "test",
            backend: "cpu",
            caches: vec!["prefix"],
            tuning: "off",
        }
    }
    fn create_cache(&self, _: &CacheOptions) -> Result<Self::Cache> {
        Ok(Vec::new())
    }
    fn clear_cache(&self, cache: &mut Self::Cache) {
        cache.clear();
    }
    async fn predict(
        &mut self,
        request: &Request,
        parameters: &usize,
        cache: &mut Self::Cache,
        cancellation: &Cancellation,
    ) -> Result<Value> {
        cancellation.check()?;
        cache.push(request.state.as_str().unwrap().into());
        if request.state == "fail" {
            return Err(Error::new("TEST_FAILURE", "failed after updating scratch"));
        }
        Ok(serde_json::json!({"retained":cache.len(),"parameter":parameters}))
    }
}
fn request(state: &str) -> Request {
    Request {
        state: Value::String(state.into()),
        questions: serde_json::from_value(serde_json::json!({"q":{"type":"noul"}})).unwrap(),
    }
}

#[test]
fn batches_reuse_one_cache_and_keep_order_and_per_item_failures() {
    let mut model = Model::with_parameters(TestBackend, 5);
    let mut cache = model.create_cache(CacheOptions::default()).unwrap();
    let outcomes = run(model.predict_batch(
        &[
            request("first"),
            request("second"),
            request("fail"),
            request("last"),
        ],
        None,
        Some(&mut cache),
        &Cancellation::default(),
    ))
    .unwrap();
    let value = serde_json::to_value(outcomes).unwrap();
    assert_eq!(value[0]["result"]["response"]["retained"], 1);
    assert_eq!(value[1]["result"]["response"]["retained"], 2);
    assert_eq!(value[1]["result"]["response"]["parameter"], 5);
    assert_eq!(value[2]["error"]["code"], "TEST_FAILURE");
    assert_eq!(value[3]["result"]["response"]["retained"], 1);
}

#[test]
fn cache_ownership_ephemeral_calls_and_cancellation() {
    let first = Model::new(TestBackend);
    let mut second = Model::new(TestBackend);
    let mut foreign = first.create_cache(CacheOptions::default()).unwrap();
    assert_eq!(
        run(second.predict(
            &request("x"),
            None,
            Some(&mut foreign),
            &Cancellation::default()
        ))
        .unwrap_err()
        .code,
        "CACHE_OWNER"
    );
    for _ in 0..2 {
        assert_eq!(
            run(second.predict(&request("x"), Some(&7), None, &Cancellation::default()))
                .unwrap()
                .response["retained"],
            1
        );
    }
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let batch = run(second.predict_batch(&[request("x"), request("y")], None, None, &cancellation))
        .unwrap();
    assert!(
        batch
            .iter()
            .all(|r| matches!(r, Outcome::Error { error } if error.code=="CANCELLED"))
    );
}

#[test]
fn parameter_types_reject_options_from_other_models() {
    assert!(serde_json::from_str::<DecimaParameters>(r#"{"maxSequenceTokens":1024}"#).is_err());
    assert!(serde_json::from_str::<PplxParameters>(r#"{"long":true}"#).is_err());
    assert!(serde_json::from_str::<KevParameters>(r#"{"long":true}"#).is_err());
    assert!(
        PplxParameters {
            max_sequence_tokens: 0
        }
        .validate()
        .is_err()
    );
    assert!(
        serde_json::from_str::<Request>(r#"{"state":"x","questions":{},"long":true}"#).is_err()
    );
}

#[test]
fn tuning_budget_stops_new_selection_stages() {
    assert!(!TuningBudget::new(false, None, 0.).permits(0.));
    assert!(!TuningBudget::new(true, Some(0), 10.).permits(10.));
    let budget = TuningBudget::new(true, Some(20), 10.);
    assert!(budget.permits(29.));
    assert!(!budget.permits(30.));
    assert!(TuningBudget::new(true, None, 0.).permits(f64::MAX));
}

#[test]
fn unsupported_browser_combinations_identify_selection_and_alternative() {
    for (model, backend, format, code, alternative) in [
        ("unknown", "wasm", "f32", "UNSUPPORTED_MODEL", "decima"),
        ("decima", "cuda", "f32", "UNSUPPORTED_BACKEND", "webgpu"),
        ("kev", "webgpu", "f32", "UNSUPPORTED_BACKEND", "native CPU"),
        (
            "jev-style",
            "webgpu",
            "f32",
            "UNSUPPORTED_BACKEND",
            "native CPU",
        ),
        ("decima", "webgpu", "int8", "UNSUPPORTED_FORMAT", "f32"),
        ("laya", "wasm", "q4-e4", "UNSUPPORTED_FORMAT", "f32, f16"),
        ("pplx-flash", "wasm", "q4-e4", "UNSUPPORTED_FORMAT", "bf16"),
    ] {
        let error = support::browser_selection(model, backend, Some(format)).unwrap_err();
        assert_eq!(error.code, code);
        assert!(error.message.contains(model), "{error}");
        assert!(error.message.contains(alternative), "{error}");
    }
}

#[test]
fn browser_build_exclusions_are_distinct_from_unsupported_combinations() {
    for (model, compiled) in [
        ("decima", cfg!(feature = "decima")),
        ("laya", cfg!(feature = "laya")),
        ("kev", cfg!(feature = "kev")),
        ("jev-style", cfg!(feature = "jev-style")),
    ] {
        let result = support::browser_selection(model, "wasm", None);
        if compiled {
            assert!(result.is_ok(), "{model}: {result:?}");
        } else {
            let error = result.unwrap_err();
            assert_eq!(error.code, "MODEL_NOT_COMPILED");
            assert!(error.message.contains(model));
        }
    }
    if cfg!(feature = "decima") && !cfg!(feature = "webgpu") {
        let error = support::browser_selection("decima", "webgpu", None).unwrap_err();
        assert_eq!(error.code, "BACKEND_NOT_COMPILED");
        assert!(error.message.contains("webgpu"));
    }
    if cfg!(feature = "decima") && !cfg!(feature = "int8") {
        let error = support::browser_selection("decima", "wasm", Some("int8")).unwrap_err();
        assert_eq!(error.code, "FORMAT_NOT_COMPILED");
        assert!(error.message.contains("int8"));
    }
}

#[test]
fn browser_memory_budget_explains_conversion_cost_and_native_alternative() {
    assert!(support::resident_budget(1024 * 1024 * 1024, 3).is_ok());
    let error = support::resident_budget(2 * 1024 * 1024 * 1024, 3).unwrap_err();
    assert_eq!(error.code, "RESOURCE_LIMIT");
    for detail in ["6.00 GiB", "3 GiB", "download size", "native runtime"] {
        assert!(error.message.contains(detail), "{error}");
    }
    assert_eq!(
        support::resident_budget(u64::MAX, 3).unwrap_err().code,
        "RESOURCE_LIMIT"
    );
}

#[test]
fn device_errors_preserve_codes_and_offer_actionable_context() {
    let error = support::load_error(
        Error::engine("required WebGPU precision is unavailable"),
        "pplx-flash",
        "webgpu",
        Some("q4-e4"),
    );
    assert_eq!(error.code, "MISSING_GPU_FEATURE");
    for detail in ["pplx-flash / webgpu / q4-e4", "shader-f16", "Decima"] {
        assert!(error.message.contains(detail), "{error}");
    }
    for message in ["no WebGPU compute adapter", "request WebGPU device"] {
        let error = support::load_error(Error::engine(message), "decima", "webgpu", None);
        assert_eq!(error.code, "GPU_UNAVAILABLE");
        assert!(error.message.contains("HTTPS or localhost"));
        assert!(error.message.contains(message));
    }
    let error = support::load_error(
        Error::engine("GPU storage buffer of 100 bytes exceeds device limits"),
        "decima",
        "webgpu",
        None,
    );
    assert_eq!(error.code, "RESOURCE_LIMIT");
    assert!(error.message.contains("smaller supported checkpoint"));
    assert_eq!(
        Error::engine("mixed FP16 checkpoints require the fp16 Cargo feature").code,
        "FORMAT_NOT_COMPILED"
    );
    assert_eq!(
        Error::engine("INT8 requires the int8 Cargo feature").code,
        "FORMAT_NOT_COMPILED"
    );
    assert_eq!(
        Error::engine("an unrelated engine failure").code,
        "INFERENCE_FAILED"
    );
}
