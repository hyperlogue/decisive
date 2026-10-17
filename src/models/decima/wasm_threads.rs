use std::sync::atomic::{AtomicUsize, Ordering};
use wasm_bindgen::prelude::*;

const STARTING: usize = usize::MAX;
static WORKERS: AtomicUsize = AtomicUsize::new(0);

pub(super) fn count() -> usize {
    match WORKERS.load(Ordering::Acquire) {
        STARTING => 0,
        n => n,
    }
}

#[wasm_bindgen(js_name = initDecimaThreadPool)]
pub async fn init(threads: usize) -> Result<(), JsValue> {
    if !(1..=64).contains(&threads) {
        return Err(JsValue::from_str("Decima threads must be in 1..=64"));
    }
    if js_sys::Reflect::has(&js_sys::global(), &JsValue::from_str("document"))? {
        return Err(JsValue::from_str(
            "Initialize the Decima thread pool in a dedicated Web Worker",
        ));
    }
    let isolated =
        js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("crossOriginIsolated"))?
            .as_bool()
            .unwrap_or(false);
    if !isolated {
        return Err(JsValue::from_str(
            "Decima threads require a cross-origin isolated page",
        ));
    }
    match WORKERS.compare_exchange(0, STARTING, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => {}
        Err(n) if n == threads => return Ok(()),
        Err(STARTING) => {
            return Err(JsValue::from_str(
                "Decima thread pool is starting or its initialization failed",
            ));
        }
        Err(_) => {
            return Err(JsValue::from_str(
                "Decima thread count cannot change after pool initialization",
            ));
        }
    }
    // Partial initialization cannot retry in this Rayon registry; reload the owning Worker.
    wasm_bindgen_futures::JsFuture::from(wasm_bindgen_rayon::init_thread_pool(threads)).await?;
    WORKERS.store(threads, Ordering::Release);
    Ok(())
}
