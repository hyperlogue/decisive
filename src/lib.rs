#[cfg(all(
    feature = "cpu",
    target_arch = "x86_64",
    any(
        feature = "laya",
        feature = "decima",
        feature = "kev",
        feature = "jev-style",
        feature = "pplx-flash"
    )
))]
#[path = "kernels/avx2_exp.rs"]
mod avx2_exp;
#[cfg(feature = "decima")]
#[path = "models/decima/mod.rs"]
pub mod decima;
#[cfg(feature = "jev-style")]
#[path = "models/jev_style/mod.rs"]
pub mod jev_style;
#[cfg(feature = "kev")]
#[path = "models/kev/mod.rs"]
pub mod kev;
#[cfg(feature = "laya")]
#[path = "models/laya/mod.rs"]
pub mod laya;
pub mod length;

#[cfg(feature = "kernel-checks")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name=wasmKernelChecks)]
pub fn wasm_kernel_checks() -> Result<String, wasm_bindgen::JsValue> {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        #[cfg(any(feature = "laya", feature = "decima", feature = "pplx-flash"))]
        let js_error = |e: anyhow::Error| wasm_bindgen::JsValue::from_str(&e.to_string());
        #[cfg(feature = "laya")]
        let laya = laya::cpu::wasm_kernel_checks().map_err(js_error)?;
        #[cfg(feature = "decima")]
        let decima = decima::wasm_kernel_checks().map_err(js_error)?;
        #[cfg(feature = "pplx-flash")]
        let pplx = pplx_flash::kernel_checks().map_err(js_error)?;
        let mut checks = serde_json::Map::new();
        #[cfg(feature = "pplx-flash")]
        checks.insert("pplx-flash".into(), pplx);
        #[cfg(feature = "laya")]
        checks.insert("laya".into(), laya);
        #[cfg(feature = "decima")]
        checks.insert("decima".into(), decima);
        return Ok(serde_json::Value::Object(checks).to_string());
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    Err(wasm_bindgen::JsValue::from_str(
        "kernel checks require SIMD128 Wasm",
    ))
}

#[cfg(feature = "pplx-flash")]
#[path = "models/pplx_flash/mod.rs"]
pub mod pplx_flash;
