use super::{Error, Result};

pub(crate) fn browser_selection(model: &str, backend: &str, format: Option<&str>) -> Result<()> {
    let compiled = match model {
        "decima" => cfg!(feature = "decima"),
        "laya" => cfg!(feature = "laya"),
        "pplx-flash" => cfg!(feature = "pplx-flash"),
        "kev" => cfg!(feature = "kev"),
        "jev-style" => cfg!(feature = "jev-style"),
        _ => {
            return Err(Error::new(
                "UNSUPPORTED_MODEL",
                format!(
                    "Unknown model '{model}'. Choose decima, laya, pplx-flash, kev, or jev-style."
                ),
            ));
        }
    };
    if !matches!(backend, "wasm" | "webgpu") {
        return Err(Error::new(
            "UNSUPPORTED_BACKEND",
            format!(
                "'{backend}' is not a browser backend for '{model}'. Choose wasm or webgpu; native CPU execution uses the Rust interface."
            ),
        ));
    }
    if matches!(model, "kev" | "jev-style") && backend == "webgpu" {
        return Err(Error::new(
            "UNSUPPORTED_BACKEND",
            format!(
                "'{model}' has no WebGPU implementation. Use the native CPU backend, or wasm if the checkpoint fits the browser memory budget."
            ),
        ));
    }
    let format = format.unwrap_or(if model == "pplx-flash" { "bf16" } else { "f32" });
    let formats: &[&str] = match (model, backend) {
        ("decima", "wasm") => &["f32", "int8", "int8-weights", "int8-full", "int8-compact"],
        ("laya", _) => &["f32", "f16"],
        ("pplx-flash", "webgpu") => &["bf16", "q4-e4"],
        ("pplx-flash", "wasm") => &["bf16"],
        _ => &["f32"],
    };
    if !formats.contains(&format) {
        return Err(Error::new(
            "UNSUPPORTED_FORMAT",
            format!(
                "'{model}' on '{backend}' does not support weight format '{format}'. Supported formats: {}. Choose a compatible checkpoint and format; no automatic conversion or backend fallback is performed.",
                formats.join(", ")
            ),
        ));
    }
    if !compiled {
        return Err(Error::new(
            "MODEL_NOT_COMPILED",
            format!(
                "'{model}' is not included in this WASM build. Rebuild with the '{model}' Cargo feature, or choose a model included in this build."
            ),
        ));
    }
    if (backend == "webgpu" || model == "pplx-flash") && !cfg!(feature = "webgpu") {
        let message = if model == "pplx-flash" && backend == "wasm" {
            "The shared Pplx browser loader is not included in this WASM build. Rebuild with 'pplx-webgpu'; the selected wasm backend will still execute on CPU.".to_owned()
        } else {
            format!(
                "'{model}' on 'webgpu' is not included in this WASM build. Rebuild with the 'webgpu' Cargo feature, or choose a compiled backend."
            )
        };
        return Err(Error::new("BACKEND_NOT_COMPILED", message));
    }
    let feature = match (model, format) {
        ("pplx-flash", "q4-e4")
            if !cfg!(any(feature = "pplx-webgpu", feature = "kernel-checks")) =>
        {
            Some("pplx-webgpu")
        }
        ("decima", f) if f.starts_with("int8") && !cfg!(feature = "int8") => Some("int8"),
        ("laya", "f16") if !cfg!(feature = "fp16") => Some("fp16"),
        _ => None,
    };
    if let Some(feature) = feature {
        return Err(Error::new(
            "FORMAT_NOT_COMPILED",
            format!(
                "'{model}' on '{backend}' supports '{format}', but this WASM build does not include it. Rebuild with the '{feature}' Cargo feature, or use a compiled weight format."
            ),
        ));
    }
    Ok(())
}

pub(crate) fn resident_budget(bytes: u64, amplification: u64) -> Result<()> {
    const LIMIT: u64 = 3 * 1024 * 1024 * 1024;
    let peak = bytes.checked_mul(amplification).ok_or_else(|| Error::new("RESOURCE_LIMIT", "The checkpoint's estimated conversion memory overflows the supported size. Use the native runtime."))?;
    if peak > LIMIT {
        return Err(Error::new(
            "RESOURCE_LIMIT",
            format!(
                "This checkpoint's resident conversion is estimated to need {peak} bytes ({:.2} GiB), exceeding the browser loader's 3 GiB budget. The download size is not the runtime memory requirement. Use a smaller supported checkpoint or the native runtime.",
                peak as f64 / (1024. * 1024. * 1024.)
            ),
        ));
    }
    Ok(())
}

pub(crate) fn load_error(
    mut error: Error,
    model: &str,
    backend: &str,
    format: Option<&str>,
) -> Error {
    let format = format.unwrap_or(if model == "pplx-flash" { "bf16" } else { "f32" });
    let selection = format!("{model} / {backend} / {format}");
    error.message = match error.code.as_str() {
        "MISSING_GPU_FEATURE" if model == "pplx-flash" && format == "q4-e4" => format!(
            "{selection} requires shader-f16, which this browser or GPU does not provide. Choose Decima with WebGPU/f32, or use a browser/device with shader-f16."
        ),
        "GPU_UNAVAILABLE" => format!(
            "Cannot create a WebGPU device for {selection}. Use a WebGPU-enabled browser/device in a secure context (HTTPS or localhost), or choose another supported backend. Details: {}",
            error.message
        ),
        "RESOURCE_LIMIT" if error.message.contains("exceeds device limits") => format!(
            "{selection}: {}. Use a smaller supported checkpoint, a shorter input, or a device with higher GPU limits.",
            error.message
        ),
        "RESOURCE_LIMIT" => format!("{selection}: {}", error.message),
        _ => error.message,
    };
    error
}
