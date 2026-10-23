use super::*;

#[derive(Clone)]
pub(super) struct Source {
    files: JsValue,
}
impl Source {
    pub fn new(files: JsValue) -> Self {
        Self { files }
    }
    pub fn resident_budget(&self, names: &[&str], amplification: u64) -> Result<()> {
        let bytes = names.iter().try_fold(0u64, |total, name| {
            total
                .checked_add(self.size(name)?)
                .ok_or_else(|| Error::new("RESOURCE_LIMIT", "Checkpoint size overflow"))
        })?;
        support::resident_budget(bytes, amplification)
    }
    pub fn file(&self, name: &str) -> Result<JsValue> {
        let value = Reflect::get(&self.files, &name.into()).map_err(from_js)?;
        if value.is_null() || value.is_undefined() {
            return Err(Error::new(
                "MISSING_FILE",
                format!("Missing checkpoint file: {name}"),
            ));
        }
        Ok(value)
    }
    pub fn size(&self, name: &str) -> Result<u64> {
        let value = self.file(name)?;
        if let Some(blob) = value.dyn_ref::<web_sys::Blob>() {
            return Ok(blob.size() as u64);
        }
        if let Some(bytes) = value.dyn_ref::<js_sys::Uint8Array>() {
            return Ok(u64::from(bytes.length()));
        }
        Err(Error::new(
            "INVALID_SOURCE",
            "Expected Blob, File or Uint8Array",
        ))
    }
    pub fn start(&self, name: &str, start: u64, end: u64) -> Result<Promise> {
        if start > end
            || end > self.size(name)?
            || end > 9_007_199_254_740_991
            || end - start > usize::MAX as u64
        {
            return Err(Error::new(
                "INVALID_CHECKPOINT",
                format!("Invalid byte range: {name}"),
            ));
        }
        let value = self.file(name)?;
        if let Some(blob) = value.dyn_ref::<web_sys::Blob>() {
            return Ok(blob
                .slice_with_f64_and_f64(start as f64, end as f64)
                .map_err(from_js)?
                .array_buffer());
        }
        let bytes = value.dyn_into::<js_sys::Uint8Array>().map_err(from_js)?;
        Ok(Promise::resolve(&JsValue::from(
            bytes.subarray(start as u32, end as u32),
        )))
    }
    pub async fn finish(promise: Promise, expected: u64) -> Result<Vec<u8>> {
        let buffer = JsFuture::from(promise).await.map_err(from_js)?;
        let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
        if bytes.len() as u64 != expected {
            return Err(Error::new("INVALID_CHECKPOINT", "Incomplete byte range"));
        }
        Ok(bytes)
    }
    pub async fn read(&self, name: &str) -> Result<Vec<u8>> {
        let size = self.size(name)?;
        Self::finish(self.start(name, 0, size)?, size).await
    }
}
