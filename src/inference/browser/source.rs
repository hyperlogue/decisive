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
    /// A `FileSystemSyncAccessHandle`-like reader (`getSize()` and `read(view, {at})`), which
    /// reads synchronously into WASM memory without an intermediate JS buffer.
    pub fn sync(&self, name: &str) -> Result<Option<JsValue>> {
        let value = self.file(name)?;
        let callable =
            |method: &str| Reflect::get(&value, &method.into()).is_ok_and(|f| f.is_function());
        Ok((value.is_object() && callable("getSize") && callable("read")).then_some(value))
    }
    fn call(handle: &JsValue, method: &str, args: &js_sys::Array) -> Result<JsValue> {
        let function: js_sys::Function = Reflect::get(handle, &method.into())
            .map_err(from_js)?
            .dyn_into()
            .map_err(from_js)?;
        function.apply(handle, args).map_err(from_js)
    }
    /// Fills `output` from `start` through a sync handle; returns false for other sources.
    pub fn read_into(&self, name: &str, start: u64, output: &mut [u8]) -> Result<bool> {
        let Some(handle) = self.sync(name)? else {
            return Ok(false);
        };
        if start + output.len() as u64 > self.size(name)? {
            return Err(Error::new(
                "INVALID_CHECKPOINT",
                format!("Invalid byte range: {name}"),
            ));
        }
        let options = js_sys::Object::new();
        Reflect::set(&options, &"at".into(), &(start as f64).into()).map_err(from_js)?;
        // The view stays valid because nothing allocates before the read returns.
        let view = unsafe { js_sys::Uint8Array::view_mut_raw(output.as_mut_ptr(), output.len()) };
        let read = Self::call(&handle, "read", &js_sys::Array::of2(&view, &options))?;
        if read.as_f64() != Some(output.len() as f64) {
            return Err(Error::new("INVALID_CHECKPOINT", "Incomplete byte range"));
        }
        Ok(true)
    }
    pub fn size(&self, name: &str) -> Result<u64> {
        if let Some(handle) = self.sync(name)? {
            return Self::call(&handle, "getSize", &js_sys::Array::new())?
                .as_f64()
                .map(|n| n as u64)
                .ok_or_else(|| Error::new("INVALID_SOURCE", "Invalid sync handle size"));
        }
        let value = self.file(name)?;
        if let Some(blob) = value.dyn_ref::<web_sys::Blob>() {
            return Ok(blob.size() as u64);
        }
        if let Some(bytes) = value.dyn_ref::<js_sys::Uint8Array>() {
            return Ok(u64::from(bytes.length()));
        }
        Err(Error::new(
            "INVALID_SOURCE",
            "Expected Blob, File, Uint8Array or FileSystemSyncAccessHandle",
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
        if self.sync(name)?.is_some() {
            let mut bytes = vec![0; (end - start) as usize];
            self.read_into(name, start, &mut bytes)?;
            return Ok(Promise::resolve(&JsValue::from(js_sys::Uint8Array::from(&bytes[..]))));
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
