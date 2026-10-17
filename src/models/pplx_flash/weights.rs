use anyhow::{Result, ensure};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::collections::HashMap;

pub(super) struct Weights<'a>(pub HashMap<String, TensorView<'a>>);
impl<'a> Weights<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        Ok(Self(
            SafeTensors::deserialize(bytes)?
                .tensors()
                .into_iter()
                .collect(),
        ))
    }
    pub fn bytes(&self, name: &str, shape: &[usize]) -> Result<&[u8]> {
        let t = self
            .0
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        ensure!(
            t.shape() == shape && t.dtype() == Dtype::BF16,
            "expected BF16 {name} {shape:?}"
        );
        Ok(t.data())
    }
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self
            .bytes(name, shape)?
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect())
    }
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}
#[derive(serde::Serialize)]
pub struct Range {
    pub name: String,
    pub shape: Vec<usize>,
    pub file: usize,
    pub start: u64,
    pub end: u64,
}
pub struct Index {
    tensors: HashMap<String, (usize, u64, TensorInfo)>,
}
impl Index {
    pub fn new(headers: &[String]) -> Result<Self> {
        let mut tensors = HashMap::new();
        for (file, header) in headers.iter().enumerate() {
            let mut values: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(header)?;
            values.remove("__metadata__");
            for (name, value) in values {
                let info: TensorInfo = serde_json::from_value(value)?;
                ensure!(
                    info.dtype == "BF16" && info.data_offsets[0] <= info.data_offsets[1],
                    "invalid BF16 tensor metadata"
                );
                let elements = info
                    .shape
                    .iter()
                    .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
                    .ok_or_else(|| anyhow::anyhow!("shape overflow"))?;
                ensure!(
                    elements.checked_mul(2) == Some(info.data_offsets[1] - info.data_offsets[0]),
                    "tensor byte size mismatch"
                );
                ensure!(
                    info.data_offsets[1]
                        .checked_add(8 + header.len() as u64)
                        .is_some(),
                    "tensor offset overflow"
                );
                ensure!(
                    tensors
                        .insert(name, (file, 8 + header.len() as u64, info))
                        .is_none(),
                    "duplicate tensor"
                );
            }
        }
        Ok(Self { tensors })
    }
    pub fn layer(&self, index: usize) -> Result<Vec<Range>> {
        ensure!(index < 32, "layer index out of range");
        let prefix = format!("language_model.layers.{index}.");
        let mut result: Vec<_> = self
            .tensors
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .map(|(name, (file, base, t))| Range {
                name: name.clone(),
                shape: t.shape.clone(),
                file: *file,
                start: base + t.data_offsets[0],
                end: base + t.data_offsets[1],
            })
            .collect();
        result.sort_by_key(|r| (r.file, r.start));
        ensure!(!result.is_empty(), "missing layer {index}");
        Ok(result)
    }
    pub fn tensor(&self, name: &str) -> Result<Range> {
        let (file, base, t) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        Ok(Range {
            name: name.into(),
            shape: t.shape.clone(),
            file: *file,
            start: base + t.data_offsets[0],
            end: base + t.data_offsets[1],
        })
    }
}
