use anyhow::{Result, ensure};
use safetensors::{Dtype, SafeTensors};

pub(super) struct Weights<'a>(SafeTensors<'a>);
impl<'a> Weights<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        Ok(Self(SafeTensors::deserialize(bytes)?))
    }
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let t = self.0.tensor(name)?;
        ensure!(
            t.shape() == shape,
            "tensor {name}: expected {shape:?}, got {:?}",
            t.shape()
        );
        let v: Vec<f32> = match t.dtype() {
            Dtype::BF16 => t
                .data()
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
            Dtype::F16 => t
                .data()
                .chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            Dtype::F32 => t
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            dt => anyhow::bail!("unsupported {name} dtype {dt:?}"),
        };
        ensure!(v.iter().all(|v| v.is_finite()), "non-finite tensor {name}");
        Ok(v)
    }
    pub fn embedding(&self, name: &str, vocab: usize, width: usize) -> Result<Vec<u16>> {
        let t = self.0.tensor(name)?;
        ensure!(
            t.shape() == [vocab, width] && t.dtype() == Dtype::BF16,
            "expected BF16 embedding [{vocab}, {width}]"
        );
        let v: Vec<u16> = t
            .data()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        ensure!(
            v.iter().all(|v| v & 0x7f80 != 0x7f80),
            "non-finite embedding"
        );
        Ok(v)
    }
}
