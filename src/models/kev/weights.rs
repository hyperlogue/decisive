use anyhow::{Result, ensure};
use safetensors::{Dtype, SafeTensors};

pub(super) struct Weights<'a> {
    base: SafeTensors<'a>,
    adapter: SafeTensors<'a>,
    consumed: std::cell::RefCell<std::collections::BTreeSet<String>>,
}
impl<'a> Weights<'a> {
    pub fn new(bytes: &'a [u8], adapter: &'a [u8], config: &serde_json::Value) -> Result<Self> {
        ensure!(
            config["peft_type"] == "LORA"
                && config["r"] == 16
                && config["lora_alpha"] == 32
                && config["bias"] == "none"
                && config["use_dora"] == false
                && config["use_rslora"] == false
                && config["fan_in_fan_out"] == false
                && config["modules_to_save"].is_null()
                && config["trainable_token_indices"].is_null(),
            "unsupported Kev adapter configuration"
        );
        let adapter = SafeTensors::deserialize(adapter)?;
        ensure!(adapter.len() == 372, "unexpected Kev adapter tensor count");
        Ok(Self {
            base: SafeTensors::deserialize(bytes)?,
            adapter,
            consumed: Default::default(),
        })
    }
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let mut values = Self::read(&self.base, name, shape)?;
        if let Some(tail) = name
            .strip_prefix("model.language_model.")
            .and_then(|n| n.strip_suffix(".weight"))
        {
            let a_name = format!("base_model.model.{tail}.lora_A.weight");
            if self.adapter.tensor(&a_name).is_ok() {
                ensure!(shape.len() == 2, "LoRA requires a matrix");
                let (output, input) = (shape[0], shape[1]);
                let a = Self::read(&self.adapter, &a_name, &[16, input])?;
                let b = Self::read(
                    &self.adapter,
                    &format!("base_model.model.{tail}.lora_B.weight"),
                    &[output, 16],
                )?;
                self.consumed.borrow_mut().insert(a_name);
                self.consumed
                    .borrow_mut()
                    .insert(format!("base_model.model.{tail}.lora_B.weight"));
                let mut delta = vec![0.; output * input];
                unsafe {
                    matrixmultiply::sgemm(
                        output,
                        16,
                        input,
                        1.,
                        b.as_ptr(),
                        16,
                        1,
                        a.as_ptr(),
                        input as isize,
                        1,
                        0.,
                        delta.as_mut_ptr(),
                        input as isize,
                        1,
                    );
                }
                for (v, d) in values.iter_mut().zip(delta) {
                    *v += d * 2.;
                }
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "non-finite merged adapter {name}"
                );
            }
        }
        Ok(values)
    }
    pub fn validate_consumed(&self) -> Result<()> {
        ensure!(
            self.consumed.borrow().len() == self.adapter.len(),
            "unsupported or unused Kev adapter tensors"
        );
        Ok(())
    }
    fn read(weights: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let t = weights.tensor(name)?;
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
        let t = self.base.tensor(name)?;
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
