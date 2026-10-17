use anyhow::{Result, ensure};
use safetensors::{Dtype, SafeTensors};

pub struct Weights<'a> {
    tensors: SafeTensors<'a>,
    #[cfg(feature = "int8")]
    int8_group: Option<usize>,
    #[cfg(feature = "int8")]
    fp16_projections: bool,
    #[cfg(feature = "int8")]
    half_vocabulary_scales: bool,
}
#[cfg(feature = "int8")]
pub(super) const INT8_FORMAT: &str = "beam-decima-int8-v1";
#[cfg(feature = "int8")]
pub(super) const SCALE_SUFFIX: &str = ".beam_scales";

#[cfg(feature = "int8")]
pub(super) struct Int8View<'a> {
    pub values: &'a [i8],
    pub scales: Vec<f32>,
    pub group: usize,
}
#[cfg(feature = "int8")]
pub(super) struct Int8HalfView<'a> {
    pub values: &'a [i8],
    pub scales: Vec<half::f16>,
    pub group: usize,
}
impl<'a> Weights<'a> {
    #[cfg(feature = "int8")]
    fn half_metadata(&self, name: &str, shape: &[usize]) -> Result<Vec<half::f16>> {
        let t = self.tensors.tensor(name)?;
        ensure!(
            t.dtype() == Dtype::F16 && t.shape() == shape,
            "invalid half metadata: {name}"
        );
        let values: Vec<half::f16> = t
            .data()
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite half metadata: {name}"
        );
        Ok(values)
    }
    #[cfg(feature = "int8")]
    pub(super) fn vocabulary_int8_half(
        &self,
        shape: &[usize; 2],
    ) -> Result<Option<Int8HalfView<'a>>> {
        if !self.half_vocabulary_scales {
            return Ok(None);
        }
        let name = "embeddings.word_embeddings.weight";
        let t = self.tensors.tensor(name)?;
        let group = self
            .int8_group
            .ok_or_else(|| anyhow::anyhow!("unmarked half-scale vocabulary"))?;
        ensure!(
            t.dtype() == Dtype::I8 && t.shape() == shape && shape[1] % group == 0,
            "invalid half-scale vocabulary shape"
        );
        let values: &[i8] = bytemuck::cast_slice(t.data());
        ensure!(
            values.iter().all(|&q| q != i8::MIN),
            "invalid INT8 vocabulary range"
        );
        let scales = self.half_metadata(
            &format!("{name}{SCALE_SUFFIX}"),
            &[shape[0], shape[1] / group],
        )?;
        ensure!(
            scales.iter().all(|v| v.to_f32() > 0.),
            "invalid half vocabulary scale"
        );
        Ok(Some(Int8HalfView {
            values,
            scales,
            group,
        }))
    }
    #[cfg(not(feature = "int8"))]
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let tensors = SafeTensors::deserialize(bytes)?;
        let (_, metadata) = SafeTensors::read_metadata(bytes)?;
        ensure!(
            metadata
                .metadata()
                .as_ref()
                .is_none_or(|m| !m.contains_key("beam_format")),
            "compact checkpoints require the int8 Cargo feature"
        );
        Ok(Self { tensors })
    }
    #[cfg(feature = "int8")]
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let tensors = SafeTensors::deserialize(bytes)?;
        let (_, metadata) = SafeTensors::read_metadata(bytes)?;
        let format = metadata
            .metadata()
            .as_ref()
            .and_then(|m| m.get("beam_format"));
        ensure!(
            format.is_none_or(|v| v == INT8_FORMAT),
            "unknown Decima compact format"
        );
        let mut int8_group = None;
        let mut fp16_projections = false;
        let mut half_vocabulary_scales = false;
        if format.is_some() {
            let m = metadata.metadata().as_ref().unwrap();
            ensure!(
                m.get("source_revision")
                    .is_some_and(|v| v == super::REVISION)
                    && m.get("group_size").is_some_and(|v| v == "16" || v == "32")
                    && m.get("recipe")
                        .is_some_and(|v| v == "symmetric-max-abs-round-away"),
                "unsupported Decima compact provenance/recipe"
            );
            ensure!(
                m.get("projection_storage")
                    .is_none_or(|v| v == "int8" || v == "f16"),
                "unsupported compact projection storage"
            );
            fp16_projections = m.get("projection_storage").is_some_and(|v| v == "f16");
            int8_group = Some(m["group_size"].parse()?);
            ensure!(
                m.get("vocabulary_scale_storage")
                    .is_none_or(|v| v == "f32" || v == "f16"),
                "unsupported vocabulary scale storage"
            );
            half_vocabulary_scales = m
                .get("vocabulary_scale_storage")
                .is_some_and(|v| v == "f16");
        }
        ensure!(
            !(fp16_projections || half_vocabulary_scales) || cfg!(feature = "fp16"),
            "mixed FP16 checkpoints require the fp16 Cargo feature"
        );
        Ok(Self {
            tensors,
            int8_group,
            fp16_projections,
            half_vocabulary_scales,
        })
    }
    #[cfg(feature = "int8")]
    pub(super) fn f16(&self, name: &str, shape: &[usize]) -> Result<Option<Vec<f32>>> {
        let t = self.tensors.tensor(name)?;
        if t.dtype() != Dtype::F16 {
            return Ok(None);
        }
        ensure!(
            self.fp16_projections && name.starts_with("encoder.layer.") && t.shape() == shape,
            "invalid compact FP16 tensor {name}"
        );
        let values: Vec<f32> = t
            .data()
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite compact FP16 tensor {name}"
        );
        Ok(Some(values))
    }
    pub fn float_bytes(&self, name: &str, shape: &[usize]) -> Result<&'a [u8]> {
        let t = self.tensors.tensor(name)?;
        ensure!(
            t.shape() == shape && t.dtype() == Dtype::F32,
            "Decima tensor {name}: expected F32 {shape:?}, got {:?} {:?}",
            t.dtype(),
            t.shape()
        );
        Ok(t.data())
    }
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let v: Vec<f32> = self
            .float_bytes(name, shape)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            v.iter().all(|x| x.is_finite()),
            "non-finite Decima tensor {name}"
        );
        Ok(v)
    }
    #[cfg(feature = "int8")]
    pub(super) fn int8(&self, name: &str, shape: &[usize; 2]) -> Result<Option<Int8View<'a>>> {
        let t = self.tensors.tensor(name)?;
        if t.dtype() == Dtype::F32 {
            return Ok(None);
        }
        let group = self
            .int8_group
            .ok_or_else(|| anyhow::anyhow!("unmarked compact INT8 tensor"))?;
        ensure!(
            t.dtype() == Dtype::I8 && t.shape() == shape && shape[1] % group == 0,
            "invalid compact Decima tensor {name}"
        );
        let values: &[i8] = bytemuck::cast_slice(t.data());
        ensure!(
            values.iter().all(|&q| q != i8::MIN),
            "INT8 uses the symmetric [-127,127] range"
        );
        let scales = self.get(
            &format!("{name}{SCALE_SUFFIX}"),
            &[shape[0], shape[1] / group],
        )?;
        ensure!(
            scales.iter().all(|&s| s >= f32::MIN_POSITIVE),
            "invalid INT8 scale in {name}"
        );
        Ok(Some(Int8View {
            values,
            scales,
            group,
        }))
    }
}

#[cfg(feature = "int8")]
pub(super) fn quantize_groups(bytes: &[u8], group: usize) -> Result<(Vec<i8>, Vec<f32>)> {
    ensure!(
        matches!(group, 16 | 32) && bytes.len() % (group * 4) == 0,
        "invalid INT8 source groups"
    );
    let mut output = Vec::with_capacity(bytes.len() / 4);
    let mut scales = Vec::with_capacity(bytes.len() / 4 / group);
    for bytes in bytes.chunks_exact(group * 4) {
        let mut values = [0_f32; 32];
        let mut max = 0_f32;
        for (v, bytes) in values.iter_mut().zip(bytes.chunks_exact(4)) {
            *v = f32::from_le_bytes(bytes.try_into().unwrap());
            ensure!(v.is_finite(), "non-finite INT8 source weight");
            max = max.max(v.abs());
        }
        let scale = (max / 127.).max(f32::MIN_POSITIVE);
        scales.push(scale);
        output.extend(
            values[..group]
                .iter()
                .map(|&v| (v / scale).round().clamp(-127., 127.) as i8),
        );
    }
    Ok((output, scales))
}

#[cfg(all(test, feature = "int8", feature = "fp16"))]
mod tests {
    use super::*;
    use safetensors::tensor::{TensorView, serialize};
    use std::collections::HashMap;
    #[test]
    fn half_int8_scales_are_explicit_and_vocabulary_only() {
        for (marked, bits, good) in [
            (true, 0x1400_u16, true),
            (false, 0x1400, false),
            (true, 0, false),
            (true, 0x7c00, false),
            (true, 0x7e00, false),
        ] {
            let values = [1_u8; 32];
            let scales = bits.to_le_bytes();
            let mut metadata = HashMap::from([
                ("beam_format".into(), INT8_FORMAT.into()),
                ("source_revision".into(), super::super::REVISION.into()),
                ("group_size".into(), "32".into()),
                ("recipe".into(), "symmetric-max-abs-round-away".into()),
            ]);
            if marked {
                metadata.insert("vocabulary_scale_storage".into(), "f16".into());
            }
            let serialized = serialize(
                [
                    (
                        "embeddings.word_embeddings.weight",
                        TensorView::new(Dtype::I8, vec![1, 32], &values).unwrap(),
                    ),
                    (
                        "embeddings.word_embeddings.weight.beam_scales",
                        TensorView::new(Dtype::F16, vec![1, 1], &scales).unwrap(),
                    ),
                ],
                Some(metadata),
            )
            .unwrap();
            let w = Weights::new(&serialized).unwrap();
            let half = w.vocabulary_int8_half(&[1, 32]);
            assert_eq!(half.as_ref().ok().and_then(|q| q.as_ref()).is_some(), good);
            if let Some(q) = half.ok().flatten() {
                assert_eq!(q.scales.len(), q.scales.capacity());
            }
            assert!(
                w.int8("embeddings.word_embeddings.weight", &[1, 32])
                    .is_err()
            );
        }
    }
    fn fixture(format: Option<&str>, q: i8, scale: f32) -> Vec<u8> {
        let data = vec![q as u8; 32];
        let bytes = scale.to_le_bytes();
        let metadata = format.map(|f| {
            HashMap::from([
                ("beam_format".into(), f.into()),
                ("source_revision".into(), super::super::REVISION.into()),
                ("group_size".into(), "32".into()),
                ("recipe".into(), "symmetric-max-abs-round-away".into()),
            ])
        });
        serialize(
            [
                (
                    "weight",
                    TensorView::new(Dtype::I8, vec![1, 32], &data).unwrap(),
                ),
                (
                    "weight.beam_scales",
                    TensorView::new(Dtype::F32, vec![1, 1], &bytes).unwrap(),
                ),
            ],
            metadata,
        )
        .unwrap()
    }
    #[test]
    fn compact_format_rejects_unmarked_invalid_or_mislabeled_data() {
        let good = fixture(Some(INT8_FORMAT), 127, 0.1);
        let w = Weights::new(&good).unwrap();
        assert_eq!(w.int8("weight", &[1, 32]).unwrap().unwrap().values[0], 127);
        assert!(w.get("weight", &[1, 32]).is_err());
        assert!(w.int8("weight", &[2, 16]).is_err());
        for (format, q, scale) in [
            (None, 1, 1.),
            (Some(INT8_FORMAT), -128, 1.),
            (Some(INT8_FORMAT), 1, 0.),
            (Some(INT8_FORMAT), 1, f32::NAN),
        ] {
            let bytes = fixture(format, q, scale);
            let w = Weights::new(&bytes).unwrap();
            assert!(w.int8("weight", &[1, 32]).is_err());
        }
        assert!(Weights::new(&fixture(Some("future-format"), 1, 1.)).is_err());
    }
    #[test]
    fn mixed_fp16_projections_require_marked_finite_data() {
        for bits in [0x3c00_u16, 0x7c00, 0x7e00] {
            let bytes = bits.to_le_bytes();
            let serialized = serialize(
                [(
                    "encoder.layer.0.test.weight",
                    TensorView::new(Dtype::F16, vec![1, 1], &bytes).unwrap(),
                )],
                Some(HashMap::from([
                    ("beam_format".into(), INT8_FORMAT.into()),
                    ("source_revision".into(), super::super::REVISION.into()),
                    ("group_size".into(), "16".into()),
                    ("projection_storage".into(), "f16".into()),
                    ("recipe".into(), "symmetric-max-abs-round-away".into()),
                ])),
            )
            .unwrap();
            let weights = Weights::new(&serialized).unwrap();
            let values = weights.f16("encoder.layer.0.test.weight", &[1, 1]);
            if bits == 0x3c00 {
                assert_eq!(values.unwrap().unwrap(), [1.]);
            } else {
                assert!(values.is_err());
            }
        }
    }
    #[test]
    fn sixteen_value_groups_keep_independent_scales_and_file_shape() {
        let values: Vec<f32> = (0..64)
            .map(|i| if i < 16 { i as f32 } else { i as f32 / 1000. })
            .collect();
        let (q, scales) = quantize_groups(bytemuck::cast_slice(&values), 16).unwrap();
        assert_eq!(q.len(), 64);
        assert_eq!(scales.len(), 4);
        assert!(scales[0] > scales[1] * 100.);
        let bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = serialize(
            [
                (
                    "weight",
                    TensorView::new(Dtype::I8, vec![2, 32], bytemuck::cast_slice(&q)).unwrap(),
                ),
                (
                    "weight.beam_scales",
                    TensorView::new(Dtype::F32, vec![2, 2], &bytes).unwrap(),
                ),
            ],
            Some(HashMap::from([
                ("beam_format".into(), INT8_FORMAT.into()),
                ("source_revision".into(), super::super::REVISION.into()),
                ("group_size".into(), "16".into()),
                ("recipe".into(), "symmetric-max-abs-round-away".into()),
            ])),
        )
        .unwrap();
        let w = Weights::new(&data).unwrap();
        let imported = w.int8("weight", &[2, 32]).unwrap().unwrap();
        assert_eq!(imported.group, 16);
        assert_eq!(imported.values, q);
        assert_eq!(imported.scales, scales);
    }
    #[test]
    fn quantizer_handles_rounding_extrema_zero_and_nonfinite() {
        let mut values = [0_f32; 32];
        values[..6].copy_from_slice(&[-127., 127., -0.5, 0.5, -1.5, 1.5]);
        let (q, s) = quantize_groups(bytemuck::cast_slice(&values), 32).unwrap();
        assert_eq!(&q[..6], &[-127, 127, -1, 1, -2, 2]);
        assert_eq!(s, [1.]);
        let (q, s) = quantize_groups(&[0; 128], 32).unwrap();
        assert!(q.iter().all(|&v| v == 0));
        assert_eq!(s, [f32::MIN_POSITIVE]);
        values[8] = f32::NAN;
        assert!(quantize_groups(bytemuck::cast_slice(&values), 32).is_err());
    }
}
