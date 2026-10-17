use super::{Precision, weights::Weights};
use anyhow::{Result, ensure};

pub(super) enum Embeddings {
    Float(Vec<f32>),
    #[cfg(feature = "int8")]
    Int8 {
        values: Vec<i8>,
        scales: Vec<f32>,
        group: usize,
    },
    #[cfg(feature = "int8")]
    Int8Half {
        values: Vec<i8>,
        scales: Vec<half::f16>,
        group: usize,
    },
}

impl Embeddings {
    #[cfg(not(feature = "int8"))]
    pub fn load(w: &Weights<'_>, precision: Precision) -> Result<Self> {
        ensure!(
            precision == Precision::F32,
            "INT8 requires the int8 Cargo feature"
        );
        Ok(Self::Float(w.get(
            "embeddings.word_embeddings.weight",
            &[250037, super::D],
        )?))
    }

    #[cfg(feature = "int8")]
    pub fn load(w: &Weights<'_>, precision: Precision) -> Result<Self> {
        let name = "embeddings.word_embeddings.weight";
        let shape = [250037, super::D];
        if let Some(q) = w.vocabulary_int8_half(&shape)? {
            ensure!(
                matches!(precision, Precision::Int8Full | Precision::Int8Compact),
                "half-scale vocabulary requires INT8 compact precision"
            );
            return Ok(Self::Int8Half {
                values: q.values.to_vec(),
                scales: q.scales,
                group: q.group,
            });
        }
        if let Some(q) = w.int8(name, &shape)? {
            ensure!(
                matches!(precision, Precision::Int8Full | Precision::Int8Compact),
                "packed Decima embeddings require int8-full or int8-compact precision"
            );
            return Ok(Self::Int8 {
                values: q.values.to_vec(),
                scales: q.scales,
                group: q.group,
            });
        }
        if !matches!(precision, Precision::Int8Full | Precision::Int8Compact) {
            return Ok(Self::Float(w.get(name, &shape)?));
        }
        let bytes = w.float_bytes(name, &shape)?;
        let (values, scales) = super::weights::quantize_groups(bytes, 32)?;
        Ok(Self::Int8 {
            values,
            scales,
            group: 32,
        })
    }

    pub fn row(&self, id: usize, output: &mut [f32]) {
        assert_eq!(output.len(), super::D);
        let start = id * super::D;
        match self {
            Self::Float(values) => output.copy_from_slice(&values[start..start + super::D]),
            #[cfg(feature = "int8")]
            Self::Int8Half {
                values,
                scales,
                group: block,
            } => {
                for (g, dst) in output.chunks_exact_mut(*block).enumerate() {
                    let scale = scales[start / *block + g].to_f32();
                    for (j, v) in dst.iter_mut().enumerate() {
                        *v = values[start + g * *block + j] as f32 * scale;
                    }
                }
            }
            #[cfg(feature = "int8")]
            Self::Int8 {
                values,
                scales,
                group: block,
            } => {
                for (g, group) in output.chunks_exact_mut(*block).enumerate() {
                    let scale = scales[start / *block + g];
                    for (i, value) in group.iter_mut().enumerate() {
                        *value = values[start + g * *block + i] as f32 * scale;
                    }
                }
            }
        }
    }
}

#[cfg(all(test, feature = "int8"))]
mod tests {
    use super::*;

    #[test]
    fn row_lookup_uses_each_groups_scale_and_requested_token() {
        let values: Vec<i8> = (0..super::super::D * 3)
            .map(|i| ((i * 31 % 255) as i16 - 127) as i8)
            .collect();
        for block in [16, 32] {
            let scales: Vec<f32> = (0..values.len() / block)
                .map(|i| (i + 1) as f32 / 127.)
                .collect();
            let embeddings = Embeddings::Int8 {
                values: values.clone(),
                scales: scales.clone(),
                group: block,
            };
            let half_scales: Vec<half::f16> =
                scales.iter().map(|s| half::f16::from_f32(*s)).collect();
            let half_embeddings = Embeddings::Int8Half {
                values: values.clone(),
                scales: half_scales.clone(),
                group: block,
            };
            for token in [2, 0, 1, 2] {
                let mut output = vec![f32::NAN; super::super::D];
                embeddings.row(token, &mut output);
                for (i, &value) in output.iter().enumerate() {
                    let index = token * super::super::D + i;
                    assert_eq!(
                        value.to_bits(),
                        (values[index] as f32 * scales[index / block]).to_bits()
                    );
                }
                half_embeddings.row(token, &mut output);
                for (i, &value) in output.iter().enumerate() {
                    let index = token * super::super::D + i;
                    assert_eq!(
                        value.to_bits(),
                        (values[index] as f32 * half_scales[index / block].to_f32()).to_bits()
                    );
                }
            }
        }
    }
}
