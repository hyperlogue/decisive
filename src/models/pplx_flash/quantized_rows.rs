use super::int8::{GROUP, group_scale_bits, pack_bits};
use anyhow::{Result, ensure};
use rayon::prelude::*;


pub(super) struct Embedding {
    data: Vec<u8>,
    scales: Vec<f32>,
    columns: usize,
    bits: u8,
    nf4: bool,
    offsets: Vec<f32>,
}

impl Embedding {
    #[cfg(test)]
    pub fn from_bf16(bytes: &[u8], columns: usize, bits: u8, avx2: bool) -> Result<Self> {
        Self::from_bf16_optimized(bytes, columns, bits, avx2, false, false, false)
    }

    pub fn from_bf16_optimized(
        bytes: &[u8],
        columns: usize,
        bits: u8,
        avx2: bool,
        optimized: bool,
        nf4: bool,
        affine: bool,
    ) -> Result<Self> {
        ensure!(!nf4 || bits == 4, "NF4 requires four-bit embeddings");
        ensure!(
            !affine || (bits == 4 && !nf4),
            "affine fitting requires integer Q4"
        );
        ensure!(
            [4, 8].contains(&bits) && columns > 0 && columns % GROUP == 0,
            "invalid embedding precision or width"
        );
        ensure!(bytes.len() % (columns * 2) == 0, "invalid embedding shape");
        let groups = bytes.len() / (GROUP * 2);
        let stride = GROUP * bits as usize / 8;
        let mut data = vec![0; groups * stride];
        let mut scales = vec![0.; groups];
        let mut offsets = vec![0.; groups];
        data.par_chunks_mut(stride)
            .zip(scales.par_iter_mut())
            .zip(offsets.par_iter_mut())
            .zip(bytes.par_chunks(GROUP * 2))
            .try_for_each(|(((out, scale), offset), source)| -> Result<()> {
                let values: [f32; GROUP] = std::array::from_fn(|i| {
                    f32::from_bits(
                        (u16::from_le_bytes([source[2 * i], source[2 * i + 1]]) as u32) << 16,
                    )
                });
                ensure!(values.iter().all(|v| v.is_finite()), "non-finite embedding");
                let mut codes = [0; GROUP];
                *scale = if affine {
                    let (s, b) = super::quantization::affine(&values, &mut codes, avx2);
                    *offset = b;
                    s
                } else if bits == 4 && nf4 {
                    super::quantization::nf4(&values, &mut codes)
                } else if bits == 4 && optimized {
                    super::quantization::mse(&values, &mut codes, avx2)
                } else {
                    let s = group_scale_bits(&values, avx2, bits);
                    pack_bits(&values, &mut codes, s, avx2, bits);
                    s
                };
                for (i, out) in out.iter_mut().enumerate() {
                    *out = if bits == 4 {
                        (codes[2 * i] + 8) as u8 | ((codes[2 * i + 1] + 8) as u8) << 4
                    } else {
                        codes[i] as u8
                    };
                }
                Ok(())
            })?;
        Ok(Self {
            data,
            scales,
            columns,
            bits,
            nf4,
            offsets: if affine { offsets } else { Vec::new() },
        })
    }

    pub fn append(&self, id: usize, out: &mut Vec<f32>) {
        let first = id * self.columns;
        let start = out.len();
        out.resize(start + self.columns, 0.);
        for (i, value) in out[start..].iter_mut().enumerate() {
            let at = first + i;
            let code = if self.bits == 4 {
                ((self.data[at / 2] >> (at % 2 * 4)) & 15) as i8 - 8
            } else {
                self.data[at] as i8
            };
            let value_code = if self.nf4 {
                super::quantization::NF4[(code + 8) as usize]
            } else {
                code as f32
            };
            *value = value_code * self.scales[at / GROUP];
            if !self.offsets.is_empty() {
                *value += self.offsets[at / GROUP];
            }
        }
    }

    pub fn bytes(&self) -> usize {
        self.data.len() + (self.scales.len() + self.offsets.len()) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_embedding_offsets_and_storage() -> Result<()> {
        let values: Vec<_> = (0..256).map(|i| i as f32 / 64.).collect();
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
            .collect();
        let rows = Embedding::from_bf16_optimized(&bytes, 128, 4, false, false, false, true)?;
        assert_eq!(rows.bytes(), 256 / 2 + 4 * 8);
        for id in [1, 0, 1] {
            let mut actual = Vec::new();
            rows.append(id, &mut actual);
            for (a, b) in actual.iter().zip(&values[id * 128..(id + 1) * 128]) {
                assert!((a - b).abs() < 0.04);
            }
        }
        Ok(())
    }

    #[test]
    fn fitted_embedding_layout_matches_independent_block_decoding() -> Result<()> {
        let bytes: Vec<_> = (0..256)
            .flat_map(|i| {
                half::bf16::from_f32(((i * 37 % 101) as f32 - 50.) / 37.)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        for nf4 in [false, true] {
            let rows = Embedding::from_bf16_optimized(&bytes, 128, 4, false, true, nf4, false)?;
            assert_eq!(rows.bytes(), 256 / 2 + 256 / 64 * 4);
            for id in [1, 0, 1] {
                let mut actual = Vec::new();
                rows.append(id, &mut actual);
                for g in 0..2 {
                    let values: Vec<_> = bytes
                        [(id * 128 + g * 64) * 2..(id * 128 + (g + 1) * 64) * 2]
                        .chunks_exact(2)
                        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                        .collect();
                    let mut codes = [0; 64];
                    let s = if nf4 {
                        super::super::quantization::nf4(&values, &mut codes)
                    } else {
                        super::super::quantization::mse(&values, &mut codes, false)
                    };
                    for (i, &q) in codes.iter().enumerate() {
                        let v = if nf4 {
                            super::super::quantization::NF4[(q + 8) as usize]
                        } else {
                            q as f32
                        };
                        assert_eq!(actual[g * 64 + i], v * s);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_rows_preserve_signs_scales_and_random_access() -> Result<()> {
        for bits in [4, 8] {
            let limit = if bits == 4 { 7 } else { 127 };
            let values: Vec<_> = (0..3 * 128)
                .map(|i| ((i % (2 * limit + 1)) as i32 - limit as i32) as f32)
                .collect();
            let bytes: Vec<_> = values
                .iter()
                .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
                .collect();
            let rows =
                Embedding::from_bf16(&bytes, 128, bits, super::super::cpu::avx2_available())?;
            assert_eq!(
                rows.bytes(),
                values.len() * bits as usize / 8 + values.len() / GROUP * 4
            );
            for id in [2, 0, 1, 2] {
                let mut actual = vec![42.];
                rows.append(id, &mut actual);
                assert_eq!(actual[0], 42.);
                for (a, b) in actual[1..].iter().zip(&values[id * 128..(id + 1) * 128]) {
                    assert!((a - b).abs() <= 0.51);
                }
            }
        }
        assert!(Embedding::from_bf16(&[0, 0], 1, 4, false).is_err());
        Ok(())
    }
}
