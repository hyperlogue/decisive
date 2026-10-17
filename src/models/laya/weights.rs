use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
#[cfg(feature = "cpu")]
use rayon::prelude::*;
use safetensors::{
    Dtype, SafeTensors,
    tensor::{Metadata, TensorInfo},
};
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    F32,
    F16,
}

pub enum Matrix {
    #[cfg(feature = "fp16")]
    WasmF16 {
        data: Vec<f16>,
        rows: usize,
        cols: usize,
    },
    WasmF32 {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    #[cfg(feature = "int8")]
    Int8(crate::laya::cpu::int8::Matrix),
    F32(Vec<f32>),
    #[cfg(feature = "fp16")]
    F16(Vec<f16>),
    PackedF32 {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    #[cfg(feature = "fp16")]
    PackedF16 {
        data: Vec<f16>,
        rows: usize,
        cols: usize,
    },
}

impl Matrix {
    #[cfg(feature = "cpu")]
    pub(crate) fn is_unpacked(&self) -> bool {
        match self {
            Self::F32(_) => true,
            #[cfg(feature = "fp16")]
            Self::F16(_) => true,
            _ => false,
        }
    }
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    pub(crate) fn is_packed(&self) -> bool {
        match self {
            Self::PackedF32 { .. } => true,
            #[cfg(feature = "fp16")]
            Self::PackedF16 { .. } => true,
            _ => false,
        }
    }
    pub fn is_quantized(&self) -> bool {
        #[cfg(feature = "int8")]
        {
            return matches!(self, Self::Int8(_));
        }
        #[cfg(not(feature = "int8"))]
        false
    }
    pub fn copy_row(&self, index: usize, dst: &mut [f32]) {
        let range = index * dst.len()..(index + 1) * dst.len();
        match self {
            Self::F32(values) => dst.copy_from_slice(&values[range]),
            #[cfg(feature = "fp16")]
            Self::F16(values) => {
                half::slice::HalfFloatSliceExt::convert_to_f32_slice(&values[range], dst)
            }
            _ => panic!("row lookup requires unpacked matrix"),
        }
    }
    pub fn values(&self) -> std::borrow::Cow<'_, [f32]> {
        match self {
            #[cfg(feature = "fp16")]
            Self::WasmF16 { data, rows, cols } => {
                let mut out = vec![0.; rows * cols];
                for n in 0..*rows {
                    for k in 0..*cols {
                        out[n * cols + k] = data[(n / 8 * cols + k) * 8 + n % 8].to_f32();
                    }
                }
                std::borrow::Cow::Owned(out)
            }
            Self::WasmF32 { data, rows, cols } => {
                let mut out = vec![0.; rows * cols];
                for n in 0..*rows {
                    for k in 0..*cols {
                        out[n * cols + k] = data[(n / 8 * cols + k) * 8 + n % 8];
                    }
                }
                std::borrow::Cow::Owned(out)
            }
            #[cfg(feature = "int8")]
            Self::Int8(w) => std::borrow::Cow::Owned(
                (0..w.rows)
                    .flat_map(|n| (0..w.cols).map(move |k| w.value(n, k)))
                    .collect(),
            ),
            #[cfg(feature = "fp16")]
            Self::PackedF16 { data, rows, cols } => {
                let mut out = vec![0.0; rows * cols];
                for n in 0..*rows {
                    for k in 0..*cols {
                        out[n * cols + k] = data[(n / 16) * cols * 16 + k * 16 + n % 16].to_f32();
                    }
                }
                std::borrow::Cow::Owned(out)
            }
            Self::PackedF32 { data, rows, cols } => {
                let mut out = vec![0.0; rows * cols];
                for n in 0..*rows {
                    for k in 0..*cols {
                        out[n * cols + k] = data[(n / 16) * cols * 16 + k * 16 + n % 16];
                    }
                }
                std::borrow::Cow::Owned(out)
            }
            Self::F32(v) => std::borrow::Cow::Borrowed(v),
            #[cfg(feature = "fp16")]
            Self::F16(v) => {
                let mut result = vec![0.0; v.len()];
                half::slice::HalfFloatSliceExt::convert_to_f32_slice(v.as_slice(), &mut result);
                std::borrow::Cow::Owned(result)
            }
        }
    }
}

enum Source<'a> {
    Bytes(&'a [u8]),
    #[cfg(feature = "cpu")]
    File {
        file: std::cell::RefCell<std::fs::File>,
        base: u64,
    },
}
pub struct Weights<'a> {
    metadata: Metadata,
    source: Source<'a>,
}

impl<'a> Weights<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let (n, metadata) = SafeTensors::read_metadata(bytes)?;
        Ok(Self {
            metadata,
            source: Source::Bytes(&bytes[n + 8..]),
        })
    }
    #[cfg(feature = "cpu")]
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // The caller must not modify/truncate the checkpoint during loading.
        let mapped = unsafe { memmap2::Mmap::map(&file)? };
        let (n, metadata) = SafeTensors::read_metadata(&mapped)?;
        drop(mapped);
        Ok(Self {
            metadata,
            source: Source::File {
                file: file.into(),
                base: (n + 8) as u64,
            },
        })
    }
    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>> {
        match &self.source {
            Source::Bytes(data) => Ok(Cow::Borrowed(&data[offset..offset + len])),
            #[cfg(feature = "cpu")]
            Source::File { .. } => {
                let mut data = vec![0; len];
                self.read_into(offset, &mut data)?;
                Ok(Cow::Owned(data))
            }
        }
    }
    #[cfg(any(feature = "cpu", feature = "fp16"))]
    fn read_into(&self, offset: usize, dst: &mut [u8]) -> Result<()> {
        match &self.source {
            Source::Bytes(data) => dst.copy_from_slice(&data[offset..offset + dst.len()]),
            #[cfg(feature = "cpu")]
            Source::File { file, base } => {
                use std::io::{Read, Seek, SeekFrom};
                let mut file = file.borrow_mut();
                file.seek(SeekFrom::Start(base + offset as u64))?;
                file.read_exact(dst)?;
            }
        }
        Ok(())
    }
    pub fn contains(&self, name: &str) -> bool {
        self.metadata.info(name).is_some()
    }
    fn info(&self, name: &str, shape: &[usize]) -> Result<&TensorInfo> {
        let info = self
            .metadata
            .info(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.shape == shape,
            "{name}: expected {shape:?}, got {:?}",
            info.shape
        );
        ensure!(
            matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16),
            "{name}: unsupported dtype {:?}",
            info.dtype
        );
        Ok(info)
    }
    pub fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        self.tensor(name, &[len])
    }
    pub fn tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.info(name, shape)?;
        self.load_values(info, shape.iter().product(), 1, false, decode_f32)
            .with_context(|| format!("{name}: invalid weights"))
    }
    pub fn matrix(&self, name: &str, rows: usize, cols: usize, storage: Storage) -> Result<Matrix> {
        self.matrix_layout(name, rows, cols, storage, false)
    }
    pub fn matrix_layout(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        storage: Storage,
        packed: bool,
    ) -> Result<Matrix> {
        ensure!(
            storage != Storage::F16 || cfg!(feature = "fp16"),
            "FP16 storage requires the fp16 Cargo feature"
        );
        ensure!(rows > 0 && cols > 0, "{name}: empty matrix");
        let info = self.info(name, &[rows, cols])?;
        #[cfg(feature = "fp16")]
        if !packed
            && storage == Storage::F16
            && info.dtype == Dtype::F16
            && cfg!(target_endian = "little")
        {
            let mut data = vec![f16::ZERO; rows * cols];
            self.read_into(info.data_offsets.0, bytemuck::cast_slice_mut(&mut data))?;
            #[cfg(feature = "cpu")]
            let finite = data.par_iter().all(|v| v.is_finite());
            #[cfg(not(feature = "cpu"))]
            let finite = data.iter().all(|v| v.is_finite());
            ensure!(finite, "{name}: non-finite weights");
            return Ok(Matrix::F16(data));
        }
        Ok(match storage {
            Storage::F32 => {
                let data = self
                    .load_values(info, rows, cols, packed, decode_f32)
                    .with_context(|| format!("{name}: invalid weights"))?;
                if packed {
                    Matrix::PackedF32 { data, rows, cols }
                } else {
                    Matrix::F32(data)
                }
            }
            #[cfg(not(feature = "fp16"))]
            Storage::F16 => unreachable!("validated FP16 feature"),
            #[cfg(feature = "fp16")]
            Storage::F16 => {
                let data = self
                    .load_values(info, rows, cols, packed, |dtype, bytes| {
                        let value = if dtype == Dtype::F16 {
                            f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap()))
                        } else {
                            f16::from_f32(decode_f32(dtype, bytes)?)
                        };
                        ensure!(value.is_finite(), "non-finite weights or FP16 overflow");
                        Ok(value)
                    })
                    .with_context(|| format!("{name}: invalid weights"))?;
                if packed {
                    Matrix::PackedF16 { data, rows, cols }
                } else {
                    Matrix::F16(data)
                }
            }
        })
    }
    fn load_values<T: Copy + Default + Send + Sync>(
        &self,
        info: &TensorInfo,
        rows: usize,
        cols: usize,
        packed: bool,
        decode: impl Fn(Dtype, &[u8]) -> Result<T> + Sync,
    ) -> Result<Vec<T>> {
        let size = info.dtype.bitsize() / 8;
        let width = if packed { 16 } else { 1 };
        let chunk_rows = (1024 * 1024 / (cols * size) / width).max(1) * width;
        let mut output = vec![T::default(); rows.div_ceil(width) * width * cols];
        for first in (0..rows).step_by(chunk_rows) {
            let count = (rows - first).min(chunk_rows);
            let bytes = self.read(
                info.data_offsets.0 + first * cols * size,
                count * cols * size,
            )?;
            let dst = &mut output[first * cols..(first + count.div_ceil(width) * width) * cols];
            let fill = |(panel, dst): (usize, &mut [T])| -> Result<()> {
                for lane in 0..width.min(count - panel * width) {
                    for col in 0..cols {
                        let index = ((panel * width + lane) * cols + col) * size;
                        dst[col * width + lane] = decode(info.dtype, &bytes[index..index + size])?;
                    }
                }
                Ok(())
            };
            #[cfg(feature = "cpu")]
            dst.par_chunks_mut(width * cols)
                .enumerate()
                .try_for_each(fill)?;
            #[cfg(not(feature = "cpu"))]
            dst.chunks_mut(width * cols)
                .enumerate()
                .try_for_each(fill)?;
        }
        Ok(output)
    }
}
fn decode_f32(dtype: Dtype, bytes: &[u8]) -> Result<f32> {
    let value = match dtype {
        Dtype::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
        Dtype::F16 => f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32(),
        Dtype::BF16 => bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32(),
        _ => bail!("unsupported dtype {dtype:?}"),
    };
    ensure!(value.is_finite(), "non-finite weights");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn checkpoint(dtype: &str, shape: &[usize], bytes: &[u8]) -> Vec<u8> {
        let header = serde_json::json!({"w": {"dtype": dtype, "shape": shape,
            "data_offsets": [0, bytes.len()]}})
        .to_string();
        let mut result = (header.len() as u64).to_le_bytes().to_vec();
        result.extend_from_slice(header.as_bytes());
        result.extend_from_slice(bytes);
        result
    }
    #[test]
    fn direct_layout_preserves_all_dtypes_and_panel_tails() -> Result<()> {
        if !cfg!(feature = "fp16") {
            let bytes = checkpoint("F32", &[1, 1], &1f32.to_le_bytes());
            assert!(
                Weights::new(&bytes)?
                    .matrix("w", 1, 1, Storage::F16)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("fp16 Cargo feature")
            );
        }
        let values: Vec<f32> = (0..35 * 19).map(|i| (i as f32 - 310.0) / 16.0).collect();
        for dtype in ["F32", "F16", "BF16"] {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|&v| match dtype {
                    "F32" => v.to_le_bytes().to_vec(),
                    "F16" => f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                    _ => bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                })
                .collect();
            let bytes = checkpoint(dtype, &[35, 19], &bytes);
            let w = Weights::new(&bytes)?;
            for storage in [Storage::F16, Storage::F32]
                .into_iter()
                .filter(|s| *s == Storage::F32 || cfg!(feature = "fp16"))
            {
                let expected = w.matrix("w", 35, 19, storage)?.values().into_owned();
                let actual = w.matrix_layout("w", 35, 19, storage, true)?;
                assert_eq!(expected, actual.values().as_ref());
                assert_eq!(expected, w.tensor("w", &[35, 19])?);
            }
        }
        Ok(())
    }
    #[test]
    fn rejects_nonfinite_and_conversion_overflow() -> Result<()> {
        for value in [f32::NAN, f32::INFINITY, 100_000.0] {
            let bytes = checkpoint("F32", &[1, 1], &value.to_le_bytes());
            let w = Weights::new(&bytes)?;
            assert!(w.matrix_layout("w", 1, 1, Storage::F16, true).is_err());
            assert!(w.matrix("w", 2, 1, Storage::F32).is_err());
        }
        for value in [f16::NAN, f16::INFINITY] {
            let bytes = checkpoint("F16", &[1, 1], &value.to_bits().to_le_bytes());
            let w = Weights::new(&bytes)?;
            for packed in [false, true] {
                assert!(w.matrix_layout("w", 1, 1, Storage::F16, packed).is_err());
            }
        }
        Ok(())
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn streamed_file_matches_bytes_across_chunks() -> Result<()> {
        let values: Vec<_> = (0..1027 * 521)
            .map(|i| f16::from_f32((i % 97) as f32 / 16.0))
            .collect();
        let bytes = checkpoint("F16", &[1027, 521], bytemuck::cast_slice(&values));
        let path = std::env::temp_dir().join(format!(
            "decisive-weight-test-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &bytes)?;
        let streamed = Weights::open(&path)?;
        for storage in [Storage::F16, Storage::F32]
            .into_iter()
            .filter(|s| *s == Storage::F32 || cfg!(feature = "fp16"))
        {
            for packed in [false, true] {
                let expected =
                    Weights::new(&bytes)?.matrix_layout("w", 1027, 521, storage, packed)?;
                let actual = streamed.matrix_layout("w", 1027, 521, storage, packed)?;
                assert_eq!(expected.values(), actual.values());
            }
        }
        drop(streamed);
        std::fs::remove_file(path)?;
        Ok(())
    }
}
