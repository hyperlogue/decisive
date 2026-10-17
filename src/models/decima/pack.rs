use super::weights::{INT8_FORMAT, SCALE_SUFFIX, quantize_groups};
use anyhow::{Context, Result, ensure};
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorView, serialize_to_file},
};
use sha2::{Digest, Sha256};
use std::{borrow::Cow, collections::HashMap, fs, path::Path};

pub fn int8_checkpoint(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    group: usize,
    fp16_projections: bool,
) -> Result<()> {
    checkpoint(
        source.as_ref(),
        destination.as_ref(),
        group,
        fp16_projections,
        false,
    )
}

pub fn int8_checkpoint_half_scales(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    group: usize,
    fp16_projections: bool,
) -> Result<()> {
    checkpoint(
        source.as_ref(),
        destination.as_ref(),
        group,
        fp16_projections,
        true,
    )
}

fn checkpoint(
    source: &Path,
    destination: &Path,
    group: usize,
    fp16_projections: bool,
    half_scales: bool,
) -> Result<()> {
    ensure!(
        !(fp16_projections || half_scales) || cfg!(feature = "fp16"),
        "mixed FP16 export requires the fp16 Cargo feature"
    );
    ensure!(
        matches!(group, 16 | 32),
        "INT8 export group must be 16 or 32"
    );
    let root = if source.join("pytorch").is_dir() {
        source.join("pytorch")
    } else {
        source.to_owned()
    };
    ensure!(!destination.exists(), "choose a new checkpoint destination");
    let file = fs::File::open(root.join("encoder/model.safetensors"))?;
    // Source checkpoint must remain immutable during export.
    let mapping = unsafe { memmap2::Mmap::map(&file)? };
    ensure!(
        format!("{:x}", Sha256::digest(&mapping[..]))
            == "2bc43026b30d506e416579b891d361a41f25bca32d91f8138261f2a819e46566",
        "export requires the pinned original Decima encoder (SHA-256 mismatch)"
    );
    for (name, expected) in [
        (
            "decima.json",
            "bfa9311b8a6ada93691b51edd26088d1a53c0071eb17e4cb014c6d7fe602d1e8",
        ),
        (
            "encoder/config.json",
            "86b060dde794ff01a1a69ae82c9be1e59eff692f3011ec10d4d9f89e65337568",
        ),
        (
            "encoder/tokenizer.json",
            "255f5e32cb320b4dc2b31e0f203832fecafe87ca4cc87c3c41eb8ccae73f441a",
        ),
        (
            "encoder/tokenizer_config.json",
            "04a6c7f6ba3ae2d7fc5727c5de4223fad062b1e131bf0ed8145ae7ffb39bc938",
        ),
        (
            "head.safetensors",
            "5f2a149e4a4f9f1b2b97df2d5ed98b006cddae53f9628a812f76eb4aae40aebf",
        ),
    ] {
        ensure!(
            format!("{:x}", Sha256::digest(fs::read(root.join(name))?)) == expected,
            "pinned Decima source hash mismatch: {name}"
        );
    }
    let tensors = SafeTensors::deserialize(&mapping)?;
    let mut owned: Vec<(String, Dtype, Vec<usize>, Cow<'_, [u8]>)> = Vec::new();
    let mut quantized = 0;
    for (name, tensor) in tensors.iter() {
        ensure!(
            tensor.dtype() == Dtype::F32,
            "export requires the original FP32 checkpoint"
        );
        let selected = name == "embeddings.word_embeddings.weight"
            || (name.starts_with("encoder.layer.")
                && name.ends_with(".weight")
                && tensor.shape().len() == 2);
        if selected && fp16_projections && name.starts_with("encoder.layer.") {
            let mut bytes = Vec::with_capacity(tensor.data().len() / 2);
            for value in tensor.data().chunks_exact(4) {
                let value = half::f16::from_f32(f32::from_le_bytes(value.try_into().unwrap()));
                ensure!(value.is_finite(), "FP16 projection overflow in {name}");
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            owned.push((
                name.into(),
                Dtype::F16,
                tensor.shape().to_vec(),
                Cow::Owned(bytes),
            ));
            quantized += 1;
        } else if selected {
            let shape = tensor.shape();
            ensure!(
                shape.len() == 2 && shape[1] % group == 0,
                "unsupported quantized shape: {name}"
            );
            let (values, scales) = quantize_groups(tensor.data(), group)?;
            owned.push((
                name.into(),
                Dtype::I8,
                shape.to_vec(),
                Cow::Owned(values.into_iter().map(|v| v as u8).collect()),
            ));
            let half = half_scales && name == "embeddings.word_embeddings.weight";
            let scale_bytes: Vec<u8> = if half {
                let scales: Vec<half::f16> = scales
                    .into_iter()
                    .map(|s| half::f16::from_f32(s.max(half::f16::from_bits(1).to_f32())))
                    .collect();
                ensure!(
                    scales.iter().all(|s| s.is_finite() && s.to_f32() > 0.),
                    "vocabulary scale cannot be represented as FP16"
                );
                scales
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect()
            } else {
                scales.iter().flat_map(|v| v.to_le_bytes()).collect()
            };
            owned.push((
                format!("{name}{SCALE_SUFFIX}"),
                if half { Dtype::F16 } else { Dtype::F32 },
                vec![shape[0], shape[1] / group],
                Cow::Owned(scale_bytes),
            ));
            quantized += 1;
        } else {
            ensure!(
                tensor
                    .data()
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                "non-finite tensor {name}"
            );
            owned.push((
                name.into(),
                tensor.dtype(),
                tensor.shape().to_vec(),
                Cow::Borrowed(tensor.data()),
            ));
        }
    }
    ensure!(
        quantized == 73,
        "expected vocabulary and 72 encoder projections, got {quantized}"
    );
    let views = owned
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            Ok((
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes.as_ref())?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = HashMap::from([
        ("beam_format".into(), INT8_FORMAT.into()),
        ("source_revision".into(), super::REVISION.into()),
        ("group_size".into(), group.to_string()),
        (
            "projection_storage".into(),
            if fp16_projections { "f16" } else { "int8" }.into(),
        ),
        ("recipe".into(), "symmetric-max-abs-round-away".into()),
        (
            "vocabulary_scale_storage".into(),
            if half_scales { "f16" } else { "f32" }.into(),
        ),
    ]);
    if let Some(parent) = destination.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(destination)?;
    let result = (|| -> Result<()> {
        fs::create_dir_all(destination.join("pytorch/encoder"))?;
        serialize_to_file(
            views,
            Some(metadata),
            &destination.join("pytorch/encoder/model.safetensors"),
        )?;
        for name in [
            "decima.json",
            "head.safetensors",
            "encoder/config.json",
            "encoder/tokenizer.json",
            "encoder/tokenizer_config.json",
        ] {
            fs::copy(root.join(name), destination.join("pytorch").join(name))
                .with_context(|| format!("copy {name}"))?;
        }
        let license = if source.join("LICENSE").is_file() {
            source.join("LICENSE")
        } else if root.join("LICENSE").is_file() {
            root.join("LICENSE")
        } else if root.file_name().is_some_and(|name| name == "pytorch") {
            root.parent().unwrap_or(&root).join("LICENSE")
        } else {
            root.join("LICENSE")
        };
        fs::copy(license, destination.join("LICENSE"))?;
        fs::write(
            destination.join("beam-revision.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "repository":super::REPOSITORY, "revision":super::REVISION,
                "format":INT8_FORMAT, "precision":"int8-compact", "group_size":group, "projection_storage":if fp16_projections {"f16"} else {"int8"},
                "vocabulary_scale_storage":if half_scales {"f16"} else {"f32"},
                "notes":if fp16_projections {"INT8 vocabulary, FP16 encoder projection storage, FP32 projection execution and scorer."} else {"INT8 vocabulary and encoder projections; original FP32 scorer. Different from upstream ONNX quantization."}
            }))?,
        )?;
        Ok(())
    })();
    if result.is_err() {
        // Cleanup owns only this newly created directory.
        let _ = fs::remove_dir_all(destination);
    }
    result
}
