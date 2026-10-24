use super::{REVISION, SHARDS, packed, weights::Index};
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

/// Group-64 WebGPU weights: Q4 (W4A8, E4, R4) or Q8 (W8A8, E8, R8) projections, embeddings and
/// readout, with the remaining tensors kept as BF16.
pub fn export_webgpu_packed(model: &Path, destination: &Path, bits: u32) -> Result<()> {
    ensure!(bits == 4 || bits == 8, "packed WebGPU weights are Q4 or Q8");
    let prefix = if bits == 4 { "q4-e4" } else { "q8-e8" };
    ensure!(
        !destination.exists(),
        "choose a new packed checkpoint directory"
    );
    fs::create_dir_all(destination)?;
    let mut sources = Vec::new();
    let mut headers = Vec::new();
    for name in SHARDS {
        let mut file = File::open(model.join(name))?;
        let mut size = [0; 8];
        file.read_exact(&mut size)?;
        let size = u64::from_le_bytes(size);
        ensure!(size <= 16 * 1024 * 1024, "checkpoint header too large");
        let mut bytes = vec![0; size as usize];
        file.read_exact(&mut bytes)?;
        headers.push(String::from_utf8(bytes)?);
        sources.push(file);
    }
    let index = Index::new(&headers)?;
    let mut read = |r: &super::weights::Range| -> Result<Vec<u8>> {
        let file = &mut sources[r.file];
        file.seek(SeekFrom::Start(r.start))?;
        let mut bytes = vec![0; (r.end - r.start) as usize];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    };
    let mut files = serde_json::Map::new();
    let mut layers = Vec::new();
    for layer in 0..32 {
        let name = format!("{prefix}-layer-{layer:02}.bin");
        let mut file = File::create(destination.join(&name))?;
        let mut tensors = Vec::new();
        let mut position = 0;
        let mut digest = Sha256::new();
        for r in index.layer(layer)? {
            let raw = read(&r)?;
            let packed = if r.shape.len() == 2 {
                packed::pack(&raw, r.shape[0], r.shape[1], bits)?
            } else {
                raw
            };
            file.write_all(&packed)?;
            digest.update(&packed);
            tensors.push(
                json!({"name":r.name,"shape":r.shape,"start":position,"end":position+packed.len()}),
            );
            position += packed.len();
        }
        files.insert(
            name.clone(),
            json!({"bytes":position,"sha256":format!("{:x}",digest.finalize())}),
        );
        layers.push(json!({"file":name,"bytes":position,"tensors":tensors}));
        eprintln!("Packed layer {}/32", layer + 1);
    }
    let embedding = index.tensor("language_model.embed_tokens.weight")?;
    let raw = read(&embedding)?;
    let embedding = packed::pack(&raw, embedding.shape[0], embedding.shape[1], bits)?;
    drop(raw);
    let readout = fs::read(model.join("readout.safetensors"))?;
    let readout = packed::pack(
        super::weights::Weights::from_bytes(&readout)?.bytes("weight", &[255, 2560])?,
        255,
        2560,
        bits,
    )?;
    let norm = read(&index.tensor("language_model.norm.weight")?)?;
    for (name, bytes) in [("embedding", embedding), ("readout", readout), ("norm", norm)] {
        let name = format!("{prefix}-{name}.bin");
        fs::write(destination.join(&name), &bytes)?;
        files.insert(
            name,
            json!({"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}),
        );
    }
    let total: u64 = layers.iter().map(|l| l["bytes"].as_u64().unwrap()).sum();
    let manifest = json!({"schema_version":1,"revision":REVISION,
        "scheme":format!("q{bits}-w{bits}a8-e{bits}-r{bits}-g64"),"group_size":64,
        "group_bytes":packed::size(1, 64, bits)?,"embedding_row_bytes":packed::size(1, 2560, bits)?,
        "layers":layers,"headers":headers,"embedding":format!("{prefix}-embedding.bin"),
        "readout":format!("{prefix}-readout.bin"),"norm":format!("{prefix}-norm.bin"),
        "decoder_bytes":total,"files":files});
    fs::write(
        destination.join(format!("packed-{prefix}.json")),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
