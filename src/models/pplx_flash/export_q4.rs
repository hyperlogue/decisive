use super::{REVISION, SHARDS, packed_q4, weights::Index};
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

pub fn export_webgpu_q4(model: &Path, destination: &Path) -> Result<()> {
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
        let name = format!("q4-e4-layer-{layer:02}.bin");
        let mut file = File::create(destination.join(&name))?;
        let mut tensors = Vec::new();
        let mut position = 0;
        let mut digest = Sha256::new();
        for r in index.layer(layer)? {
            let raw = read(&r)?;
            let packed = if r.shape.len() == 2 {
                packed_q4::pack(&raw, r.shape[0], r.shape[1])?
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
    let embedding = packed_q4::pack(&raw, embedding.shape[0], embedding.shape[1])?;
    drop(raw);
    let readout = fs::read(model.join("readout.safetensors"))?;
    let readout = packed_q4::pack(
        super::weights::Weights::from_bytes(&readout)?.bytes("weight", &[255, 2560])?,
        255,
        2560,
    )?;
    let norm = read(&index.tensor("language_model.norm.weight")?)?;
    for (name, bytes) in [
        ("q4-e4-embedding.bin", embedding),
        ("q4-e4-readout.bin", readout),
        ("q4-e4-norm.bin", norm),
    ] {
        fs::write(destination.join(name), &bytes)?;
        files.insert(
            name.into(),
            json!({"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}),
        );
    }
    let total: u64 = layers.iter().map(|l| l["bytes"].as_u64().unwrap()).sum();
    let manifest = json!({"schema_version":1,"revision":REVISION,"scheme":"q4-w4a8-e4-r4-g64",
        "group_size":64,"group_bytes":36,"embedding_row_bytes":1440,"layers":layers,"headers":headers,
        "embedding":"q4-e4-embedding.bin","readout":"q4-e4-readout.bin","norm":"q4-e4-norm.bin",
        "decoder_bytes":total,"files":files});
    fs::write(
        destination.join("packed-q4-e4.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
