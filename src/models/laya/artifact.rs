use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{fs, path::Path};

pub fn fetch(variant: &str, directory: &Path) -> Result<()> {
    let manifest: Value = serde_json::from_str(include_str!("../../../references.json"))?;
    let model = &manifest["models"][variant];
    let repository = model["repository"]
        .as_str()
        .context("unknown model variant")?;
    let revision = model["revision"].as_str().unwrap();
    fs::create_dir_all(directory)?;
    let stamp = directory.join("beam-revision.json");
    if stamp.exists() {
        ensure!(
            serde_json::from_slice::<Value>(&fs::read(&stamp)?)? == *model,
            "destination contains a different model revision"
        );
    }
    for name in [
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "model.safetensors",
    ] {
        let path = directory.join(name);
        if path.is_file() && stamp.exists() {
            continue;
        }
        fs::create_dir_all(path.parent().unwrap())?;
        let temp = path.with_extension("download");
        let url = format!("https://huggingface.co/{repository}/resolve/{revision}/{name}");
        eprintln!("fetching {variant}/{name}");
        let mut response = ureq::get(&url).call()?;
        let expected = response
            .headers()
            .get("Content-Length")
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.parse::<u64>().ok());
        let mut file = fs::File::create(&temp)?;
        let bytes = std::io::copy(&mut response.body_mut().as_reader(), &mut file)?;
        if let Some(n) = expected {
            ensure!(bytes == n, "incomplete download: {name}");
        }
        file.sync_all()?;
        fs::rename(temp, path)?;
    }
    fs::write(stamp, serde_json::to_vec_pretty(model)?)?;
    Ok(())
}
