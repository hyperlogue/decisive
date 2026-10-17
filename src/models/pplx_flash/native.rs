use super::*;
use memmap2::Mmap;
use serde_json::Map;
use std::{path::Path, time::Instant};
pub struct Model {
    pub engine: Engine,
    shards: Vec<Mmap>,
    layers: Vec<Layer>,
    int8: bool,
    pub profile: bool,
}
impl Model {
    pub fn load(root: &Path, portable: bool) -> Result<Self> {
        let read = |name: &str| std::fs::read(root.join(name));
        let engine = Engine::new(
            &read("config.json")?,
            &read("decision_config.json")?,
            &read("joint_config.json")?,
            &read("tokenizer.json")?,
            &read("readout.safetensors")?,
            portable,
        )?;
        let mut shards = Vec::new();
        for name in SHARDS {
            shards.push(unsafe { Mmap::map(&std::fs::File::open(root.join(name))?)? });
        }
        Ok(Self {
            engine,
            shards,
            layers: Vec::new(),
            int8: false,
            profile: false,
        })
    }
    #[cfg(feature = "int8")]
    pub fn load_int8(root: &Path, portable: bool) -> Result<Self> {
        let mut model = Self::load(root, portable)?;
        let mut tensors = std::collections::HashMap::new();
        for shard in &model.shards {
            tensors.extend(Weights::from_bytes(shard)?.0);
        }
        let weights = Weights(tensors);
        model.layers = (0..32)
            .map(|i| Layer::load_int8(&weights, &model.engine.config, i, model.engine.avx2))
            .collect::<Result<Vec<_>>>()?;
        model.int8 = true;
        Ok(model)
    }
    pub fn preload_weights(&mut self) -> Result<()> {
        if !self.layers.is_empty() {
            return Ok(());
        }
        let mut tensors = std::collections::HashMap::new();
        for shard in &self.shards {
            tensors.extend(Weights::from_bytes(shard)?.0);
        }
        let weights = Weights(tensors);
        let layers = (0..32)
            .map(|i| Layer::load(&weights, &self.engine.config, i, self.engine.avx2))
            .collect::<Result<Vec<_>>>()?;
        self.layers = layers;
        Ok(())
    }
    pub fn predict(&self, request: &Value, max_length: usize) -> Result<Value> {
        Ok(self.evaluate(request, max_length)?.0)
    }
    pub fn evaluate(&self, request: &Value, max_length: usize) -> Result<(Value, Value)> {
        let start = Instant::now();
        let chunks = self.engine.input.plan(request, max_length)?;
        let mut tensors = std::collections::HashMap::new();
        for shard in &self.shards {
            tensors.extend(Weights::from_bytes(shard)?.0);
        }
        let w = Weights(tensors);
        let embedding = w.bytes("language_model.embed_tokens.weight", &[V, D])?;
        let norm = w.get("language_model.norm.weight", &[D])?;
        let mut answers = Map::new();
        let mut logits = Map::new();
        let mut timings = Vec::new();
        for chunk in &chunks {
            let mut work = Work::default();
            work.linear_scratch.profile = self.profile;
            for &id in &chunk.ids {
                ensure!((id as usize) < V, "token out of range");
                let at = id as usize * D * 2;
                work.hidden.extend(
                    embedding[at..at + D * 2]
                        .chunks_exact(2)
                        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)),
                );
            }
            for index in 0..32 {
                let t = Instant::now();
                let loaded;
                let layer = if self.layers.is_empty() {
                    loaded = Layer::load(&w, &self.engine.config, index, self.engine.avx2)?;
                    &loaded
                } else {
                    &self.layers[index]
                };
                let packed = t.elapsed().as_secs_f64() * 1000.;
                let t = Instant::now();
                work.linear_scratch.projection_ms = 0.;
                layer.forward(
                    &mut work,
                    Plan::default(),
                    self.engine.avx2,
                    if index == 31 && self.engine.prune_last {
                        Some(&chunk.markers)
                    } else {
                        None
                    },
                )?;
                let compute_ms = t.elapsed().as_secs_f64() * 1000.;
                let mut timing = json!({"layer":index,"pack_ms":packed,"compute_ms":compute_ms});
                if self.profile {
                    timing["projection_ms"] = json!(work.linear_scratch.projection_ms);
                    timing["non_projection_ms"] =
                        json!(compute_ms - work.linear_scratch.projection_ms);
                }
                timings.push(timing);
            }
            for (key, values) in chunk
                .keys
                .iter()
                .zip(self.engine.logits(&mut work, &norm, chunk)?)
            {
                answers.insert(
                    key.clone(),
                    self.engine
                        .input
                        .answer(&request["questions"][key], &values)?,
                );
                logits.insert(key.clone(), json!(values));
            }
        }
        let response = json!({"answers":answers,"usage":{"input_tokens":chunks.iter().map(|c|c.ids.len()).sum::<usize>(),"output_tokens":0,"sequences":chunks.len()}});
        let metrics = json!({"revision":REVISION,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,"layers":timings,"logits":logits,"prune_last":self.engine.prune_last,"precision":if self.int8 {"int8-w8a8-g64"} else {"bf16-storage-f32-compute"},"packed_projection_bytes":self.layers.iter().map(Layer::projection_bytes).sum::<usize>(),"weight_loading":if self.layers.is_empty(){"mmap, pack one layer per request"}else{"resident packed decoder weights"}});
        Ok((response, metrics))
    }
}
