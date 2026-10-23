use super::*;
use memmap2::Mmap;
use serde_json::Map;
use std::{path::Path, time::Instant};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    Bf16,

    #[cfg(feature = "int8")]
    Int8,
    #[cfg(feature = "int8")]
    Q8,
    #[cfg(feature = "int8")]
    Q4,

    #[cfg(feature = "int8")]
    Q4E4,

    #[cfg(feature = "int8")]
    Q4Mse,

    #[cfg(feature = "int8")]
    Q4E4Mse,

    #[cfg(feature = "int8")]
    Q4E4Nf4,

    #[cfg(feature = "int8")]
    Q4E4Affine,
    #[cfg(feature = "int8")]
    Q4All,
}
impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16-storage-f32-compute",
            #[cfg(feature = "int8")]
            Self::Int8 => "int8-w8a8-g64",
            #[cfg(feature = "int8")]
            Self::Q8 => "q8-w8a8-e8-r8-g64",
            #[cfg(feature = "int8")]
            Self::Q4 => "q4-w4a8-e8-r4-g64",
            #[cfg(feature = "int8")]
            Self::Q4E4 => "q4-w4a8-e4-r4-g64",
            #[cfg(feature = "int8")]
            Self::Q4Mse => "q4-w4a8-e8-r4-g64-mse",
            #[cfg(feature = "int8")]
            Self::Q4E4Mse => "q4-w4a8-e4-r4-g64-mse",
            #[cfg(feature = "int8")]
            Self::Q4E4Nf4 => "q4-w4a8-e4-r4-g64-mse-nf4",
            #[cfg(feature = "int8")]
            Self::Q4E4Affine => "q4-w4a8-e4-r4-g64-affine",
            #[cfg(feature = "int8")]
            Self::Q4All => "q4-w4a4-e4-r4-g64",
        }
    }
    #[cfg(feature = "int8")]
    fn bits(self) -> (u8, u8, u8) {
        match self {
            Self::Bf16 => (16, 32, 16),
            Self::Int8 => (8, 8, 16),
            Self::Q8 => (8, 8, 8),
            Self::Q4 | Self::Q4Mse => (4, 8, 8),
            Self::Q4E4 | Self::Q4E4Mse | Self::Q4E4Nf4 | Self::Q4E4Affine => (4, 8, 4),
            Self::Q4All => (4, 4, 4),
        }
    }
}

pub struct Model {
    pub engine: Engine,
    shards: Vec<Mmap>,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    precision: Precision,
    #[cfg(feature = "int8")]
    embedding: Option<super::quantized_rows::Embedding>,
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
        let mut tensors = std::collections::HashMap::new();
        for shard in &shards {
            tensors.extend(Weights::from_bytes(shard)?.0);
        }
        let norm = Weights(tensors).get("language_model.norm.weight", &[D])?;
        Ok(Self {
            engine,
            shards,
            layers: Vec::new(),
            norm,
            precision: Precision::Bf16,
            #[cfg(feature = "int8")]
            embedding: None,
            profile: false,
        })
    }
    #[cfg(feature = "int8")]
    pub fn load_int8(root: &Path, portable: bool) -> Result<Self> {
        Self::load_with_precision(root, portable, Precision::Int8)
    }
    pub fn load_with_precision(root: &Path, portable: bool, precision: Precision) -> Result<Self> {
        let mut model = Self::load(root, portable)?;
        if precision == Precision::Bf16 {
            return Ok(model);
        }
        #[cfg(feature = "int8")]
        {
            let (weight_bits, activation_bits, embedding_bits) = precision.bits();
            let optimized = matches!(
                precision,
                Precision::Q4Mse | Precision::Q4E4Mse | Precision::Q4E4Nf4
            );
            let mut tensors = std::collections::HashMap::new();
            for shard in &model.shards {
                tensors.extend(Weights::from_bytes(shard)?.0);
            }
            let weights = Weights(tensors);
            model.layers = (0..32)
                .map(|i| {
                    Layer::load_quantized(
                        &weights,
                        &model.engine.config,
                        i,
                        model.engine.avx2,
                        weight_bits,
                        activation_bits,
                        precision != Precision::Int8,
                        optimized,
                        precision == Precision::Q4E4Affine,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            if embedding_bits != 16 {
                model.embedding = Some(super::quantized_rows::Embedding::from_bf16_optimized(
                    weights.bytes("language_model.embed_tokens.weight", &[V, D])?,
                    D,
                    embedding_bits,
                    model.engine.avx2,
                    optimized,
                    precision == Precision::Q4E4Nf4,
                    precision == Precision::Q4E4Affine,
                )?);
                let bytes = std::fs::read(root.join("readout.safetensors"))?;
                model.engine.quantized_readout = Some(super::int8::Matrix::from_bf16_optimized(
                    Weights::from_bytes(&bytes)?.bytes("weight", &[255, D])?,
                    D,
                    255,
                    model.engine.avx2,
                    weight_bits,
                    activation_bits,
                    optimized,
                    precision == Precision::Q4E4Affine,
                )?);
                model.engine.readout.clear();
                model.engine.readout.shrink_to_fit();
                model.shards.clear();
            }
        }
        model.precision = precision;
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
        let embedding = if self.shards.is_empty() {
            &[][..]
        } else {
            w.bytes("language_model.embed_tokens.weight", &[V, D])?
        };
        let mut answers = Map::new();
        let mut logits = Map::new();
        let mut timings = Vec::new();
        for chunk in &chunks {
            let mut work = Work::default();
            work.linear_scratch.profile = self.profile;
            for &id in &chunk.ids {
                ensure!((id as usize) < V, "token out of range");
                #[cfg(feature = "int8")]
                if let Some(embedding) = &self.embedding {
                    embedding.append(id as usize, &mut work.hidden);
                    continue;
                }
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
                .zip(self.engine.logits(&mut work, &self.norm, chunk)?)
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
        let embedding_bytes = V * D * 2;
        let readout_bytes = self.engine.readout.len() * 4;
        #[cfg(feature = "int8")]
        let embedding_bytes = self
            .embedding
            .as_ref()
            .map_or(embedding_bytes, |e| e.bytes());
        #[cfg(feature = "int8")]
        let readout_bytes = self
            .engine
            .quantized_readout
            .as_ref()
            .map_or(readout_bytes, |r| r.bytes());
        let metrics = json!({"revision":REVISION,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,"layers":timings,"logits":logits,"prune_last":self.engine.prune_last,"precision":self.precision.name(),"embedding_bytes":embedding_bytes,"readout_bytes":readout_bytes,"packed_projection_bytes":self.layers.iter().map(Layer::projection_bytes).sum::<usize>(),"weight_loading":if self.layers.is_empty(){"mmap, pack one layer per request"}else{"resident packed decoder weights"}});
        Ok((response, metrics))
    }
}
