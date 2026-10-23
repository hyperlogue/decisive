mod device;
mod readout;
mod shaders;
use super::{D, Kind, Model as CpuModel, Precision, Question, cpu, input};
use anyhow::{Context, Result, ensure};
use device::{Bindings, Device, Frame, Kernel};
use serde_json::{Value, json};
use shaders::{Accumulate, Epilogue, SLICE};
use std::{collections::VecDeque, sync::Arc};

const FF: usize = 1536;
const STATE_ROWS: usize = 1024;
const READOUT_ROWS: usize = 2048;
const CHUNK_ROWS: usize = 512;
const ENCODER_ROWS: usize = STATE_ROWS + CHUNK_ROWS;
const MAX_CHOICES: usize = 255;
const VOCABULARY: usize = 250037;
const VOCABULARY_SPLIT: usize = 1 << 16;
const ROW_BYTES: usize = D * 4 + 4;
// GEMM workgroup tiles as (rows, rows per thread), for small, medium and large batches.
const GEMM_TILES: [(usize, usize); 3] = [(16, 2), (32, 4), (32, 8)];
// Long residual GEMMs up to this many rows split k across workgroups.
const SPLIT_ROWS: usize = 192;
// Queries per encoder attention tile.
const ENCODER_QUERIES: usize = 16;

struct Linear {
    input: usize,
    output: usize,
    weight: wgpu::Buffer,
    bias: wgpu::Buffer,
}
struct Norm {
    eps: f32,
    weight: wgpu::Buffer,
    bias: wgpu::Buffer,
}
struct EncoderLayer {
    qkv: Linear,
    o: Linear,
    n1: Norm,
    n2: Norm,
    ff1: Linear,
    ff2: Linear,
}
struct ScorerLayer {
    qkv: Linear,
    o: Linear,
    cross_q: Linear,
    cross_o: Linear,
    n1: Norm,
    n2: Norm,
    n3: Norm,
    ff1: Linear,
    ff2: Linear,
}
struct Kernels {
    embed: Kernel,
    gemm: [[Kernel; 3]; 3],
    sliced: [Kernel; 2],
    split: [Kernel; 2],
    reduce: Kernel,
    norm: Kernel,
    encoder_attention: Kernel,
    scorer_attention: Kernel,
    mean: Kernel,
    copy: Kernel,
}
pub struct Model {
    device: Device,
    config: super::Config,
    input: input::Input,
    vocabulary: [wgpu::Buffer; 4],
    prelude: wgpu::Buffer,
    embedding_eps: f32,
    readout: readout::Readout,
    encoder: Vec<EncoderLayer>,
    scorer: Vec<ScorerLayer>,
    memory: Linear,
    out_norm: Norm,
    kernels: Kernels,
    // Scorer attention query tile: 32 where workgroup memory allows, else 16.
    queries: usize,
    identity: Arc<()>,
}

struct Entry {
    ids: Vec<u32>,
    rows: Vec<u32>,
    request: u64,
}
pub struct Session {
    identity: Option<Arc<()>>,
    arena: Option<Arena>,
    cache: VecDeque<Entry>,
    free: Vec<u32>,
    bytes: usize,
    limit: usize,
    request: u64,
    pub choice_cache_hits: u64,
}
impl Default for Session {
    fn default() -> Self {
        Self {
            identity: None,
            arena: None,
            cache: VecDeque::new(),
            free: Vec::new(),
            bytes: 0,
            limit: 32 * 1024 * 1024,
            request: 0,
            choice_cache_hits: 0,
        }
    }
}
impl Session {
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.bytes = 0;
        self.free = self
            .arena
            .as_ref()
            .map_or(Vec::new(), |a| (0..a.cache_rows as u32).rev().collect());
    }
    pub fn set_cache_bytes(&mut self, bytes: usize) {
        self.limit = bytes;
        self.arena = None;
        self.clear_cache();
    }
    pub fn cache_bytes(&self) -> usize {
        self.bytes
    }
}
struct Arena {
    hidden: wgpu::Buffer,
    update: wgpu::Buffer,
    partials: wgpu::Buffer,
    qkv: wgpu::Buffer,
    context: wgpu::Buffer,
    ff: wgpu::Buffer,
    memory: wgpu::Buffer,
    stream: [wgpu::Buffer; 2],
    normed: wgpu::Buffer,
    cache: wgpu::Buffer,
    cache_rows: usize,
    tables: wgpu::Buffer,
    table_capacity: usize,
    readout: wgpu::Buffer,
    staging: wgpu::Buffer,
    bindings: Bindings,
}
impl Arena {
    fn new(device: &Device, limit: usize) -> Result<Self> {
        let max_rows = device.device.limits().max_storage_buffer_binding_size as usize / (D * 4);
        let cache_rows = (limit / ROW_BYTES).min(max_rows - CHUNK_ROWS);
        let table_capacity = 32 * 1024;
        let readout = READOUT_ROWS * D;
        Ok(Self {
            hidden: device.storage(ENCODER_ROWS * D, "Decima hidden")?,
            update: device.storage(ENCODER_ROWS * D, "Decima update")?,
            partials: device.storage(FF / SLICE * SPLIT_ROWS * D, "Decima split partials")?,
            qkv: device.storage(ENCODER_ROWS * 3 * D, "Decima qkv")?,
            context: device.storage(ENCODER_ROWS * D, "Decima context")?,
            ff: device.storage(ENCODER_ROWS * FF, "Decima feed-forward")?,
            memory: device.storage(STATE_ROWS * 4 * D, "Decima state memory")?,
            stream: [
                device.storage(CHUNK_ROWS * D, "Decima scorer stream")?,
                device.storage(CHUNK_ROWS * D, "Decima scorer stream")?,
            ],
            normed: device.storage(CHUNK_ROWS * D, "Decima scorer norm")?,
            cache: device.storage((cache_rows + CHUNK_ROWS) * D, "Decima choice cache")?,
            cache_rows,
            tables: device.storage(table_capacity, "Decima tables")?,
            table_capacity,
            readout: device.storage(readout, "Decima readout vectors")?,
            staging: device.empty(
                readout * 4,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                "Decima result",
            )?,
            bindings: Bindings::new(),
        })
    }
}

struct Group {
    questions: Vec<Placed>,
    chunks: Vec<Chunk>,
    state_rows: usize,
    readout_rows: usize,
}
struct Placed {
    state: usize,
    tokens: usize,
    readout: usize,
    choices: usize,
}
#[derive(Default)]
struct Chunk {
    encode: Vec<(usize, Vec<u32>)>,
    scatter: Vec<[u32; 2]>,
    choices: Vec<(usize, usize, Vec<u32>)>,
}

impl Model {
    pub async fn from_cpu(mut source: CpuModel) -> Result<Self> {
        let super::embedding::Embeddings::Float(vocabulary) = std::mem::replace(
            &mut source.embeddings,
            super::embedding::Embeddings::Float(Vec::new()),
        ) else {
            anyhow::bail!("WebGPU requires float Decima embeddings");
        };
        let model = Self::without_vocabulary(source).await?;
        model.write_vocabulary(0, bytemuck::cast_slice(&vocabulary))?;
        Ok(model)
    }
    async fn without_vocabulary(source: CpuModel) -> Result<Self> {
        ensure!(
            source.precision() == Precision::F32,
            "WebGPU currently requires float Decima projections"
        );
        let device = Device::new().await?;
        let scope = device
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let gemm = |(rows, per_thread): (usize, usize)| {
            [Epilogue::Bias, Epilogue::Gelu, Epilogue::Residual].map(|e| {
                device.kernel(
                    "Decima GEMM",
                    shaders::gemm(rows, per_thread, e, Accumulate::Whole),
                    3 + (e == Epilogue::Residual) as usize,
                    1,
                )
            })
        };
        let long = |(rows, per_thread): (usize, usize), accumulate| {
            let reads = if accumulate == Accumulate::Split {
                2
            } else {
                4
            };
            let code = shaders::gemm(rows, per_thread, Epilogue::Residual, accumulate);
            device.kernel("Decima long GEMM", code, reads, 1)
        };
        let storage = device.device.limits().max_compute_workgroup_storage_size as usize;
        let queries = if shaders::attention_tiled_storage(64, 32, 16) <= storage {
            32
        } else {
            16
        };
        let kernels = Kernels {
            embed: device.kernel("Decima embedding", shaders::embed(), 6, 1),
            gemm: GEMM_TILES.map(gemm),
            sliced: [
                long(GEMM_TILES[1], Accumulate::Sliced),
                long(GEMM_TILES[2], Accumulate::Sliced),
            ],
            split: [
                long(GEMM_TILES[0], Accumulate::Split),
                long(GEMM_TILES[1], Accumulate::Split),
            ],
            reduce: device.kernel("Decima reduce", shaders::reduce(), 3, 1),
            norm: device.kernel("Decima norm", shaders::norm(), 3, 1),
            encoder_attention: device.kernel(
                "Decima attention",
                shaders::attention_rows(32, ENCODER_QUERIES, 4, 16),
                3,
                1,
            ),
            scorer_attention: device.kernel(
                "Decima attention",
                shaders::attention_tiled(64, queries, 16),
                3,
                1,
            ),
            mean: device.kernel("Decima mean", shaders::mean(), 2, 1),
            copy: device.kernel("Decima copy", shaders::copy(), 2, 1),
        };
        if let Some(error) = scope.pop().await {
            anyhow::bail!("Decima GPU pipeline: {error}");
        }
        let linear = |parts: Vec<cpu::Linear>| Linear::upload(&device, parts);
        let encoder = source
            .encoder
            .into_iter()
            .map(|l| {
                Ok(EncoderLayer {
                    qkv: linear(vec![l.q, l.k, l.v])?,
                    o: linear(vec![l.o])?,
                    n1: Norm::upload(&device, &l.n1)?,
                    n2: Norm::upload(&device, &l.n2)?,
                    ff1: linear(vec![l.ff1])?,
                    ff2: linear(vec![l.ff2])?,
                })
            })
            .collect::<Result<_>>()?;
        let mut memory = Vec::new();
        let scorer = source
            .scorer
            .into_iter()
            .map(|l| {
                memory.push(l.cross_attn.k);
                memory.push(l.cross_attn.v);
                Ok(ScorerLayer {
                    qkv: linear(vec![l.self_attn.q, l.self_attn.k, l.self_attn.v])?,
                    o: linear(vec![l.self_attn.o])?,
                    cross_q: linear(vec![l.cross_attn.q])?,
                    cross_o: linear(vec![l.cross_attn.o])?,
                    n1: Norm::upload(&device, &l.n1)?,
                    n2: Norm::upload(&device, &l.n2)?,
                    n3: Norm::upload(&device, &l.n3)?,
                    ff1: linear(vec![l.ff1])?,
                    ff2: linear(vec![l.ff2])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(scorer.len() * 2 * D <= 4 * D, "Decima scorer memory layout");
        let memory = linear(memory)?;
        let n = &source.embedding_norm;
        ensure!(
            source.types.len() >= D && n.weight.len() == D && source.positions.len() == 512 * D,
            "Decima embedding shapes"
        );
        let prelude: Vec<f32> =
            [&source.types[..D], &n.weight, &n.bias, &source.positions].concat();
        let split = |i: usize| {
            let rows = VOCABULARY_SPLIT.min(VOCABULARY - i * VOCABULARY_SPLIT);
            device.storage(rows * D, "Decima vocabulary")
        };
        let vocabulary = [split(0)?, split(1)?, split(2)?, split(3)?];
        Ok(Self {
            vocabulary,
            prelude: device.buffer(&prelude, "Decima embedding prelude")?,
            embedding_eps: n.eps,
            out_norm: Norm::upload(&device, &source.out_norm)?,
            config: source.config,
            input: source.input,
            readout: readout::Readout::new(
                source.score,
                source.ord_g,
                source.ord_gap,
                source.sim_scale,
                source.sim_center,
            )?,
            encoder,
            scorer,
            memory,
            kernels,
            queries,
            device,
            identity: Arc::new(()),
        })
    }
    fn write_vocabulary(&self, first: usize, bytes: &[u8]) -> Result<()> {
        let width = D * 4;
        ensure!(
            bytes.len() % width == 0 && first + bytes.len() / width <= VOCABULARY,
            "Decima vocabulary range"
        );
        let (mut row, mut rest) = (first, bytes);
        while !rest.is_empty() {
            let (split, within) = (row / VOCABULARY_SPLIT, row % VOCABULARY_SPLIT);
            let rows = (VOCABULARY_SPLIT - within).min(rest.len() / width);
            let (head, tail) = rest.split_at(rows * width);
            self.device
                .queue
                .write_buffer(&self.vocabulary[split], (within * width) as u64, head);
            (row, rest) = (row + rows, tail);
        }
        Ok(())
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.device.info
    }

    fn gemm(
        &self,
        f: &mut Frame,
        a: &mut Arena,
        l: &Linear,
        x: &wgpu::Buffer,
        residual: Option<&wgpu::Buffer>,
        gelu: bool,
        y: &wgpu::Buffer,
        rows: usize,
    ) -> Result<()> {
        // Short tiles keep small batches spread over the GPU; more rows per thread pay off once
        // the grid fills it (measured on M3 Safari and Dawn). All tiles give identical results.
        let tile = match rows {
            0..=32 => 0,
            33..320 => 1,
            _ => 2,
        };
        let groups = |tile: usize| {
            [
                (l.output / 64) as u32,
                rows.div_ceil(GEMM_TILES[tile].0) as u32,
                1,
            ]
        };
        let p = [rows as u32, l.output as u32, l.input as u32];
        if l.input > SLICE {
            // Long products: few rows split k over more workgroups, then reduce in slice order.
            let r = residual.context("long Decima projections must have a residual")?;
            ensure!(
                !gelu && l.input % SLICE == 0 && l.input <= FF && l.output <= D,
                "Decima long projection shape"
            );
            if rows > SPLIT_ROWS {
                let kernel = &self.kernels.sliced[tile - 1];
                return f.dispatch(
                    &self.device,
                    &mut a.bindings,
                    kernel,
                    &[x, &l.weight, &l.bias, r, y],
                    &p,
                    groups(tile),
                );
            }
            let tile = (rows > 32) as usize;
            let slices = l.input / SLICE;
            let [g0, g1, _] = groups(tile);
            let partials = a.partials.clone();
            f.dispatch(
                &self.device,
                &mut a.bindings,
                &self.kernels.split[tile],
                &[x, &l.weight, &partials],
                &p,
                [g0, g1, slices as u32],
            )?;
            let count = (rows * l.output / 4).div_ceil(64) as u32;
            return f.dispatch(
                &self.device,
                &mut a.bindings,
                &self.kernels.reduce,
                &[&partials, &l.bias, r, y],
                &[rows as u32, l.output as u32, slices as u32],
                [count, 1, 1],
            );
        }
        let kernels = &self.kernels.gemm[tile];
        match residual {
            Some(r) => f.dispatch(
                &self.device,
                &mut a.bindings,
                &kernels[2],
                &[x, &l.weight, &l.bias, r, y],
                &p,
                groups(tile),
            ),
            None => f.dispatch(
                &self.device,
                &mut a.bindings,
                &kernels[gelu as usize],
                &[x, &l.weight, &l.bias, y],
                &p,
                groups(tile),
            ),
        }
    }
    fn norm(
        &self,
        f: &mut Frame,
        a: &mut Arena,
        n: &Norm,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
    ) -> Result<()> {
        f.dispatch(
            &self.device,
            &mut a.bindings,
            &self.kernels.norm,
            &[x, &n.weight, &n.bias, y],
            &[n.eps.to_bits()],
            [rows as u32, 1, 1],
        )
    }
    fn attention(
        &self,
        f: &mut Frame,
        a: &mut Arena,
        kernel: &Kernel,
        q: &wgpu::Buffer,
        kv: &wgpu::Buffer,
        y: &wgpu::Buffer,
        strides: [usize; 5],
        heads: usize,
        table: usize,
        tiles: usize,
    ) -> Result<()> {
        if tiles == 0 {
            return Ok(());
        }
        let scale = 1. / ((D / heads) as f32).sqrt();
        let [qs, qo, kvs, ko, vo] = strides.map(|v| v as u32);
        f.dispatch(
            &self.device,
            &mut a.bindings,
            kernel,
            &[q, kv, &a.tables.clone(), y],
            &[qs, qo, kvs, ko, vo, D as u32, scale.to_bits(), table as u32],
            [tiles as u32, heads as u32, 1],
        )
    }
    fn rows_op(
        &self,
        f: &mut Frame,
        a: &mut Arena,
        kernel: &Kernel,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        table: usize,
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let tables = a.tables.clone();
        f.dispatch(
            &self.device,
            &mut a.bindings,
            kernel,
            &[x, &tables, y],
            &[table as u32],
            [count as u32, 1, 1],
        )
    }

    pub async fn decide(
        &self,
        state: &str,
        q: &Question,
        session: &mut Session,
    ) -> Result<Vec<f32>> {
        Ok(self.decide_many(&[(state, q)], session).await?.remove(0))
    }
    pub async fn decide_many(
        &self,
        items: &[(&str, &Question)],
        session: &mut Session,
    ) -> Result<Vec<Vec<f32>>> {
        let result = self.decide_inner(items, session).await;
        if result.is_err() {
            session.clear_cache();
        }
        result
    }
    async fn decide_inner(
        &self,
        items: &[(&str, &Question)],
        session: &mut Session,
    ) -> Result<Vec<Vec<f32>>> {
        if !session
            .identity
            .as_ref()
            .is_some_and(|id| Arc::ptr_eq(id, &self.identity))
        {
            session.arena = None;
            session.identity = Some(self.identity.clone());
        }
        if session.arena.is_none() {
            session.arena = Some(Arena::new(&self.device, session.limit)?);
            session.clear_cache();
        }
        let c = &self.config;
        let mut prepared = Vec::with_capacity(items.len());
        for &(state, q) in items {
            ensure!(
                (2..=MAX_CHOICES).contains(&q.choices.len()),
                "need 2–255 choices"
            );
            if q.kind == Kind::Score {
                ensure!(q.choices.len() <= 10, "score allows at most 10 levels");
            }
            let body = if c.question_in_state {
                format!("{}\n{state}", q.text)
            } else {
                state.into()
            };
            let text = format!("{}{}", c.state_prefix, input::normalize(&body, &q.lang));
            let ids = self.input.ids(&text)?;
            ensure!(
                ids.len() <= self.input.max_state && ids.len() <= 512,
                "STATE_TRUNCATED: state + question is {} tokens; maximum {}",
                ids.len(),
                self.input.max_state
            );
            ensure!(
                !ids.is_empty() && ids.iter().all(|&id| (id as usize) < 250037),
                "invalid Decima tokens"
            );
            let rows =
                self.input
                    .choices(q, &c.choice_prefix, c.question_in_choices, c.choice_fit)?;
            ensure!(
                rows.iter().all(|r| !r.is_empty()
                    && r.len() <= 64
                    && r.iter().all(|&id| (id as usize) < 250037)),
                "invalid Decima choice tokens"
            );
            prepared.push((ids, rows));
        }
        let groups = plan(session, prepared);
        let mut out = Vec::with_capacity(items.len());
        let mut item = 0;
        for g in &groups {
            let mut encoder = None;
            for index in 0..g.chunks.len() {
                let arena = session.arena.as_mut().unwrap();
                let mut e = self
                    .device
                    .device
                    .create_command_encoder(&Default::default());
                self.record(arena, g, index, &mut e)?;
                if index + 1 < g.chunks.len() {
                    self.device.queue.submit([e.finish()]);
                } else {
                    encoder = Some(e);
                }
            }
            let arena = session.arena.as_ref().unwrap();
            let pooled = device::read(
                &self.device,
                encoder.context("Decima request has no work")?,
                &arena.readout,
                &arena.staging,
                g.readout_rows * D,
            )
            .await?;
            for q in &g.questions {
                let rows = &pooled[q.readout * D..(q.readout + 1 + 2 * q.choices) * D];
                out.push(
                    self.readout
                        .probabilities(rows, items[item].1.kind, c.temperature)?,
                );
                item += 1;
            }
        }
        Ok(out)
    }

    fn record(
        &self,
        a: &mut Arena,
        g: &Group,
        index: usize,
        e: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        let chunk = &g.chunks[index];
        let first = index == 0;
        let mut tables: Vec<u32> = Vec::new();
        let mut encoder_rows = 0;
        for (first_row, ids) in &chunk.encode {
            ensure!(
                *first_row == encoder_rows,
                "Decima encoder rows must be contiguous"
            );
            tables.extend(ids.iter().enumerate().flat_map(|(p, &id)| [id, p as u32]));
            encoder_rows += ids.len();
        }
        let encoder_tiles = tables.len();
        for (first_row, ids) in &chunk.encode {
            tiles(
                &mut tables,
                ENCODER_QUERIES,
                *first_row,
                ids.len(),
                *first_row,
                ids.len(),
            );
        }
        let encoder_tile_count = (tables.len() - encoder_tiles) / 4;
        let state_means = tables.len();
        for q in &g.questions {
            tables.extend([q.state as u32, q.tokens as u32, q.readout as u32]);
        }
        let scatter = tables.len();
        tables.extend(chunk.scatter.iter().flatten());
        let gather = tables.len();
        let mut segments = Vec::new();
        let mut scorer_rows = 0;
        for (question, readout, rows) in &chunk.choices {
            for &row in rows {
                tables.extend([row, scorer_rows as u32]);
                scorer_rows += 1;
            }
            segments.push((scorer_rows - rows.len(), rows.len(), *question, *readout));
        }
        let self_tiles = tables.len();
        for &(start, count, _, _) in &segments {
            tiles(&mut tables, self.queries, start, count, start, count);
        }
        let self_tile_count = (tables.len() - self_tiles) / 4;
        let cross_tiles = tables.len();
        for run in segments.chunk_by(|a, b| a.2 == b.2) {
            let (start, question) = (run[0].0, &g.questions[run[0].2]);
            let count = run.iter().map(|s| s.1).sum();
            tiles(
                &mut tables,
                self.queries,
                start,
                count,
                question.state,
                question.tokens,
            );
        }
        let cross_tile_count = (tables.len() - cross_tiles) / 4;
        let originals = tables.len();
        for &(start, count, _, readout) in &segments {
            tables.extend([start as u32, count as u32, readout as u32]);
        }
        let finals = tables.len();
        for &(start, count, _, readout) in &segments {
            tables.extend([start as u32, count as u32, (readout + 1) as u32]);
        }
        ensure!(
            tables.len() <= a.table_capacity,
            "Decima request tables too large"
        );
        self.device
            .queue
            .write_buffer(&a.tables, 0, bytemuck::cast_slice(&tables));
        a.bindings.reserve(&self.device, 160);
        let mut f = Frame::new();
        let (hidden, update, qkv, context, ff) = (
            a.hidden.clone(),
            a.update.clone(),
            a.qkv.clone(),
            a.context.clone(),
            a.ff.clone(),
        );
        if encoder_rows > 0 {
            let n = encoder_rows;
            let [v0, v1, v2, v3] = &self.vocabulary;
            let tables = a.tables.clone();
            f.dispatch(
                &self.device,
                &mut a.bindings,
                &self.kernels.embed,
                &[v0, v1, v2, v3, &self.prelude, &tables, &hidden],
                &[0, self.embedding_eps.to_bits()],
                [n as u32, 1, 1],
            )?;
            for l in &self.encoder {
                self.gemm(&mut f, a, &l.qkv, &hidden, None, false, &qkv, n)?;
                self.attention(
                    &mut f,
                    a,
                    &self.kernels.encoder_attention,
                    &qkv,
                    &qkv,
                    &context,
                    [3 * D, 0, 3 * D, D, 2 * D],
                    12,
                    encoder_tiles,
                    encoder_tile_count,
                )?;
                self.gemm(&mut f, a, &l.o, &context, Some(&hidden), false, &update, n)?;
                self.norm(&mut f, a, &l.n1, &update, &hidden, n)?;
                self.gemm(&mut f, a, &l.ff1, &hidden, None, true, &ff, n)?;
                self.gemm(&mut f, a, &l.ff2, &ff, Some(&hidden), false, &update, n)?;
                self.norm(&mut f, a, &l.n2, &update, &hidden, n)?;
            }
        }
        let (memory, cache, readout, normed) = (
            a.memory.clone(),
            a.cache.clone(),
            a.readout.clone(),
            a.normed.clone(),
        );
        if first {
            self.gemm(
                &mut f,
                a,
                &self.memory,
                &hidden,
                None,
                false,
                &memory,
                g.state_rows,
            )?;
            self.rows_op(
                &mut f,
                a,
                &self.kernels.mean,
                &hidden,
                &readout,
                state_means,
                g.questions.len(),
            )?;
        }
        self.rows_op(
            &mut f,
            a,
            &self.kernels.copy,
            &hidden,
            &cache,
            scatter,
            chunk.scatter.len(),
        )?;
        let [mut x, mut y] = a.stream.clone();
        if scorer_rows > 0 {
            self.rows_op(
                &mut f,
                a,
                &self.kernels.copy,
                &cache,
                &x,
                gather,
                scorer_rows,
            )?;
            self.rows_op(
                &mut f,
                a,
                &self.kernels.mean,
                &x,
                &readout,
                originals,
                segments.len(),
            )?;
            let n = scorer_rows;
            let attention = &self.kernels.scorer_attention;
            for (index, l) in self.scorer.iter().enumerate() {
                self.norm(&mut f, a, &l.n1, &x, &normed, n)?;
                self.gemm(&mut f, a, &l.qkv, &normed, None, false, &qkv, n)?;
                self.attention(
                    &mut f,
                    a,
                    attention,
                    &qkv,
                    &qkv,
                    &context,
                    [3 * D, 0, 3 * D, D, 2 * D],
                    6,
                    self_tiles,
                    self_tile_count,
                )?;
                self.gemm(&mut f, a, &l.o, &context, Some(&x), false, &y, n)?;
                self.norm(&mut f, a, &l.n2, &y, &normed, n)?;
                self.gemm(&mut f, a, &l.cross_q, &normed, None, false, &update, n)?;
                let offset = index * 2 * D;
                self.attention(
                    &mut f,
                    a,
                    attention,
                    &update,
                    &memory,
                    &context,
                    [D, 0, 4 * D, offset, offset + D],
                    6,
                    cross_tiles,
                    cross_tile_count,
                )?;
                self.gemm(&mut f, a, &l.cross_o, &context, Some(&y), false, &x, n)?;
                self.norm(&mut f, a, &l.n3, &x, &normed, n)?;
                self.gemm(&mut f, a, &l.ff1, &normed, None, true, &ff, n)?;
                self.gemm(&mut f, a, &l.ff2, &ff, Some(&x), false, &y, n)?;
                std::mem::swap(&mut x, &mut y);
            }
            self.norm(&mut f, a, &self.out_norm, &x, &normed, n)?;
            self.rows_op(
                &mut f,
                a,
                &self.kernels.mean,
                &normed,
                &readout,
                finals,
                segments.len(),
            )?;
        }
        f.encode(&self.device, &a.bindings, e);
        Ok(())
    }
}

fn tiles(t: &mut Vec<u32>, size: usize, first: usize, count: usize, keys: usize, key_count: usize) {
    for q in (0..count).step_by(size) {
        t.extend([
            (first + q) as u32,
            size.min(count - q) as u32,
            keys as u32,
            key_count as u32,
        ]);
    }
}

fn plan(session: &mut Session, items: Vec<(Vec<u32>, Vec<Vec<u32>>)>) -> Vec<Group> {
    session.request += 1;
    let request = session.request;
    let cache_rows = session.arena.as_ref().unwrap().cache_rows as u32;
    let mut groups: Vec<Group> = Vec::new();
    let mut pending: Vec<Vec<Vec<Vec<u32>>>> = Vec::new();
    for (state, choices) in items {
        let need = 1 + 2 * choices.len();
        let fits = groups.last().is_some_and(|g| {
            g.state_rows + state.len() <= STATE_ROWS && g.readout_rows + need <= READOUT_ROWS
        });
        if !fits {
            groups.push(Group {
                questions: Vec::new(),
                chunks: vec![Chunk::default()],
                state_rows: 0,
                readout_rows: 0,
            });
            pending.push(Vec::new());
        }
        let g = groups.last_mut().unwrap();
        g.questions.push(Placed {
            state: g.state_rows,
            tokens: state.len(),
            readout: g.readout_rows,
            choices: choices.len(),
        });
        g.state_rows += state.len();
        g.readout_rows += need;
        g.chunks[0]
            .encode
            .push((g.questions.last().unwrap().state, state));
        pending.last_mut().unwrap().push(choices);
    }
    for (g, pending) in groups.iter_mut().zip(pending) {
        let mut used = 0;
        let mut encoded = g.state_rows;
        for (question, choices) in pending.into_iter().enumerate() {
            let readout = g.questions[question].readout;
            for (j, ids) in choices.into_iter().enumerate() {
                if used + ids.len() > CHUNK_ROWS {
                    g.chunks.push(Chunk::default());
                    used = 0;
                    encoded = 0;
                }
                let chunk = g.chunks.last_mut().unwrap();
                let row = readout + 1 + 2 * j;
                let n = ids.len();
                used += n;
                if let Some(i) = session.cache.iter().position(|e| e.ids == ids) {
                    let mut entry = session.cache.remove(i).unwrap();
                    entry.request = request;
                    chunk.choices.push((question, row, entry.rows.clone()));
                    session.cache.push_back(entry);
                    session.choice_cache_hits += 1;
                    continue;
                }
                let bytes = n * ROW_BYTES;
                let mut rows = None;
                if bytes <= session.limit && n <= cache_rows as usize {
                    while (session.bytes + bytes > session.limit || session.free.len() < n)
                        && session.cache.front().is_some_and(|e| e.request != request)
                    {
                        let old = session.cache.pop_front().unwrap();
                        session.bytes -= old.ids.len() * ROW_BYTES;
                        session.free.extend(old.rows);
                    }
                    if session.bytes + bytes <= session.limit && session.free.len() >= n {
                        rows = Some(session.free.split_off(session.free.len() - n));
                    }
                }
                let cacheable = rows.is_some();
                let first = (used - n) as u32;
                let rows =
                    rows.unwrap_or_else(|| (0..n as u32).map(|i| cache_rows + first + i).collect());
                for (i, &row) in rows.iter().enumerate() {
                    chunk.scatter.push([(encoded + i) as u32, row]);
                }
                chunk.choices.push((question, row, rows.clone()));
                chunk.encode.push((encoded, ids.clone()));
                encoded += n;
                if cacheable {
                    session.bytes += bytes;
                    session.cache.push_back(Entry { ids, rows, request });
                }
            }
        }
    }
    groups
}

impl Linear {
    fn upload(device: &Device, parts: Vec<cpu::Linear>) -> Result<Self> {
        let parts = parts
            .into_iter()
            .map(|l| l.into_float_parts())
            .collect::<Result<Vec<_>>>()?;
        let input = parts[0].0;
        ensure!(
            parts.iter().all(|p| p.0 == input),
            "Decima projection widths differ"
        );
        let output: usize = parts.iter().map(|p| p.1).sum();
        ensure!(
            input % 16 == 0 && output % 64 == 0,
            "Decima projection shape"
        );
        let mut weight = vec![0.; input * output];
        let mut bias = Vec::with_capacity(output);
        let mut column = 0;
        for (_, n, w, b) in &parts {
            for o in 0..*n {
                for k in 0..input {
                    weight[k * output + column + o] = w[o * input + k];
                }
            }
            bias.extend_from_slice(b);
            column += n;
        }
        Ok(Self {
            input,
            output,
            weight: device.buffer(&weight, "Decima projection")?,
            bias: device.buffer(&bias, "Decima bias")?,
        })
    }
}
impl Norm {
    fn upload(device: &Device, l: &cpu::Norm) -> Result<Self> {
        ensure!(l.weight.len() == D, "Decima norm width");
        Ok(Self {
            eps: l.eps,
            weight: device.buffer(&l.weight, "Decima norm weight")?,
            bias: device.buffer(&l.bias, "Decima norm bias")?,
        })
    }
}

impl Model {
    pub async fn predict_json(&self, request: &Value, s: &mut Session) -> Result<Value> {
        if let Some(rows) = request.as_array() {
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                out.push(Box::pin(self.predict_json(row, s)).await?);
            }
            return Ok(Value::Array(out));
        }
        let parsed = super::long::parse(&self.input, request)?;
        let batch: Vec<_> = parsed
            .states
            .iter()
            .flat_map(|state| {
                parsed
                    .questions
                    .iter()
                    .map(move |(_, q, _)| (state.as_str(), q))
            })
            .collect();
        let scores = self.decide_many(&batch, s).await?;
        let scores: Vec<Vec<Vec<f32>>> = scores
            .chunks(parsed.questions.len())
            .map(<[_]>::to_vec)
            .collect();
        Ok(super::long::respond(&parsed, &scores))
    }
}

#[cfg(feature = "wasm")]
fn without_vocabulary(header: &[u8]) -> Result<(Vec<u8>, std::ops::Range<usize>)> {
    let mut tensors: serde_json::Map<String, Value> = serde_json::from_slice(header)?;
    let name = "embeddings.word_embeddings.weight";
    let entry = tensors
        .remove(name)
        .context("encoder checkpoint has no word embeddings")?;
    let offsets = |v: &Value| -> Option<[usize; 2]> {
        let o = v["data_offsets"].as_array()?;
        Some([o.first()?.as_u64()? as usize, o.get(1)?.as_u64()? as usize])
    };
    let [start, end] = offsets(&entry).context("invalid word embedding offsets")?;
    ensure!(
        entry["dtype"] == "F32"
            && entry["shape"] == json!([VOCABULARY, D])
            && end - start == VOCABULARY * D * 4,
        "WebGPU requires float Decima embeddings"
    );
    for (key, value) in tensors.iter_mut() {
        if key == "__metadata__" {
            continue;
        }
        let [a, b] = offsets(value).with_context(|| format!("invalid offsets for {key}"))?;
        ensure!(
            b <= start || a >= end,
            "tensor {key} overlaps the word embeddings"
        );
        if a >= end {
            value["data_offsets"] = json!([a - (end - start), b - (end - start)]);
        }
    }
    let mut out = serde_json::to_vec(&tensors)?;
    out.resize(out.len().next_multiple_of(8), b' ');
    Ok((out, start..end))
}

#[cfg(feature = "wasm")]
pub(crate) mod browser {
    use super::*;
    use wasm_bindgen::prelude::*;
    fn error(e: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&e.to_string())
    }
    enum Source {
        Blob(web_sys::Blob),
        Bytes(js_sys::Uint8Array),
    }
    impl Source {
        fn new(value: JsValue) -> Result<Self> {
            if let Some(blob) = value.dyn_ref::<web_sys::Blob>() {
                return Ok(Self::Blob(blob.clone()));
            }
            if let Some(buffer) = value.dyn_ref::<js_sys::ArrayBuffer>() {
                return Ok(Self::Bytes(js_sys::Uint8Array::new(buffer)));
            }
            match value.dyn_into::<js_sys::Uint8Array>() {
                Ok(bytes) => Ok(Self::Bytes(bytes)),
                Err(_) => {
                    anyhow::bail!("encoder weights must be a Blob, Uint8Array or ArrayBuffer")
                }
            }
        }
        fn len(&self) -> usize {
            match self {
                Self::Blob(blob) => blob.size() as usize,
                Self::Bytes(bytes) => bytes.length() as usize,
            }
        }
        async fn read(&self, start: usize, end: usize) -> Result<Vec<u8>> {
            ensure!(
                start <= end && end <= self.len(),
                "encoder weights are truncated"
            );
            match self {
                Self::Bytes(bytes) => Ok(bytes.subarray(start as u32, end as u32).to_vec()),
                Self::Blob(blob) => {
                    let js = |e: JsValue| anyhow::anyhow!("reading encoder weights: {e:?}");
                    let part = blob
                        .slice_with_f64_and_f64(start as f64, end as f64)
                        .map_err(js)?;
                    let buffer = wasm_bindgen_futures::JsFuture::from(part.array_buffer())
                        .await
                        .map_err(js)?;
                    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
                }
            }
        }
        async fn without_vocabulary(&self) -> Result<(Vec<u8>, std::ops::Range<usize>)> {
            let size = u64::from_le_bytes(self.read(0, 8).await?.try_into().unwrap()) as usize;
            ensure!(size < 1 << 24, "invalid encoder checkpoint header");
            let data = 8 + size;
            let (header, gap) = super::without_vocabulary(&self.read(8, data).await?)?;
            let mut out = Vec::with_capacity(8 + header.len() + self.len() - data - gap.len());
            out.extend((header.len() as u64).to_le_bytes());
            out.extend(header);
            out.extend(self.read(data, data + gap.start).await?);
            out.extend(self.read(data + gap.end, self.len()).await?);
            Ok((out, data + gap.start..data + gap.end))
        }
    }
    #[wasm_bindgen(js_name=DecimaWebGpuModel)]
    pub struct BrowserModel {
        pub(crate) model: Model,
        pub(crate) session: Session,
    }
    #[wasm_bindgen(js_class=DecimaWebGpuModel)]
    impl BrowserModel {
        pub async fn create(
            config: Vec<u8>,
            encoder_config: Vec<u8>,
            tokenizer: Vec<u8>,
            encoder_weights: JsValue,
            head_weights: Vec<u8>,
        ) -> Result<BrowserModel, JsValue> {
            let source = Source::new(encoder_weights).map_err(error)?;
            let (encoder, vocabulary) = source.without_vocabulary().await.map_err(error)?;
            let cpu = CpuModel::without_vocabulary(
                &config,
                &encoder_config,
                &tokenizer,
                &encoder,
                &head_weights,
                super::super::Options {
                    precision: Precision::F32,
                    kernel: super::super::Kernel::Portable,
                },
            )
            .map_err(error)?;
            drop((config, encoder_config, tokenizer, encoder, head_weights));
            let model = Model::without_vocabulary(cpu).await.map_err(error)?;
            let step = 4096 * D * 4;
            for at in vocabulary.clone().step_by(step) {
                let bytes = source
                    .read(at, (at + step).min(vocabulary.end))
                    .await
                    .map_err(error)?;
                model
                    .write_vocabulary((at - vocabulary.start) / (D * 4), &bytes)
                    .map_err(error)?;
            }
            Ok(Self {
                model,
                session: Session::default(),
            })
        }
        pub async fn predict(&mut self, json: &str) -> Result<String, JsValue> {
            let request = serde_json::from_str(json).map_err(error)?;
            serde_json::to_string(
                &self
                    .model
                    .predict_json(&request, &mut self.session)
                    .await
                    .map_err(error)?,
            )
            .map_err(error)
        }
        pub async fn decide(
            &mut self,
            state: &str,
            question_json: &str,
        ) -> Result<String, JsValue> {
            let question = serde_json::from_str(question_json).map_err(error)?;
            serde_json::to_string(
                &self
                    .model
                    .decide(state, &question, &mut self.session)
                    .await
                    .map_err(error)?,
            )
            .map_err(error)
        }
        // Length rules tune the CPU kernels; the GPU path sizes its work per batch.
        #[wasm_bindgen(js_name=setLengthRules)]
        pub fn set_length_rules(&mut self, _json: &str) -> Result<(), JsValue> {
            Err(error("length rules apply only to the CPU Decima model"))
        }
        #[wasm_bindgen(js_name=clearCache)]
        pub fn clear_cache(&mut self) {
            self.session.clear_cache();
        }
        #[wasm_bindgen(js_name=setCacheBytes)]
        pub fn set_cache_bytes(&mut self, bytes: usize) {
            self.session.set_cache_bytes(bytes);
        }
        #[wasm_bindgen(js_name=adapterInfo)]
        pub fn adapter_info(&self) -> String {
            let i = self.model.adapter_info();
            json!({"name":i.name,"backend":format!("{:?}",i.backend),"device_type":format!("{:?}",i.device_type)}).to_string()
        }
    }
}
