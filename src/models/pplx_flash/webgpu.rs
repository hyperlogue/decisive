#[cfg(feature = "kernel-checks")]
mod checkpoint;
#[cfg(feature = "kernel-checks")]
pub(super) mod checks;
mod device;
pub(super) mod streaming;
#[cfg(feature = "kernel-checks")]
mod streaming_check;
#[cfg(feature = "kernel-checks")]
mod int8;
#[cfg(feature = "kernel-checks")]
pub(super) mod q4;
#[cfg(feature = "kernel-checks")]
pub(super) mod grouped_q4;
mod profile;
mod shaders;
#[cfg(feature = "kernel-checks")]
pub(super) mod synthetic;
use super::D;
use anyhow::{Context, Result, ensure};
use device::Device;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};
use wgpu::{Buffer, CommandEncoder, ComputePipeline};

#[derive(Clone)]
pub(super) struct Tensor {
    pub shape: Vec<usize>,
    pub buffer: Buffer,
}
pub(super) struct QueuedLayer {
    completion: device::Completion,
    scratch: Vec<Buffer>,
    #[cfg(feature = "kernel-checks")]
    quantized: Vec<(Buffer, Buffer)>,
    #[cfg(feature = "kernel-checks")]
    q4_expansion: Option<Buffer>,
}
pub(super) struct Gpu {
    device: Device,
    pipelines: HashMap<&'static str, ComputePipeline>,
    dummy: Buffer,
    hidden: Option<Buffer>,
    rope: Option<Buffer>,
    rows: usize,
    scratch: RefCell<Vec<Buffer>>,
    fp16_projections: bool,
    gemm_tile: [u32; 2],
    packed_inputs: bool,
    packed_cache: RefCell<Vec<(Buffer, Buffer)>>,
    mlp_tile: Option<[u32; 2]>,
    tiled_attention: bool,
    cooperative_norm: bool,
    small_tile: Option<[u32; 2]>,
    attention_block: usize,
    reuse_scratch: bool,
    scratch_pool: RefCell<Vec<Buffer>>,
    allocation_count: Cell<usize>,
    dispatch_count: Cell<usize>,
    delta_scan: &'static str,
    delta_scan_selection: serde_json::Value,
    gemm_selection: serde_json::Value,
    mlp_selection: serde_json::Value,
    // Whether layers use the q4_mlp kernel when it is installed; startup selection may prefer the
    // bitwise-equal unfused path.
    fused_mlp: bool,
    // The selected int8_gemm kernel's [rows, columns] tile, when it replaced the tuned one.
    int8_gemm_tile: Option<[u32; 2]>,
    #[cfg(feature = "kernel-checks")]
    int8: Option<int8::State>,
    #[cfg(feature = "kernel-checks")]
    reuse_weight_buffers: bool,
    #[cfg(feature = "kernel-checks")]
    weight_buffers: RefCell<Vec<Buffer>>,
    #[cfg(feature = "kernel-checks")]
    q4: Option<q4::Mode>,
    #[cfg(feature = "kernel-checks")]
    q4_expansion: Option<Buffer>,
}
impl Gpu {
    pub(crate) fn grouped_q4(&self) -> bool {
        #[cfg(feature = "kernel-checks")]
        {
            matches!(self.q4, Some(q4::Mode::Group64 | q4::Mode::Group64Signed))
        }
        #[cfg(not(feature = "kernel-checks"))]
        {
            false
        }
    }
    pub async fn new() -> Result<Self> {
        Self::with_precision(false).await
    }
    pub(super) async fn with_precision(fp16_projections: bool) -> Result<Self> {
        Self::with_options(fp16_projections, false).await
    }
    pub(super) async fn with_options(fp16_projections: bool, profile: bool) -> Result<Self> {
        let mut features = wgpu::Features::empty();
        if fp16_projections {
            features |= wgpu::Features::SHADER_F16;
        }
        if profile {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let device = Device::with_features(features).await?;
        let scope = device
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 4 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 3,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Pplx Flash bindings"),
                entries: &entries,
            });
        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pplx Flash layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let mut pipelines = HashMap::new();
        for (name, body) in [
            ("gemm", shaders::GEMM),
            ("norm", shaders::NORM),
            ("point", shaders::POINT),
            ("conv", shaders::CONV),
            ("delta_qk", shaders::DELTA_QK),
            ("delta_params", shaders::DELTA_PARAMS),
            ("delta_constants", shaders::DELTA_CONSTANTS),
            ("delta_scan", shaders::DELTA_SCAN),
            ("delta_gate", shaders::DELTA_GATE),
            ("attention_norm", shaders::ATTENTION_NORM),
            ("scores", shaders::SCORES),
            ("softmax", shaders::SOFTMAX),
            ("values", shaders::VALUES),
        ] {
            let module = device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(if name == "gemm" && fp16_projections {
                        shaders::fp16_gemm().into()
                    } else {
                        shaders::source(body).into()
                    }),
                });
            pipelines.insert(
                name,
                device
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(name),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );
        }
        if let Some(error) = scope.pop().await {
            anyhow::bail!("Pplx WebGPU pipeline: {error}");
        }
        let dummy = device.buffer(&[0.], "unused")?;
        Ok(Self {
            device,
            pipelines,
            dummy,
            hidden: None,
            rope: None,
            rows: 0,
            scratch: RefCell::new(Vec::new()),
            fp16_projections,
            gemm_tile: [32, 32],
            packed_inputs: false,
            packed_cache: RefCell::new(Vec::new()),
            mlp_tile: None,
            tiled_attention: false,
            cooperative_norm: false,
            small_tile: None,
            attention_block: 16,
            reuse_scratch: false,
            scratch_pool: RefCell::new(Vec::new()),
            allocation_count: Cell::new(0),
            dispatch_count: Cell::new(0),
            delta_scan: "delta_scan",
            delta_scan_selection: serde_json::Value::Null,
            gemm_selection: serde_json::Value::Null,
            mlp_selection: serde_json::Value::Null,
            fused_mlp: true,
            int8_gemm_tile: None,
            #[cfg(feature = "kernel-checks")]
            int8: None,
            #[cfg(feature = "kernel-checks")]
            reuse_weight_buffers: false,
            #[cfg(feature = "kernel-checks")]
            weight_buffers: RefCell::new(Vec::new()),
            #[cfg(feature = "kernel-checks")]
            q4: None,
            #[cfg(feature = "kernel-checks")]
            q4_expansion: None,
        })
    }
    pub fn info(&self) -> serde_json::Value {
        let info = &self.device.info;
        let projection_precision = if self.fp16_projections {
            "fp16-storage-f16-tile-f32-outer"
        } else {
            "bf16-storage-f32-compute"
        };
        #[cfg(feature = "kernel-checks")]
        let projection_precision = match &self.int8 {
            Some(state) if self.grouped_q4() && state.float_partials => "w4a8-fp32-group-dots",
            Some(_) if self.grouped_q4() => "w4a8-packed-int32-group-dots",
            Some(state) if state.pure && state.float_partials => {
                "w8a8-exact-fp32-partials-int32-outer-no-fp16"
            }
            Some(state) if state.float_partials => "hybrid-w8a8-exact-fp32-partials-and-fp16",
            Some(state) if state.pure => "w8a8-all-decoder-projections-no-fp16-fallback",
            Some(state) if state.fp16_mlp => {
                "hybrid-w8a8-projections-fp16-mlp-gate-up-and-small-fallback"
            }
            Some(_) => "w8a8-unsigned-int32-row-scales-with-fp16-small-fallback",
            None => projection_precision,
        };
        serde_json::json!({"name":info.name,"vendor":info.vendor,"device":info.device,
            "device_type":format!("{:?}",info.device_type),"driver":info.driver,
            "backend":format!("{:?}",info.backend),"max_storage_buffer_bytes":self.device.device.limits().max_storage_buffer_binding_size,
            "weight_residency":"one-layer","decoder":"webgpu","readout":"wasm-f32",
            "projection_precision":projection_precision,
            "attention_query_block":self.attention_block,"reuse_scratch":self.reuse_scratch,
            "gemm_tile":self.gemm_tile,"packed_projection_inputs":self.packed_inputs,
            "fused_mlp_tile":self.mlp_tile,"small_row_tile":self.small_tile,
            "tiled_attention":self.tiled_attention,"cooperative_norm":self.cooperative_norm,"batched_compute_passes":self.device.batch_passes.get(),
            "cached_bindings":true,"fused_mlp_quantization":self.fused_mlp&&self.pipelines.contains_key("q4_mlp"),
            "delta_scan":self.delta_scan,"delta_scan_selection":self.delta_scan_selection,
            "gemm_selection":self.gemm_selection,"mlp_selection":self.mlp_selection,
            "timing":"wall-clock, synchronized per layer; not GPU timestamps"})
    }
    fn gemm_candidate_tile(name: &str) -> [u32; 2] {
        if name == "int8_gemm_unrolled_32x64" { [32, 64] } else { [64, 64] }
    }
    pub fn discard_scratch(&self) {
        self.packed_cache.borrow_mut().clear();
        for b in self.scratch.borrow_mut().drain(..) {
            b.destroy();
        }
    }
    pub fn reset(&mut self) {
        self.device.discard_pending();
        self.discard_scratch();
        for b in self.scratch_pool.borrow_mut().drain(..) {
            b.destroy();
        }
        if let Some(b) = self.hidden.take() {
            b.destroy();
        }
        if let Some(b) = self.rope.take() {
            b.destroy();
        }
        self.rows = 0;
    }
    pub fn upload(&self, bytes: &[u8], shape: Vec<usize>) -> Result<Tensor> {
        #[cfg(not(feature = "kernel-checks"))]
        let buffer = self.device.bytes(bytes, "BF16 weight")?;
        #[cfg(feature = "kernel-checks")]
        let buffer = if let Some(buffer) = self.reused_weight_buffer(bytes.len()) {
            self.device.queue.write_buffer(&buffer, 0, bytes);
            buffer
        } else {
            self.device.bytes(bytes, "BF16 weight")?
        };
        #[cfg(feature = "kernel-checks")]
        let buffer = if self.fp16_projections && shape.len() == 2 && !self.grouped_q4() {
            let converted = self.checkpoint_fp16(&buffer, bytes.len())?;
            if self.reuse_weight_buffers {
                self.retain_weight_buffer(buffer);
            }
            converted
        } else {
            buffer
        };
        Ok(Tensor { shape, buffer })
    }
    pub fn begin(&mut self, embeddings: &[f32]) -> Result<()> {
        ensure!(
            !embeddings.is_empty() && embeddings.len() % D == 0,
            "invalid GPU embeddings"
        );
        self.reset();
        self.rows = embeddings.len() / D;
        ensure!(
            self.rows <= 2048,
            "Pplx WebGPU currently supports at most 2048 tokens per sequence"
        );
        self.hidden = Some(self.device.buffer(embeddings, "hidden")?);
        self.rope = Some(
            self.device
                .buffer(&rope_table(&(0..self.rows).collect::<Vec<_>>()), "RoPE")?,
        );
        Ok(())
    }
    fn empty(&self, count: usize) -> Result<Buffer> {
        if self.reuse_scratch {
            let mut pool = self.scratch_pool.borrow_mut();
            if let Some(index) = pool.iter().position(|b| b.size() == count as u64 * 4) {
                let buffer = pool.swap_remove(index);
                self.scratch.borrow_mut().push(buffer.clone());
                return Ok(buffer);
            }
        }
        let buffer = self.device.empty(count, "Pplx activation")?;
        self.allocation_count.set(self.allocation_count.get() + 1);
        self.scratch.borrow_mut().push(buffer.clone());
        Ok(buffer)
    }
    fn release_scratch(&self, buffer: Buffer) {
        if self.reuse_scratch {
            let mut pool = self.scratch_pool.borrow_mut();
            if pool.iter().map(Buffer::size).sum::<u64>() + buffer.size() <= 512 * 1024 * 1024 {
                pool.push(buffer);
                return;
            }
        }
        buffer.destroy();
    }
    fn run(
        &self,
        e: &mut CommandEncoder,
        name: &str,
        buffers: [&Buffer; 4],
        values: &[usize],
        count: usize,
    ) -> Result<()> {
        let mut buffers = buffers;
        let packed;
        if self.packed_inputs && (name == "gemm" || name == "mlp_fused") {
            let input = buffers[0];
            let existing = self
                .packed_cache
                .borrow()
                .iter()
                .find(|(source, _)| source == input)
                .map(|(_, b)| b.clone());
            packed = if let Some(buffer) = existing {
                buffer
            } else {
                let count = values[0]
                    .checked_mul(values[2])
                    .context("packed input size overflow")?;
                let buffer = self.empty(count.div_ceil(2))?;
                self.run(
                    e,
                    "pack_input",
                    [input, &self.dummy, &self.dummy, &buffer],
                    &[count],
                    count,
                )?;
                self.packed_cache
                    .borrow_mut()
                    .push((input.clone(), buffer.clone()));
                buffer
            };
            buffers[0] = &packed;
        }
        let small = self
            .small_tile
            .filter(|_| values.first().is_some_and(|&rows| rows <= 8));
        let name = match (name, small) {
            ("gemm", Some(_)) => "gemm_small",
            ("mlp_fused", Some(_)) => "mlp_small",
            _ => name,
        };
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        let mut params = [0u32; 16];
        for (p, &v) in params.iter_mut().zip(values) {
            *p = u32::try_from(v)?;
        }
        let int8_tile = [64, 64];
        #[cfg(feature = "kernel-checks")]
        let int8_tile = self.int8.as_ref().map_or(int8_tile, |s| s.tile);
        let groups = if (name == "int8_input" || name == "int8_silu_input") && self.grouped_q4() {
            [params[0], params[2].div_ceil(256), 1]
        } else if name == "int8_input" || name == "int8_weight" || name == "q4_weight" {
            [params[0], 1, 1]
        } else if matches!(
            name,
            "int8_gemm" | "int8_raw" | "int8_gemm_unrolled_64x64" | "int8_gemm_unrolled_32x64" | "int8_gemm_r4x8"
        ) {
            let tile = match name {
                "int8_gemm" => self.int8_gemm_tile.unwrap_or(int8_tile),
                "int8_raw" => int8_tile,
                candidate => Self::gemm_candidate_tile(candidate),
            };
            [params[1].div_ceil(tile[1]), params[0].div_ceil(tile[0]), 1]
        } else if name == "int8_gemm_narrow" {
            [params[1].div_ceil(32), params[0].div_ceil(32), 1]
        } else if name == "int8_mlp" || name == "q4_mlp" {
            [params[1].div_ceil(64), params[0].div_ceil(64), 1]
        } else if name == "delta_scan_chained" {
            // 32 state columns per workgroup: 32 heads * 128 columns / 32.
            [128, 1, 1]
        } else if name == "delta_scan_staged" {
            [64, 1, 1]
        } else if (self.cooperative_norm && name == "norm") || name == "norm_cooperative" {
            [params[0], 1, 1]
        } else if self.tiled_attention && name == "scores" {
            // 32 keys x 32 (query, head) rows per workgroup; one KV head per z.
            [params[0].div_ceil(32), (params[1] * 4).div_ceil(32), 4]
        } else if self.tiled_attention && name == "values" {
            [8, (params[1] * 4).div_ceil(64), 4]
        } else if name == "gemm_small" || name == "mlp_small" {
            let tile = small.unwrap();
            [params[1].div_ceil(tile[1]), params[0].div_ceil(tile[0]), 1]
        } else if name == "mlp_fused" {
            let tile = self.mlp_tile.context("fused MLP not configured")?;
            [params[1].div_ceil(tile[1]), params[0].div_ceil(tile[0]), 1]
        } else if name == "gemm" {
            [
                params[1].div_ceil(self.gemm_tile[1]),
                params[0].div_ceil(self.gemm_tile[0]),
                1,
            ]
        } else {
            let groups = u32::try_from(count)?.div_ceil(64);
            let width = groups.min(
                self.device
                    .device
                    .limits()
                    .max_compute_workgroups_per_dimension,
            );
            params[15] = width * 64;
            [width, groups.div_ceil(width), 1]
        };
        self.device
            .dispatch(e, name, &self.pipelines[name], &buffers, &params, groups)
    }
    fn linear(
        &self,
        e: &mut CommandEncoder,
        x: &Buffer,
        w: &Buffer,
        n: usize,
        input: usize,
        output: usize,
    ) -> Result<Buffer> {
        let y = self.empty(n * output)?;
        #[cfg(feature = "kernel-checks")]
        if self.grouped_q4() {
            let input_buffer = self.int8_input(e, x, n, input)?;
            // The 64x64 tile leaves the 32-wide recurrent projections few workgroups.
            let name = if output <= 32 && self.pipelines.contains_key("int8_gemm_narrow") { "int8_gemm_narrow" } else { "int8_gemm" };
            self.run(e, name, [&input_buffer, w, &self.dummy, &y], &[n, output, input], 0)?;
            return Ok(y);
        }
        #[cfg(feature = "kernel-checks")]
        if let Some(weight) = self
            .int8
            .as_ref()
            .filter(|state| n > 8 || state.pure)
            .and_then(|state| state.weight(w))
        {
            let weight = if self.q4 == Some(q4::Mode::Buffered) {
                self.expand_q4(e, &weight, output, input)?
            } else {
                weight
            };
            let input_buffer = self.int8_input(e, x, n, input)?;
            self.run(
                e,
                "int8_gemm",
                [&input_buffer, &weight, &self.dummy, &y],
                &[n, output, input],
                0,
            )?;
            return Ok(y);
        }
        #[cfg(feature = "kernel-checks")]
        ensure!(
            !self.int8.as_ref().is_some_and(|s| s.pure),
            "pure INT8 projection is missing quantized weights"
        );
        self.run(e, "gemm", [x, w, &self.dummy, &y], &[n, output, input], 0)?;
        Ok(y)
    }
    fn norm(&self, e: &mut CommandEncoder, x: &Buffer, w: &Buffer, n: usize) -> Result<Buffer> {
        let y = self.empty(n * D)?;
        self.run(e, "norm", [x, w, &self.dummy, &y], &[n, D], n)?;
        Ok(y)
    }
    fn point(
        &self,
        e: &mut CommandEncoder,
        x: &Buffer,
        w: &Buffer,
        n: usize,
        op: usize,
    ) -> Result<Buffer> {
        let y = self.empty(n)?;
        self.run(e, "point", [x, w, &self.dummy, &y], &[n, op], n)?;
        Ok(y)
    }
    fn gather(&self, e: &mut CommandEncoder, x: &Buffer, markers: &[usize]) -> Result<Buffer> {
        let ids: Vec<u32> = markers.iter().map(|&i| i as u32).collect();
        let ids = self.device.bytes(bytemuck::cast_slice(&ids), "markers")?;
        let y = self.empty(markers.len() * D)?;
        self.run(
            e,
            "point",
            [x, &self.dummy, &ids, &y],
            &[markers.len() * D, 2, D],
            markers.len() * D,
        )?;
        Ok(y)
    }
    #[cfg(feature = "kernel-checks")]
    pub async fn layer(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        index: usize,
        markers: Option<&[usize]>,
    ) -> Result<()> {
        self.layer_submitted(tensors, index, markers, || {}).await
    }
    pub async fn layer_submitted(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        index: usize,
        markers: Option<&[usize]>,
        submitted: impl FnOnce(),
    ) -> Result<()> {
        let (encoder, output, rows) = self.encode_layer(tensors, index, markers)?;
        self.device.read_submitted(encoder, &output, 1, submitted).await?;
        for buffer in self.advance_layer(output, rows) { self.release_scratch(buffer); }
        Ok(())
    }
    pub fn supports_queued(&self) -> bool { self.device.profile.is_none() }
    pub async fn wait_idle(&self) -> Result<()> { self.device.idle_completion().wait().await }
    pub fn queue_layer(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        index: usize,
        markers: Option<&[usize]>,
    ) -> Result<QueuedLayer> {
        ensure!(self.supports_queued(), "queued execution does not support per-layer profiling");
        let (encoder, output, rows) = self.encode_layer(tensors, index, markers)?;
        Ok(self.queue_encoded(encoder, output, rows))
    }
    fn queue_encoded(&mut self, encoder: CommandEncoder, output: Buffer, rows: usize) -> QueuedLayer {
        let completion = self.device.submit_deferred(encoder);
        let scratch = self.advance_layer(output, rows);
        QueuedLayer {
            completion, scratch,
            #[cfg(feature = "kernel-checks")]
            quantized: self.int8.as_mut().map(|s| std::mem::take(&mut s.weights)).unwrap_or_default(),
            #[cfg(feature = "kernel-checks")]
            q4_expansion: self.q4_expansion.take(),
        }
    }
    pub async fn collect_layer(&mut self, layer: QueuedLayer) -> Result<()> {
        let result = layer.completion.wait().await;
        for buffer in layer.scratch { self.release_scratch(buffer); }
        #[cfg(feature = "kernel-checks")]
        if let Some(state) = &self.int8 { state.recycle(layer.quantized); }
        #[cfg(feature = "kernel-checks")]
        if let Some(buffer) = layer.q4_expansion {
            if self.q4_expansion.is_none() { self.q4_expansion = Some(buffer); }
            else { buffer.destroy(); }
        }
        result
    }
    fn encode_layer(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        index: usize,
        markers: Option<&[usize]>,
    ) -> Result<(CommandEncoder, Buffer, usize)> {
        ensure!(
            index < 32 && (markers.is_none() || index == 31),
            "invalid GPU layer/pruning"
        );
        self.packed_cache.borrow_mut().clear();
        let n = self.rows;
        if let Some(m) = markers {
            ensure!(
                !m.is_empty() && m.iter().all(|&i| i < n),
                "invalid marker positions"
            );
        }
        let get = |name: &str, shape: &[usize]| -> Result<&Buffer> {
            let name = format!("language_model.layers.{index}.{name}");
            let t = tensors
                .get(&name)
                .with_context(|| format!("missing {name}"))?;
            ensure!(t.shape == shape, "unexpected tensor shape: {name}");
            Ok(&t.buffer)
        };
        let mut e = self.device.encoder();
        let hidden = self.hidden.as_ref().context("no GPU sequence")?;
        let norm = self.norm(&mut e, hidden, get("input_layernorm.weight", &[D])?, n)?;
        let nq = markers.map_or(n, |m| m.len());
        let projected;
        if index % 4 == 3 {
            let k = self.linear(
                &mut e,
                &norm,
                get("self_attn.k_proj.weight", &[1024, D])?,
                n,
                D,
                1024,
            )?;
            let v = self.linear(
                &mut e,
                &norm,
                get("self_attn.v_proj.weight", &[1024, D])?,
                n,
                D,
                1024,
            )?;
            let selected;
            let qin = if let Some(m) = markers {
                selected = self.gather(&mut e, &norm, m)?;
                &selected
            } else {
                &norm
            };
            let q = self.linear(
                &mut e,
                qin,
                get("self_attn.q_proj.weight", &[8192, D])?,
                nq,
                D,
                8192,
            )?;
            let query = self.empty(nq * 4096)?;
            let key = self.empty(n * 1024)?;
            let selected_rope;
            let qrope = if let Some(m) = markers {
                selected_rope = self.device.buffer(&rope_table(m), "marker RoPE")?;
                &selected_rope
            } else {
                self.rope.as_ref().unwrap()
            };
            self.run(
                &mut e,
                "attention_norm",
                [&q, get("self_attn.q_norm.weight", &[256])?, qrope, &query],
                &[nq, 16, 512],
                nq * 16,
            )?;
            self.run(
                &mut e,
                "attention_norm",
                [
                    &k,
                    get("self_attn.k_norm.weight", &[256])?,
                    self.rope.as_ref().unwrap(),
                    &key,
                ],
                &[n, 4, 256],
                n * 4,
            )?;
            let context = self.empty(nq * 4096)?;
            let block = if self.attention_block == 16 {
                16
            } else {
                self.attention_block.min(nq)
            };
            let scores = self.empty(n * block * 16)?;
            let probabilities = self.empty(n * block * 16)?;
            for start in (0..nq).step_by(block) {
                let rows = (nq - start).min(block);
                self.run(
                    &mut e,
                    "scores",
                    [&query, &key, &self.dummy, &scores],
                    &[n, rows, start],
                    rows * 16 * n,
                )?;
                self.run(
                    &mut e,
                    "softmax",
                    [&scores, &self.dummy, &self.dummy, &probabilities],
                    &[rows * 16, n],
                    rows * 16,
                )?;
                self.run(
                    &mut e,
                    "values",
                    [&probabilities, &v, &q, &context],
                    &[n, rows, start],
                    rows * 4096,
                )?;
            }
            projected = self.linear(
                &mut e,
                &context,
                get("self_attn.o_proj.weight", &[D, 4096])?,
                nq,
                4096,
                D,
            )?;
        } else {
            let qkv = self.linear(
                &mut e,
                &norm,
                get("linear_attn.in_proj_qkv.weight", &[8192, D])?,
                n,
                D,
                8192,
            )?;
            let z = self.linear(
                &mut e,
                &norm,
                get("linear_attn.in_proj_z.weight", &[4096, D])?,
                n,
                D,
                4096,
            )?;
            let a = self.linear(
                &mut e,
                &norm,
                get("linear_attn.in_proj_a.weight", &[32, D])?,
                n,
                D,
                32,
            )?;
            let b = self.linear(
                &mut e,
                &norm,
                get("linear_attn.in_proj_b.weight", &[32, D])?,
                n,
                D,
                32,
            )?;
            let conv = self.empty(n * 8192)?;
            let qk = self.empty(n * 8192)?;
            self.run(
                &mut e,
                "conv",
                [
                    &qkv,
                    get("linear_attn.conv1d.weight", &[8192, 1, 4])?,
                    &self.dummy,
                    &conv,
                ],
                &[n],
                n * 8192,
            )?;
            self.run(
                &mut e,
                "delta_qk",
                [&conv, &self.dummy, &self.dummy, &qk],
                &[n],
                n * 64,
            )?;
            // Small BF16 decay parameters remain in the same streamed GPU weight format.
            let gates = self.empty(n * 64)?;
            let params = self.empty(64)?;
            self.run(
                &mut e,
                "delta_constants",
                [
                    &self.dummy,
                    get("linear_attn.A_log", &[32])?,
                    get("linear_attn.dt_bias", &[32])?,
                    &params,
                ],
                &[],
                32,
            )?;
            self.run(
                &mut e,
                "delta_params",
                [&a, &b, &params, &gates],
                &[n],
                n * 32,
            )?;
            let heads = self.empty(n * 4096)?;
            let context = self.empty(n * 4096)?;
            self.run(
                &mut e,
                self.delta_scan,
                [&qk, &gates, &self.dummy, &heads],
                &[n],
                4096,
            )?;
            self.run(
                &mut e,
                "delta_gate",
                [
                    &heads,
                    get("linear_attn.norm.weight", &[128])?,
                    &z,
                    &context,
                ],
                &[n],
                n * 32,
            )?;
            projected = self.linear(
                &mut e,
                &context,
                get("linear_attn.out_proj.weight", &[D, 4096])?,
                n,
                4096,
                D,
            )?;
        }
        let selected;
        let residual = if let Some(m) = markers {
            selected = self.gather(&mut e, hidden, m)?;
            &selected
        } else {
            hidden
        };
        let residual = self.point(&mut e, residual, &projected, nq * D, 0)?;
        let norm = self.norm(
            &mut e,
            &residual,
            get("post_attention_layernorm.weight", &[D])?,
            nq,
        )?;
        #[cfg(feature = "kernel-checks")]
        let int8_mlp = self
            .int8
            .as_ref()
            .filter(|state| (nq > 8 || state.pure) && !state.fp16_mlp);
        #[cfg(not(feature = "kernel-checks"))]
        let int8_mlp: Option<()> = None;
        #[cfg(feature = "kernel-checks")]
        let mut quantized_activated = None;
        #[cfg(not(feature = "kernel-checks"))]
        let quantized_activated: Option<Buffer> = None;
        // Gate, up, SiLU product and down_proj activation quantization in one kernel.
        #[allow(unused_mut)]
        let mut quantized_down: Option<Buffer> = None;
        #[cfg(feature = "kernel-checks")]
        if int8_mlp.is_some() && self.grouped_q4() && self.fused_mlp && self.pipelines.contains_key("q4_mlp") {
            let input = self.int8_input(&mut e, &norm, nq, D)?;
            let output = self.empty(nq * (9216 / 64) * 17)?;
            self.run(
                &mut e,
                "q4_mlp",
                [
                    &input,
                    get("mlp.gate_proj.weight", &[9216, D])?,
                    get("mlp.up_proj.weight", &[9216, D])?,
                    &output,
                ],
                &[nq, 9216, D],
                0,
            )?;
            quantized_down = Some(output);
        }
        #[cfg(feature = "kernel-checks")]
        if let Some(state) = int8_mlp.filter(|state| state.fused && quantized_down.is_none()) {
            let gate = state
                .weight(get("mlp.gate_proj.weight", &[9216, D])?)
                .context("missing INT8 gate")?;
            let up = state
                .weight(get("mlp.up_proj.weight", &[9216, D])?)
                .context("missing INT8 up")?;
            let input = self.int8_input(&mut e, &norm, nq, D)?;
            let output = self.empty(nq * 9216)?;
            self.run(
                &mut e,
                "int8_mlp",
                [&input, &gate, &up, &output],
                &[nq, 9216, D],
                0,
            )?;
            quantized_activated = Some(output);
        }
        // Unfused grouped path: the SiLU product goes straight into down_proj's quantized input.
        #[cfg(feature = "kernel-checks")]
        if int8_mlp.is_some()
            && self.grouped_q4()
            && quantized_down.is_none()
            && quantized_activated.is_none()
            && self.pipelines.contains_key("int8_silu_input")
        {
            let gate = self.linear(&mut e, &norm, get("mlp.gate_proj.weight", &[9216, D])?, nq, D, 9216)?;
            let up = self.linear(&mut e, &norm, get("mlp.up_proj.weight", &[9216, D])?, nq, D, 9216)?;
            let output = self.empty(nq * (9216 / 64) * 17)?;
            self.run(&mut e, "int8_silu_input", [&gate, &up, &self.dummy, &output], &[nq, 0, 9216], 0)?;
            quantized_down = Some(output);
        }
        let activated = if let Some(output) = quantized_activated {
            output
        } else if let Some(input) = &quantized_down {
            input.clone()
        } else if self.mlp_tile.is_some() && int8_mlp.is_none() {
            let output = self.empty(nq * 9216)?;
            self.run(
                &mut e,
                "mlp_fused",
                [
                    &norm,
                    get("mlp.gate_proj.weight", &[9216, D])?,
                    get("mlp.up_proj.weight", &[9216, D])?,
                    &output,
                ],
                &[nq, 9216, D],
                0,
            )?;
            output
        } else {
            let gate = self.linear(
                &mut e,
                &norm,
                get("mlp.gate_proj.weight", &[9216, D])?,
                nq,
                D,
                9216,
            )?;
            let up = self.linear(
                &mut e,
                &norm,
                get("mlp.up_proj.weight", &[9216, D])?,
                nq,
                D,
                9216,
            )?;
            self.point(&mut e, &gate, &up, nq * 9216, 1)?
        };
        let down = if quantized_down.is_some() {
            let y = self.empty(nq * D)?;
            self.run(
                &mut e,
                "int8_gemm",
                [&activated, get("mlp.down_proj.weight", &[D, 9216])?, &self.dummy, &y],
                &[nq, D, 9216],
                0,
            )?;
            y
        } else {
            self.linear(
                &mut e,
                &activated,
                get("mlp.down_proj.weight", &[D, 9216])?,
                nq,
                9216,
                D,
            )?
        };
        let output = self.point(&mut e, &residual, &down, nq * D, 0)?;
        Ok((e, output, nq))
    }
    fn advance_layer(&mut self, output: Buffer, rows: usize) -> Vec<Buffer> {
        self.packed_cache.borrow_mut().clear();
        let mut retired: Vec<_> = self.scratch.borrow_mut().drain(..).filter(|b| b != &output).collect();
        if let Some(old) = self.hidden.take() {
            retired.push(old);
        }
        self.hidden = Some(output);
        self.rows = rows;
        retired
    }
    pub async fn read_hidden(&self) -> Result<Vec<f32>> {
        self.device
            .read(
                self.device.encoder(),
                self.hidden.as_ref().context("no GPU result")?,
                self.rows * D,
            )
            .await
    }
}
fn rope_table(positions: &[usize]) -> Vec<f32> {
    let freq: Vec<_> = (0..32)
        .map(|i| 1. / 10_000_000f32.powf((2 * i) as f32 / 64.))
        .collect();
    let mut values = Vec::with_capacity(positions.len() * 64);
    for &t in positions {
        values.extend(freq.iter().map(|&f| (t as f32 * f).cos()));
        values.extend(freq.iter().map(|&f| (t as f32 * f).sin()));
    }
    values
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.device.device.destroy();
    }
}
