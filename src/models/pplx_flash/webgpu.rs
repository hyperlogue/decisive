#[cfg(feature = "kernel-checks")]
pub(super) mod checks;
mod device;
mod shaders;
use super::D;
use anyhow::{Context, Result, ensure};
use device::Device;
use std::{cell::RefCell, collections::HashMap};
use wgpu::{Buffer, CommandEncoder, ComputePipeline};

pub(super) struct Tensor {
    pub shape: Vec<usize>,
    pub buffer: Buffer,
}
pub(super) struct Gpu {
    device: Device,
    pipelines: HashMap<&'static str, ComputePipeline>,
    dummy: Buffer,
    hidden: Option<Buffer>,
    rope: Option<Buffer>,
    rows: usize,
    scratch: RefCell<Vec<Buffer>>,
}
impl Gpu {
    pub async fn new() -> Result<Self> {
        let device = Device::new().await?;
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
                    source: wgpu::ShaderSource::Wgsl(shaders::source(body).into()),
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
        })
    }
    pub fn info(&self) -> serde_json::Value {
        let info = &self.device.info;
        serde_json::json!({"name":info.name,"vendor":info.vendor,"device":info.device,
            "device_type":format!("{:?}",info.device_type),"driver":info.driver,
            "backend":format!("{:?}",info.backend),"max_storage_buffer_bytes":self.device.device.limits().max_storage_buffer_binding_size,
            "weight_residency":"one-layer","decoder":"webgpu","readout":"wasm-f32",
            "timing":"wall-clock, synchronized per layer; not GPU timestamps"})
    }
    pub fn discard_scratch(&self) {
        for b in self.scratch.borrow_mut().drain(..) {
            b.destroy();
        }
    }
    pub fn reset(&mut self) {
        self.discard_scratch();
        if let Some(b) = self.hidden.take() {
            b.destroy();
        }
        if let Some(b) = self.rope.take() {
            b.destroy();
        }
        self.rows = 0;
    }
    pub fn upload(&self, bytes: &[u8], shape: Vec<usize>) -> Result<Tensor> {
        Ok(Tensor {
            shape,
            buffer: self.device.bytes(bytes, "BF16 weight")?,
        })
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
        let buffer = self.device.empty(count, "Pplx activation")?;
        self.scratch.borrow_mut().push(buffer.clone());
        Ok(buffer)
    }
    fn run(
        &self,
        e: &mut CommandEncoder,
        name: &str,
        buffers: [&Buffer; 4],
        values: &[usize],
        count: usize,
    ) -> Result<()> {
        let mut params = [0u32; 16];
        for (p, &v) in params.iter_mut().zip(values) {
            *p = u32::try_from(v)?;
        }
        let groups = if name == "gemm" {
            [params[1].div_ceil(32), params[0].div_ceil(32), 1]
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
            .dispatch(e, &self.pipelines[name], &buffers, &params, groups)
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
    pub async fn layer(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        index: usize,
        markers: Option<&[usize]>,
    ) -> Result<()> {
        ensure!(
            index < 32 && (markers.is_none() || index == 31),
            "invalid GPU layer/pruning"
        );
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
            let scores = self.empty(n * 16 * 16)?;
            let probabilities = self.empty(n * 16 * 16)?;
            for start in (0..nq).step_by(16) {
                let rows = (nq - start).min(16);
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
                "delta_scan",
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
        let activated = self.point(&mut e, &gate, &up, nq * 9216, 1)?;
        let down = self.linear(
            &mut e,
            &activated,
            get("mlp.down_proj.weight", &[D, 9216])?,
            nq,
            9216,
            D,
        )?;
        let output = self.point(&mut e, &residual, &down, nq * D, 0)?;
        // A four-byte fence bounds in-flight weights to one layer without reading activations back.
        self.device.read(e, &output, 1).await?;
        for b in self.scratch.borrow_mut().drain(..) {
            if b != output {
                b.destroy();
            }
        }
        if let Some(old) = self.hidden.take() {
            old.destroy();
        }
        self.hidden = Some(output);
        self.rows = nq;
        Ok(())
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
