use super::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn now_ms() -> f64;
}
#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1e3
}

impl Gpu {
    pub(super) async fn install_pipeline(
        &mut self,
        name: &'static str,
        source: String,
    ) -> Result<()> {
        let device = &self.device.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let bindings = self.pipelines["gemm"].get_bind_group_layout(0);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &[Some(&bindings)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = scope.pop().await {
            anyhow::bail!("{name}: {error}");
        }
        self.pipelines.insert(name, pipeline);
        Ok(())
    }

    pub(crate) async fn checkpoint(mode: &str, profile: bool) -> Result<Self> {
        let reuse = mode.ends_with("-reuse");
        let mode = mode.strip_suffix("-reuse").unwrap_or(mode);
        let (mode, forced_scan) = if let Some(mode) = mode.strip_suffix("-scan-chained") {
            (mode, Some("delta_scan_chained"))
        } else if let Some(mode) = mode.strip_suffix("-scan-baseline") {
            (mode, Some("delta_scan"))
        } else if let Some(mode) = mode.strip_suffix("-scan-staged") {
            (mode, Some("delta_scan_staged"))
        } else {
            (mode, None)
        };
        let requested_wide = mode == "q4-e4";
        let grouped_float = requested_wide || mode == "q4-e4-baseline";
        let mode = if ["q4-e4-int8", "q4-e4-baseline"].contains(&mode) { "q4-e4" } else { mode };
        ensure!(
            [
                "bf16",
                "fp16",
                "fp16-scan",
                "hybrid",
                "hybrid-scan",
                "hybrid-f32",
                "int8",
                "int8-tuned",
                "w8a8-f32",
                "q4-fused",
                "q4-buffered",
                "q4-e4",
                "q4-e4-signed"
            ]
            .contains(&mode),
            "unknown checkpoint configuration"
        );
        let mut gpu = Self::with_options(mode != "bf16", profile).await?;
        gpu.reuse_weight_buffers = reuse;
        let wide = requested_wide && gpu.device.device.limits().max_compute_workgroup_storage_size >= 16896;
        gpu.q4 = match mode {
            "q4-fused" => Some(q4::Mode::Fused),
            "q4-buffered" => Some(q4::Mode::Buffered),
            "q4-e4" => Some(q4::Mode::Group64),
            "q4-e4-signed" => Some(q4::Mode::Group64Signed),
            _ => None,
        };
        if mode == "bf16" {
            return Ok(gpu);
        }
        gpu.gemm_tile = [64, 64];
        gpu.mlp_tile = Some([64, 64]);
        gpu.attention_block = 128;
        gpu.reuse_scratch = true;
        gpu.tiled_attention = true;
        gpu.install_pipeline(
            "gemm",
            shaders::fp16_padded(shaders::fp16_gemm_tiled(64, 64, false), 64),
        )
        .await?;
        gpu.install_pipeline(
            "mlp_fused",
            shaders::fp16_padded(shaders::fp16_mlp(64, 64), 64),
        )
        .await?;
        gpu.install_pipeline("scores", shaders::source(shaders::SCORES_TILED))
            .await?;
        gpu.install_pipeline("values", shaders::source(shaders::VALUES_TILED))
            .await?;
        gpu.install_pipeline(
            "checkpoint_fp16",
            shaders::source(
                r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;if(i>=p.a.x){return;}
 let bits=bitcast<u32>(x[i]);
 y[i]=pack2x16float(vec2<f32>(bitcast<f32>((bits&65535u)<<16u),bitcast<f32>(bits&4294901760u)));
}"#,
            )
            .replace("y: array<f32>", "y: array<u32>"),
        )
        .await?;
        if !["fp16", "hybrid", "int8"].contains(&mode) {
            gpu.install_pipeline("delta_scan", shaders::source(&shaders::delta_scan_four()))
                .await?;
        }
        if mode.starts_with("fp16") {
            return Ok(gpu);
        }
        let fp16_mlp = mode.starts_with("hybrid");
        let float_partials = grouped_float || mode.ends_with("f32");
        for (name, source) in [
            ("int8_input", if gpu.grouped_q4() { grouped_q4::quantize() } else { int8::quantize(false) }),
            ("int8_weight", int8::quantize(true)),
            (
                "int8_gemm",
                if wide { include_str!("grouped_q4_tuned.wgsl").into() } else if grouped_float { grouped_q4::float_gemm() } else if gpu.grouped_q4() { grouped_q4::gemm(gpu.q4==Some(q4::Mode::Group64)) } else if gpu.q4 == Some(q4::Mode::Fused) {
                    q4::gemm(false)
                } else if float_partials {
                    int8::fp32_partials(false)
                } else {
                    int8::gemm(false, false)
                },
            ),
            (
                "int8_raw",
                if wide { include_str!("grouped_q4_tuned.wgsl").into() } else if grouped_float { grouped_q4::float_gemm() } else if gpu.grouped_q4() { grouped_q4::gemm(gpu.q4==Some(q4::Mode::Group64)) } else if gpu.q4 == Some(q4::Mode::Fused) {
                    q4::gemm(true)
                } else if float_partials {
                    int8::fp32_partials(true)
                } else {
                    int8::gemm(false, true)
                },
            ),
        ] {
            gpu.install_pipeline(name, source).await?;
        }
        gpu.int8 = Some(int8::State {
            weights: Vec::new(),
            pool: RefCell::new(Vec::new()),
            fused: false,
            fp16_mlp,
            pure: !fp16_mlp,
            tile: if wide { [64, 64] } else if grouped_float { [64, 32] } else { [64, 64] },
            float_partials,
        });
        if gpu.q4.is_some() {
            gpu.install_pipeline("q4_weight", q4::quantize()).await?;
            gpu.install_pipeline("q4_expand", q4::expand()).await?;
        }
        if gpu.grouped_q4() {
            gpu.install_pipeline("delta_scan_chained", shaders::source(shaders::DELTA_SCAN_CHAINED))
                .await?;
            gpu.install_pipeline("delta_scan_staged", shaders::source(shaders::DELTA_SCAN_STAGED))
                .await?;
            gpu.select_delta_scan(forced_scan).await?;
            // Both match the tuned projection kernel bit for bit, so they require that kernel.
            if wide {
                gpu.install_pipeline("int8_gemm_narrow", include_str!("grouped_q4_narrow.wgsl").into())
                    .await?;
                if gpu.device.device.limits().max_compute_workgroup_storage_size >= 29696 {
                    gpu.install_pipeline("q4_mlp", include_str!("grouped_q4_mlp.wgsl").into())
                        .await?;
                }
            }
            gpu.select_norm().await?;
            // Per-dispatch timestamps need their own passes.
            gpu.device.batch_passes.set(!profile);
        }
        Ok(gpu)
    }

    // The chained and staged scans keep the baseline's summation order, so they must match bit for bit;
    // the fastest one this device's compiler preserves that for is kept if measurably faster.
    async fn select_delta_scan(&mut self, forced: Option<&'static str>) -> Result<()> {
        const ROWS: usize = 128;
        const CANDIDATES: [&str; 3] = ["delta_scan", "delta_scan_chained", "delta_scan_staged"];
        // Unit-scale q/k/v and decays below one, like delta_qk and delta_params outputs.
        let qk: Vec<f32> = (0..ROWS * 8192)
            .map(|i| ((i * 7919 % 2003) as f32 / 1001.5 - 1.) * 0.09)
            .collect();
        let gates: Vec<f32> = (0..ROWS * 64)
            .map(|i| if i % 2 == 0 { 0.9 + (i / 2 % 7) as f32 * 0.01 } else { 0.5 })
            .collect();
        let qk = self.device.buffer(&qk, "scan selection inputs")?;
        let gates = self.device.buffer(&gates, "scan selection gates")?;
        let outputs = [
            self.device.empty(ROWS * 4096, "scan selection output")?,
            self.device.empty(ROWS * 4096, "scan selection output")?,
            self.device.empty(ROWS * 4096, "scan selection output")?,
        ];
        let mut best = [f64::INFINITY; 3];
        for round in 0..4 {
            for (i, name) in CANDIDATES.iter().enumerate() {
                let mut encoder = self.device.encoder();
                for _ in 0..4 {
                    self.run(&mut encoder, name, [&qk, &gates, &self.dummy, &outputs[i]], &[ROWS], 4096)?;
                }
                let start = now_ms();
                self.device.submit_deferred(encoder).wait().await?;
                // The first round compiles and warms up both kernels.
                if round > 0 {
                    best[i] = best[i].min(now_ms() - start);
                }
            }
        }
        let baseline = self.device.read(self.device.encoder(), &outputs[0], ROWS * 4096).await?;
        let mut difference = 0f32;
        let mut agrees = [true; 3];
        for i in 1..3 {
            let candidate = self.device.read(self.device.encoder(), &outputs[i], ROWS * 4096).await?;
            difference = baseline.iter().zip(&candidate).fold(difference, |a, (x, y)| a.max((x - y).abs()));
            agrees[i] = baseline.iter().zip(&candidate).all(|(x, y)| x.to_bits() == y.to_bits());
        }
        let fastest = (1..3)
            .filter(|&i| agrees[i] && best[i] < best[0] * 0.97)
            .min_by(|&a, &b| best[a].total_cmp(&best[b]));
        self.delta_scan = match forced {
            Some(name) => name,
            None => CANDIDATES[fastest.unwrap_or(0)],
        };
        self.delta_scan_selection = serde_json::json!({"forced":forced.is_some(),"rows":ROWS,
            "baseline_ms":best[0],"chained_ms":best[1],"staged_ms":best[2],"max_abs_difference":difference,
            "agrees":agrees[1]&&agrees[2],"chained_agrees":agrees[1],"staged_agrees":agrees[2]});
        for buffer in [qk, gates].into_iter().chain(outputs) {
            buffer.destroy();
        }
        self.device.discard_pending();
        let forced_agrees = CANDIDATES.iter().zip(agrees).all(|(name, ok)| ok || forced != Some(*name));
        ensure!(forced_agrees, "forced delta scan disagrees with the baseline");
        Ok(())
    }

    // The cooperative norm keeps NORM's summation order; use it only where this device's
    // compiler reproduces NORM bit for bit.
    async fn select_norm(&mut self) -> Result<()> {
        const ROWS: usize = 64;
        self.install_pipeline("norm_cooperative", shaders::source(shaders::NORM_COOPERATIVE)).await?;
        let x: Vec<f32> = (0..ROWS * D).map(|i| ((i * 7919 % 2003) as f32 - 1001.) / 97.).collect();
        let w: Vec<u8> = (0..D)
            .flat_map(|i| ((((i * 13 % 97) as f32 - 48.) / 50.).to_bits() >> 16).to_le_bytes()[..2].to_vec())
            .collect();
        let x = self.device.buffer(&x, "norm selection input")?;
        let w = self.device.bytes(&w, "norm selection weights")?;
        let outputs = [
            self.device.empty(ROWS * D, "norm selection output")?,
            self.device.empty(ROWS * D, "norm selection output")?,
        ];
        let mut encoder = self.device.encoder();
        for (name, y) in ["norm", "norm_cooperative"].into_iter().zip(&outputs) {
            self.run(&mut encoder, name, [&x, &w, &self.dummy, y], &[ROWS, D], ROWS)?;
        }
        let serial = self.device.read_words(encoder, &outputs[0], ROWS * D).await?;
        let cooperative = self.device.read_words(self.device.encoder(), &outputs[1], ROWS * D).await?;
        let cooperative_pipeline = self.pipelines.remove("norm_cooperative").context("cooperative norm")?;
        if serial == cooperative {
            self.pipelines.insert("norm", cooperative_pipeline);
            self.cooperative_norm = true;
        }
        for buffer in [x, w].into_iter().chain(outputs) {
            buffer.destroy();
        }
        self.device.discard_pending();
        Ok(())
    }

    pub(super) fn checkpoint_fp16(&self, input: &Buffer, bytes: usize) -> Result<Buffer> {
        ensure!(
            bytes % 4 == 0,
            "projection byte count must be divisible by four"
        );
        let output = match self.reused_weight_buffer(bytes) {
            Some(buffer) => buffer,
            None => self.device.empty(bytes / 4, "FP16 checkpoint weight")?,
        };
        let mut encoder = self.device.encoder();
        self.run(
            &mut encoder,
            "checkpoint_fp16",
            [input, &self.dummy, &self.dummy, &output],
            &[bytes / 4],
            bytes / 4,
        )?;
        self.device.queue.submit([encoder.finish()]);
        Ok(output)
    }

    pub(super) fn reused_weight_buffer(&self, bytes: usize) -> Option<Buffer> {
        if !self.reuse_weight_buffers {
            return None;
        }
        let mut pool = self.weight_buffers.borrow_mut();
        let index = pool.iter().position(|b| b.size() == bytes as u64)?;
        Some(pool.swap_remove(index))
    }
    pub(crate) fn retain_weight_buffer(&self, buffer: Buffer) {
        if self.reuse_weight_buffers {
            let mut pool = self.weight_buffers.borrow_mut();
            if pool.iter().map(Buffer::size).sum::<u64>() + buffer.size() <= 384 * 1024 * 1024 {
                pool.push(buffer);
            }
        } else {
            buffer.destroy();
        }
    }

    pub(crate) async fn prepare_checkpoint_weights(
        &mut self,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<()> {
        if self.grouped_q4() { return Ok(()); }
        let encoder = self.encode_checkpoint_weights(tensors)?;
        self.device.read(encoder, &self.dummy, 1).await?;
        Ok(())
    }
    pub(crate) fn prepare_queued_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        if self.grouped_q4() { return Ok(()); }
        if self.int8.is_some() {
            let encoder = self.encode_checkpoint_weights(tensors)?;
            self.device.submit_commands(encoder);
        }
        Ok(())
    }
    fn encode_checkpoint_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<CommandEncoder> {
        let mut encoder = self.device.encoder();
        if let Some(state) = self.int8.as_mut() {
            state.clear();
        }
        for (name, tensor) in tensors {
            let Some(state) = self.int8.as_ref() else {
                break;
            };
            if tensor.shape.len() != 2
                || (!state.pure && tensor.shape[0] < 1024)
                || (state.fp16_mlp
                    && (name.ends_with("mlp.gate_proj.weight")
                        || name.ends_with("mlp.up_proj.weight")))
            {
                continue;
            }
            let packed = if self.q4.is_some() {
                self.quantize_q4(&mut encoder, &tensor.buffer, tensor.shape[0], tensor.shape[1])?
            } else {
                self.quantize_int8(
                    &mut encoder,
                    &tensor.buffer,
                    tensor.shape[0],
                    tensor.shape[1],
                    true,
                )?
            };
            self.int8
                .as_mut()
                .unwrap()
                .weights
                .push((tensor.buffer.clone(), packed));
        }
        Ok(encoder)
    }
    pub(crate) fn clear_checkpoint_weights(&mut self) {
        if let Some(state) = self.int8.as_mut() {
            state.clear();
        }
    }
    pub(crate) async fn prepare_resident_weights(
        &mut self,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        if self.grouped_q4() { return self.wait_idle().await; }
        ensure!(
            self.int8.as_ref().is_none_or(|state| state.pure),
            "resident weights currently support FP16, BF16 and all-INT8 projections"
        );
        let mut encoder = self.device.encoder();
        let mut packed = Vec::new();
        if self.int8.is_some() {
            for (name, tensor) in tensors.iter() {
                if tensor.shape.len() == 2 {
                    let buffer = if self.q4.is_some() {
                        self.quantize_q4(&mut encoder, &tensor.buffer, tensor.shape[0], tensor.shape[1])?
                    } else {
                        self.quantize_int8(&mut encoder, &tensor.buffer, tensor.shape[0], tensor.shape[1], true)?
                    };
                    packed.push((name.clone(), buffer));
                }
            }
        }
        self.device.read(encoder, &self.dummy, 1).await?;
        for (name, buffer) in packed {
            let tensor = tensors.get_mut(&name).unwrap();
            tensor.buffer.destroy();
            tensor.buffer = buffer.clone();
            self.int8.as_mut().unwrap().weights.push((buffer.clone(), buffer));
        }
        for buffer in self.weight_buffers.borrow_mut().drain(..) {
            buffer.destroy();
        }
        Ok(())
    }
    pub(crate) async fn upload_resident(&self, bytes: &[u8], shape: Vec<usize>) -> Result<Tensor> {
        let memory = self.device.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let tensor = self.upload(bytes, shape);
        let validation = validation.pop().await;
        let memory = memory.pop().await;
        if let Some(error) = validation.or(memory) {
            if let Ok(tensor) = tensor { tensor.buffer.destroy(); }
            anyhow::bail!("resident weight upload ({} bytes): {error}", bytes.len());
        }
        tensor
    }
    pub(crate) fn start_profile(&self) {
        if let Some(profile) = &self.device.profile {
            profile.begin();
        }
    }
    pub(crate) async fn finish_profile(&self) -> Result<serde_json::Value> {
        if let Some(profile) = &self.device.profile {
            profile.finish(&self.device).await
        } else {
            Ok(serde_json::Value::Null)
        }
    }
}
