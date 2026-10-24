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
        Self::checkpoint_with_tuning(mode, profile, true, None).await
    }

    pub(crate) async fn checkpoint_with_tuning(mode: &str, profile: bool, tune: bool, tuning_budget_ms: Option<u32>) -> Result<Self> {
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
        let requested_wide = mode == "q4-e4" || mode == "q8-e8";
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
                "q4-e4-signed",
                "q8-e8"
            ]
            .contains(&mode),
            "unknown checkpoint configuration"
        );
        let mut gpu = Self::with_options(mode != "bf16", profile).await?;
        gpu.reuse_weight_buffers = reuse;
        let wide = requested_wide && gpu.device.device.limits().max_compute_workgroup_storage_size >= 16896;
        ensure!(wide || mode != "q8-e8", "Q8 weights require 16896 bytes of workgroup storage");
        gpu.q4 = match mode {
            "q4-fused" => Some(q4::Mode::Fused),
            "q4-buffered" => Some(q4::Mode::Buffered),
            "q4-e4" => Some(q4::Mode::Group64),
            "q4-e4-signed" => Some(q4::Mode::Group64Signed),
            "q8-e8" => Some(q4::Mode::Group64Q8),
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
            ("int8_input", if gpu.grouped() { grouped_q4::quantize() } else { int8::quantize(false) }),
            ("int8_weight", int8::quantize(true)),
            (
                "int8_gemm",
                if wide { gpu.grouped_kernel(include_str!("grouped_q4_tuned.wgsl")) } else if grouped_float { grouped_q4::float_gemm() } else if gpu.grouped() { grouped_q4::gemm(gpu.q4==Some(q4::Mode::Group64)) } else if gpu.q4 == Some(q4::Mode::Fused) {
                    q4::gemm(false)
                } else if float_partials {
                    int8::fp32_partials(false)
                } else {
                    int8::gemm(false, false)
                },
            ),
            (
                "int8_raw",
                if wide { gpu.grouped_kernel(include_str!("grouped_q4_tuned.wgsl")) } else if grouped_float { grouped_q4::float_gemm() } else if gpu.grouped() { grouped_q4::gemm(gpu.q4==Some(q4::Mode::Group64)) } else if gpu.q4 == Some(q4::Mode::Fused) {
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
        if gpu.grouped() {
            let budget = crate::inference::TuningBudget::new(tune, tuning_budget_ms, now_ms());
            gpu.install_pipeline("delta_scan_chained", shaders::source(shaders::DELTA_SCAN_CHAINED))
                .await?;
            gpu.install_pipeline("delta_scan_staged", shaders::source(shaders::DELTA_SCAN_STAGED))
                .await?;
            if budget.permits(now_ms()) { gpu.select_delta_scan(forced_scan).await?; }
            // Both match the tuned projection kernel bit for bit, so they require that kernel.
            if wide {
                gpu.install_pipeline("int8_gemm_narrow", gpu.grouped_kernel(include_str!("grouped_q4_narrow.wgsl")))
                    .await?;
                if gpu.device.device.limits().max_compute_workgroup_storage_size >= 29696 {
                    gpu.install_pipeline("q4_mlp", gpu.grouped_kernel(include_str!("grouped_q4_mlp.wgsl")))
                        .await?;
                }
            }
            if budget.permits(now_ms()) { gpu.select_norm().await?; }
            if wide {
                if budget.permits(now_ms()) { gpu.select_gemm().await?; }
                gpu.install_pipeline("int8_silu_input", grouped_q4::quantize_silu()).await?;
                if budget.permits(now_ms()) && gpu.pipelines.contains_key("q4_mlp") {
                    gpu.select_mlp().await?;
                } else { gpu.fused_mlp = false; }
            }
            // Per-dispatch timestamps need their own passes.
            gpu.device.batch_passes.set(!profile);
        }
        Ok(gpu)
    }

    // The chained and staged scans keep the baseline's summation order, so they must match bit for bit;
    // the fastest one this device's compiler preserves that for is kept if measurably faster.
    async fn select_delta_scan(&mut self, forced: Option<&'static str>) -> Result<()> {
        let started = now_ms();
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
            "agrees":agrees[1]&&agrees[2],"chained_agrees":agrees[1],"staged_agrees":agrees[2],
            "elapsed_ms":now_ms() - started});
        for buffer in [qk, gates].into_iter().chain(outputs) {
            buffer.destroy();
        }
        self.device.discard_pending();
        let forced_agrees = CANDIDATES.iter().zip(agrees).all(|(name, ok)| ok || forced != Some(*name));
        ensure!(forced_agrees, "forced delta scan disagrees with the baseline");
        Ok(())
    }

    // Best time per run of each candidate in ms. Safari rounds performance.now() to whole
    // milliseconds, so each timed batch repeats a candidate until it takes about 30 ms; rounds
    // rotate the order so no candidate always runs first while clocks ramp up.
    async fn time_candidates<F>(&self, count: usize, encode: F) -> Result<Vec<f64>>
    where
        F: Fn(usize, &mut wgpu::CommandEncoder) -> Result<()>,
    {
        let batch = |i: usize, reps: usize| -> Result<wgpu::CommandEncoder> {
            let mut encoder = self.device.encoder();
            for _ in 0..reps {
                encode(i, &mut encoder)?;
            }
            Ok(encoder)
        };
        // The first runs compile and warm up every candidate.
        for i in 0..count {
            self.device.submit_deferred(batch(i, 1)?).wait().await?;
        }
        let start = now_ms();
        self.device.submit_deferred(batch(0, 2)?).wait().await?;
        let per_run = ((now_ms() - start) / 2.).max(0.1);
        let reps = ((30. / per_run).ceil() as usize).clamp(2, 64);
        let mut best = vec![f64::INFINITY; count];
        for round in 0..3 {
            for i in (0..count).map(|j| (j + round) % count) {
                let encoder = batch(i, reps)?;
                let start = now_ms();
                self.device.submit_deferred(encoder).wait().await?;
                best[i] = best[i].min((now_ms() - start) / reps as f64);
            }
        }
        Ok(best)
    }

    // Every candidate keeps the tuned kernel's per-output fma order: the k loop fully unrolled on
    // 64x64 or 32x64 tiles, or 8x16 threads with 4x8 outputs each. The fastest one that is bitwise
    // equal to the tuned kernel on this device, and measurably faster, replaces int8_gemm.
    async fn select_gemm(&mut self) -> Result<()> {
        let started = now_ms();
        const M: usize = 256;
        const N: usize = 4096;
        const K: usize = 2560;
        const CANDIDATES: [(&str, &str); 4] = [
            ("tuned", "int8_gemm"),
            ("unrolled-64x64", "int8_gemm_unrolled_64x64"),
            ("unrolled-32x64", "int8_gemm_unrolled_32x64"),
            ("r4x8", "int8_gemm_r4x8"),
        ];
        for (name, source) in [
            (CANDIDATES[1].1, include_str!("grouped_q4_unrolled_64x64.wgsl")),
            (CANDIDATES[2].1, include_str!("grouped_q4_unrolled_32x64.wgsl")),
            (CANDIDATES[3].1, include_str!("grouped_q4_r4x8.wgsl")),
        ] {
            self.install_pipeline(name, self.grouped_kernel(source)).await?;
        }
        let x = self.device.bytes(bytemuck::cast_slice(&packed_test_words(M, K, 17, 0)), "GEMM selection input")?;
        let w = self.device.bytes(bytemuck::cast_slice(&packed_test_words(N, K, self.group_words(), 7)), "GEMM selection weights")?;
        let mut outputs = Vec::new();
        for _ in CANDIDATES {
            outputs.push(self.device.empty(M * N, "GEMM selection output")?);
        }
        let best = self
            .time_candidates(CANDIDATES.len(), |i, encoder| {
                self.run(encoder, CANDIDATES[i].1, [&x, &w, &self.dummy, &outputs[i]], &[M, N, K], 0)
            })
            .await?;
        let tuned = self.device.read_words(self.device.encoder(), &outputs[0], M * N).await?;
        let mut agrees = [true; 4];
        for i in 1..4 {
            agrees[i] = tuned == self.device.read_words(self.device.encoder(), &outputs[i], M * N).await?;
        }
        let fastest = (1..4)
            .filter(|&i| agrees[i] && best[i] < best[0] * 0.97)
            .min_by(|&a, &b| best[a].total_cmp(&best[b]))
            .unwrap_or(0);
        for (i, (_, name)) in CANDIDATES.iter().enumerate().skip(1) {
            let pipeline = self.pipelines.remove(name).context("GEMM candidate")?;
            if i == fastest {
                self.pipelines.insert("int8_gemm", pipeline);
                self.int8_gemm_tile = Some(Self::gemm_candidate_tile(name));
            }
        }
        self.gemm_selection = serde_json::json!({"shape":[M,N,K],"candidates":CANDIDATES.map(|c| c.0),
            "ms_per_run":best,"agrees":agrees,"selected":CANDIDATES[fastest].0,"elapsed_ms":now_ms() - started});
        for buffer in [x, w].into_iter().chain(outputs) {
            buffer.destroy();
        }
        self.device.discard_pending();
        Ok(())
    }

    // The fused gate/up/SiLU/quantize kernel emits exactly the unfused path's down_proj input,
    // so keep it only where it beats two int8_gemm projections and the SiLU-product quantizer.
    async fn select_mlp(&mut self) -> Result<()> {
        let started = now_ms();
        const M: usize = 128;
        const N: usize = 9216;
        const K: usize = D;
        let x = self.device.bytes(bytemuck::cast_slice(&packed_test_words(M, K, 17, 0)), "MLP selection input")?;
        let gate = self.device.bytes(bytemuck::cast_slice(&packed_test_words(N, K, self.group_words(), 7)), "MLP selection gate")?;
        let up = self.device.bytes(bytemuck::cast_slice(&packed_test_words(N, K, self.group_words(), 11)), "MLP selection up")?;
        let g = self.device.empty(M * N, "MLP selection gate output")?;
        let u = self.device.empty(M * N, "MLP selection up output")?;
        let words = M * (N / 64) * 17;
        let outputs = [self.device.empty(words, "MLP selection output")?, self.device.empty(words, "MLP selection output")?];
        let best = self
            .time_candidates(2, |i, encoder| {
                if i == 0 {
                    self.run(encoder, "q4_mlp", [&x, &gate, &up, &outputs[0]], &[M, N, K], 0)
                } else {
                    self.run(encoder, "int8_gemm", [&x, &gate, &self.dummy, &g], &[M, N, K], 0)?;
                    self.run(encoder, "int8_gemm", [&x, &up, &self.dummy, &u], &[M, N, K], 0)?;
                    self.run(encoder, "int8_silu_input", [&g, &u, &self.dummy, &outputs[1]], &[M, 0, N], 0)
                }
            })
            .await?;
        let fused = self.device.read_words(self.device.encoder(), &outputs[0], words).await?;
        let agrees = fused == self.device.read_words(self.device.encoder(), &outputs[1], words).await?;
        ensure!(agrees, "fused MLP differs from the unfused path");
        // Ties keep the fused kernel, the established default.
        let keep_fused = best[1] >= best[0] * 0.97;
        self.fused_mlp = keep_fused;
        self.mlp_selection = serde_json::json!({"shape":[M,N,K],"fused_ms_per_run":best[0],"unfused_ms_per_run":best[1],
            "selected":if keep_fused { "fused" } else { "unfused" },"elapsed_ms":now_ms() - started});
        for buffer in [x, gate, up, g, u].into_iter().chain(outputs) {
            buffer.destroy();
        }
        self.device.discard_pending();
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
        if self.grouped() { return Ok(()); }
        let encoder = self.encode_checkpoint_weights(tensors)?;
        self.device.read(encoder, &self.dummy, 1).await?;
        Ok(())
    }
    pub(crate) fn prepare_queued_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        if self.grouped() { return Ok(()); }
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
        if self.grouped() { return self.wait_idle().await; }
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

// Group-64 packed rows for kernel selection: `stride` words per group whose last word is a small
// positive scale; the packed int8 or q4 codes before it may be any bits.
fn packed_test_words(rows: usize, k: usize, stride: usize, seed: usize) -> Vec<u32> {
    let hash = |i: usize| ((i + seed) as u32).wrapping_mul(2654435761).rotate_left(13).wrapping_mul(2246822519);
    (0..rows * k / 64 * stride)
        .map(|i| if i % stride == stride - 1 { (0.001 + (hash(i) % 1000) as f32 * 1e-5).to_bits() } else { hash(i) })
        .collect()
}
