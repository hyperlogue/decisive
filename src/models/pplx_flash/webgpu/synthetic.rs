use super::*;
use serde_json::json;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn now() -> f64;
}
fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

const GENERATE: &str = r#"
fn random_weight(i:u32)->f32 {
    var z=i^p.a.y;
    z=(z^(z>>16u))*2146121005u;
    z=(z^(z>>15u))*2221713035u;
    z=z^(z>>16u);
    return (f32(z>>8u)/8388608.0-1.0)*bitcast<f32>(p.a.z)+bitcast<f32>(p.a.w);
}
fn rounded(i:u32)->u32 {
    let b=bitcast<u32>(random_weight(i));
    return (b+32767u+((b>>16u)&1u))>>16u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    if(i>=p.a.x/2u) {return;}
    let a=rounded(2u*i);let b=rounded(2u*i+1u);
    if(p.b.x==1u) {y[i]=pack2x16float(vec2<f32>(bitcast<f32>(a<<16u),bitcast<f32>(b<<16u)));}
    else {y[i]=a|(b<<16u);}
}
"#;

fn seed_for(name: &str, seed: u32) -> u32 {
    name.bytes().fold(seed ^ 2166136261, |s, b| {
        (s ^ b as u32).wrapping_mul(16777619)
    })
}
fn random(i: u32, seed: u32) -> f32 {
    let mut z = i ^ seed;
    z = (z ^ (z >> 16)).wrapping_mul(2146121005);
    z = (z ^ (z >> 15)).wrapping_mul(2221713035);
    z ^= z >> 16;
    half::bf16::from_f32((z >> 8) as f32 / 8388608. - 1.).to_f32()
}
fn tensors(index: usize) -> Vec<(&'static str, Vec<usize>, f32, f32)> {
    let mut out = vec![
        ("input_layernorm.weight", vec![D], 0.01, 0.),
        ("post_attention_layernorm.weight", vec![D], 0.01, 0.),
    ];
    let mut linear = |name, output, input, residual: bool| {
        out.push((
            name,
            vec![output, input],
            (3. / input as f32).sqrt() * if residual { 0.125 } else { 1. },
            0.,
        ));
    };
    if index % 4 == 3 {
        linear("self_attn.q_proj.weight", 8192, D, false);
        linear("self_attn.k_proj.weight", 1024, D, false);
        linear("self_attn.v_proj.weight", 1024, D, false);
        linear("self_attn.o_proj.weight", D, 4096, true);
    } else {
        linear("linear_attn.in_proj_qkv.weight", 8192, D, false);
        linear("linear_attn.in_proj_z.weight", 4096, D, false);
        linear("linear_attn.in_proj_a.weight", 32, D, false);
        linear("linear_attn.in_proj_b.weight", 32, D, false);
        linear("linear_attn.out_proj.weight", D, 4096, true);
    }
    linear("mlp.gate_proj.weight", 9216, D, false);
    linear("mlp.up_proj.weight", 9216, D, false);
    linear("mlp.down_proj.weight", D, 9216, true);
    if index % 4 == 3 {
        out.extend([
            ("self_attn.q_norm.weight", vec![256], 0.01, 0.),
            ("self_attn.k_norm.weight", vec![256], 0.01, 0.),
        ]);
    } else {
        out.extend([
            ("linear_attn.conv1d.weight", vec![8192, 1, 4], 0.5, 0.),
            ("linear_attn.A_log", vec![32], 1., 1.),
            ("linear_attn.dt_bias", vec![32], 0.1, -2.),
            ("linear_attn.norm.weight", vec![128], 0.01, 1.),
        ]);
    }
    out
}

#[wasm_bindgen(js_name=PplxFlashSyntheticGpu)]
pub struct SyntheticGpu {
    gpu: Gpu,
    tensors: HashMap<String, Tensor>,
    weight_pool: Vec<Buffer>,
    layer: usize,
    markers: Vec<usize>,
    readout: Vec<f32>,
    seed: u32,
    input_change: Option<(usize, usize, u32)>,
    dependency_audit: bool,
}
#[wasm_bindgen(js_class=PplxFlashSyntheticGpu)]
impl SyntheticGpu {
    pub async fn create(fp16: bool, seed: u32) -> Result<SyntheticGpu, JsValue> {
        Self::create_internal(fp16, seed, false).await
    }
    #[wasm_bindgen(js_name=createProfiled)]
    pub async fn create_profiled(seed: u32) -> Result<SyntheticGpu, JsValue> {
        Self::create_internal(true, seed, true).await
    }
}
impl SyntheticGpu {
    async fn install_pipeline(
        &mut self,
        name: &'static str,
        source: String,
    ) -> Result<(), JsValue> {
        self.gpu.install_pipeline(name, source).await.map_err(error)
    }
    async fn create_internal(
        fp16: bool,
        seed: u32,
        profile: bool,
    ) -> Result<SyntheticGpu, JsValue> {
        let mut gpu = Gpu::with_options(fp16, profile).await.map_err(error)?;
        let module = gpu
            .device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("synthetic BF16 / FP16 weights"),
                source: wgpu::ShaderSource::Wgsl(
                    shaders::source(GENERATE)
                        .replace("y: array<f32>", "y: array<u32>")
                        .into(),
                ),
            });
        let layout = gpu.pipelines["gemm"].get_bind_group_layout(0);
        let layout = gpu
            .device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("synthetic weight layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline =
            gpu.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("synthetic weight generation"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
        gpu.pipelines.insert("synthetic_weights", pipeline);
        let readout = (0..255 * D)
            .map(|i| {
                half::bf16::from_f32(
                    random(i as u32, seed_for("readout", seed)) / (D as f32).sqrt(),
                )
                .to_f32()
            })
            .collect();
        Ok(Self {
            gpu,
            tensors: HashMap::new(),
            weight_pool: Vec::new(),
            layer: 32,
            markers: Vec::new(),
            readout,
            seed,
            input_change: None,
            dependency_audit: false,
        })
    }
}
#[wasm_bindgen(js_class=PplxFlashSyntheticGpu)]
impl SyntheticGpu {
    pub fn adapter(&self) -> String {
        self.gpu.info().to_string()
    }
    #[wasm_bindgen(js_name=configureInputChange)]
    pub fn configure_input_change(
        &mut self,
        first: usize,
        last: usize,
        seed: u32,
    ) -> Result<(), JsValue> {
        if self.layer != 32 || first >= last || last > 2048 {
            return Err(error(
                "input change requires an idle model and a nonempty range",
            ));
        }
        self.input_change = if seed == 0 {
            None
        } else {
            Some((first, last, seed))
        };
        self.dependency_audit = true;
        Ok(())
    }
    #[wasm_bindgen(js_name=configureGemm)]
    pub async fn configure_gemm(
        &mut self,
        tile_m: usize,
        tile_n: usize,
        unroll: bool,
    ) -> Result<(), JsValue> {
        if self.layer != 32
            || !self.gpu.fp16_projections
            || ![32, 64].contains(&tile_m)
            || ![32, 64].contains(&tile_n)
        {
            return Err(error(
                "configure an idle FP16 model with 32/64 projection tiles",
            ));
        }
        self.install_pipeline("gemm", shaders::fp16_gemm_tiled(tile_m, tile_n, unroll))
            .await?;
        self.gpu.gemm_tile = [tile_m as u32, tile_n as u32];
        Ok(())
    }
    #[wasm_bindgen(js_name=configureMlp)]
    pub async fn configure_mlp(&mut self, tile: usize) -> Result<(), JsValue> {
        if self.layer != 32 || !self.gpu.fp16_projections || ![32, 64].contains(&tile) {
            return Err(error(
                "configure an idle FP16 model with a 32/64 fused MLP tile",
            ));
        }
        self.install_pipeline("mlp_fused", shaders::fp16_mlp(tile, tile))
            .await?;
        self.gpu.mlp_tile = Some([tile as u32, tile as u32]);
        Ok(())
    }
    #[wasm_bindgen(js_name=configurePadding)]
    pub async fn configure_padding(&mut self) -> Result<(), JsValue> {
        if self.layer != 32 || !self.gpu.fp16_projections || self.gpu.packed_inputs {
            return Err(error(
                "configure padding on an idle FP16 model before packing inputs",
            ));
        }
        let [m, n] = self.gpu.gemm_tile.map(|v| v as usize);
        self.install_pipeline(
            "gemm",
            shaders::fp16_padded(shaders::fp16_gemm_tiled(m, n, false), n),
        )
        .await?;
        if let Some([m, n]) = self.gpu.mlp_tile.map(|tile| tile.map(|v| v as usize)) {
            self.install_pipeline(
                "mlp_fused",
                shaders::fp16_padded(shaders::fp16_mlp(m, n), n),
            )
            .await?;
        }
        Ok(())
    }
    #[wasm_bindgen(js_name=configureInt8)]
    pub async fn configure_int8(&mut self, fused: bool, fp16_mlp: bool) -> Result<(), JsValue> {
        if self.layer != 32
            || !self.gpu.fp16_projections
            || self.gpu.packed_inputs
            || self.gpu.int8.is_some()
            || (fused && fp16_mlp)
        {
            return Err(error(
                "configure INT8 on an idle FP16 model with unpacked activations",
            ));
        }
        for (name, source) in [
            ("int8_input", int8::quantize(false)),
            ("int8_weight", int8::quantize(true)),
            ("int8_gemm", int8::gemm(false, false)),
            ("int8_raw", int8::gemm(false, true)),
            ("int8_mlp", int8::gemm(true, false)),
        ] {
            self.install_pipeline(name, source).await?;
        }
        self.gpu.int8 = Some(int8::State {
            weights: Vec::new(),
            pool: RefCell::new(Vec::new()),
            fused,
            fp16_mlp,
            pure: false,
            tile: [64, 64],
            float_partials: false,
        });
        Ok(())
    }
    #[wasm_bindgen(js_name=configurePureInt8)]
    pub fn configure_pure_int8(&mut self) -> Result<(), JsValue> {
        if self.layer != 32 || self.gpu.int8.is_none() || self.gpu.int8.as_ref().unwrap().fp16_mlp {
            return Err(error(
                "configure pure INT8 on an idle INT8 model without FP16 MLP",
            ));
        }
        self.gpu.int8.as_mut().unwrap().pure = true;
        Ok(())
    }
    #[wasm_bindgen(js_name=configureDeltaScan)]
    pub async fn configure_delta_scan(&mut self, width: usize) -> Result<(), JsValue> {
        if self.layer != 32 || width != 4 {
            return Err(error("configure an idle model with a four-wide delta scan"));
        }
        self.install_pipeline("delta_scan", shaders::source(&shaders::delta_scan_four()))
            .await
    }
    #[wasm_bindgen(js_name=configureInt8FloatPartials)]
    pub async fn configure_int8_float_partials(&mut self, rows: usize) -> Result<(), JsValue> {
        if self.layer != 32 || self.gpu.int8.is_none() || rows != 8 {
            return Err(error("configure an idle INT8 model with 4/8 register rows"));
        }
        self.install_pipeline("int8_gemm", int8::fp32_partials(false))
            .await?;
        self.install_pipeline("int8_raw", int8::fp32_partials(true))
            .await?;
        let state = self.gpu.int8.as_mut().unwrap();
        state.tile = [64, 64];
        state.float_partials = true;
        Ok(())
    }
    #[wasm_bindgen(js_name=checkInt8)]
    pub async fn check_int8(&self) -> Result<String, JsValue> {
        if self.gpu.int8.is_none() {
            return Ok(json!({"skipped":true}).to_string());
        }
        int8::check(&self.gpu)
            .await
            .map(|v| v.to_string())
            .map_err(error)
    }
    #[wasm_bindgen(js_name=configureGemmK64)]
    pub async fn configure_gemm_k64(&mut self, tile: usize) -> Result<(), JsValue> {
        if self.layer != 32 || !self.gpu.fp16_projections || ![32, 64].contains(&tile) {
            return Err(error(
                "configure an idle FP16 model with a 32/64 projection tile",
            ));
        }
        self.install_pipeline("gemm", shaders::fp16_gemm_k64(tile))
            .await?;
        self.gpu.gemm_tile = [tile as u32, tile as u32];
        Ok(())
    }
    #[wasm_bindgen(js_name=configureGemmSubtile)]
    pub async fn configure_gemm_subtile(
        &mut self,
        tile: usize,
        load_k: usize,
    ) -> Result<(), JsValue> {
        if self.layer != 32
            || !self.gpu.fp16_projections
            || ![32, 64].contains(&tile)
            || ![8, 16].contains(&load_k)
        {
            return Err(error(
                "configure an idle FP16 model with a 32/64 tile and 8/16 load chunk",
            ));
        }
        self.install_pipeline("gemm", shaders::fp16_gemm_subtile(tile, load_k))
            .await?;
        self.gpu.gemm_tile = [tile as u32, tile as u32];
        Ok(())
    }
    #[wasm_bindgen(js_name=configureSmallRows)]
    pub async fn configure_small_rows(&mut self, tile: usize) -> Result<(), JsValue> {
        if self.layer != 32
            || !self.gpu.fp16_projections
            || self.gpu.packed_inputs
            || ![4, 8].contains(&tile)
        {
            return Err(error(
                "configure an idle FP16 model with 4/8-row tiles and unpacked inputs",
            ));
        }
        self.install_pipeline("gemm_small", shaders::fp16_gemm_tiled(tile, 64, false))
            .await?;
        if self.gpu.mlp_tile.is_some() {
            self.install_pipeline("mlp_small", shaders::fp16_mlp(tile, 64))
                .await?;
        }
        self.gpu.small_tile = Some([tile as u32, 64]);
        Ok(())
    }
    #[wasm_bindgen(js_name=configureNorm)]
    pub async fn configure_norm(&mut self) -> Result<(), JsValue> {
        if self.layer != 32 {
            return Err(error("configure an idle model"));
        }
        self.install_pipeline("norm", shaders::source(shaders::NORM_COOPERATIVE))
            .await?;
        self.gpu.cooperative_norm = true;
        Ok(())
    }
    #[wasm_bindgen(js_name=configureBatchedPasses)]
    pub fn configure_batched_passes(&mut self) -> Result<(), JsValue> {
        if self.layer != 32 || self.gpu.device.profile.is_some() {
            return Err(error("batch an idle, unprofiled model"));
        }
        self.gpu.device.batch_passes.set(true);
        Ok(())
    }
    #[wasm_bindgen(js_name=configurePackedInputs)]
    pub async fn configure_packed_inputs(&mut self) -> Result<(), JsValue> {
        if self.layer != 32 || !self.gpu.fp16_projections || self.gpu.small_tile.is_some() {
            return Err(error("configure an idle FP16 model"));
        }
        let [m, n] = self.gpu.gemm_tile;
        self.install_pipeline("pack_input", shaders::fp16_pack_input())
            .await?;
        self.install_pipeline(
            "gemm",
            shaders::fp16_gemm_tiled(m as usize, n as usize, false)
                .replace("x: array<f32>", "x: array<f16>"),
        )
        .await?;
        if let Some([m, n]) = self.gpu.mlp_tile {
            self.install_pipeline(
                "mlp_fused",
                shaders::fp16_mlp(m as usize, n as usize).replace("x: array<f32>", "x: array<f16>"),
            )
            .await?;
        }
        self.gpu.packed_inputs = true;
        Ok(())
    }
    #[wasm_bindgen(js_name=configureAttention)]
    pub async fn configure_attention(&mut self) -> Result<(), JsValue> {
        if self.layer != 32 || !self.gpu.fp16_projections {
            return Err(error("configure an idle FP16 model"));
        }
        self.install_pipeline("scores", shaders::source(shaders::SCORES_TILED))
            .await?;
        self.install_pipeline("values", shaders::source(shaders::VALUES_TILED))
            .await?;
        self.gpu.tiled_attention = true;
        Ok(())
    }
    #[wasm_bindgen(js_name=checkAttention)]
    pub async fn check_attention(&mut self) -> Result<String, JsValue> {
        if !self.gpu.tiled_attention {
            return Ok(json!({"skipped":true}).to_string());
        }
        self.install_pipeline("scores_reference", shaders::source(shaders::SCORES))
            .await?;
        self.install_pipeline("values_reference", shaders::source(shaders::VALUES))
            .await?;
        let mut checks = Vec::new();
        for (rows, nk, start) in [(3, 19, 2), (2, 17, 0), (18, 33, 1)] {
            let d = &self.gpu.device;
            let q = d
                .buffer(
                    &(0..(rows + start) * 4096)
                        .map(|i| random(i as u32, 9))
                        .collect::<Vec<_>>(),
                    "attention query",
                )
                .map_err(error)?;
            let k = d
                .buffer(
                    &(0..nk * 1024)
                        .map(|i| random(i as u32, 10))
                        .collect::<Vec<_>>(),
                    "attention key",
                )
                .map_err(error)?;
            let v = d
                .buffer(
                    &(0..nk * 1024)
                        .map(|i| random(i as u32, 11))
                        .collect::<Vec<_>>(),
                    "attention value",
                )
                .map_err(error)?;
            let z = d
                .buffer(
                    &(0..(rows + start) * 8192)
                        .map(|i| random(i as u32, 12))
                        .collect::<Vec<_>>(),
                    "attention gate",
                )
                .map_err(error)?;
            let mut outputs = Vec::new();
            for reference in [true, false] {
                let scores = self.gpu.empty(rows * 16 * nk).map_err(error)?;
                let probs = self.gpu.empty(rows * 16 * nk).map_err(error)?;
                let out = self.gpu.empty((rows + start) * 4096).map_err(error)?;
                let mut e = d.encoder();
                self.gpu
                    .run(
                        &mut e,
                        if reference {
                            "scores_reference"
                        } else {
                            "scores"
                        },
                        [&q, &k, &self.gpu.dummy, &scores],
                        &[nk, rows, start],
                        rows * 16 * nk,
                    )
                    .map_err(error)?;
                self.gpu
                    .run(
                        &mut e,
                        "softmax",
                        [&scores, &self.gpu.dummy, &self.gpu.dummy, &probs],
                        &[rows * 16, nk],
                        rows * 16,
                    )
                    .map_err(error)?;
                self.gpu
                    .run(
                        &mut e,
                        if reference {
                            "values_reference"
                        } else {
                            "values"
                        },
                        [&probs, &v, &z, &out],
                        &[nk, rows, start],
                        rows * 4096,
                    )
                    .map_err(error)?;
                outputs.push(
                    d.read(e, &out, (rows + start) * 4096)
                        .await
                        .map_err(error)?,
                );
            }
            let max_error = outputs[0][start * 4096..]
                .iter()
                .zip(&outputs[1][start * 4096..])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            q.destroy();
            k.destroy();
            v.destroy();
            z.destroy();
            self.gpu.discard_scratch();
            if max_error != 0. {
                return Err(error(format!(
                    "tiled attention changed output by {max_error}"
                )));
            }
            checks.push(json!({"rows":rows,"keys":nk,"start":start,"values":rows*4096,"max_absolute_error":max_error}));
        }
        Ok(json!({"passed":true,"checks":checks}).to_string())
    }
    #[wasm_bindgen(js_name=checkMlp)]
    pub async fn check_mlp(&self) -> Result<String, JsValue> {
        if self.gpu.int8.as_ref().is_some_and(|s| s.pure) {
            return Ok(
                json!({"skipped":true,"reason":"Pure INT8 MLP validated by checkInt8"}).to_string(),
            );
        }
        if self.gpu.mlp_tile.is_none() {
            return Ok(json!({"skipped":true}).to_string());
        }
        let (m, k, n) = (3, 66, 34);
        let x: Vec<_> = (0..m * k).map(|i| random(i as u32, 7)).collect();
        let w: Vec<_> = (0..n * k)
            .map(|i| half::f16::from_f32(random(i as u32, 11) * 0.125).to_bits())
            .collect();
        let u: Vec<_> = (0..n * k)
            .map(|i| half::f16::from_f32(random(i as u32, 19) * 0.125).to_bits())
            .collect();
        let xb = self.gpu.device.buffer(&x, "MLP input").map_err(error)?;
        let wb = self
            .gpu
            .device
            .bytes(bytemuck::cast_slice(&w), "MLP gate")
            .map_err(error)?;
        let ub = self
            .gpu
            .device
            .bytes(bytemuck::cast_slice(&u), "MLP up")
            .map_err(error)?;
        let mut encoder = self.gpu.device.encoder();
        let gate = self
            .gpu
            .linear(&mut encoder, &xb, &wb, m, k, n)
            .map_err(error)?;
        let up = self
            .gpu
            .linear(&mut encoder, &xb, &ub, m, k, n)
            .map_err(error)?;
        let reference = self
            .gpu
            .point(&mut encoder, &gate, &up, m * n, 1)
            .map_err(error)?;
        let expected = self
            .gpu
            .device
            .read(encoder, &reference, m * n)
            .await
            .map_err(error)?;
        let mut encoder = self.gpu.device.encoder();
        let result = self.gpu.empty(m * n).map_err(error)?;
        self.gpu
            .run(
                &mut encoder,
                "mlp_fused",
                [&xb, &wb, &ub, &result],
                &[m, n, k],
                0,
            )
            .map_err(error)?;
        let actual = self
            .gpu
            .device
            .read(encoder, &result, m * n)
            .await
            .map_err(error)?;
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        xb.destroy();
        wb.destroy();
        ub.destroy();
        self.gpu.discard_scratch();
        if max_error != 0. {
            return Err(error(format!("fused MLP changed output by {max_error}")));
        }
        Ok(json!({"passed":true,"max_absolute_error":max_error,"values":m*n}).to_string())
    }
    pub fn configure(
        &mut self,
        attention_block: usize,
        reuse_scratch: bool,
    ) -> Result<(), JsValue> {
        if self.layer != 32
            || !self.gpu.fp16_projections
            || ![16, 32, 64, 128, 256].contains(&attention_block)
        {
            return Err(error(
                "configure an idle FP16 model with attention block 16..256",
            ));
        }
        self.gpu.reset();
        self.gpu.attention_block = attention_block;
        self.gpu.reuse_scratch = reuse_scratch;
        Ok(())
    }
    #[wasm_bindgen(js_name=checkProjection)]
    pub async fn check_projection(&self) -> Result<String, JsValue> {
        if self.gpu.int8.as_ref().is_some_and(|s| s.pure) {
            return Ok(
                json!({"skipped":true,"reason":"Pure INT8 projections validated by checkInt8"})
                    .to_string(),
            );
        }
        let (m, k, n) = (3, 66, 34);
        let x: Vec<_> = (0..m * k).map(|i| random(i as u32, 7)).collect();
        let w: Vec<_> = (0..n * k).map(|i| random(i as u32, 11) * 0.125).collect();
        let packed: Vec<u16> = w
            .iter()
            .map(|&v| {
                if self.gpu.fp16_projections {
                    half::f16::from_f32(v).to_bits()
                } else {
                    half::bf16::from_f32(v).to_bits()
                }
            })
            .collect();
        let xb = self
            .gpu
            .device
            .buffer(&x, "projection check input")
            .map_err(error)?;
        let wb = self
            .gpu
            .device
            .bytes(bytemuck::cast_slice(&packed), "projection check weights")
            .map_err(error)?;
        let mut encoder = self.gpu.device.encoder();
        let out = self
            .gpu
            .linear(&mut encoder, &xb, &wb, m, k, n)
            .map_err(error)?;
        let actual = self
            .gpu
            .device
            .read(encoder, &out, m * n)
            .await
            .map_err(error)?;
        let mut max_error = 0f32;
        for r in 0..m {
            for c in 0..n {
                let mut expected = 0.;
                for first in (0..k).step_by(32) {
                    let mut partial = 0.;
                    for j in first..(first + 32).min(k) {
                        let product = x[r * k + j] * w[c * k + j];
                        if self.gpu.fp16_projections {
                            partial = half::f16::from_f32(product + partial).to_f32();
                        } else {
                            partial += product;
                        }
                    }
                    expected += partial;
                }
                max_error = max_error.max((actual[r * n + c] - expected).abs());
            }
        }
        xb.destroy();
        wb.destroy();
        self.gpu.discard_scratch();
        if max_error > 1e-4 {
            return Err(error(format!(
                "projection CPU reference error: {max_error}"
            )));
        }
        Ok(json!({"passed":true,"max_absolute_error":max_error,"values":m*n}).to_string())
    }
    pub fn prepare(&mut self, rows: usize) -> Result<String, JsValue> {
        if !(3..=2048).contains(&rows) || !self.tensors.is_empty() {
            return Err(error("expected 3..2048 rows and no pending layer"));
        }
        let start = now();
        let embedding_seed = seed_for("embedding", self.seed);
        if self.input_change.is_some_and(|(_, last, _)| last > rows) {
            return Err(error("input change outside rows"));
        }
        let embedding: Vec<_> = (0..rows)
            .flat_map(|r| {
                let token = (r * 37 + 17) % super::super::backbone::V;
                let seed = self
                    .input_change
                    .filter(|(first, last, _)| r >= *first && r < *last)
                    .map_or(embedding_seed, |(_, _, seed)| {
                        seed_for("changed input", seed)
                    });
                (0..D).map(move |c| random((token * D + c) as u32, seed))
            })
            .collect();
        self.gpu.begin(&embedding).map_err(error)?;
        self.layer = 0;
        self.markers = vec![rows / 3, rows - 1];
        Ok(json!({"embedding_ms":now()-start,"tokens":rows,"markers":self.markers}).to_string())
    }
    #[wasm_bindgen(js_name=prepareLayer)]
    pub async fn prepare_layer(&mut self) -> Result<String, JsValue> {
        if self.layer >= 32 || !self.tensors.is_empty() {
            return Err(error("no layer to prepare, or layer already prepared"));
        }
        let start = now();
        let mut encoder = self.gpu.device.encoder();
        let mut bytes = 0;
        for (name, shape, scale, bias) in tensors(self.layer) {
            let elements = shape.iter().product::<usize>();
            let buffer = if let Some(index) = self
                .weight_pool
                .iter()
                .position(|b| b.size() == elements as u64 * 2)
            {
                self.weight_pool.swap_remove(index)
            } else {
                self.gpu
                    .device
                    .empty(elements / 2, "synthetic weight")
                    .map_err(error)?
            };
            let full_name = format!("language_model.layers.{}.{name}", self.layer);
            self.gpu
                .run(
                    &mut encoder,
                    "synthetic_weights",
                    [&self.gpu.dummy, &self.gpu.dummy, &self.gpu.dummy, &buffer],
                    &[
                        elements,
                        seed_for(&full_name, self.seed) as usize,
                        scale.to_bits() as usize,
                        bias.to_bits() as usize,
                        usize::from(self.gpu.fp16_projections && shape.len() == 2),
                    ],
                    elements / 2,
                )
                .map_err(error)?;
            bytes += elements * 2;
            let fp16_mlp = self.gpu.int8.as_ref().is_some_and(|state| state.fp16_mlp)
                && matches!(name, "mlp.gate_proj.weight" | "mlp.up_proj.weight");
            if self.gpu.int8.is_some()
                && !fp16_mlp
                && shape.len() == 2
                && (shape[0] >= 1024 || self.gpu.int8.as_ref().unwrap().pure)
            {
                let quantized = self
                    .gpu
                    .quantize_int8(&mut encoder, &buffer, shape[0], shape[1], true)
                    .map_err(error)?;
                bytes += quantized.size() as usize;
                self.gpu
                    .int8
                    .as_mut()
                    .unwrap()
                    .weights
                    .push((buffer.clone(), quantized));
            }
            self.tensors.insert(full_name, Tensor { shape, buffer });
        }
        self.gpu
            .device
            .read(encoder, &self.gpu.dummy, 1)
            .await
            .map_err(error)?;
        Ok(
            json!({"generation_ms":now()-start,"weight_bytes":bytes,"layer":self.layer})
                .to_string(),
        )
    }
    #[wasm_bindgen(js_name=runLayer)]
    pub async fn run_layer(&mut self) -> Result<String, JsValue> {
        if self.layer >= 32 || self.tensors.is_empty() {
            return Err(error("layer weights not prepared"));
        }
        let start = now();
        let allocations = self.gpu.allocation_count.get();
        let dispatches = self.gpu.dispatch_count.get();
        if let Some(profile) = &self.gpu.device.profile {
            profile.begin();
        }
        let result = self
            .gpu
            .layer(
                &self.tensors,
                self.layer,
                if self.layer == 31 {
                    Some(&self.markers)
                } else {
                    None
                },
            )
            .await;
        let compute_ms = now() - start;
        if let Some(state) = &mut self.gpu.int8 {
            state.clear();
        }
        for (_, t) in self.tensors.drain() {
            if self.weight_pool.iter().map(Buffer::size).sum::<u64>() + t.buffer.size()
                <= 384 * 1024 * 1024
            {
                self.weight_pool.push(t.buffer);
            } else {
                t.buffer.destroy();
            }
        }
        result.map_err(error)?;
        let profile = if let Some(profile) = &self.gpu.device.profile {
            profile.finish(&self.gpu.device).await.map_err(error)?
        } else {
            serde_json::Value::Null
        };
        let start = now();
        let hidden = self.gpu.read_hidden().await.map_err(error)?;
        let max_abs = hidden.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let rms =
            (hidden.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / hidden.len() as f64).sqrt();
        let dependency = if self.dependency_audit {
            let split = (self.gpu.rows / 2) * D;
            let hash = |values: &[f32]| {
                format!(
                    "{:016x}",
                    values
                        .iter()
                        .fold(14695981039346656037u64, |h, v| (h ^ v.to_bits() as u64)
                            .wrapping_mul(1099511628211))
                )
            };
            json!({"prefix_hash":hash(&hidden[..split]),"suffix_hash":hash(&hidden[split..]),"split_row":self.gpu.rows/2})
        } else {
            serde_json::Value::Null
        };
        let layer = self.layer;
        self.layer += 1;
        Ok(
            json!({"layer":layer,"compute_ms":compute_ms,"audit_ms":now()-start,"profile":profile,"dependency":dependency,
            "scratch_allocations":self.gpu.allocation_count.get()-allocations,
            "dispatches":self.gpu.dispatch_count.get()-dispatches,
            "retained_scratch_bytes":self.gpu.scratch_pool.borrow().iter().map(Buffer::size).sum::<u64>(),
            "hidden_values":hidden.len(),"hidden_max_abs":max_abs,"hidden_rms":rms,"finite":true})
            .to_string(),
        )
    }
    pub async fn finish(&mut self) -> Result<String, JsValue> {
        if self.layer != 32 || self.markers.is_empty() {
            return Err(error("decoder is unfinished"));
        }
        let start = now();
        let mut hidden = self.gpu.read_hidden().await.map_err(error)?;
        let norm: Vec<_> = (0..D)
            .map(|i| {
                half::bf16::from_f32(random(i as u32, seed_for("final_norm", self.seed)) * 0.01)
                    .to_f32()
            })
            .collect();
        super::super::cpu::rms(&mut hidden, &norm, 1e-6, true);
        let mut logits = Vec::new();
        let mut probabilities = Vec::new();
        for x in hidden.chunks_exact(D) {
            let values: Vec<f32> = self
                .readout
                .chunks_exact(D)
                .map(|w| w.iter().zip(x).map(|(w, x)| w * x).sum())
                .collect();
            if !values.iter().all(|v| v.is_finite()) {
                return Err(error("non-finite readout"));
            }
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<_> = values.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = p.iter().sum();
            for v in &mut p {
                *v /= sum;
            }
            probabilities.push(p);
            logits.push(values);
        }
        Ok(json!({"readout_ms":now()-start,"logits":logits,"probabilities":probabilities,"completed_layers":32,
            "markers":self.markers,"finite":true}).to_string())
    }
}
impl Drop for SyntheticGpu {
    fn drop(&mut self) {
        self.gpu.reset();
        for t in self.tensors.values() {
            t.buffer.destroy();
        }
        for buffer in self.weight_pool.drain(..) {
            buffer.destroy();
        }
        self.gpu.device.device.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_decoder_weight_shapes() {
        let mut projection_bytes = 0;
        for i in 0..32 {
            let weights = tensors(i);
            assert_eq!(weights.len(), if i % 4 == 3 { 11 } else { 14 });
            for (_, shape, scale, bias) in weights {
                assert_eq!(shape.iter().product::<usize>() % 2, 0);
                assert!(scale.is_finite() && bias.is_finite());
                if shape.len() == 2 {
                    projection_bytes += shape.iter().product::<usize>() * 2;
                }
            }
        }
        assert_eq!(projection_bytes, 7138181120usize);
    }
}
