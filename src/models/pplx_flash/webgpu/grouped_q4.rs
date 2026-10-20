use super::*;

pub(super) fn quantize() -> String {
    r#"
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> maxima:array<f32,64>;
fn even(v:f32)->i32 {let lo=floor(v);let f=v-lo;return i32(lo)+select(0,1,f>0.5||(f==0.5&&(i32(lo)&1)==1));}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) l:u32){
 let first=g.x*p.a.z+g.y*64u;let out=(g.x*(p.a.z/64u)+g.y)*17u;
 maxima[l]=abs(x[first+l]);workgroupBarrier();
 for(var s=32u;s>0u;s/=2u){if(l<s){maxima[l]=max(maxima[l],maxima[l+s]);}workgroupBarrier();}
 let scale=max(maxima[0]/127.0,1.1754943508222875e-38);let inverse=1.0/scale;
 if(l==0u){y[out+16u]=bitcast<u32>(scale);}
 if(l<16u){var word=0u;for(var j=0u;j<4u;j++){
  let q=clamp(even(x[first+l*4u+j]*inverse),-127,127);word|=(u32(q)&255u)<<(8u*j);
 }y[out+l]=word;}
}"#.into()
}

pub(super) fn gemm(unsigned: bool) -> String {
    let mut s = r#"requires packed_4x8_integer_dot_product;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<u32,1024>;var<workgroup> bt:array<u32,1024>;
var<workgroup> sa:array<f32,64>;var<workgroup> sb:array<f32,64>;
fn unpack(word:u32,half:u32)->u32 {
 let v=(word>>(half*16u))&65535u;
 let bytes=(v&15u)|((v&240u)<<4u)|((v&3840u)<<8u)|((v&61440u)<<12u);
 return (bytes+2021161080u)^2155905152u;
}
@compute @workgroup_size(16,8)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
 let m=p.a.x;let n=p.a.y;let groups=p.a.z/64u;let lane=l.y*16u+l.x;
 let row=g.y*64u+l.y*8u;let col=g.x*64u+l.x*4u;
"#
    .to_string();
    if unsigned {
        s=s.replace("return (bytes+2021161080u)^2155905152u;","return bytes;")
            .replace("var<workgroup> sa:","var<workgroup> suma:array<i32,64>;var<workgroup> sumb:array<i32,64>;\nvar<workgroup> sa:");
    }
    for r in 0..8 {
        s += &format!("var total{r}=vec4<f32>(0.0);\n");
    }
    s += r#"for(var group=0u;group<groups;group++){
 for(var i=lane;i<1024u;i+=128u){let r=i/16u;let c=i%16u;at[i]=0u;bt[c*64u+r]=0u;
  if(g.y*64u+r<m){at[i]=x[((g.y*64u+r)*groups+group)*17u+c];}
  if(g.x*64u+r<n){bt[c*64u+r]=unpack(w[((g.x*64u+r)*groups+group)*9u+c/2u],c%2u);}
 }
 if(lane<64u){sa[lane]=0.0;sb[lane]=0.0;
  if(g.y*64u+lane<m){sa[lane]=bitcast<f32>(x[((g.y*64u+lane)*groups+group)*17u+16u]);}
  if(g.x*64u+lane<n){sb[lane]=bitcast<f32>(w[((g.x*64u+lane)*groups+group)*9u+8u]);}
 }
 workgroupBarrier();
"#;
    if unsigned {
        s = s
            .replace(
                "at[i]=x[((g.y*64u+r)*groups+group)*17u+c];",
                "at[i]=x[((g.y*64u+r)*groups+group)*17u+c]^2155905152u;",
            )
            .replace(
                "if(lane<64u){sa[lane]=0.0;sb[lane]=0.0;",
                r#"if(lane<64u){sa[lane]=0.0;sb[lane]=0.0;suma[lane]=0;sumb[lane]=0;
  if(g.y*64u+lane<m){var sum=0;for(var j=0u;j<16u;j++){
   sum+=i32(dot4U8Packed(x[((g.y*64u+lane)*groups+group)*17u+j]^2155905152u,16843009u));
  }suma[lane]=sum-8192;}
  if(g.x*64u+lane<n){var sum=0;for(var j=0u;j<8u;j++){
   let word=w[((g.x*64u+lane)*groups+group)*9u+j];
   sum+=i32(dot4U8Packed((word&252645135u)+((word>>4u)&252645135u),16843009u));
  }sumb[lane]=sum-512;}
"#,
            );
    }
    for r in 0..8 {
        s += &format!("var dot{r}=vec4<i32>(0);\n");
    }
    s += "for(var j=0u;j<16u;j++){\n";
    for c in 0..4 {
        s += &format!("let b{c}=bt[j*64u+l.x*4u+{c}u];\n");
    }
    for r in 0..8 {
        let dots = (0..4)
            .map(|c| {
                if unsigned {
                    format!("i32(dot4U8Packed(a{r},b{c}))")
                } else {
                    format!("dot4I8Packed(a{r},b{c})")
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        s += &format!("let a{r}=at[(l.y*8u+{r}u)*16u+j];dot{r}+=vec4<i32>({dots});\n");
    }
    s += "}\nlet scales=vec4<f32>(sb[l.x*4u],sb[l.x*4u+1u],sb[l.x*4u+2u],sb[l.x*4u+3u]);\n";
    for r in 0..8 {
        if unsigned {
            s += &format!(
                "dot{r}-=128*vec4<i32>(sumb[l.x*4u],sumb[l.x*4u+1u],sumb[l.x*4u+2u],sumb[l.x*4u+3u])+vec4<i32>(8*suma[l.y*8u+{r}u]+64*128*8);\n"
            );
        }
        s += &format!("total{r}=fma(vec4<f32>(dot{r}),sa[l.y*8u+{r}u]*scales,total{r});\n");
    }
    s += "workgroupBarrier();}\n";
    for r in 0..8 {
        for c in 0..4 {
            s +=
                &format!("if(row+{r}u<m&&col+{c}u<n){{y[(row+{r}u)*n+col+{c}u]=total{r}[{c}];}}\n");
        }
    }
    s + "}"
}

pub(super) fn float_gemm() -> String {
    let mut s = r#"enable f16;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<vec4<f16>,1024>;var<workgroup> bt:array<vec4<f16>,512>;
var<workgroup> sa:array<f32,64>;var<workgroup> sb:array<f32,32>;
fn unpack_a(v:u32)->vec4<f16>{
 return vec4<f16>(bitcast<vec4<i32>>(vec4<u32>(v)<<vec4<u32>(24u,16u,8u,0u))>>vec4<u32>(24u));
}
fn unpack_b(word:u32,half:u32)->vec4<f16>{
 let v=word>>(half*16u);
 return vec4<f16>(vec4<i32>(i32(v&15u),i32((v>>4u)&15u),i32((v>>8u)&15u),i32((v>>12u)&15u))-vec4<i32>(8));
}
@compute @workgroup_size(16,8)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
 let m=p.a.x;let n=p.a.y;let groups=p.a.z/64u;let lane=l.y*16u+l.x;
 let row=g.y*64u+l.y*8u;let col=g.x*32u+l.x*2u;
"#.to_string();
    for r in 0..8 { s += &format!("var total{r}=vec2<f32>(0.0);\n"); }
    s += r#"for(var group=0u;group<groups;group++){
 for(var i=lane;i<1024u;i+=128u){let r=i/16u;let c=i%16u;at[i]=vec4<f16>(0.0h);
  if(g.y*64u+r<m){at[i]=unpack_a(x[((g.y*64u+r)*groups+group)*17u+c]);}
 }
 for(var i=lane;i<512u;i+=128u){let r=i/16u;let c=i%16u;bt[c*32u+r]=vec4<f16>(0.0h);
  if(g.x*32u+r<n){bt[c*32u+r]=unpack_b(w[((g.x*32u+r)*groups+group)*9u+c/2u],c%2u);}
 }
 if(lane<64u){sa[lane]=0.0;if(g.y*64u+lane<m){sa[lane]=bitcast<f32>(x[((g.y*64u+lane)*groups+group)*17u+16u]);}}
 if(lane<32u){sb[lane]=0.0;if(g.x*32u+lane<n){sb[lane]=bitcast<f32>(w[((g.x*32u+lane)*groups+group)*9u+8u]);}}
 workgroupBarrier();
"#;
    for r in 0..8 { s += &format!("var dot{r}=vec2<f32>(0.0);\n"); }
    s += "for(var j=0u;j<16u;j++){let b0=vec4<f32>(bt[j*32u+l.x*2u]);let b1=vec4<f32>(bt[j*32u+l.x*2u+1u]);\n";
    for r in 0..8 { s += &format!("let a{r}=vec4<f32>(at[(l.y*8u+{r}u)*16u+j]);dot{r}+=vec2<f32>(dot(a{r},b0),dot(a{r},b1));\n"); }
    s += "}\nlet scales=vec2<f32>(sb[l.x*2u],sb[l.x*2u+1u]);\n";
    for r in 0..8 { s += &format!("total{r}=fma(dot{r},sa[l.y*8u+{r}u]*scales,total{r});\n"); }
    s += "workgroupBarrier();}\n";
    for r in 0..8 {
        for c in 0..2 { s += &format!("if(row+{r}u<m&&col+{c}u<n){{y[(row+{r}u)*n+col+{c}u]=total{r}[{c}];}}\n"); }
    }
    s + "}"
}

pub(crate) async fn check() -> Result<serde_json::Value> {
    let mut checks = Vec::new();
    let mut baseline = Vec::<Vec<f32>>::new();
    for (variant, mode) in ["q4-e4-signed", "q4-e4-int8", "q4-e4-baseline", "q4-e4"].into_iter().enumerate() {
        let gpu = Gpu::checkpoint(mode, false).await?;
        for (case, (m, n, k)) in [(1, 7, 64), (5, 67, 128), (65, 33, 256), (7, 11, 192), (3, 5, 128)]
            .into_iter()
            .enumerate()
        {
            let a: Vec<f32> = (0..m * k)
                .map(|i| {
                    if case == 4 {
                        if (i / k) % 2 == 0 {127.0} else {-127.0}
                    } else if case == 3 {
                        ((i * 73 % 1021) as i32 - 510) as f32 / 113.0
                    } else {
                        ((i * 17 % 255) as i32 - 127) as f32 * if i % k < 64 { 0.125 } else { 4. }
                    }
                })
                .collect();
            let b: Vec<f32> = (0..n * k)
                .map(|i| {
                    if case == 4 {
                        if ((i / k) + (i % k) / 64) % 3 == 0 {-7.0} else {7.0}
                    } else if case == 3 {
                        ((i * 47 % 257) as i32 - 128) as f32 / 193.0
                    } else {
                        ((i * 11 % 15) as i32 - 7) as f32 * if i % k < 64 { 0.25 } else { 8. }
                    }
                })
                .collect();
            let raw: Vec<u8> = b
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let packed = crate::pplx_flash::packed_q4::pack(&raw, n, k)?;
            let ab = gpu.device.buffer(&a, "grouped activations")?;
            let wb = gpu.device.bytes(&packed, "grouped Q4 weights")?;
            let mut encoder = gpu.device.encoder();
            let out = gpu.linear(&mut encoder, &ab, &wb, m, k, n)?;
            let actual = gpu.device.read(encoder, &out, m * n).await?;
            if variant == 0 {
                baseline.push(actual.clone());
            } else {
                let maximum = actual
                    .iter()
                    .zip(&baseline[case])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                ensure!(
                    actual == baseline[case],
                    "grouped Q4 {mode} differs from signed kernel: case {case}, max error {maximum}"
                );
            }
            let mut maximum = 0f32;
            for row in 0..m {
                let expected = crate::pplx_flash::packed_q4::project(
                    &packed,
                    k,
                    &a[row * k..(row + 1) * k],
                    n,
                )?;
                for col in 0..n {
                    let error = (actual[row * n + col] - expected[col]).abs();
                    maximum = maximum.max(error);
                    ensure!(
                        error <= 1e-4 + expected[col].abs() * 2e-6,
                        "grouped Q4 dot mismatch"
                    );
                }
            }
            checks
            .push(serde_json::json!({"mode":mode,"rows":m,"outputs":n,"input":k,"max_absolute_error":maximum}));
            ab.destroy();
            wb.destroy();
            gpu.discard_scratch();
        }
    }
    checks.push(fused_mlp_check().await?);
    checks.push(norm_check().await?);
    Ok(serde_json::json!({"passed":true,"scheme":"q4-w4a8-e4-r4-g64","checks":checks}))
}

// The fused gate/up/SiLU/quantize kernel must emit exactly the down_proj input of the
// unfused path: two projections, the SiLU product, then activation quantization.
async fn fused_mlp_check() -> Result<serde_json::Value> {
    let gpu = Gpu::checkpoint("q4-e4", false).await?;
    ensure!(gpu.pipelines.contains_key("q4_mlp"), "fused MLP kernel unavailable on this device");
    let (m, n, k) = (67, 192, 256);
    let weight = |seed: usize| -> Result<Buffer> {
        let raw: Vec<u8> = (0..n * k)
            .flat_map(|i| {
                let v = ((i * seed % 251) as f32 - 125.) / 400.;
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        gpu.device.bytes(&crate::pplx_flash::packed_q4::pack(&raw, n, k)?, "MLP check weights")
    };
    let (gate, up) = (weight(37)?, weight(53)?);
    let x: Vec<f32> = (0..m * k).map(|i| ((i * 29 % 509) as f32 - 254.) / 97.).collect();
    let x = gpu.device.buffer(&x, "MLP check input")?;
    let mut e = gpu.device.encoder();
    let input = gpu.int8_input(&mut e, &x, m, k)?;
    let g = gpu.empty(m * n)?;
    let u = gpu.empty(m * n)?;
    gpu.run(&mut e, "int8_gemm", [&input, &gate, &gpu.dummy, &g], &[m, n, k], 0)?;
    gpu.run(&mut e, "int8_gemm", [&input, &up, &gpu.dummy, &u], &[m, n, k], 0)?;
    let activated = gpu.point(&mut e, &g, &u, m * n, 1)?;
    let expected = gpu.quantize_int8(&mut e, &activated, m, n, false)?;
    let fused = gpu.empty(m * (n / 64) * 17)?;
    gpu.run(&mut e, "q4_mlp", [&input, &gate, &up, &fused], &[m, n, k], 0)?;
    let words = m * (n / 64) * 17;
    let expected = gpu.device.read_words(e, &expected, words).await?;
    let fused = gpu.device.read_words(gpu.device.encoder(), &fused, words).await?;
    let mismatches = expected.iter().zip(&fused).filter(|(a, b)| a != b).count();
    ensure!(mismatches == 0, "fused MLP differs from the unfused path in {mismatches} words");
    for buffer in [gate, up, x] {
        buffer.destroy();
    }
    gpu.discard_scratch();
    Ok(serde_json::json!({"mode":"q4-e4-fused-mlp","rows":m,"outputs":n,"input":k,"exact_words":words}))
}

// The cooperative norm keeps NORM's summation order, so it must match NORM bit for bit,
// including rows that end inside a tile.
async fn norm_check() -> Result<serde_json::Value> {
    let mut gpu = Gpu::checkpoint("q4-e4", false).await?;
    ensure!(gpu.cooperative_norm, "cooperative norm rejected by its startup comparison");
    gpu.install_pipeline("norm_serial", shaders::source(shaders::NORM)).await?;
    let mut cases = Vec::new();
    for (rows, d) in [(37, 2560), (5, 1028)] {
        let x: Vec<f32> = (0..rows * d).map(|i| ((i * 31 % 1009) as f32 - 504.) / 71.).collect();
        let w: Vec<u8> = (0..d)
            .flat_map(|i| ((((i * 13 % 97) as f32 - 48.) / 50.).to_bits() >> 16).to_le_bytes()[..2].to_vec())
            .collect();
        let x = gpu.device.buffer(&x, "norm check input")?;
        let w = gpu.device.bytes(&w, "norm check weights")?;
        let (cooperative, serial) = (gpu.empty(rows * d)?, gpu.empty(rows * d)?);
        let mut e = gpu.device.encoder();
        gpu.run(&mut e, "norm", [&x, &w, &gpu.dummy, &cooperative], &[rows, d], rows)?;
        gpu.run(&mut e, "norm_serial", [&x, &w, &gpu.dummy, &serial], &[rows, d], rows)?;
        let serial = gpu.device.read_words(e, &serial, rows * d).await?;
        let cooperative = gpu.device.read_words(gpu.device.encoder(), &cooperative, rows * d).await?;
        ensure!(serial == cooperative, "cooperative norm differs from NORM: {rows}x{d}");
        cases.push([rows, d]);
        x.destroy();
        w.destroy();
        gpu.discard_scratch();
    }
    Ok(serde_json::json!({"mode":"q4-e4-cooperative-norm","cases":cases,"exact":true}))
}
