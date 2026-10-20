const HEADER: &str = r#"
struct Params { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32> }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> aux: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
fn bf(i:u32)->f32 { return bitcast<f32>(((w[i/2u] >> ((i%2u)*16u)) & 65535u)<<16u); }
fn af(i:u32)->f32 { return bitcast<f32>(aux[i]); }
fn wf(i:u32)->f32 { return bitcast<f32>(w[i]); }
fn sigmoid(v:f32)->f32 { return 1.0/(1.0+exp(-v)); }
fn silu(v:f32)->f32 { return v*sigmoid(v); }
"#;
pub(super) fn source(body: &str) -> String {
    format!("{HEADER}\n{}", body.replace("id.x", "(id.x+id.y*p.d.w)"))
}

pub(super) fn fp16_gemm() -> String {
    let header = HEADER
        .replace("w: array<u32>", "w: array<f16>")
        .replace(
            "fn bf(i:u32)->f32 { return bitcast<f32>(((w[i/2u] >> ((i%2u)*16u)) & 65535u)<<16u); }",
            "fn bf(i:u32)->f16 { return w[i]; }",
        )
        .replace(
            "fn wf(i:u32)->f32 { return bitcast<f32>(w[i]); }",
            "fn wf(i:u32)->f32 { return f32(w[i]); }",
        );
    let mut gemm = GEMM
        .replace("array<f32,1024>", "array<f16,1024>")
        .replace("at[j]=x[(g.y*32u+r)*k+first+c];", "at[j]=f16(x[(g.y*32u+r)*k+first+c]);")
        .replace("bt[j]=bf((g.x*32u+c)*k+first+r);", "bt[j]=bf((g.x*32u+c)*k+first+r);")
        .replace("for(var j=0u;j<32u;j++) {", "var t0=vec4<f16>(0.0h);var t1=t0;var t2=t0;var t3=t0;\n        for(var j=0u;j<32u;j++) {")
        .replace("let v=vec4<f32>", "let v=vec4<f16>")
        .replace("        workgroupBarrier();\n    }", "        s0+=vec4<f32>(t0);s1+=vec4<f32>(t1);s2+=vec4<f32>(t2);s3+=vec4<f32>(t3);\n        workgroupBarrier();\n    }");
    for i in 0..4 {
        gemm = gemm
            .replace(
                &format!("s{i}=fma(vec4<f32>"),
                &format!("t{i}=fma(vec4<f16>"),
            )
            .replace(&format!("v,s{i});"), &format!("v,t{i});"));
    }
    format!("enable f16;\n{header}\n{gemm}")
}

#[cfg(feature = "kernel-checks")]
pub(super) fn fp16_gemm_tiled(m: usize, n: usize, unroll: bool) -> String {
    let shader = fp16_gemm();
    let start = shader
        .find("        for(var j=l.y*8u+l.x;j<1024u;j+=64u) {")
        .unwrap();
    let end = start + shader[start..].find("        workgroupBarrier();").unwrap();
    let threads = m * n / 16;
    let loads = format!(
        r#"
        let lane=l.y*{wx}u+l.x;
        for(var i=lane;i<{a}u;i+={threads}u) {{
            let r=i/32u;let c=i%32u;at[i]=0.0h;
            if(g.y*{m}u+r<p.a.x && first+c<k) {{ at[i]=f16(x[(g.y*{m}u+r)*k+first+c]); }}
        }}
        for(var i=lane;i<{b}u;i+={threads}u) {{
            let r=i/32u;let c=i%32u;bt[c*{n}u+r]=0.0h;
            if(g.x*{n}u+r<p.a.y && first+c<k) {{ bt[c*{n}u+r]=w[(g.x*{n}u+r)*k+first+c]; }}
        }}
"#,
        wx = n / 4,
        a = m * 32,
        b = n * 32
    );
    let mut shader = format!("{}{}{}", &shader[..start], loads, &shader[end..]);
    shader = shader
        .replace("at: array<f16,1024>", &format!("at: array<f16,{}>", m * 32))
        .replace("bt: array<f16,1024>", &format!("bt: array<f16,{}>", n * 32))
        .replace(
            "@workgroup_size(8,8)",
            &format!("@workgroup_size({},{})", n / 4, m / 4),
        )
        .replace(
            "let row=g.y*32u+l.y*4u; let col=g.x*32u+l.x*4u;",
            &format!("let row=g.y*{m}u+l.y*4u; let col=g.x*{n}u+l.x*4u;"),
        )
        .replace("let i=j*32u+l.x*4u;", &format!("let i=j*{n}u+l.x*4u;"));
    if unroll {
        let marker = "        for(var j=0u;j<32u;j++) {";
        let start = shader.find(marker).unwrap();
        let end = shader[start..].find("        s0+=").unwrap() + start;
        let body_end = shader[..end].rfind("        }").unwrap();
        let body = &shader[start + marker.len()..body_end];
        let expanded = (0..32)
            .map(|j| format!("{{let j={j}u;{body}}}\n"))
            .collect::<String>();
        shader.replace_range(start..end, &expanded);
    }
    shader
}

#[cfg(feature = "kernel-checks")]
pub(super) fn fp16_mlp(m: usize, n: usize) -> String {
    let mut shader = fp16_gemm_tiled(m, n, false)
        .replace("aux: array<u32>", "aux: array<f16>")
        .replace("return bitcast<f32>(aux[i]);", "return f32(aux[i]);")
        .replace("var<workgroup> bt:", &format!("var<workgroup> ut: array<f16,{}>;\nvar<workgroup> bt:", n*32))
        .replace(&format!("bt[c*{n}u+r]=0.0h;"), &format!("bt[c*{n}u+r]=0.0h;ut[c*{n}u+r]=0.0h;"))
        .replace(&format!("bt[c*{n}u+r]=w[(g.x*{n}u+r)*k+first+c];"), &format!("bt[c*{n}u+r]=w[(g.x*{n}u+r)*k+first+c];ut[c*{n}u+r]=aux[(g.x*{n}u+r)*k+first+c];"))
        .replace("var s3=s0;", "var s3=s0;var u0=s0;var u1=s0;var u2=s0;var u3=s0;")
        .replace("var t3=t0;", "var t3=t0;var v0=t0;var v1=t0;var v2=t0;var v3=t0;")
        .replace("let v=vec4<f16>(bt[i],bt[i+1u],bt[i+2u],bt[i+3u]);", "let v=vec4<f16>(bt[i],bt[i+1u],bt[i+2u],bt[i+3u]);let uv=vec4<f16>(ut[i],ut[i+1u],ut[i+2u],ut[i+3u]);");
    for r in 0..4 {
        let a = if r == 0 {
            "(l.y*4u)".to_string()
        } else {
            format!("(l.y*4u+{r}u)")
        };
        let operation = format!("t{r}=fma(vec4<f16>(at[{a}*32u+j]),v,t{r});");
        shader = shader
            .replace(
                &operation,
                &format!("{operation}v{r}=fma(vec4<f16>(at[{a}*32u+j]),uv,v{r});"),
            )
            .replace(
                &format!("s{r}+=vec4<f32>(t{r});"),
                &format!("s{r}+=vec4<f32>(t{r});u{r}+=vec4<f32>(v{r});"),
            )
            .replace(&format!("=s{r}[c];"), &format!("=silu(s{r}[c])*u{r}[c];"));
    }
    shader
}

#[cfg(feature = "kernel-checks")]
pub(super) fn fp16_padded(shader: String, n: usize) -> String {
    let mut shader = shader;
    for name in ["bt", "ut"] {
        shader = shader
            .replace(
                &format!("{name}: array<f16,{}>", n * 32),
                &format!("{name}: array<f16,{}>", (n + 1) * 32),
            )
            .replace(
                &format!("{name}[c*{n}u+r]"),
                &format!("{name}[c*{}u+r]", n + 1),
            );
    }
    shader.replace(
        &format!("let i=j*{n}u+l.x*4u;"),
        &format!("let i=j*{}u+l.x*4u;", n + 1),
    )
}

#[cfg(feature = "kernel-checks")]
pub(super) fn fp16_gemm_k64(tile: usize) -> String {
    let (m, n) = (tile, tile);
    let mut shader = fp16_gemm_tiled(m, n, false);
    shader = shader
        .replace(
            &format!("at: array<f16,{}>", m * 32),
            &format!("at: array<f16,{}>", m * 64),
        )
        .replace(
            &format!("bt: array<f16,{}>", n * 32),
            &format!("bt: array<f16,{}>", n * 64),
        )
        .replace("first+=32u", "first+=64u")
        .replace(&format!("i<{}u", m * 32), &format!("i<{}u", m * 64))
        .replace(&format!("i<{}u", n * 32), &format!("i<{}u", n * 64))
        .replace("r=i/32u;let c=i%32u", "r=i/64u;let c=i%64u")
        .replace(
            "var t0=vec4<f16>",
            "for(var inner=0u;inner<64u;inner+=32u){var t0=vec4<f16>",
        )
        .replace(
            &format!("let i=j*{n}u+l.x*4u;"),
            &format!("let i=(inner+j)*{n}u+l.x*4u;"),
        )
        .replace("*32u+j]", "*64u+inner+j]")
        .replace("s3+=vec4<f32>(t3);", "s3+=vec4<f32>(t3);}");
    shader
}

#[cfg(feature = "kernel-checks")]
pub(super) fn fp16_gemm_subtile(tile: usize, load_k: usize) -> String {
    assert!([8, 16].contains(&load_k));
    let original = fp16_gemm_tiled(tile, tile, false);
    let header = &original[..original.find("var<workgroup>").unwrap()];
    let threads = tile * tile / 16;
    let size = tile * load_k;
    let wg = tile / 4;
    format!(
        r#"{header}
var<workgroup> at: array<f16,{size}>;
var<workgroup> bt: array<f16,{size}>;
@compute @workgroup_size({wg},{wg})
fn main(@builtin(workgroup_id) g:vec3<u32>, @builtin(local_invocation_id) l:vec3<u32>) {{
    let row=g.y*{tile}u+l.y*4u; let col=g.x*{tile}u+l.x*4u;
    let m=p.a.x; let n=p.a.y; let k=p.a.z;
    let lane=l.y*{wg}u+l.x;
    var s0=vec4<f32>(0.0);var s1=s0;var s2=s0;var s3=s0;
    for(var first=0u;first<k;first+=32u) {{
        var t0=vec4<f16>(0.0h);var t1=t0;var t2=t0;var t3=t0;
        for(var chunk=0u;chunk<32u;chunk+={load_k}u) {{
            for(var i=lane;i<{size}u;i+={threads}u) {{
                let r=i/{load_k}u;let c=i%{load_k}u;at[i]=0.0h;
                if(g.y*{tile}u+r<m && first+chunk+c<k) {{at[i]=f16(x[(g.y*{tile}u+r)*k+first+chunk+c]);}}
                bt[c*{tile}u+r]=0.0h;
                if(g.x*{tile}u+r<n && first+chunk+c<k) {{bt[c*{tile}u+r]=w[(g.x*{tile}u+r)*k+first+chunk+c];}}
            }}
            workgroupBarrier();
            for(var j=0u;j<{load_k}u;j++) {{
                let i=j*{tile}u+l.x*4u;
                let v=vec4<f16>(bt[i],bt[i+1u],bt[i+2u],bt[i+3u]);
                t0=fma(vec4<f16>(at[(l.y*4u)*{load_k}u+j]),v,t0);
                t1=fma(vec4<f16>(at[(l.y*4u+1u)*{load_k}u+j]),v,t1);
                t2=fma(vec4<f16>(at[(l.y*4u+2u)*{load_k}u+j]),v,t2);
                t3=fma(vec4<f16>(at[(l.y*4u+3u)*{load_k}u+j]),v,t3);
            }}
            workgroupBarrier();
        }}
        s0+=vec4<f32>(t0);s1+=vec4<f32>(t1);s2+=vec4<f32>(t2);s3+=vec4<f32>(t3);
    }}
    for(var c=0u;c<4u;c++) {{if(col+c<n) {{
        if(row<m) {{y[row*n+col+c]=s0[c];}}
        if(row+1u<m) {{y[(row+1u)*n+col+c]=s1[c];}}
        if(row+2u<m) {{y[(row+2u)*n+col+c]=s2[c];}}
        if(row+3u<m) {{y[(row+3u)*n+col+c]=s3[c];}}
    }}}}
}}
"#
    )
}

#[cfg(feature = "kernel-checks")]
pub(super) fn fp16_pack_input() -> String {
    format!(
        "enable f16;\n{}\n{}",
        HEADER.replace("y: array<f32>", "y: array<f16>"),
        r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x+id.y*p.d.w; if(i<p.a.x) { y[i]=f16(x[i]); }
}"#
    )
}

pub(super) const GEMM: &str = r#"
var<workgroup> at: array<f32,1024>;
var<workgroup> bt: array<f32,1024>;
@compute @workgroup_size(8,8)
fn main(@builtin(workgroup_id) g:vec3<u32>, @builtin(local_invocation_id) l:vec3<u32>) {
    let row=g.y*32u+l.y*4u; let col=g.x*32u+l.x*4u;
    let m=p.a.x; let n=p.a.y; let k=p.a.z;
    var s0=vec4<f32>(0.0);var s1=s0;var s2=s0;var s3=s0;
    for(var first=0u;first<k;first+=32u) {
        for(var j=l.y*8u+l.x;j<1024u;j+=64u) {
            let r=j/32u;let c=j%32u;
            at[j]=0.0;bt[j]=0.0;
            if(g.y*32u+r<m && first+c<k) { at[j]=x[(g.y*32u+r)*k+first+c]; }
            if(g.x*32u+c<n && first+r<k) { bt[j]=bf((g.x*32u+c)*k+first+r); }
        }
        workgroupBarrier();
        for(var j=0u;j<32u;j++) {
            let i=j*32u+l.x*4u;
            let v=vec4<f32>(bt[i],bt[i+1u],bt[i+2u],bt[i+3u]);
            s0=fma(vec4<f32>(at[(l.y*4u)*32u+j]),v,s0);
            s1=fma(vec4<f32>(at[(l.y*4u+1u)*32u+j]),v,s1);
            s2=fma(vec4<f32>(at[(l.y*4u+2u)*32u+j]),v,s2);
            s3=fma(vec4<f32>(at[(l.y*4u+3u)*32u+j]),v,s3);
        }
        workgroupBarrier();
    }
    for(var c=0u;c<4u;c++) {if(col+c<n) {
        if(row<m) {y[row*n+col+c]=s0[c];}
        if(row+1u<m) {y[(row+1u)*n+col+c]=s1[c];}
        if(row+2u<m) {y[(row+2u)*n+col+c]=s2[c];}
        if(row+3u<m) {y[(row+3u)*n+col+c]=s3[c];}
    }}
}
"#;
pub(super) const NORM: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let row=id.x; let d=p.a.y; if(row>=p.a.x) { return; }
    var sum=0.0; for(var i=0u;i<d;i++) { let v=x[row*d+i]; sum+=v*v; }
    let scale=1.0/sqrt(sum/f32(d)+1e-6);
    for(var i=0u;i<d;i++) { y[row*d+i]=(x[row*d+i]*scale)*(1.0+bf(i)); }
}
"#;
// One workgroup per row: coalesced 512-value tiles in workgroup memory, with one thread
// summing them in NORM's order so the output matches NORM bit for bit. Requires d % 4 == 0.
#[cfg(feature = "kernel-checks")]
pub(super) const NORM_COOPERATIVE: &str = r#"
var<workgroup> tile:array<vec4<f32>,128>;
var<workgroup> row_scale:f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g:vec3<u32>, @builtin(local_invocation_index) l:u32) {
    let row=g.x; let d=p.a.y;
    var sum=0.0;
    for(var k=0u;k<d;k+=512u) {
        for(var i=l;i<128u;i+=64u) {
            let c=min(k+i*4u,d-4u); let o=row*d+c;
            tile[i]=vec4<f32>(x[o],x[o+1u],x[o+2u],x[o+3u]);
        }
        workgroupBarrier();
        if(l==0u) {
            for(var j=0u;j<min(128u,(d-k)/4u);j++) { let v=tile[j]; sum+=v.x*v.x; sum+=v.y*v.y; sum+=v.z*v.z; sum+=v.w*v.w; }
        }
        workgroupBarrier();
    }
    if(l==0u) { row_scale=1.0/sqrt(sum/f32(d)+1e-6); }
    workgroupBarrier();
    for(var i=l;i<d;i+=64u) { y[row*d+i]=(x[row*d+i]*row_scale)*(1.0+bf(i)); }
}
"#;

pub(super) const POINT: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x; if(i>=p.a.x) { return; }
    if(p.a.y==0u) { y[i]=x[i]+wf(i); }
    else if(p.a.y==1u) { y[i]=silu(x[i])*wf(i); }
    else if(p.a.y==2u) { y[i]=x[bitcast<u32>(af(i/p.a.z))*p.a.z+i%p.a.z]; }
}
"#;
pub(super) const CONV: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x; if(i>=p.a.x*8192u) { return; }
    let t=i/8192u; let c=i%8192u; var sum=0.0;
    for(var tap=0u;tap<4u;tap++) { if(t+tap>=3u) { sum+=x[(t+tap-3u)*8192u+c]*bf(c*4u+tap); } }
    y[i]=silu(sum);
}
"#;
pub(super) const DELTA_QK: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let h=id.x; if(h>=p.a.x*64u) { return; }
    let start=h*128u;
    var scale=1.0;
    if(h%64u<32u) {
        var sum=0.0; for(var i=0u;i<128u;i++) { let v=x[start+i]; sum+=v*v; }
        scale=1.0/sqrt(sum+1e-6);
    }
    for(var i=0u;i<128u;i++) {
        var v=x[start+i]*scale;
        if(h%64u<16u) { v*=0.08838834764831845; }
        y[start+i]=v;
    }
}
"#;
pub(super) const DELTA_PARAMS: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x; if(i>=p.a.x*32u) { return; }
    let v=x[i]+af(32u+i%32u);
    var sp=v; if(v<=20.0) { sp=log(1.0+exp(v)); }
    y[i*2u]=exp(-af(i%32u)*sp); y[i*2u+1u]=sigmoid(wf(i));
}
"#;
pub(super) const DELTA_SCAN: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let h=id.x/128u; let c=id.x%128u; if(h>=32u) { return; }
    var state:array<f32,128>;
    for(var t=0u;t<p.a.x;t++) {
        let g=wf((t*32u+h)*2u); let beta=wf((t*32u+h)*2u+1u);
        let q=t*8192u+(h/2u)*128u; let k=q+2048u;
        var memory=0.0;
        for(var i=0u;i<128u;i++) { state[i]*=g; memory+=state[i]*x[k+i]; }
        let delta=(x[t*8192u+4096u+h*128u+c]-memory)*beta;
        var sum=0.0;
        for(var i=0u;i<128u;i++) { state[i]+=x[k+i]*delta; sum+=state[i]*x[q+i]; }
        y[t*4096u+h*128u+c]=sum;
    }
}
"#;
#[cfg(feature = "kernel-checks")]
pub(super) fn delta_scan_four() -> String {
    let width = 4;
    let loop_body = |memory: bool| {
        let mut text = format!("for(var base=0u;base<128u;base+={width}u) {{\n");
        for lane in 0..width {
            let index = format!("base+{lane}u");
            if memory {
                text.push_str(&format!(
                    "state[{index}]*=g; memory+=state[{index}]*x[k+{index}];\n"
                ));
            } else {
                text.push_str(&format!(
                    "state[{index}]+=x[k+{index}]*delta; sum+=state[{index}]*x[q+{index}];\n"
                ));
            }
        }
        text.push('}');
        text
    };
    DELTA_SCAN
        .replace(
            "for(var i=0u;i<128u;i++) { state[i]*=g; memory+=state[i]*x[k+i]; }",
            &loop_body(true),
        )
        .replace(
            "for(var i=0u;i<128u;i++) { state[i]+=x[k+i]*delta; sum+=state[i]*x[q+i]; }",
            &loop_body(false),
        )
}
// Two threads hold halves of each 128-element state column, 32 columns per workgroup. Each
// serial sum runs over the first half, then continues from the carried partial over the second,
// so the result matches DELTA_SCAN bit for bit while halving per-thread state. q/k for eight
// tokens are staged in workgroup memory.
#[cfg(feature = "kernel-checks")]
pub(super) const DELTA_SCAN_CHAINED: &str = r#"
var<workgroup> kq:array<vec4<f32>,512>;
var<workgroup> gb:array<f32,16>;
var<workgroup> carry:array<f32,32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_index) l:u32) {
    let part=l/32u; let col=l%32u; let gid=wg.x*32u+col; let h=gid/128u; let c=gid%128u;
    let first=part*16u;
    var state:array<f32,64>;
    for(var t0=0u;t0<p.a.x;t0+=8u) {
        workgroupBarrier();
        for(var i=l;i<512u;i+=64u) {
            let t=t0+i/64u; let j=i%64u; var v=vec4<f32>(0.0);
            if(t<p.a.x) { let o=t*8192u+(h/2u)*128u+select(j*4u,2048u+(j-32u)*4u,j>=32u); v=vec4<f32>(x[o],x[o+1u],x[o+2u],x[o+3u]); }
            kq[i]=v;
        }
        for(var i=l;i<16u;i+=64u) { let t=t0+i/2u; gb[i]=select(0.0,wf((t*32u+h)*2u+i%2u),t<p.a.x); }
        workgroupBarrier();
        let end=min(8u,p.a.x-t0);
        for(var tt=0u;tt<end;tt++) {
            let t=t0+tt; let g=gb[tt*2u]; let beta=gb[tt*2u+1u];
            for(var i=0u;i<64u;i++) { state[i]*=g; }
            for(var s=0u;s<2u;s++) {
                if(part==s) {
                    var memory=select(carry[col],0.0,s==0u);
                    for(var b=0u;b<16u;b++) { let kv=kq[tt*64u+32u+first+b];
                        memory+=state[b*4u]*kv.x; memory+=state[b*4u+1u]*kv.y; memory+=state[b*4u+2u]*kv.z; memory+=state[b*4u+3u]*kv.w; }
                    carry[col]=memory;
                }
                workgroupBarrier();
            }
            let delta=(x[t*8192u+4096u+h*128u+c]-carry[col])*beta;
            workgroupBarrier();
            for(var b=0u;b<16u;b++) { let kv=kq[tt*64u+32u+first+b];
                state[b*4u]+=kv.x*delta; state[b*4u+1u]+=kv.y*delta; state[b*4u+2u]+=kv.z*delta; state[b*4u+3u]+=kv.w*delta; }
            for(var s=0u;s<2u;s++) {
                if(part==s) {
                    var sum=select(carry[col],0.0,s==0u);
                    for(var b=0u;b<16u;b++) { let qv=kq[tt*64u+first+b];
                        sum+=state[b*4u]*qv.x; sum+=state[b*4u+1u]*qv.y; sum+=state[b*4u+2u]*qv.z; sum+=state[b*4u+3u]*qv.w; }
                    carry[col]=sum;
                    if(s==1u) { y[t*4096u+h*128u+c]=sum; }
                }
                workgroupBarrier();
            }
        }
    }
}
"#;
// One thread per state column like DELTA_SCAN, with q/k for four tokens staged in workgroup
// memory; the arithmetic order is DELTA_SCAN's.
#[cfg(feature = "kernel-checks")]
pub(super) const DELTA_SCAN_STAGED: &str = r#"
var<workgroup> kq:array<vec4<f32>,256>;
var<workgroup> gb:array<f32,8>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_index) l:u32) {
    let gid=wg.x*64u+l; let h=gid/128u; let c=gid%128u;
    var state:array<f32,128>;
    for(var t0=0u;t0<p.a.x;t0+=4u) {
        workgroupBarrier();
        for(var i=l;i<256u;i+=64u) {
            let t=t0+i/64u; let j=i%64u; var v=vec4<f32>(0.0);
            if(t<p.a.x) { let o=t*8192u+(h/2u)*128u+select(j*4u,2048u+(j-32u)*4u,j>=32u); v=vec4<f32>(x[o],x[o+1u],x[o+2u],x[o+3u]); }
            kq[i]=v;
        }
        for(var i=l;i<8u;i+=64u) { let t=t0+i/2u; gb[i]=select(0.0,wf((t*32u+h)*2u+i%2u),t<p.a.x); }
        workgroupBarrier();
        let end=min(4u,p.a.x-t0);
        for(var tt=0u;tt<end;tt++) {
            let t=t0+tt; let g=gb[tt*2u]; let beta=gb[tt*2u+1u];
            var memory=0.0;
            for(var b=0u;b<32u;b++) { let kv=kq[tt*64u+32u+b];
                state[b*4u+0u]*=g; memory+=state[b*4u+0u]*kv.x;
                state[b*4u+1u]*=g; memory+=state[b*4u+1u]*kv.y;
                state[b*4u+2u]*=g; memory+=state[b*4u+2u]*kv.z;
                state[b*4u+3u]*=g; memory+=state[b*4u+3u]*kv.w;
            }
            let delta=(x[t*8192u+4096u+h*128u+c]-memory)*beta;
            var sum=0.0;
            for(var b=0u;b<32u;b++) { let kv=kq[tt*64u+32u+b]; let qv=kq[tt*64u+b];
                state[b*4u+0u]+=kv.x*delta; sum+=state[b*4u+0u]*qv.x;
                state[b*4u+1u]+=kv.y*delta; sum+=state[b*4u+1u]*qv.y;
                state[b*4u+2u]+=kv.z*delta; sum+=state[b*4u+2u]*qv.z;
                state[b*4u+3u]+=kv.w*delta; sum+=state[b*4u+3u]*qv.w;
            }
            y[t*4096u+h*128u+c]=sum;
        }
    }
}

"#;
pub(super) const DELTA_GATE: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let h=id.x; if(h>=p.a.x*32u) { return; }
    var sum=0.0; for(var i=0u;i<128u;i++) { let v=x[h*128u+i]; sum+=v*v; }
    let scale=1.0/sqrt(sum/128.0+1e-6);
    for(var i=0u;i<128u;i++) { let at=h*128u+i; y[at]=((x[at]*scale)*bf(i))*silu(af(at)); }
}
"#;
pub(super) const ATTENTION_NORM: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let h=id.x; if(h>=p.a.x*p.a.y) { return; }
    let base=h*p.a.z;
    var sum=0.0; for(var i=0u;i<256u;i++) { let v=x[base+i]; sum+=v*v; }
    let scale=1.0/sqrt(sum/256.0+1e-6);
    for(var i=0u;i<256u;i++) { y[h*256u+i]=(x[base+i]*scale)*(1.0+bf(i)); }
    for(var i=0u;i<32u;i++) {
        let c=af((h/p.a.y)*64u+i); let s=af((h/p.a.y)*64u+i+32u);
        let a=y[h*256u+i]; let b=y[h*256u+i+32u];
        y[h*256u+i]=a*c-b*s; y[h*256u+i+32u]=b*c+a*s;
    }
}
"#;
pub(super) const SCORES: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x; let nk=p.a.x; if(i>=p.a.y*16u*nk) { return; }
    let k=i%nk; let h=(i/nk)%16u; let q=i/nk/16u+p.a.z;
    var sum=0.0;
    for(var j=0u;j<256u;j++) { sum=fma(x[q*4096u+h*256u+j],wf(k*1024u+(h/4u)*256u+j),sum); }
    y[i]=sum/16.0;
}
"#;
pub(super) const SOFTMAX: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let r=id.x; let n=p.a.y; if(r>=p.a.x) { return; }
    var maximum=-3.402823e38;
    for(var j=0u;j<n;j++) { maximum=max(maximum,x[r*n+j]); }
    var sum=0.0;
    for(var j=0u;j<n;j++) { let v=exp(x[r*n+j]-maximum); y[r*n+j]=v; sum+=v; }
    for(var j=0u;j<n;j++) { y[r*n+j]/=sum; }
}
"#;
pub(super) const VALUES: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x; let nk=p.a.x; if(i>=p.a.y*4096u) { return; }
    let q=i/4096u; let h=(i%4096u)/256u; let c=i%256u;
    var sum=0.0;
    for(var j=0u;j<nk;j++) { sum=fma(x[(q*16u+h)*nk+j],wf(j*1024u+(h/4u)*256u+c),sum); }
    y[(q+p.a.z)*4096u+h*256u+c]=sum*sigmoid(af(((q+p.a.z)*16u+h)*512u+256u+c));
}
"#;

pub(super) const DELTA_CONSTANTS: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x; if(i>=32u) { return; }
    y[i]=exp(bf(i));
    y[32u+i]=bitcast<f32>(((aux[i/2u]>>((i%2u)*16u))&65535u)<<16u);
}
"#;

// Tiled attention: rows are (query, head) pairs of one KV head so the four query heads share each K/V tile.
// Every output keeps the reference kernels' sequential fma order, so results are bitwise identical.
#[cfg(feature = "kernel-checks")]
pub(super) const SCORES_TILED: &str = r#"
var<workgroup> qa:array<vec4<f32>,288>;
var<workgroup> kb:array<vec4<f32>,288>;
@compute @workgroup_size(8,8)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>) {
 let nk=p.a.x;let rows=p.a.y;
 var s0_0=0.0; var s0_1=0.0; var s0_2=0.0; var s0_3=0.0;
 var s1_0=0.0; var s1_1=0.0; var s1_2=0.0; var s1_3=0.0;
 var s2_0=0.0; var s2_1=0.0; var s2_2=0.0; var s2_3=0.0;
 var s3_0=0.0; var s3_1=0.0; var s3_2=0.0; var s3_3=0.0;
 for(var first=0u;first<256u;first+=32u){
  for(var i=l.y*8u+l.x;i<256u;i+=64u){
   let r=i/8u;let c=(i%8u)*4u;let m=g.y*32u+r;var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);
   if(m<rows*4u){let o=(m/4u+p.a.z)*4096u+(g.z*4u+m%4u)*256u+first+c;a=vec4<f32>(x[o],x[o+1u],x[o+2u],x[o+3u]);}
   if(g.x*32u+r<nk){let o=(g.x*32u+r)*1024u+g.z*256u+first+c;b=vec4<f32>(wf(o),wf(o+1u),wf(o+2u),wf(o+3u));}
   qa[r*9u+i%8u]=a;kb[r*9u+i%8u]=b;
  }
  workgroupBarrier();
  for(var j=0u;j<8u;j++){
   let b0=kb[(l.x+0u)*9u+j];
   let b1=kb[(l.x+8u)*9u+j];
   let b2=kb[(l.x+16u)*9u+j];
   let b3=kb[(l.x+24u)*9u+j];
   let a0=qa[(l.y+0u)*9u+j];s0_0=fma(a0.w,b0.w,fma(a0.z,b0.z,fma(a0.y,b0.y,fma(a0.x,b0.x,s0_0))));s0_1=fma(a0.w,b1.w,fma(a0.z,b1.z,fma(a0.y,b1.y,fma(a0.x,b1.x,s0_1))));s0_2=fma(a0.w,b2.w,fma(a0.z,b2.z,fma(a0.y,b2.y,fma(a0.x,b2.x,s0_2))));s0_3=fma(a0.w,b3.w,fma(a0.z,b3.z,fma(a0.y,b3.y,fma(a0.x,b3.x,s0_3))));
   let a1=qa[(l.y+8u)*9u+j];s1_0=fma(a1.w,b0.w,fma(a1.z,b0.z,fma(a1.y,b0.y,fma(a1.x,b0.x,s1_0))));s1_1=fma(a1.w,b1.w,fma(a1.z,b1.z,fma(a1.y,b1.y,fma(a1.x,b1.x,s1_1))));s1_2=fma(a1.w,b2.w,fma(a1.z,b2.z,fma(a1.y,b2.y,fma(a1.x,b2.x,s1_2))));s1_3=fma(a1.w,b3.w,fma(a1.z,b3.z,fma(a1.y,b3.y,fma(a1.x,b3.x,s1_3))));
   let a2=qa[(l.y+16u)*9u+j];s2_0=fma(a2.w,b0.w,fma(a2.z,b0.z,fma(a2.y,b0.y,fma(a2.x,b0.x,s2_0))));s2_1=fma(a2.w,b1.w,fma(a2.z,b1.z,fma(a2.y,b1.y,fma(a2.x,b1.x,s2_1))));s2_2=fma(a2.w,b2.w,fma(a2.z,b2.z,fma(a2.y,b2.y,fma(a2.x,b2.x,s2_2))));s2_3=fma(a2.w,b3.w,fma(a2.z,b3.z,fma(a2.y,b3.y,fma(a2.x,b3.x,s2_3))));
   let a3=qa[(l.y+24u)*9u+j];s3_0=fma(a3.w,b0.w,fma(a3.z,b0.z,fma(a3.y,b0.y,fma(a3.x,b0.x,s3_0))));s3_1=fma(a3.w,b1.w,fma(a3.z,b1.z,fma(a3.y,b1.y,fma(a3.x,b1.x,s3_1))));s3_2=fma(a3.w,b2.w,fma(a3.z,b2.z,fma(a3.y,b2.y,fma(a3.x,b2.x,s3_2))));s3_3=fma(a3.w,b3.w,fma(a3.z,b3.z,fma(a3.y,b3.y,fma(a3.x,b3.x,s3_3))));
  }
  workgroupBarrier();
 }
 {let m=g.y*32u+l.y+0u;let col=g.x*32u+l.x+0u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s0_0/16.0;}}
 {let m=g.y*32u+l.y+0u;let col=g.x*32u+l.x+8u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s0_1/16.0;}}
 {let m=g.y*32u+l.y+0u;let col=g.x*32u+l.x+16u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s0_2/16.0;}}
 {let m=g.y*32u+l.y+0u;let col=g.x*32u+l.x+24u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s0_3/16.0;}}
 {let m=g.y*32u+l.y+8u;let col=g.x*32u+l.x+0u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s1_0/16.0;}}
 {let m=g.y*32u+l.y+8u;let col=g.x*32u+l.x+8u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s1_1/16.0;}}
 {let m=g.y*32u+l.y+8u;let col=g.x*32u+l.x+16u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s1_2/16.0;}}
 {let m=g.y*32u+l.y+8u;let col=g.x*32u+l.x+24u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s1_3/16.0;}}
 {let m=g.y*32u+l.y+16u;let col=g.x*32u+l.x+0u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s2_0/16.0;}}
 {let m=g.y*32u+l.y+16u;let col=g.x*32u+l.x+8u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s2_1/16.0;}}
 {let m=g.y*32u+l.y+16u;let col=g.x*32u+l.x+16u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s2_2/16.0;}}
 {let m=g.y*32u+l.y+16u;let col=g.x*32u+l.x+24u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s2_3/16.0;}}
 {let m=g.y*32u+l.y+24u;let col=g.x*32u+l.x+0u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s3_0/16.0;}}
 {let m=g.y*32u+l.y+24u;let col=g.x*32u+l.x+8u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s3_1/16.0;}}
 {let m=g.y*32u+l.y+24u;let col=g.x*32u+l.x+16u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s3_2/16.0;}}
 {let m=g.y*32u+l.y+24u;let col=g.x*32u+l.x+24u;if(m<rows*4u&&col<nk){y[((m/4u)*16u+g.z*4u+m%4u)*nk+col]=s3_3/16.0;}}
}
"#;

#[cfg(feature = "kernel-checks")]
pub(super) const VALUES_TILED: &str = r#"
var<workgroup> pa:array<vec4<f32>,320>;
var<workgroup> vb:array<vec4<f32>,128>;
@compute @workgroup_size(8,8)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>) {
 let nk=p.a.x;let rows=p.a.y;let lane=l.y*8u+l.x;
 var s0_0=vec4<f32>(0.0);
 var s1_0=vec4<f32>(0.0);
 var s2_0=vec4<f32>(0.0);
 var s3_0=vec4<f32>(0.0);
 var s4_0=vec4<f32>(0.0);
 var s5_0=vec4<f32>(0.0);
 var s6_0=vec4<f32>(0.0);
 var s7_0=vec4<f32>(0.0);
 for(var first=0u;first<nk;first+=16u){
  for(var i=lane;i<256u;i+=64u){
   let r=i/4u;let c=(i%4u)*4u;let m=g.y*64u+r;var v=vec4<f32>(0.0);
   if(m<rows*4u){let o=((m/4u)*16u+g.z*4u+m%4u)*nk+first+c;
    for(var e=0u;e<4u;e++){if(first+c+e<nk){v[e]=x[o+e];}}}
   pa[r*5u+i%4u]=v;
  }
  for(var i=lane;i<128u;i+=64u){
   let r=i/8u;let c=(i%8u)*4u;vb[i]=vec4<f32>(0.0);
   if(first+r<nk){let o=(first+r)*1024u+g.z*256u+g.x*32u+c;vb[i]=vec4<f32>(wf(o),wf(o+1u),wf(o+2u),wf(o+3u));}
  }
  workgroupBarrier();
  for(var j=0u;j<4u;j++){
   let a0=pa[(l.y+0u)*5u+j];
   let a1=pa[(l.y+8u)*5u+j];
   let a2=pa[(l.y+16u)*5u+j];
   let a3=pa[(l.y+24u)*5u+j];
   let a4=pa[(l.y+32u)*5u+j];
   let a5=pa[(l.y+40u)*5u+j];
   let a6=pa[(l.y+48u)*5u+j];
   let a7=pa[(l.y+56u)*5u+j];
   for(var e=0u;e<4u;e++){
    let b0=vb[(j*4u+e)*8u+l.x+0u];
    s0_0=fma(vec4<f32>(a0[e]),b0,s0_0);
    s1_0=fma(vec4<f32>(a1[e]),b0,s1_0);
    s2_0=fma(vec4<f32>(a2[e]),b0,s2_0);
    s3_0=fma(vec4<f32>(a3[e]),b0,s3_0);
    s4_0=fma(vec4<f32>(a4[e]),b0,s4_0);
    s5_0=fma(vec4<f32>(a5[e]),b0,s5_0);
    s6_0=fma(vec4<f32>(a6[e]),b0,s6_0);
    s7_0=fma(vec4<f32>(a7[e]),b0,s7_0);
   }
  }
  workgroupBarrier();
 }
 {let m=g.y*64u+l.y+0u;if(m<rows*4u){let q=m/4u+p.a.z;let h=g.z*4u+m%4u;let col=g.x*32u+(l.x+0u)*4u;
  for(var e=0u;e<4u;e++){y[q*4096u+h*256u+col+e]=s0_0[e]*sigmoid(af((q*16u+h)*512u+256u+col+e));}}}
 {let m=g.y*64u+l.y+8u;if(m<rows*4u){let q=m/4u+p.a.z;let h=g.z*4u+m%4u;let col=g.x*32u+(l.x+0u)*4u;
  for(var e=0u;e<4u;e++){y[q*4096u+h*256u+col+e]=s1_0[e]*sigmoid(af((q*16u+h)*512u+256u+col+e));}}}
 {let m=g.y*64u+l.y+16u;if(m<rows*4u){let q=m/4u+p.a.z;let h=g.z*4u+m%4u;let col=g.x*32u+(l.x+0u)*4u;
  for(var e=0u;e<4u;e++){y[q*4096u+h*256u+col+e]=s2_0[e]*sigmoid(af((q*16u+h)*512u+256u+col+e));}}}
 {let m=g.y*64u+l.y+24u;if(m<rows*4u){let q=m/4u+p.a.z;let h=g.z*4u+m%4u;let col=g.x*32u+(l.x+0u)*4u;
  for(var e=0u;e<4u;e++){y[q*4096u+h*256u+col+e]=s3_0[e]*sigmoid(af((q*16u+h)*512u+256u+col+e));}}}
 {let m=g.y*64u+l.y+32u;if(m<rows*4u){let q=m/4u+p.a.z;let h=g.z*4u+m%4u;let col=g.x*32u+(l.x+0u)*4u;
  for(var e=0u;e<4u;e++){y[q*4096u+h*256u+col+e]=s4_0[e]*sigmoid(af((q*16u+h)*512u+256u+col+e));}}}
 {let m=g.y*64u+l.y+40u;if(m<rows*4u){let q=m/4u+p.a.z;let h=g.z*4u+m%4u;let col=g.x*32u+(l.x+0u)*4u;
  for(var e=0u;e<4u;e++){y[q*4096u+h*256u+col+e]=s5_0[e]*sigmoid(af((q*16u+h)*512u+256u+col+e));}}}
 {let m=g.y*64u+l.y+48u;if(m<rows*4u){let q=m/4u+p.a.z;let h=g.z*4u+m%4u;let col=g.x*32u+(l.x+0u)*4u;
  for(var e=0u;e<4u;e++){y[q*4096u+h*256u+col+e]=s6_0[e]*sigmoid(af((q*16u+h)*512u+256u+col+e));}}}
 {let m=g.y*64u+l.y+56u;if(m<rows*4u){let q=m/4u+p.a.z;let h=g.z*4u+m%4u;let col=g.x*32u+(l.x+0u)*4u;
  for(var e=0u;e<4u;e++){y[q*4096u+h*256u+col+e]=s7_0[e]*sigmoid(af((q*16u+h)*512u+256u+col+e));}}}
}
"#;
