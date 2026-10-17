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
