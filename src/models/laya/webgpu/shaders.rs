const HEADER: &str = r#"
struct Params { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32> }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
fn gelu(v: f32) -> f32 {
    let a = abs(v * 0.7071067811865475);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = (((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;
    let erf = sign(v) * (1.0 - poly * exp(-a*a));
    return 0.5*v*(1.0+erf);
}
"#;
pub(super) fn source(body: &str) -> String {
    format!("{HEADER}\n{body}")
}
pub(super) fn gemm(tile: usize) -> String {
    source(
        &r#"
var<workgroup> at: array<f32, $COUNT>;
var<workgroup> bt: array<f32, $COUNT>;
@compute @workgroup_size($TILE, $TILE)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let row=g.y*$TILE+l.y; let col=g.x*$TILE+l.x;
    let m=p.a.x; let n=p.a.y; let k=p.a.z;
    var sum=0.0;
    for(var first=0u;first<k;first+=$TILE) {
        let ak=first+l.x; let bk=first+l.y; let index=l.y*$TILE+l.x;
        at[index]=0.0; bt[index]=0.0;
        if(row<m && ak<k) { at[index]=x[row*k+ak]; }
        if(col<n && bk<k) { bt[index]=w[col*k+bk]; }
        workgroupBarrier();
        for(var j=0u;j<$TILE;j++) { sum=fma(at[l.y*$TILE+j],bt[j*$TILE+l.x],sum); }
        workgroupBarrier();
    }
    if(row<m && col<n) { y[row*n+col]=sum+bias[col]; }
}
"#
        .replace("$COUNT", &(tile * tile).to_string())
        .replace("$TILE", &format!("{tile}u")),
    )
}
pub(super) const NORM: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row=id.x+id.y*p.d.w; let d=p.a.y; if(row>=p.a.x) { return; }
    var sum=0.0; var correction=0.0;
    for(var i=0u;i<d;i++) { let v=x[row*d+i]-correction; let t=sum+v; correction=(t-sum)-v; sum=t; }
    let mean=sum/f32(d); var variance=0.0; correction=0.0;
    for(var i=0u;i<d;i++) { let z=x[row*d+i]-mean; let v=z*z-correction; let t=variance+v; correction=(t-variance)-v; variance=t; }
    let inv=1.0/sqrt(variance/f32(d)+bitcast<f32>(p.a.z));
    for(var i=0u;i<d;i++) { y[row*d+i]=(x[row*d+i]-mean)*inv*w[i]+bias[i]; }
}
"#;
pub(super) const POINTWISE: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x+id.y*p.d.w; if(i>=p.a.x) { return; }
    let op=p.a.y;
    if(op==0u) { y[i]=x[i]+w[i]; }
    else if(op==1u) { y[i]=x[i]+w[p.a.w*p.a.z+i%p.a.z]; }
    else if(op==2u) { let d=p.a.z; let r=i/d; let c=i%d; y[i]=gelu(x[r*2u*d+c])*x[r*2u*d+d+c]; }
    else if(op==3u) { y[i]=max(x[i],0.0); }
    else { y[i]=gelu(x[i]); }
}
"#;
pub(super) const ROPE: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x+id.y*p.d.w; let d=p.a.y; let hd=p.a.z; let half=hd/2u;
    if(i>=p.a.x*3u*d) { return; }
    let which=(i/d)%3u; if(which==2u) { y[i]=x[i]; return; }
    let c=i%hd; let start=i-c; let j=c%half;
    let angle=((i/(3u*d))*half+j)*2u;
    let s=w[angle]; let co=w[angle+1u]; let a=x[start+j]; let b=x[start+j+half];
    if(c<half) { y[i]=a*co-b*s; } else { y[i]=b*co+a*s; }
}
"#;
pub(super) const SCORES: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x+id.y*p.d.w; let n=p.a.x; let heads=p.a.y; let hd=p.a.z; let rows=p.a.w;
    if(i>=rows*heads*n) { return; }
    let key=i%n; let query=i/n/heads+p.b.x; let h=(i/n)%heads;
    if(p.b.y>0u && (key+p.b.y-1u<query || query+p.b.y-1u<key)) { y[i]=-3.402823e38; return; }
    let d=heads*hd; var sum=0.0;
    for(var j=0u;j<hd;j++) { sum=fma(x[query*3u*d+h*hd+j],x[key*3u*d+d+h*hd+j],sum); }
    y[i]=sum/sqrt(f32(hd));
}
"#;
pub(super) const SOFTMAX: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row=id.x+id.y*p.d.w; let n=p.a.y; if(row>=p.a.x) { return; }
    var maximum=-3.402823e38;
    for(var j=0u;j<n;j++) { maximum=max(maximum,x[row*n+j]); }
    var sum=0.0;
    for(var j=0u;j<n;j++) { let e=exp(x[row*n+j]-maximum); y[row*n+j]=e; sum+=e; }
    for(var j=0u;j<n;j++) { y[row*n+j]/=sum; }
}
"#;
pub(super) const VALUES: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x+id.y*p.d.w; let n=p.a.x; let heads=p.a.y; let hd=p.a.z; let d=heads*hd;
    if(i>=p.a.w*d) { return; }
    let query=i/d; let h=(i%d)/hd; let c=i%hd; var sum=0.0;
    for(var j=0u;j<n;j++) { sum=fma(x[(query*heads+h)*n+j],w[j*3u*d+2u*d+h*hd+c],sum); }
    y[(query+p.b.x)*d+h*hd+c]=sum;
}
"#;
