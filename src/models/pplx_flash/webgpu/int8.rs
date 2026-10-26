use super::*;

pub(super) struct State {
    pub weights: Vec<(Buffer, Buffer)>,
    pub fused: bool,
    pub fp16_mlp: bool,
    pub pure: bool,
    pub tile: [u32; 2],
    pub float_partials: bool,
    pub pool: RefCell<Vec<Buffer>>,
}
impl State {
    pub fn weight(&self, original: &Buffer) -> Option<Buffer> {
        self.weights
            .iter()
            .find(|(source, _)| source == original)
            .map(|(_, packed)| packed.clone())
    }
    pub fn clear(&mut self) {
        let weights = std::mem::take(&mut self.weights);
        self.recycle(weights);
    }
    pub fn recycle(&self, weights: Vec<(Buffer, Buffer)>) {
        let mut pool = self.pool.borrow_mut();
        for (_, packed) in weights {
            if pool.iter().map(Buffer::size).sum::<u64>() + packed.size() <= 192 * 1024 * 1024 {
                pool.push(packed);
            } else {
                packed.destroy();
            }
        }
    }
}
impl Drop for State {
    fn drop(&mut self) {
        self.clear();
        for buffer in self.pool.get_mut().drain(..) {
            buffer.destroy();
        }
    }
}
impl Gpu {
    pub(super) fn quantize_int8(
        &self,
        e: &mut CommandEncoder,
        input: &Buffer,
        rows: usize,
        k: usize,
        weight: bool,
    ) -> Result<Buffer> {
        ensure!(
            k % 4 == 0 && k <= 32768,
            "INT8 reduction must be a multiple of four, at most 32768"
        );
        let size = if self.grouped() {
            ensure!(k % 64 == 0 && !weight, "grouped weights must be prepacked");
            rows * (k / 64) * 17
        } else { rows * (k / 4 + 2) };
        let reused = if weight {
            self.int8.as_ref().and_then(|state| {
                let mut pool = state.pool.borrow_mut();
                let index = pool.iter().position(|b| b.size() == size as u64 * 4)?;
                Some(pool.swap_remove(index))
            })
        } else {
            None
        };
        let output = if let Some(output) = reused {
            output
        } else if weight {
            self.device.empty(size, "INT8 weight and row metadata")?
        } else {
            self.empty(size)?
        };
        self.run(
            e,
            if weight { "int8_weight" } else { "int8_input" },
            [input, &self.dummy, &self.dummy, &output],
            &[rows, 0, k],
            0,
        )?;
        Ok(output)
    }
    pub(super) fn int8_input(
        &self,
        e: &mut CommandEncoder,
        input: &Buffer,
        rows: usize,
        k: usize,
    ) -> Result<Buffer> {
        if let Some((_, packed)) = self
            .packed_cache
            .borrow()
            .iter()
            .find(|(source, _)| source == input)
        {
            return Ok(packed.clone());
        }
        let packed = self.quantize_int8(e, input, rows, k, false)?;
        self.packed_cache
            .borrow_mut()
            .push((input.clone(), packed.clone()));
        Ok(packed)
    }
}

pub(super) fn quantize(weight: bool) -> String {
    let source = r#"
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> maxima:array<f32,64>;
var<workgroup> sums:array<i32,64>;
fn even(v:f32)->i32 {let lo=floor(v);let f=v-lo;return i32(lo)+select(0,1,f>0.5||(f==0.5&&(i32(lo)&1)==1));}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) l:u32) {
 let row=g.x;let rows=p.a.x;let k=p.a.z;let words=k/4u;
 var maximum=0.0;
 for(var j=l;j<k;j+=64u){maximum=max(maximum,abs(f32(x[row*k+j])));}
 maxima[l]=maximum;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){maxima[l]=max(maxima[l],maxima[l+stride]);}workgroupBarrier();}
 let scale=maxima[0]/127.0;
 if(l==0u){y[rows*words+row]=bitcast<u32>(scale);}
 var total=0;
 for(var j=l;j<words;j+=64u){
  let at=row*k+j*4u;var q=vec4<i32>(0);
  if(scale>0.0){q=clamp(vec4<i32>(even(f32(x[at])/scale),even(f32(x[at+1u])/scale),even(f32(x[at+2u])/scale),even(f32(x[at+3u])/scale)),vec4<i32>(-127),vec4<i32>(127));}
  total+=q.x+q.y+q.z+q.w;
  let u=vec4<u32>(q+vec4<i32>(128));y[row*words+j]=u.x|(u.y<<8u)|(u.z<<16u)|(u.w<<24u);
 }
 sums[l]=total;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){sums[l]+=sums[l+stride];}workgroupBarrier();}
 if(l==0u){y[rows*words+rows+row]=bitcast<u32>(sums[0]);}
}
"#;
    if weight {
        format!(
            "enable f16;\n{}",
            source.replace("x:array<f32>", "x:array<f16>")
        )
    } else {
        source.to_string()
    }
}

pub(super) fn gemm(fused: bool, raw: bool) -> String {
    let rows = 8;
    let mut s = format!(
        r#"requires packed_4x8_integer_dot_product;
struct Params {{ a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }}
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(2) var<storage,read> aux:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<{}>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<u32,1024>;var<workgroup> bt:array<u32,1024>;
@compute @workgroup_size(16,8)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){{
 let m=p.a.x;let n=p.a.y;let k=p.a.z/4u;let lane=l.y*16u+l.x;
 let row=g.y*64u+l.y*8u;let col=g.x*64u+l.x*4u;
"#,
        if raw { "i32" } else { "f32" }
    );
    if fused {
        s = s.replace(
            "var<workgroup> bt:",
            "var<workgroup> ut:array<u32,1024>;var<workgroup> bt:",
        );
    }
    for r in 0..rows {
        s += &format!("var s{r}=vec4<u32>(0);\n");
        if fused {
            s += &format!("var u{r}=vec4<u32>(0);\n");
        }
    }
    s += r#"for(var first=0u;first<k;first+=16u){
 for(var i=lane;i<1024u;i+=128u){let r=i/16u;let c=i%16u;at[i]=0u;
 if(g.y*64u+r<m&&first+c<k){at[i]=x[(g.y*64u+r)*k+first+c];}}
 for(var i=lane;i<1024u;i+=128u){let r=i/16u;let c=i%16u;bt[c*64u+r]=0u;
 if(g.x*64u+r<n&&first+c<k){bt[c*64u+r]=w[(g.x*64u+r)*k+first+c];}}
 workgroupBarrier();for(var j=0u;j<16u;j++){
"#;
    if fused {
        s = s
            .replace("bt[c*64u+r]=0u;", "bt[c*64u+r]=0u;ut[c*64u+r]=0u;")
            .replace(
                "bt[c*64u+r]=w[(g.x*64u+r)*k+first+c];",
                "bt[c*64u+r]=w[(g.x*64u+r)*k+first+c];ut[c*64u+r]=aux[(g.x*64u+r)*k+first+c];",
            );
    }
    for r in 0..rows {
        s += &format!("let a{r}=at[(l.y*8u+{r}u)*16u+j];\n");
    }
    for c in 0..4 {
        s += &format!("let b{c}=bt[j*64u+l.x*4u+{c}u];\n");
        if fused {
            s += &format!("let v{c}=ut[j*64u+l.x*4u+{c}u];\n");
        }
    }
    for r in 0..rows {
        s += &format!(
            "s{r}+=vec4<u32>(dot4U8Packed(a{r},b0),dot4U8Packed(a{r},b1),dot4U8Packed(a{r},b2),dot4U8Packed(a{r},b3));\n"
        );
        if fused {
            s += &format!(
                "u{r}+=vec4<u32>(dot4U8Packed(a{r},v0),dot4U8Packed(a{r},v1),dot4U8Packed(a{r},v2),dot4U8Packed(a{r},v3));\n"
            );
        }
    }
    s += "}workgroupBarrier();}\n";
    for r in 0..rows {
        for c in 0..4 {
            let correction = |value: &str, weight: &str| {
                format!(
                    "i32({value})-128*(bitcast<i32>(x[m*k+m+row+{r}u])+bitcast<i32>({weight}[n*k+n+col+{c}u]))-i32(p.a.z)*16384"
                )
            };
            let dot = correction(&format!("s{r}[{c}]"), "w");
            let scale = |dot: &str, weight: &str| {
                format!(
                    "(f32({dot})*bitcast<f32>(x[m*k+row+{r}u]))*bitcast<f32>({weight}[n*k+col+{c}u])"
                )
            };
            s += &format!("if(row+{r}u<m&&col+{c}u<n){{");
            if fused {
                let up = scale(&correction(&format!("u{r}[{c}]"), "aux"), "aux");
                s += &format!(
                    "let gate={};let up={up};y[(row+{r}u)*n+col+{c}u]=(gate/(1.0+exp(-gate)))*up;",
                    scale(&dot, "w")
                );
            } else {
                s += &format!(
                    "y[(row+{r}u)*n+col+{c}u]={};",
                    if raw { dot } else { scale(&dot, "w") }
                );
            }
            s += "}\n";
        }
    }
    s + "}"
}

fn tiled(raw: bool) -> String {
    let (tm, tn, rm, rn, kw) = (64usize, 64usize, 8usize, 4usize, 16usize);
    assert!(tm % rm == 0 && tn % rn == 0 && rn % 4 == 0);
    let wx = tn / rn;
    let wy = tm / rm;
    let threads = wx * wy;
    assert!(threads <= 256 && kw * (tm + tn) * 4 <= 16384);
    let mut s = format!(
        r#"requires packed_4x8_integer_dot_product;
struct Params {{ a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }}
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<{output}>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<u32,{asize}>;var<workgroup> bt:array<u32,{bsize}>;
@compute @workgroup_size({wx},{wy})
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){{
let m=p.a.x;let n=p.a.y;let k=p.a.z/4u;let lane=l.y*{wx}u+l.x;
let row=g.y*{tm}u+l.y*{rm}u;let col=g.x*{tn}u+l.x*{rn}u;
"#,
        output = if raw { "i32" } else { "f32" },
        asize = tm * kw,
        bsize = tn * kw
    );
    for r in 0..rm {
        for c in (0..rn).step_by(4) {
            s += &format!("var s{r}_{c}=vec4<u32>(0);\n");
        }
    }
    s += &format!(
        r#"for(var first=0u;first<k;first+={kw}u){{
for(var i=lane;i<{asize}u;i+={threads}u){{let r=i/{kw}u;let c=i%{kw}u;at[i]=0u;
if(g.y*{tm}u+r<m&&first+c<k){{at[i]=x[(g.y*{tm}u+r)*k+first+c];}}}}
for(var i=lane;i<{bsize}u;i+={threads}u){{let r=i/{kw}u;let c=i%{kw}u;bt[c*{tn}u+r]=0u;
if(g.x*{tn}u+r<n&&first+c<k){{bt[c*{tn}u+r]=w[(g.x*{tn}u+r)*k+first+c];}}}}
workgroupBarrier();for(var j=0u;j<{kw}u;j++){{
"#,
        asize = tm * kw,
        bsize = tn * kw
    );
    for r in 0..rm {
        s += &format!("let a{r}=at[(l.y*{rm}u+{r}u)*{kw}u+j];\n");
    }
    for c in 0..rn {
        s += &format!("let b{c}=bt[j*{tn}u+l.x*{rn}u+{c}u];\n");
    }
    for r in 0..rm {
        for c in (0..rn).step_by(4) {
            let dots = (c..c + 4)
                .map(|j| format!("dot4U8Packed(a{r},b{j})"))
                .collect::<Vec<_>>()
                .join(",");
            s += &format!("s{r}_{c}+=vec4<u32>({dots});\n");
        }
    }
    s += "}workgroupBarrier();}\n";
    for r in 0..rm {
        for c in 0..rn {
            let part = c - c % 4;
            let lane = c % 4;
            let dot = format!(
                "i32(s{r}_{part}[{lane}])-128*(bitcast<i32>(x[m*k+m+row+{r}u])+bitcast<i32>(w[n*k+n+col+{c}u]))-i32(p.a.z)*16384"
            );
            let value = if raw {
                dot
            } else {
                format!("(f32({dot})*bitcast<f32>(x[m*k+row+{r}u]))*bitcast<f32>(w[n*k+col+{c}u])")
            };
            s += &format!("if(row+{r}u<m&&col+{c}u<n){{y[(row+{r}u)*n+col+{c}u]={value};}}\n");
        }
    }
    s + "}"
}

pub(super) fn fp32_partials(raw: bool) -> String {
    let (tn, rm, rn, kw) = (64, 8, 4, 16);
    let mut s = tiled(raw);
    let mut temporaries = String::new();
    let mut finish = String::new();
    for r in 0..rm {
        temporaries += &format!("var t{r}_0=vec4<f32>(0.0);\n");
        finish += &format!("s{r}_0+=vec4<u32>(t{r}_0);\n");
        s = s.replace(
            &format!("let a{r}=at[(l.y*{rm}u+{r}u)*{kw}u+j];"),
            &format!("let a{r}=vec4<f32>(unpack4xU8(at[(l.y*{rm}u+{r}u)*{kw}u+j]));"),
        );
    }
    s = s.replace(
        "workgroupBarrier();for(var j",
        &format!("workgroupBarrier();{temporaries}for(var j"),
    );
    s = s.replace(
        "}workgroupBarrier();}",
        &format!("}}{finish}workgroupBarrier();}}"),
    );
    for c in 0..4 {
        s = s.replace(
            &format!("let b{c}=bt[j*{tn}u+l.x*{rn}u+{c}u];"),
            &format!("let b{c}=vec4<f32>(unpack4xU8(bt[j*{tn}u+l.x*{rn}u+{c}u]));"),
        );
    }
    let mut vectors = String::new();
    for lane in ["x", "y", "z", "w"] {
        vectors += &format!("let bv_{lane}=vec4<f32>(b0.{lane},b1.{lane},b2.{lane},b3.{lane});\n");
    }
    for r in 0..rm {
        let from = format!(
            "s{r}_0+=vec4<u32>(dot4U8Packed(a{r},b0),dot4U8Packed(a{r},b1),dot4U8Packed(a{r},b2),dot4U8Packed(a{r},b3));"
        );
        assert!(s.contains(&from));
        let mut operations = if r == 0 {
            vectors.clone()
        } else {
            String::new()
        };
        for lane in ["x", "y", "z", "w"] {
            operations += &format!("t{r}_0=fma(vec4<f32>(a{r}.{lane}),bv_{lane},t{r}_0);\n");
        }
        s = s.replace(&from, &operations);
    }
    s
}
