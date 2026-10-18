use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode { Fused, Buffered }

impl Gpu {
    pub(super) fn quantize_q4(&mut self, e: &mut CommandEncoder, input: &Buffer, rows: usize, k: usize) -> Result<Buffer> {
        ensure!(k % 8 == 0 && k <= 32768, "Q4 reduction must be a multiple of eight, at most 32768");
        let out = self.device.empty(rows * (k / 8 + 2), "Q4 codes, row scales and sums")?;
        self.run(e, "q4_weight", [input, &self.dummy, &self.dummy, &out], &[rows, 0, k], 0)?;
        if self.q4 == Some(Mode::Buffered) {
            let words = rows * (k / 4 + 2);
            if self.q4_expansion.as_ref().is_none_or(|b| b.size() < words as u64 * 4) {
                if let Some(old) = self.q4_expansion.take() { old.destroy(); }
                self.q4_expansion = Some(self.device.empty(words, "reusable Q4 to Q8 expansion")?);
            }
        }
        Ok(out)
    }
    pub(super) fn expand_q4(&self, e: &mut CommandEncoder, weight: &Buffer, rows: usize, k: usize) -> Result<Buffer> {
        let buffer = self.q4_expansion.as_ref().context("missing Q4 expansion workspace")?;
        ensure!(buffer.size() >= (rows * (k / 4 + 2) * 4) as u64, "Q4 expansion exceeds workspace");
        self.run(e, "q4_expand", [weight, &self.dummy, &self.dummy, buffer], &[rows, 0, k], rows * (k / 4 + 2))?;
        Ok(buffer.clone())
    }
    pub(crate) fn q4_info(&self) -> serde_json::Value {
        serde_json::json!({"weight_bits":if self.q4.is_some(){4}else if self.int8.is_some(){8}else{16},
            "q4_scheme":self.q4.map(|_| "symmetric-per-output-channel-absmax-round-even-[-7,7]"),
            "q4_unpack":self.q4.map(|m|if m==Mode::Fused{"workgroup-tile"}else{"reusable-matrix-buffer"}),
            "q4_expansion_bytes":self.q4_expansion.as_ref().map_or(0, Buffer::size)})
    }
}

pub(super) fn quantize() -> String {
    r#"enable f16;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<f16>;
@group(0) @binding(3) var<storage,read_write> y:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> maxima:array<f32,64>;var<workgroup> sums:array<i32,64>;
fn even(v:f32)->i32 {let lo=floor(v);let f=v-lo;return i32(lo)+select(0,1,f>0.5||(f==0.5&&(i32(lo)&1)==1));}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) l:u32){
 let row=g.x;let rows=p.a.x;let k=p.a.z;let words=k/8u;var maximum=0.0;
 for(var j=l;j<k;j+=64u){maximum=max(maximum,abs(f32(x[row*k+j])));}
 maxima[l]=maximum;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){maxima[l]=max(maxima[l],maxima[l+stride]);}workgroupBarrier();}
 let scale=maxima[0]/7.0;if(l==0u){y[rows*words+row]=bitcast<u32>(scale);}
 var total=0;
 for(var j=l;j<words;j+=64u){var packed=0u;
  for(var t=0u;t<8u;t++){var q=0;if(scale>0.0){q=clamp(even(f32(x[row*k+j*8u+t])/scale),-7,7);}
   total+=q;packed|=u32(q+8)<<(4u*t);}
  y[row*words+j]=packed;
 }
 sums[l]=total;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){sums[l]+=sums[l+stride];}workgroupBarrier();}
 if(l==0u){y[rows*words+rows+row]=bitcast<u32>(sums[0]);}
}"#.into()
}

const UNPACK: &str = r#"
fn q4_bytes(word:u32, half:u32)->u32 {
 let v=word>>(half*16u);
 return ((v&15u)|((v&240u)<<4u)|((v&3840u)<<8u)|((v&61440u)<<12u))+2021161080u;
}
"#;

pub(super) fn expand() -> String {
    format!(r#"
struct Params {{ a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }}
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
{UNPACK}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {{
 let i=id.x+id.y*p.d.w;let codes=p.a.x*(p.a.z/4u);
 if(i>=codes+p.a.x*2u){{return;}}
 if(i<codes){{y[i]=q4_bytes(x[i/2u],i%2u);}}
 else{{y[i]=x[codes/2u+i-codes];}}
}}
"#)
}

pub(super) fn gemm(raw: bool) -> String {
    let source = int8::gemm(false, raw)
        .replace("bt[c*64u+r]=w[(g.x*64u+r)*k+first+c];",
                 "bt[c*64u+r]=q4_bytes(w[((g.x*64u+r)*k+first+c)/2u],(first+c)%2u);")
        .replace("w[n*k", "w[n*(k/2u)");
    source + UNPACK
}

pub(crate) async fn check() -> Result<serde_json::Value> {
    let mut checks=Vec::new();
    for mode in ["q4-fused","q4-buffered"] {
        let mut gpu=Gpu::checkpoint(mode,false).await?;
        for (m,n,k) in [(5,7,72),(3,4,64)] {
            let a:Vec<f32>=(0..m*k).map(|i|if i%k==0{127.0}else{((i*17%255) as i32-127) as f32}).collect();
            let b:Vec<half::f16>=(0..n*k).map(|i|half::f16::from_f32(if i%k==0{7.0}else{((i*11%15) as i32-7) as f32})).collect();
            let ab=gpu.device.buffer(&a,"Q4 check activations")?;
            let wb=gpu.device.bytes(bytemuck::cast_slice(&b),"Q4 check weights")?;
            let mut e=gpu.device.encoder();
            let qa=gpu.quantize_int8(&mut e,&ab,m,k,false)?;
            let qb=gpu.quantize_q4(&mut e,&wb,n,k)?;
            let packed=gpu.device.read_words(e,&qb,n*(k/8+2)).await?;
            for row in 0..n {
                let mut sum=0i32;
                for col in 0..k {
                    let expected=b[row*k+col].to_f32() as i32;
                    ensure!(((packed[row*k/8+col/8]>>(4*(col%8)))&15) as i32-8==expected,"Q4 code mismatch");sum+=expected;
                }
                ensure!(packed[n*k/8+row]==1f32.to_bits()&&packed[n*k/8+n+row] as i32==sum,"Q4 metadata mismatch");
            }
            let mut e=gpu.device.encoder();
            let weight=if mode=="q4-buffered"{gpu.expand_q4(&mut e,&qb,n,k)?}else{qb.clone()};
            let out=gpu.device.empty(m*n,"Q4 exact raw check")?;
            gpu.run(&mut e,"int8_raw",[&qa,&weight,&gpu.dummy,&out],&[m,n,k],0)?;
            let raw=gpu.device.read_words(e,&out,m*n).await?;
            for row in 0..m {for col in 0..n {
                let expected:i32=(0..k).map(|i|a[row*k+i] as i32*b[col*k+i].to_f32() as i32).sum();
                ensure!(raw[row*n+col] as i32==expected,"Q4 raw dot mismatch");
            }}
            checks.push(serde_json::json!({"mode":mode,"rows":m,"output":n,"input":k,"exact_raw_values":m*n,"validated_weight_codes":n*k}));
            ab.destroy();wb.destroy();qa.destroy();qb.destroy();out.destroy();gpu.discard_scratch();
        }
    }
    Ok(serde_json::json!({"passed":true,"reference":"Independent CPU signed dots and every quantized weight code, scale and sum","checks":checks}))
}
