enable f16;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<vec4<f16>,1024>;var<workgroup> bt:array<vec4<f16>,1024>;
var<workgroup> sa:array<f32,64>;var<workgroup> sb:array<f32,64>;
fn unpack_a(v:u32)->vec4<f16>{
 return vec4<f16>(bitcast<vec4<i32>>(vec4<u32>(v)<<vec4<u32>(24u,16u,8u,0u))>>vec4<u32>(24u));
}
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>){
 let l=local&vec3<u32>(15u);
 let m=p.a.x;let n=p.a.y;let groups=p.a.z/64u;let lane=l.y*16u+l.x;
 let row=g.y*64u+l.y*4u;let col=g.x*64u+l.x;
var total0=vec4<f32>(0.0);
var total1=vec4<f32>(0.0);
var total2=vec4<f32>(0.0);
var total3=vec4<f32>(0.0);
for(var group=0u;group<groups;group++){
 for(var i=lane;i<1024u;i+=256u){let r=i/16u;let c=i%16u;at[i]=vec4<f16>(0.0h);
  if(g.y*64u+r<m){at[i]=unpack_a(x[((g.y*64u+r)*groups+group)*17u+c]);}
 }
 for(var i=lane;i<512u;i+=256u){let r=i/8u;let c=i%8u;
  bt[c*2u*64u+r]=vec4<f16>(0.0h);bt[(c*2u+1u)*64u+r]=vec4<f16>(0.0h);
  if(g.x*64u+r<n){
   let word=w[((g.x*64u+r)*groups+group)*9u+c];
   let lo=vec4<f16>(vec4<i32>((vec4<u32>(word)>>vec4<u32>(0u,8u,16u,24u))&vec4<u32>(15u))-vec4<i32>(8));
   let hi=vec4<f16>(vec4<i32>((vec4<u32>(word)>>vec4<u32>(4u,12u,20u,28u))&vec4<u32>(15u))-vec4<i32>(8));
   bt[c*2u*64u+r]=vec4<f16>(lo.x,hi.x,lo.y,hi.y);
   bt[(c*2u+1u)*64u+r]=vec4<f16>(lo.z,hi.z,lo.w,hi.w);
  }
 }
 if(lane<64u){sa[lane]=0.0;if(g.y*64u+lane<m){sa[lane]=bitcast<f32>(x[((g.y*64u+lane)*groups+group)*17u+16u]);}}
 if(lane<64u){sb[lane]=0.0;if(g.x*64u+lane<n){sb[lane]=bitcast<f32>(w[((g.x*64u+lane)*groups+group)*9u+8u]);}}
 workgroupBarrier();
var dot0=vec4<f32>(0.0);
var dot1=vec4<f32>(0.0);
var dot2=vec4<f32>(0.0);
var dot3=vec4<f32>(0.0);
for(var base=0u;base<16u;base+=2u){{let j=base+0u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);
let b1=vec4<f32>(bt[j*64u+l.x+16u]);
let b2=vec4<f32>(bt[j*64u+l.x+32u]);
let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);dot0+=vec4<f32>(dot(a0,b0),dot(a0,b1),dot(a0,b2),dot(a0,b3));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);dot1+=vec4<f32>(dot(a1,b0),dot(a1,b1),dot(a1,b2),dot(a1,b3));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);dot2+=vec4<f32>(dot(a2,b0),dot(a2,b1),dot(a2,b2),dot(a2,b3));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);dot3+=vec4<f32>(dot(a3,b0),dot(a3,b1),dot(a3,b2),dot(a3,b3));
}
{let j=base+1u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);
let b1=vec4<f32>(bt[j*64u+l.x+16u]);
let b2=vec4<f32>(bt[j*64u+l.x+32u]);
let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);dot0+=vec4<f32>(dot(a0,b0),dot(a0,b1),dot(a0,b2),dot(a0,b3));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);dot1+=vec4<f32>(dot(a1,b0),dot(a1,b1),dot(a1,b2),dot(a1,b3));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);dot2+=vec4<f32>(dot(a2,b0),dot(a2,b1),dot(a2,b2),dot(a2,b3));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);dot3+=vec4<f32>(dot(a3,b0),dot(a3,b1),dot(a3,b2),dot(a3,b3));
}
}
let scales=vec4<f32>(sb[l.x+0u],sb[l.x+16u],sb[l.x+32u],sb[l.x+48u]);
total0=fma(dot0,sa[l.y*4u+0u]*scales,total0);
total1=fma(dot1,sa[l.y*4u+1u]*scales,total1);
total2=fma(dot2,sa[l.y*4u+2u]*scales,total2);
total3=fma(dot3,sa[l.y*4u+3u]*scales,total3);
workgroupBarrier();}
if(row+0u<m&&col+0u<n){y[(row+0u)*n+col+0u]=total0[0];}
if(row+0u<m&&col+16u<n){y[(row+0u)*n+col+16u]=total0[1];}
if(row+0u<m&&col+32u<n){y[(row+0u)*n+col+32u]=total0[2];}
if(row+0u<m&&col+48u<n){y[(row+0u)*n+col+48u]=total0[3];}
if(row+1u<m&&col+0u<n){y[(row+1u)*n+col+0u]=total1[0];}
if(row+1u<m&&col+16u<n){y[(row+1u)*n+col+16u]=total1[1];}
if(row+1u<m&&col+32u<n){y[(row+1u)*n+col+32u]=total1[2];}
if(row+1u<m&&col+48u<n){y[(row+1u)*n+col+48u]=total1[3];}
if(row+2u<m&&col+0u<n){y[(row+2u)*n+col+0u]=total2[0];}
if(row+2u<m&&col+16u<n){y[(row+2u)*n+col+16u]=total2[1];}
if(row+2u<m&&col+32u<n){y[(row+2u)*n+col+32u]=total2[2];}
if(row+2u<m&&col+48u<n){y[(row+2u)*n+col+48u]=total2[3];}
if(row+3u<m&&col+0u<n){y[(row+3u)*n+col+0u]=total3[0];}
if(row+3u<m&&col+16u<n){y[(row+3u)*n+col+16u]=total3[1];}
if(row+3u<m&&col+32u<n){y[(row+3u)*n+col+32u]=total3[2];}
if(row+3u<m&&col+48u<n){y[(row+3u)*n+col+48u]=total3[3];}
}