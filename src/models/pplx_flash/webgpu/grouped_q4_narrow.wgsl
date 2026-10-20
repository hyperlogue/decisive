enable f16;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<vec4<f16>,512>;var<workgroup> bt:array<vec4<f16>,512>;
var<workgroup> sa:array<f32,32>;var<workgroup> sb:array<f32,32>;
fn unpack_a(v:u32)->vec4<f16>{
 return vec4<f16>(bitcast<vec4<i32>>(vec4<u32>(v)<<vec4<u32>(24u,16u,8u,0u))>>vec4<u32>(24u));
}
@compute @workgroup_size(8,32)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
 let m=p.a.x;let n=p.a.y;let groups=p.a.z/64u;let lane=l.y*8u+l.x;
 let row=g.y*32u+l.y*1u;let col=g.x*32u+l.x;
var t0_0=vec4<f32>(0.0);
for(var group=0u;group<groups;group++){
 for(var i=lane;i<512u;i+=256u){let r=i/16u;let c=i%16u;at[i]=vec4<f16>(0.0h);
  if(g.y*32u+r<m){at[i]=unpack_a(x[((g.y*32u+r)*groups+group)*17u+c]);}
 }
 for(var i=lane;i<256u;i+=256u){let r=i/8u;let c=i%8u;
  bt[c*2u*32u+r]=vec4<f16>(0.0h);bt[(c*2u+1u)*32u+r]=vec4<f16>(0.0h);
  if(g.x*32u+r<n){
   let word=w[((g.x*32u+r)*groups+group)*9u+c];
   let lo=vec4<f16>(vec4<i32>((vec4<u32>(word)>>vec4<u32>(0u,8u,16u,24u))&vec4<u32>(15u))-vec4<i32>(8));
   let hi=vec4<f16>(vec4<i32>((vec4<u32>(word)>>vec4<u32>(4u,12u,20u,28u))&vec4<u32>(15u))-vec4<i32>(8));
   bt[c*2u*32u+r]=vec4<f16>(lo.x,hi.x,lo.y,hi.y);
   bt[(c*2u+1u)*32u+r]=vec4<f16>(lo.z,hi.z,lo.w,hi.w);
  }
 }
 for(var i=lane;i<32u;i+=256u){sa[i]=0.0;if(g.y*32u+i<m){sa[i]=bitcast<f32>(x[((g.y*32u+i)*groups+group)*17u+16u]);}}
 for(var i=lane;i<32u;i+=256u){sb[i]=0.0;if(g.x*32u+i<n){sb[i]=bitcast<f32>(w[((g.x*32u+i)*groups+group)*9u+8u]);}}
 workgroupBarrier();
var d0_0=vec4<f32>(0.0);
for(var base=0u;base<16u;base+=2u){
{let j=base+0u;
let b0=vec4<f32>(bt[j*32u+l.x+0u]);let b1=vec4<f32>(bt[j*32u+l.x+8u]);let b2=vec4<f32>(bt[j*32u+l.x+16u]);let b3=vec4<f32>(bt[j*32u+l.x+24u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*1u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
}
{let j=base+1u;
let b0=vec4<f32>(bt[j*32u+l.x+0u]);let b1=vec4<f32>(bt[j*32u+l.x+8u]);let b2=vec4<f32>(bt[j*32u+l.x+16u]);let b3=vec4<f32>(bt[j*32u+l.x+24u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*1u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
}
}
let sc0=vec4<f32>(sb[l.x+0u],sb[l.x+8u],sb[l.x+16u],sb[l.x+24u]);
t0_0=fma(d0_0,sa[l.y*1u+0u]*sc0,t0_0);
workgroupBarrier();}
if(row+0u<m&&col+0u<n){y[(row+0u)*n+col+0u]=t0_0[0];}
if(row+0u<m&&col+8u<n){y[(row+0u)*n+col+8u]=t0_0[1];}
if(row+0u<m&&col+16u<n){y[(row+0u)*n+col+16u]=t0_0[2];}
if(row+0u<m&&col+24u<n){y[(row+0u)*n+col+24u]=t0_0[3];}
}
