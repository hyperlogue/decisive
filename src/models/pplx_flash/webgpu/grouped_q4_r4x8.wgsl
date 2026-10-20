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
@compute @workgroup_size(8,16)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>){
 let l=local&vec3<u32>(7u,15u,0u);
 let m=p.a.x;let n=p.a.y;let groups=p.a.z/64u;let lane=l.y*8u+l.x;
 let row=g.y*64u+l.y*4u;let col=g.x*64u+l.x;
var t0_0=vec4<f32>(0.0);var t0_1=vec4<f32>(0.0);var t1_0=vec4<f32>(0.0);var t1_1=vec4<f32>(0.0);var t2_0=vec4<f32>(0.0);var t2_1=vec4<f32>(0.0);var t3_0=vec4<f32>(0.0);var t3_1=vec4<f32>(0.0);
for(var group=0u;group<groups;group++){
 for(var i=lane;i<1024u;i+=128u){let r=i/16u;let c=i%16u;at[i]=vec4<f16>(0.0h);
  if(g.y*64u+r<m){at[i]=unpack_a(x[((g.y*64u+r)*groups+group)*17u+c]);}
 }
 for(var i=lane;i<512u;i+=128u){let r=i/8u;let c=i%8u;
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
var d0_0=vec4<f32>(0.0);var d0_1=vec4<f32>(0.0);var d1_0=vec4<f32>(0.0);var d1_1=vec4<f32>(0.0);var d2_0=vec4<f32>(0.0);var d2_1=vec4<f32>(0.0);var d3_0=vec4<f32>(0.0);var d3_1=vec4<f32>(0.0);
for(var base=0u;base<16u;base+=2u){
{let j=base+0u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+8u]);let b2=vec4<f32>(bt[j*64u+l.x+16u]);let b3=vec4<f32>(bt[j*64u+l.x+24u]);let b4=vec4<f32>(bt[j*64u+l.x+32u]);let b5=vec4<f32>(bt[j*64u+l.x+40u]);let b6=vec4<f32>(bt[j*64u+l.x+48u]);let b7=vec4<f32>(bt[j*64u+l.x+56u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);let b1x=vec4<f32>(b4.x,b5.x,b6.x,b7.x);let b1y=vec4<f32>(b4.y,b5.y,b6.y,b7.y);let b1z=vec4<f32>(b4.z,b5.z,b6.z,b7.z);let b1w=vec4<f32>(b4.w,b5.w,b6.w,b7.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));d0_1=fma(vec4<f32>(a0.w),b1w,fma(vec4<f32>(a0.z),b1z,fma(vec4<f32>(a0.y),b1y,fma(vec4<f32>(a0.x),b1x,d0_1))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));d1_1=fma(vec4<f32>(a1.w),b1w,fma(vec4<f32>(a1.z),b1z,fma(vec4<f32>(a1.y),b1y,fma(vec4<f32>(a1.x),b1x,d1_1))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));d2_1=fma(vec4<f32>(a2.w),b1w,fma(vec4<f32>(a2.z),b1z,fma(vec4<f32>(a2.y),b1y,fma(vec4<f32>(a2.x),b1x,d2_1))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));d3_1=fma(vec4<f32>(a3.w),b1w,fma(vec4<f32>(a3.z),b1z,fma(vec4<f32>(a3.y),b1y,fma(vec4<f32>(a3.x),b1x,d3_1))));
}
{let j=base+1u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+8u]);let b2=vec4<f32>(bt[j*64u+l.x+16u]);let b3=vec4<f32>(bt[j*64u+l.x+24u]);let b4=vec4<f32>(bt[j*64u+l.x+32u]);let b5=vec4<f32>(bt[j*64u+l.x+40u]);let b6=vec4<f32>(bt[j*64u+l.x+48u]);let b7=vec4<f32>(bt[j*64u+l.x+56u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);let b1x=vec4<f32>(b4.x,b5.x,b6.x,b7.x);let b1y=vec4<f32>(b4.y,b5.y,b6.y,b7.y);let b1z=vec4<f32>(b4.z,b5.z,b6.z,b7.z);let b1w=vec4<f32>(b4.w,b5.w,b6.w,b7.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));d0_1=fma(vec4<f32>(a0.w),b1w,fma(vec4<f32>(a0.z),b1z,fma(vec4<f32>(a0.y),b1y,fma(vec4<f32>(a0.x),b1x,d0_1))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));d1_1=fma(vec4<f32>(a1.w),b1w,fma(vec4<f32>(a1.z),b1z,fma(vec4<f32>(a1.y),b1y,fma(vec4<f32>(a1.x),b1x,d1_1))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));d2_1=fma(vec4<f32>(a2.w),b1w,fma(vec4<f32>(a2.z),b1z,fma(vec4<f32>(a2.y),b1y,fma(vec4<f32>(a2.x),b1x,d2_1))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));d3_1=fma(vec4<f32>(a3.w),b1w,fma(vec4<f32>(a3.z),b1z,fma(vec4<f32>(a3.y),b1y,fma(vec4<f32>(a3.x),b1x,d3_1))));
}
}
let sc0=vec4<f32>(sb[l.x+0u],sb[l.x+8u],sb[l.x+16u],sb[l.x+24u]);
let sc1=vec4<f32>(sb[l.x+32u],sb[l.x+40u],sb[l.x+48u],sb[l.x+56u]);
t0_0=fma(d0_0,sa[l.y*4u+0u]*sc0,t0_0);
t0_1=fma(d0_1,sa[l.y*4u+0u]*sc1,t0_1);
t1_0=fma(d1_0,sa[l.y*4u+1u]*sc0,t1_0);
t1_1=fma(d1_1,sa[l.y*4u+1u]*sc1,t1_1);
t2_0=fma(d2_0,sa[l.y*4u+2u]*sc0,t2_0);
t2_1=fma(d2_1,sa[l.y*4u+2u]*sc1,t2_1);
t3_0=fma(d3_0,sa[l.y*4u+3u]*sc0,t3_0);
t3_1=fma(d3_1,sa[l.y*4u+3u]*sc1,t3_1);
workgroupBarrier();}
if(row+0u<m&&col+0u<n){y[(row+0u)*n+col+0u]=t0_0[0];}
if(row+0u<m&&col+8u<n){y[(row+0u)*n+col+8u]=t0_0[1];}
if(row+0u<m&&col+16u<n){y[(row+0u)*n+col+16u]=t0_0[2];}
if(row+0u<m&&col+24u<n){y[(row+0u)*n+col+24u]=t0_0[3];}
if(row+0u<m&&col+32u<n){y[(row+0u)*n+col+32u]=t0_1[0];}
if(row+0u<m&&col+40u<n){y[(row+0u)*n+col+40u]=t0_1[1];}
if(row+0u<m&&col+48u<n){y[(row+0u)*n+col+48u]=t0_1[2];}
if(row+0u<m&&col+56u<n){y[(row+0u)*n+col+56u]=t0_1[3];}
if(row+1u<m&&col+0u<n){y[(row+1u)*n+col+0u]=t1_0[0];}
if(row+1u<m&&col+8u<n){y[(row+1u)*n+col+8u]=t1_0[1];}
if(row+1u<m&&col+16u<n){y[(row+1u)*n+col+16u]=t1_0[2];}
if(row+1u<m&&col+24u<n){y[(row+1u)*n+col+24u]=t1_0[3];}
if(row+1u<m&&col+32u<n){y[(row+1u)*n+col+32u]=t1_1[0];}
if(row+1u<m&&col+40u<n){y[(row+1u)*n+col+40u]=t1_1[1];}
if(row+1u<m&&col+48u<n){y[(row+1u)*n+col+48u]=t1_1[2];}
if(row+1u<m&&col+56u<n){y[(row+1u)*n+col+56u]=t1_1[3];}
if(row+2u<m&&col+0u<n){y[(row+2u)*n+col+0u]=t2_0[0];}
if(row+2u<m&&col+8u<n){y[(row+2u)*n+col+8u]=t2_0[1];}
if(row+2u<m&&col+16u<n){y[(row+2u)*n+col+16u]=t2_0[2];}
if(row+2u<m&&col+24u<n){y[(row+2u)*n+col+24u]=t2_0[3];}
if(row+2u<m&&col+32u<n){y[(row+2u)*n+col+32u]=t2_1[0];}
if(row+2u<m&&col+40u<n){y[(row+2u)*n+col+40u]=t2_1[1];}
if(row+2u<m&&col+48u<n){y[(row+2u)*n+col+48u]=t2_1[2];}
if(row+2u<m&&col+56u<n){y[(row+2u)*n+col+56u]=t2_1[3];}
if(row+3u<m&&col+0u<n){y[(row+3u)*n+col+0u]=t3_0[0];}
if(row+3u<m&&col+8u<n){y[(row+3u)*n+col+8u]=t3_0[1];}
if(row+3u<m&&col+16u<n){y[(row+3u)*n+col+16u]=t3_0[2];}
if(row+3u<m&&col+24u<n){y[(row+3u)*n+col+24u]=t3_0[3];}
if(row+3u<m&&col+32u<n){y[(row+3u)*n+col+32u]=t3_1[0];}
if(row+3u<m&&col+40u<n){y[(row+3u)*n+col+40u]=t3_1[1];}
if(row+3u<m&&col+48u<n){y[(row+3u)*n+col+48u]=t3_1[2];}
if(row+3u<m&&col+56u<n){y[(row+3u)*n+col+56u]=t3_1[3];}
}
