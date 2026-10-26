enable f16;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
fn unpack_a(v:u32)->vec4<f16>{
 let lo=bitcast<vec2<f16>>(((v&0x00ff00ffu)^0x00800080u)|0x64006400u)-vec2<f16>(1152.0h);
 let hi=bitcast<vec2<f16>>((((v>>8u)&0x00ff00ffu)^0x00800080u)|0x64006400u)-vec2<f16>(1152.0h);
 return vec4<f16>(lo.x,hi.x,lo.y,hi.y);
}
fn nib2(word:u32,s:u32)->vec2<f16>{return bitcast<vec2<f16>>(((word>>s)&0x000f000fu)|0x64006400u)-vec2<f16>(1032.0h);}
fn nib4(word:u32)->vec4<f32>{return vec4<f32>((vec4<u32>(word)>>vec4<u32>(0u,4u,8u,12u))&vec4<u32>(15u))-vec4<f32>(8.0);}
var<workgroup> at:array<vec4<f16>,512>;var<workgroup> bt:array<vec4<f16>,1024>;var<workgroup> sb:array<f32,64>;var<workgroup> sa:array<f32,32>;
@compute @workgroup_size(16,8)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>){
 let l=local&vec3<u32>(15u,7u,0u);
 let m=p.a.x;let n=p.a.y;let groups=p.a.z/64u;let lane=l.y*16u+l.x;
 let row=g.y*32u+l.y*4u;let col=g.x*64u+l.x;
var t0_0=vec4<f32>(0.0);var t1_0=vec4<f32>(0.0);var t2_0=vec4<f32>(0.0);var t3_0=vec4<f32>(0.0);
for(var group=0u;group<groups;group++){
 for(var i=lane;i<512u;i+=128u){let r=i/16u;let c=i%16u;at[i]=vec4<f16>(0.0h);
  if(g.y*32u+r<m){at[i]=unpack_a(x[((g.y*32u+r)*groups+group)*17u+c]);}
 }
 for(var i=lane;i<1024u;i+=128u){let r=i/16u;let c=i%16u;bt[c*64u+r]=vec4<f16>(0.0h);
  if(g.x*64u+r<n){bt[c*64u+r]=unpack_a(w[((g.x*64u+r)*groups+group)*17u+c]);}
 }
 if(lane<32u){let i=lane;sa[i]=0.0;if(g.y*32u+i<m){sa[i]=bitcast<f32>(x[((g.y*32u+i)*groups+group)*17u+16u]);}}
 if(lane<64u){let i=lane;sb[i]=0.0;if(g.x*64u+i<n){sb[i]=bitcast<f32>(w[((g.x*64u+i)*groups+group)*17u+16u]);}}
 workgroupBarrier();
var d0_0=vec4<f32>(0.0);var d1_0=vec4<f32>(0.0);var d2_0=vec4<f32>(0.0);var d3_0=vec4<f32>(0.0);
for(var base=0u;base<16u;base+=8u){
{let j=base+0u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));
}
{let j=base+1u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));
}
{let j=base+2u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));
}
{let j=base+3u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));
}
{let j=base+4u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));
}
{let j=base+5u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));
}
{let j=base+6u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));
}
{let j=base+7u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));
}
}
let dsc0=vec4<f32>(sb[l.x+0u],sb[l.x+16u],sb[l.x+32u],sb[l.x+48u]);
t0_0=fma(d0_0,sa[l.y*4u+0u]*dsc0,t0_0);
t1_0=fma(d1_0,sa[l.y*4u+1u]*dsc0,t1_0);
t2_0=fma(d2_0,sa[l.y*4u+2u]*dsc0,t2_0);
t3_0=fma(d3_0,sa[l.y*4u+3u]*dsc0,t3_0);
workgroupBarrier();}
if(row+0u<m&&col+0u<n){y[(row+0u)*n+col+0u]=t0_0[0];}
if(row+0u<m&&col+16u<n){y[(row+0u)*n+col+16u]=t0_0[1];}
if(row+0u<m&&col+32u<n){y[(row+0u)*n+col+32u]=t0_0[2];}
if(row+0u<m&&col+48u<n){y[(row+0u)*n+col+48u]=t0_0[3];}
if(row+1u<m&&col+0u<n){y[(row+1u)*n+col+0u]=t1_0[0];}
if(row+1u<m&&col+16u<n){y[(row+1u)*n+col+16u]=t1_0[1];}
if(row+1u<m&&col+32u<n){y[(row+1u)*n+col+32u]=t1_0[2];}
if(row+1u<m&&col+48u<n){y[(row+1u)*n+col+48u]=t1_0[3];}
if(row+2u<m&&col+0u<n){y[(row+2u)*n+col+0u]=t2_0[0];}
if(row+2u<m&&col+16u<n){y[(row+2u)*n+col+16u]=t2_0[1];}
if(row+2u<m&&col+32u<n){y[(row+2u)*n+col+32u]=t2_0[2];}
if(row+2u<m&&col+48u<n){y[(row+2u)*n+col+48u]=t2_0[3];}
if(row+3u<m&&col+0u<n){y[(row+3u)*n+col+0u]=t3_0[0];}
if(row+3u<m&&col+16u<n){y[(row+3u)*n+col+16u]=t3_0[1];}
if(row+3u<m&&col+32u<n){y[(row+3u)*n+col+32u]=t3_0[2];}
if(row+3u<m&&col+48u<n){y[(row+3u)*n+col+48u]=t3_0[3];}
}
