enable f16;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<vec4<f16>,512>;var<workgroup> bt:array<vec4<f16>,1024>;
var<workgroup> sa:array<f32,32>;var<workgroup> sb:array<f32,64>;
// Bytes 0/2 and 1/3 as f16 1024+(b^128), minus 1152: the exact signed byte.
fn unpack_a(v:u32)->vec4<f16>{
 let lo=bitcast<vec2<f16>>(((v&0x00ff00ffu)^0x00800080u)|0x64006400u)-vec2<f16>(1152.0h);
 let hi=bitcast<vec2<f16>>((((v>>8u)&0x00ff00ffu)^0x00800080u)|0x64006400u)-vec2<f16>(1152.0h);
 return vec4<f16>(lo.x,hi.x,lo.y,hi.y);
}
// Nibbles i and i+4 as f16 1024+n, minus 1032: the exact signed code.
fn nib2(word:u32,s:u32)->vec2<f16>{return bitcast<vec2<f16>>(((word>>s)&0x000f000fu)|0x64006400u)-vec2<f16>(1032.0h);}
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
 for(var i=lane;i<512u;i+=128u){let r=i/8u;let c=i%8u;
  bt[c*2u*64u+r]=vec4<f16>(0.0h);bt[(c*2u+1u)*64u+r]=vec4<f16>(0.0h);
  if(g.x*64u+r<n){
   let word=w[((g.x*64u+r)*groups+group)*9u+c];
   let p0=nib2(word,0u);let p1=nib2(word,4u);let p2=nib2(word,8u);let p3=nib2(word,12u);
   bt[c*2u*64u+r]=vec4<f16>(p0.x,p1.x,p2.x,p3.x);
   bt[(c*2u+1u)*64u+r]=vec4<f16>(p0.y,p1.y,p2.y,p3.y);
  }
 }
 if(lane<32u){sa[lane]=0.0;if(g.y*32u+lane<m){sa[lane]=bitcast<f32>(x[((g.y*32u+lane)*groups+group)*17u+16u]);}}
 if(lane<64u){sb[lane]=0.0;if(g.x*64u+lane<n){sb[lane]=bitcast<f32>(w[((g.x*64u+lane)*groups+group)*9u+8u]);}}
 workgroupBarrier();
var d0_0=vec4<f32>(0.0);var d1_0=vec4<f32>(0.0);var d2_0=vec4<f32>(0.0);var d3_0=vec4<f32>(0.0);
for(var base=0u;base<16u;base+=16u){
{let j=base+0u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+1u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+2u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+3u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+4u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+5u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+6u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+7u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+8u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+9u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+10u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+11u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+12u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+13u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+14u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
{let j=base+15u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let b1=vec4<f32>(bt[j*64u+l.x+16u]);let b2=vec4<f32>(bt[j*64u+l.x+32u]);let b3=vec4<f32>(bt[j*64u+l.x+48u]);
let b0x=vec4<f32>(b0.x,b1.x,b2.x,b3.x);let b0y=vec4<f32>(b0.y,b1.y,b2.y,b3.y);let b0z=vec4<f32>(b0.z,b1.z,b2.z,b3.z);let b0w=vec4<f32>(b0.w,b1.w,b2.w,b3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),b0w,fma(vec4<f32>(a0.z),b0z,fma(vec4<f32>(a0.y),b0y,fma(vec4<f32>(a0.x),b0x,d0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),b0w,fma(vec4<f32>(a1.z),b0z,fma(vec4<f32>(a1.y),b0y,fma(vec4<f32>(a1.x),b0x,d1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),b0w,fma(vec4<f32>(a2.z),b0z,fma(vec4<f32>(a2.y),b0y,fma(vec4<f32>(a2.x),b0x,d2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),b0w,fma(vec4<f32>(a3.z),b0z,fma(vec4<f32>(a3.y),b0y,fma(vec4<f32>(a3.x),b0x,d3_0))));
}
}
let sc0=vec4<f32>(sb[l.x+0u],sb[l.x+16u],sb[l.x+32u],sb[l.x+48u]);
t0_0=fma(d0_0,sa[l.y*4u+0u]*sc0,t0_0);
t1_0=fma(d1_0,sa[l.y*4u+1u]*sc0,t1_0);
t2_0=fma(d2_0,sa[l.y*4u+2u]*sc0,t2_0);
t3_0=fma(d3_0,sa[l.y*4u+3u]*sc0,t3_0);
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
