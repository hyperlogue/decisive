enable f16;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(2) var<storage,read> u:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<vec4<f16>,1024>;var<workgroup> bt:array<vec4<f16>,1024>;var<workgroup> ut:array<vec4<f16>,1024>;
var<workgroup> sa:array<f32,64>;var<workgroup> sb:array<f32,64>;var<workgroup> su:array<f32,64>;
var<workgroup> rmax:array<atomic<u32>,64>;var<workgroup> codes:array<atomic<u32>,1024>;
fn even(v:f32)->i32 {let lo=floor(v);let f=v-lo;return i32(lo)+select(0,1,f>0.5||(f==0.5&&(i32(lo)&1)==1));}
fn sigmoid(v:f32)->f32 { return 1.0/(1.0+exp(-v)); }
fn silu(v:f32)->f32 { return v*sigmoid(v); }
fn unpack_a(v:u32)->vec4<f16>{
 return vec4<f16>(bitcast<vec4<i32>>(vec4<u32>(v)<<vec4<u32>(24u,16u,8u,0u))>>vec4<u32>(24u));
}
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>){
 let l=local&vec3<u32>(15u);
 let m=p.a.x;let n=p.a.y;let groups=p.a.z/64u;let lane=l.y*16u+l.x;
 let row=g.y*64u+l.y*4u;let col=g.x*64u+l.x;
 if(lane<64u){atomicStore(&rmax[lane],0u);}for(var i=lane;i<1024u;i+=256u){atomicStore(&codes[i],0u);}
var total0=vec4<f32>(0.0);var totalu0=vec4<f32>(0.0);
var total1=vec4<f32>(0.0);var totalu1=vec4<f32>(0.0);
var total2=vec4<f32>(0.0);var totalu2=vec4<f32>(0.0);
var total3=vec4<f32>(0.0);var totalu3=vec4<f32>(0.0);
for(var group=0u;group<groups;group++){
 for(var i=lane;i<1024u;i+=256u){let r=i/16u;let c=i%16u;at[i]=vec4<f16>(0.0h);
  if(g.y*64u+r<m){at[i]=unpack_a(x[((g.y*64u+r)*groups+group)*17u+c]);}
 }
 for(var i=lane;i<512u;i+=256u){let r=i/8u;let c=i%8u;
  bt[c*2u*64u+r]=vec4<f16>(0.0h);bt[(c*2u+1u)*64u+r]=vec4<f16>(0.0h);ut[c*2u*64u+r]=vec4<f16>(0.0h);ut[(c*2u+1u)*64u+r]=vec4<f16>(0.0h);
  if(g.x*64u+r<n){
   let o=((g.x*64u+r)*groups+group)*9u+c;
   let word=w[o];
   let lo=vec4<f16>(vec4<i32>((vec4<u32>(word)>>vec4<u32>(0u,8u,16u,24u))&vec4<u32>(15u))-vec4<i32>(8));
   let hi=vec4<f16>(vec4<i32>((vec4<u32>(word)>>vec4<u32>(4u,12u,20u,28u))&vec4<u32>(15u))-vec4<i32>(8));
   bt[c*2u*64u+r]=vec4<f16>(lo.x,hi.x,lo.y,hi.y);
   bt[(c*2u+1u)*64u+r]=vec4<f16>(lo.z,hi.z,lo.w,hi.w);
   let uword=u[o];
   let ulo=vec4<f16>(vec4<i32>((vec4<u32>(uword)>>vec4<u32>(0u,8u,16u,24u))&vec4<u32>(15u))-vec4<i32>(8));
   let uhi=vec4<f16>(vec4<i32>((vec4<u32>(uword)>>vec4<u32>(4u,12u,20u,28u))&vec4<u32>(15u))-vec4<i32>(8));
   ut[c*2u*64u+r]=vec4<f16>(ulo.x,uhi.x,ulo.y,uhi.y);
   ut[(c*2u+1u)*64u+r]=vec4<f16>(ulo.z,uhi.z,ulo.w,uhi.w);
  }
 }
 if(lane<64u){sa[lane]=0.0;if(g.y*64u+lane<m){sa[lane]=bitcast<f32>(x[((g.y*64u+lane)*groups+group)*17u+16u]);}}
 if(lane<64u){sb[lane]=0.0;su[lane]=0.0;if(g.x*64u+lane<n){let o=((g.x*64u+lane)*groups+group)*9u+8u;sb[lane]=bitcast<f32>(w[o]);su[lane]=bitcast<f32>(u[o]);}}
 workgroupBarrier();
var dot0=vec4<f32>(0.0);var dotu0=vec4<f32>(0.0);
var dot1=vec4<f32>(0.0);var dotu1=vec4<f32>(0.0);
var dot2=vec4<f32>(0.0);var dotu2=vec4<f32>(0.0);
var dot3=vec4<f32>(0.0);var dotu3=vec4<f32>(0.0);
for(var base=0u;base<16u;base+=2u){{let j=base+0u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let c0=vec4<f32>(ut[j*64u+l.x+0u]);
let b1=vec4<f32>(bt[j*64u+l.x+16u]);let c1=vec4<f32>(ut[j*64u+l.x+16u]);
let b2=vec4<f32>(bt[j*64u+l.x+32u]);let c2=vec4<f32>(ut[j*64u+l.x+32u]);
let b3=vec4<f32>(bt[j*64u+l.x+48u]);let c3=vec4<f32>(ut[j*64u+l.x+48u]);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);dot0=fma(vec4<f32>(a0.w),vec4<f32>(b0.w,b1.w,b2.w,b3.w),fma(vec4<f32>(a0.z),vec4<f32>(b0.z,b1.z,b2.z,b3.z),fma(vec4<f32>(a0.y),vec4<f32>(b0.y,b1.y,b2.y,b3.y),fma(vec4<f32>(a0.x),vec4<f32>(b0.x,b1.x,b2.x,b3.x),dot0))));dotu0=fma(vec4<f32>(a0.w),vec4<f32>(c0.w,c1.w,c2.w,c3.w),fma(vec4<f32>(a0.z),vec4<f32>(c0.z,c1.z,c2.z,c3.z),fma(vec4<f32>(a0.y),vec4<f32>(c0.y,c1.y,c2.y,c3.y),fma(vec4<f32>(a0.x),vec4<f32>(c0.x,c1.x,c2.x,c3.x),dotu0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);dot1=fma(vec4<f32>(a1.w),vec4<f32>(b0.w,b1.w,b2.w,b3.w),fma(vec4<f32>(a1.z),vec4<f32>(b0.z,b1.z,b2.z,b3.z),fma(vec4<f32>(a1.y),vec4<f32>(b0.y,b1.y,b2.y,b3.y),fma(vec4<f32>(a1.x),vec4<f32>(b0.x,b1.x,b2.x,b3.x),dot1))));dotu1=fma(vec4<f32>(a1.w),vec4<f32>(c0.w,c1.w,c2.w,c3.w),fma(vec4<f32>(a1.z),vec4<f32>(c0.z,c1.z,c2.z,c3.z),fma(vec4<f32>(a1.y),vec4<f32>(c0.y,c1.y,c2.y,c3.y),fma(vec4<f32>(a1.x),vec4<f32>(c0.x,c1.x,c2.x,c3.x),dotu1))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);dot2=fma(vec4<f32>(a2.w),vec4<f32>(b0.w,b1.w,b2.w,b3.w),fma(vec4<f32>(a2.z),vec4<f32>(b0.z,b1.z,b2.z,b3.z),fma(vec4<f32>(a2.y),vec4<f32>(b0.y,b1.y,b2.y,b3.y),fma(vec4<f32>(a2.x),vec4<f32>(b0.x,b1.x,b2.x,b3.x),dot2))));dotu2=fma(vec4<f32>(a2.w),vec4<f32>(c0.w,c1.w,c2.w,c3.w),fma(vec4<f32>(a2.z),vec4<f32>(c0.z,c1.z,c2.z,c3.z),fma(vec4<f32>(a2.y),vec4<f32>(c0.y,c1.y,c2.y,c3.y),fma(vec4<f32>(a2.x),vec4<f32>(c0.x,c1.x,c2.x,c3.x),dotu2))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);dot3=fma(vec4<f32>(a3.w),vec4<f32>(b0.w,b1.w,b2.w,b3.w),fma(vec4<f32>(a3.z),vec4<f32>(b0.z,b1.z,b2.z,b3.z),fma(vec4<f32>(a3.y),vec4<f32>(b0.y,b1.y,b2.y,b3.y),fma(vec4<f32>(a3.x),vec4<f32>(b0.x,b1.x,b2.x,b3.x),dot3))));dotu3=fma(vec4<f32>(a3.w),vec4<f32>(c0.w,c1.w,c2.w,c3.w),fma(vec4<f32>(a3.z),vec4<f32>(c0.z,c1.z,c2.z,c3.z),fma(vec4<f32>(a3.y),vec4<f32>(c0.y,c1.y,c2.y,c3.y),fma(vec4<f32>(a3.x),vec4<f32>(c0.x,c1.x,c2.x,c3.x),dotu3))));
}
{let j=base+1u;
let b0=vec4<f32>(bt[j*64u+l.x+0u]);let c0=vec4<f32>(ut[j*64u+l.x+0u]);
let b1=vec4<f32>(bt[j*64u+l.x+16u]);let c1=vec4<f32>(ut[j*64u+l.x+16u]);
let b2=vec4<f32>(bt[j*64u+l.x+32u]);let c2=vec4<f32>(ut[j*64u+l.x+32u]);
let b3=vec4<f32>(bt[j*64u+l.x+48u]);let c3=vec4<f32>(ut[j*64u+l.x+48u]);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);dot0=fma(vec4<f32>(a0.w),vec4<f32>(b0.w,b1.w,b2.w,b3.w),fma(vec4<f32>(a0.z),vec4<f32>(b0.z,b1.z,b2.z,b3.z),fma(vec4<f32>(a0.y),vec4<f32>(b0.y,b1.y,b2.y,b3.y),fma(vec4<f32>(a0.x),vec4<f32>(b0.x,b1.x,b2.x,b3.x),dot0))));dotu0=fma(vec4<f32>(a0.w),vec4<f32>(c0.w,c1.w,c2.w,c3.w),fma(vec4<f32>(a0.z),vec4<f32>(c0.z,c1.z,c2.z,c3.z),fma(vec4<f32>(a0.y),vec4<f32>(c0.y,c1.y,c2.y,c3.y),fma(vec4<f32>(a0.x),vec4<f32>(c0.x,c1.x,c2.x,c3.x),dotu0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);dot1=fma(vec4<f32>(a1.w),vec4<f32>(b0.w,b1.w,b2.w,b3.w),fma(vec4<f32>(a1.z),vec4<f32>(b0.z,b1.z,b2.z,b3.z),fma(vec4<f32>(a1.y),vec4<f32>(b0.y,b1.y,b2.y,b3.y),fma(vec4<f32>(a1.x),vec4<f32>(b0.x,b1.x,b2.x,b3.x),dot1))));dotu1=fma(vec4<f32>(a1.w),vec4<f32>(c0.w,c1.w,c2.w,c3.w),fma(vec4<f32>(a1.z),vec4<f32>(c0.z,c1.z,c2.z,c3.z),fma(vec4<f32>(a1.y),vec4<f32>(c0.y,c1.y,c2.y,c3.y),fma(vec4<f32>(a1.x),vec4<f32>(c0.x,c1.x,c2.x,c3.x),dotu1))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);dot2=fma(vec4<f32>(a2.w),vec4<f32>(b0.w,b1.w,b2.w,b3.w),fma(vec4<f32>(a2.z),vec4<f32>(b0.z,b1.z,b2.z,b3.z),fma(vec4<f32>(a2.y),vec4<f32>(b0.y,b1.y,b2.y,b3.y),fma(vec4<f32>(a2.x),vec4<f32>(b0.x,b1.x,b2.x,b3.x),dot2))));dotu2=fma(vec4<f32>(a2.w),vec4<f32>(c0.w,c1.w,c2.w,c3.w),fma(vec4<f32>(a2.z),vec4<f32>(c0.z,c1.z,c2.z,c3.z),fma(vec4<f32>(a2.y),vec4<f32>(c0.y,c1.y,c2.y,c3.y),fma(vec4<f32>(a2.x),vec4<f32>(c0.x,c1.x,c2.x,c3.x),dotu2))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);dot3=fma(vec4<f32>(a3.w),vec4<f32>(b0.w,b1.w,b2.w,b3.w),fma(vec4<f32>(a3.z),vec4<f32>(b0.z,b1.z,b2.z,b3.z),fma(vec4<f32>(a3.y),vec4<f32>(b0.y,b1.y,b2.y,b3.y),fma(vec4<f32>(a3.x),vec4<f32>(b0.x,b1.x,b2.x,b3.x),dot3))));dotu3=fma(vec4<f32>(a3.w),vec4<f32>(c0.w,c1.w,c2.w,c3.w),fma(vec4<f32>(a3.z),vec4<f32>(c0.z,c1.z,c2.z,c3.z),fma(vec4<f32>(a3.y),vec4<f32>(c0.y,c1.y,c2.y,c3.y),fma(vec4<f32>(a3.x),vec4<f32>(c0.x,c1.x,c2.x,c3.x),dotu3))));
}
}
let scales=vec4<f32>(sb[l.x+0u],sb[l.x+16u],sb[l.x+32u],sb[l.x+48u]);
let scalesu=vec4<f32>(su[l.x+0u],su[l.x+16u],su[l.x+32u],su[l.x+48u]);
total0=fma(dot0,sa[l.y*4u+0u]*scales,total0);totalu0=fma(dotu0,sa[l.y*4u+0u]*scalesu,totalu0);
total1=fma(dot1,sa[l.y*4u+1u]*scales,total1);totalu1=fma(dotu1,sa[l.y*4u+1u]*scalesu,totalu1);
total2=fma(dot2,sa[l.y*4u+2u]*scales,total2);totalu2=fma(dotu2,sa[l.y*4u+2u]*scalesu,totalu2);
total3=fma(dot3,sa[l.y*4u+3u]*scales,total3);totalu3=fma(dotu3,sa[l.y*4u+3u]*scalesu,totalu3);
workgroupBarrier();}
var v0=vec4<f32>(0.0);if(row+0u<m){v0=vec4<f32>(silu(total0[0])*totalu0[0],silu(total0[1])*totalu0[1],silu(total0[2])*totalu0[2],silu(total0[3])*totalu0[3]);}
let x0=abs(v0);atomicMax(&rmax[l.y*4u+0u],bitcast<u32>(max(max(x0.x,x0.y),max(x0.z,x0.w))));
var v1=vec4<f32>(0.0);if(row+1u<m){v1=vec4<f32>(silu(total1[0])*totalu1[0],silu(total1[1])*totalu1[1],silu(total1[2])*totalu1[2],silu(total1[3])*totalu1[3]);}
let x1=abs(v1);atomicMax(&rmax[l.y*4u+1u],bitcast<u32>(max(max(x1.x,x1.y),max(x1.z,x1.w))));
var v2=vec4<f32>(0.0);if(row+2u<m){v2=vec4<f32>(silu(total2[0])*totalu2[0],silu(total2[1])*totalu2[1],silu(total2[2])*totalu2[2],silu(total2[3])*totalu2[3]);}
let x2=abs(v2);atomicMax(&rmax[l.y*4u+2u],bitcast<u32>(max(max(x2.x,x2.y),max(x2.z,x2.w))));
var v3=vec4<f32>(0.0);if(row+3u<m){v3=vec4<f32>(silu(total3[0])*totalu3[0],silu(total3[1])*totalu3[1],silu(total3[2])*totalu3[2],silu(total3[3])*totalu3[3]);}
let x3=abs(v3);atomicMax(&rmax[l.y*4u+3u],bitcast<u32>(max(max(x3.x,x3.y),max(x3.z,x3.w))));
workgroupBarrier();
{let scale=max(bitcast<f32>(atomicLoad(&rmax[l.y*4u+0u]))/127.0,1.1754943508222875e-38);let inverse=1.0/scale;
for(var c=0u;c<4u;c++){let q=clamp(even(v0[c]*inverse),-127,127);let gc=l.x+16u*c;atomicOr(&codes[(l.y*4u+0u)*16u+gc/4u],(u32(q)&255u)<<(8u*(gc%4u)));}
if(l.x==0u&&row+0u<m){y[((row+0u)*(n/64u)+g.x)*17u+16u]=bitcast<u32>(scale);}}
{let scale=max(bitcast<f32>(atomicLoad(&rmax[l.y*4u+1u]))/127.0,1.1754943508222875e-38);let inverse=1.0/scale;
for(var c=0u;c<4u;c++){let q=clamp(even(v1[c]*inverse),-127,127);let gc=l.x+16u*c;atomicOr(&codes[(l.y*4u+1u)*16u+gc/4u],(u32(q)&255u)<<(8u*(gc%4u)));}
if(l.x==0u&&row+1u<m){y[((row+1u)*(n/64u)+g.x)*17u+16u]=bitcast<u32>(scale);}}
{let scale=max(bitcast<f32>(atomicLoad(&rmax[l.y*4u+2u]))/127.0,1.1754943508222875e-38);let inverse=1.0/scale;
for(var c=0u;c<4u;c++){let q=clamp(even(v2[c]*inverse),-127,127);let gc=l.x+16u*c;atomicOr(&codes[(l.y*4u+2u)*16u+gc/4u],(u32(q)&255u)<<(8u*(gc%4u)));}
if(l.x==0u&&row+2u<m){y[((row+2u)*(n/64u)+g.x)*17u+16u]=bitcast<u32>(scale);}}
{let scale=max(bitcast<f32>(atomicLoad(&rmax[l.y*4u+3u]))/127.0,1.1754943508222875e-38);let inverse=1.0/scale;
for(var c=0u;c<4u;c++){let q=clamp(even(v3[c]*inverse),-127,127);let gc=l.x+16u*c;atomicOr(&codes[(l.y*4u+3u)*16u+gc/4u],(u32(q)&255u)<<(8u*(gc%4u)));}
if(l.x==0u&&row+3u<m){y[((row+3u)*(n/64u)+g.x)*17u+16u]=bitcast<u32>(scale);}}
workgroupBarrier();
for(var i=lane;i<1024u;i+=256u){let r=i/16u;if(g.y*64u+r<m){y[((g.y*64u+r)*(n/64u)+g.x)*17u+i%16u]=atomicLoad(&codes[i]);}}
}