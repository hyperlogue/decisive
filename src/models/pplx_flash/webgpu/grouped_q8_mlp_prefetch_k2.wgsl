enable f16;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(2) var<storage,read> u:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
fn unpack_a(v:u32)->vec4<f16>{
 let lo=bitcast<vec2<f16>>(((v&0x00ff00ffu)^0x00800080u)|0x64006400u)-vec2<f16>(1152.0h);
 let hi=bitcast<vec2<f16>>((((v>>8u)&0x00ff00ffu)^0x00800080u)|0x64006400u)-vec2<f16>(1152.0h);
 return vec4<f16>(lo.x,hi.x,lo.y,hi.y);
}
fn nib2(word:u32,s:u32)->vec2<f16>{return bitcast<vec2<f16>>(((word>>s)&0x000f000fu)|0x64006400u)-vec2<f16>(1032.0h);}
fn nib4(word:u32)->vec4<f32>{return vec4<f32>((vec4<u32>(word)>>vec4<u32>(0u,4u,8u,12u))&vec4<u32>(15u))-vec4<f32>(8.0);}
var<workgroup> at:array<vec4<f16>,1024>;var<workgroup> bt:array<vec4<f16>,1024>;var<workgroup> sb:array<f32,64>;var<workgroup> ut:array<vec4<f16>,1024>;var<workgroup> su:array<f32,64>;var<workgroup> sa:array<f32,64>;
var<workgroup> rmax:array<atomic<u32>,64>;var<workgroup> codes:array<atomic<u32>,1024>;
fn even(v:f32)->i32 {let lo=floor(v);let f=v-lo;return i32(lo)+select(0,1,f>0.5||(f==0.5&&(i32(lo)&1)==1));}
fn sigmoid(v:f32)->f32 { return 1.0/(1.0+exp(-v)); }
fn silu(v:f32)->f32 { return v*sigmoid(v); }
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>){
 let l=local&vec3<u32>(15u,15u,0u);
 let m=p.a.x;let n=p.a.y;let groups=p.a.z/64u;let lane=l.y*16u+l.x;
 let row=g.y*64u+l.y*4u;let col=g.x*64u+l.x;
 if(lane<64u){atomicStore(&rmax[lane],0u);}for(var i=lane;i<1024u;i+=256u){atomicStore(&codes[i],0u);}
var t0_0=vec4<f32>(0.0);var t1_0=vec4<f32>(0.0);var t2_0=vec4<f32>(0.0);var t3_0=vec4<f32>(0.0);var v0_0=vec4<f32>(0.0);var v1_0=vec4<f32>(0.0);var v2_0=vec4<f32>(0.0);var v3_0=vec4<f32>(0.0);
var ra0=0u;var ra1=0u;var ra2=0u;var ra3=0u;var rw0=0u;var rw1=0u;var rw2=0u;var rw3=0u;var ru0=0u;var ru1=0u;var ru2=0u;var ru3=0u;
 {let grp=0u;
 {let i=lane+0u;ra0=0u;if(g.y*64u+i/16u<m){ra0=x[((g.y*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+256u;ra1=0u;if(g.y*64u+i/16u<m){ra1=x[((g.y*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+512u;ra2=0u;if(g.y*64u+i/16u<m){ra2=x[((g.y*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+768u;ra3=0u;if(g.y*64u+i/16u<m){ra3=x[((g.y*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+0u;rw0=0u;if(g.x*64u+i/16u<n){rw0=w[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+256u;rw1=0u;if(g.x*64u+i/16u<n){rw1=w[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+512u;rw2=0u;if(g.x*64u+i/16u<n){rw2=w[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+768u;rw3=0u;if(g.x*64u+i/16u<n){rw3=w[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+0u;ru0=0u;if(g.x*64u+i/16u<n){ru0=u[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+256u;ru1=0u;if(g.x*64u+i/16u<n){ru1=u[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+512u;ru2=0u;if(g.x*64u+i/16u<n){ru2=u[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+768u;ru3=0u;if(g.x*64u+i/16u<n){ru3=u[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 }
for(var group=0u;group<groups;group++){
 {let i=lane+0u;at[i]=unpack_a(ra0);}
 {let i=lane+256u;at[i]=unpack_a(ra1);}
 {let i=lane+512u;at[i]=unpack_a(ra2);}
 {let i=lane+768u;at[i]=unpack_a(ra3);}
 {let i=lane+0u;bt[((i)%16u)*64u+(i)/16u]=unpack_a(rw0);}
 {let i=lane+256u;bt[((i)%16u)*64u+(i)/16u]=unpack_a(rw1);}
 {let i=lane+512u;bt[((i)%16u)*64u+(i)/16u]=unpack_a(rw2);}
 {let i=lane+768u;bt[((i)%16u)*64u+(i)/16u]=unpack_a(rw3);}
 {let i=lane+0u;ut[((i)%16u)*64u+(i)/16u]=unpack_a(ru0);}
 {let i=lane+256u;ut[((i)%16u)*64u+(i)/16u]=unpack_a(ru1);}
 {let i=lane+512u;ut[((i)%16u)*64u+(i)/16u]=unpack_a(ru2);}
 {let i=lane+768u;ut[((i)%16u)*64u+(i)/16u]=unpack_a(ru3);}
 if(lane<64u){let i=lane;sa[i]=0.0;if(g.y*64u+i<m){sa[i]=bitcast<f32>(x[((g.y*64u+i)*groups+group)*17u+16u]);}}
 if(lane<64u){let i=lane;sb[i]=0.0;if(g.x*64u+i<n){sb[i]=bitcast<f32>(w[((g.x*64u+i)*groups+group)*17u+16u]);}}
 if(lane<64u){let i=lane;su[i]=0.0;if(g.x*64u+i<n){su[i]=bitcast<f32>(u[((g.x*64u+i)*groups+group)*17u+16u]);}}
 workgroupBarrier();
 if(group+1u<groups) {let grp=group+1u;
 {let i=lane+0u;ra0=0u;if(g.y*64u+i/16u<m){ra0=x[((g.y*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+256u;ra1=0u;if(g.y*64u+i/16u<m){ra1=x[((g.y*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+512u;ra2=0u;if(g.y*64u+i/16u<m){ra2=x[((g.y*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+768u;ra3=0u;if(g.y*64u+i/16u<m){ra3=x[((g.y*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+0u;rw0=0u;if(g.x*64u+i/16u<n){rw0=w[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+256u;rw1=0u;if(g.x*64u+i/16u<n){rw1=w[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+512u;rw2=0u;if(g.x*64u+i/16u<n){rw2=w[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+768u;rw3=0u;if(g.x*64u+i/16u<n){rw3=w[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+0u;ru0=0u;if(g.x*64u+i/16u<n){ru0=u[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+256u;ru1=0u;if(g.x*64u+i/16u<n){ru1=u[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+512u;ru2=0u;if(g.x*64u+i/16u<n){ru2=u[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 {let i=lane+768u;ru3=0u;if(g.x*64u+i/16u<n){ru3=u[((g.x*64u+(i)/16u)*groups+grp)*17u+(i)%16u];}}
 }
var d0_0=vec4<f32>(0.0);var d1_0=vec4<f32>(0.0);var d2_0=vec4<f32>(0.0);var d3_0=vec4<f32>(0.0);var e0_0=vec4<f32>(0.0);var e1_0=vec4<f32>(0.0);var e2_0=vec4<f32>(0.0);var e3_0=vec4<f32>(0.0);
for(var base=0u;base<16u;base+=2u){
{let j=base+0u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let eb0=vec4<f32>(ut[j*64u+l.x+0u]);let eb1=vec4<f32>(ut[j*64u+l.x+16u]);let eb2=vec4<f32>(ut[j*64u+l.x+32u]);let eb3=vec4<f32>(ut[j*64u+l.x+48u]);
let eb0x=vec4<f32>(eb0.x,eb1.x,eb2.x,eb3.x);let eb0y=vec4<f32>(eb0.y,eb1.y,eb2.y,eb3.y);let eb0z=vec4<f32>(eb0.z,eb1.z,eb2.z,eb3.z);let eb0w=vec4<f32>(eb0.w,eb1.w,eb2.w,eb3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));e0_0=fma(vec4<f32>(a0.w),eb0w,fma(vec4<f32>(a0.z),eb0z,fma(vec4<f32>(a0.y),eb0y,fma(vec4<f32>(a0.x),eb0x,e0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));e1_0=fma(vec4<f32>(a1.w),eb0w,fma(vec4<f32>(a1.z),eb0z,fma(vec4<f32>(a1.y),eb0y,fma(vec4<f32>(a1.x),eb0x,e1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));e2_0=fma(vec4<f32>(a2.w),eb0w,fma(vec4<f32>(a2.z),eb0z,fma(vec4<f32>(a2.y),eb0y,fma(vec4<f32>(a2.x),eb0x,e2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));e3_0=fma(vec4<f32>(a3.w),eb0w,fma(vec4<f32>(a3.z),eb0z,fma(vec4<f32>(a3.y),eb0y,fma(vec4<f32>(a3.x),eb0x,e3_0))));
}
{let j=base+1u;
let db0=vec4<f32>(bt[j*64u+l.x+0u]);let db1=vec4<f32>(bt[j*64u+l.x+16u]);let db2=vec4<f32>(bt[j*64u+l.x+32u]);let db3=vec4<f32>(bt[j*64u+l.x+48u]);
let db0x=vec4<f32>(db0.x,db1.x,db2.x,db3.x);let db0y=vec4<f32>(db0.y,db1.y,db2.y,db3.y);let db0z=vec4<f32>(db0.z,db1.z,db2.z,db3.z);let db0w=vec4<f32>(db0.w,db1.w,db2.w,db3.w);
let eb0=vec4<f32>(ut[j*64u+l.x+0u]);let eb1=vec4<f32>(ut[j*64u+l.x+16u]);let eb2=vec4<f32>(ut[j*64u+l.x+32u]);let eb3=vec4<f32>(ut[j*64u+l.x+48u]);
let eb0x=vec4<f32>(eb0.x,eb1.x,eb2.x,eb3.x);let eb0y=vec4<f32>(eb0.y,eb1.y,eb2.y,eb3.y);let eb0z=vec4<f32>(eb0.z,eb1.z,eb2.z,eb3.z);let eb0w=vec4<f32>(eb0.w,eb1.w,eb2.w,eb3.w);
let a0=vec4<f32>(at[(l.y*4u+0u)*16u+j]);d0_0=fma(vec4<f32>(a0.w),db0w,fma(vec4<f32>(a0.z),db0z,fma(vec4<f32>(a0.y),db0y,fma(vec4<f32>(a0.x),db0x,d0_0))));e0_0=fma(vec4<f32>(a0.w),eb0w,fma(vec4<f32>(a0.z),eb0z,fma(vec4<f32>(a0.y),eb0y,fma(vec4<f32>(a0.x),eb0x,e0_0))));
let a1=vec4<f32>(at[(l.y*4u+1u)*16u+j]);d1_0=fma(vec4<f32>(a1.w),db0w,fma(vec4<f32>(a1.z),db0z,fma(vec4<f32>(a1.y),db0y,fma(vec4<f32>(a1.x),db0x,d1_0))));e1_0=fma(vec4<f32>(a1.w),eb0w,fma(vec4<f32>(a1.z),eb0z,fma(vec4<f32>(a1.y),eb0y,fma(vec4<f32>(a1.x),eb0x,e1_0))));
let a2=vec4<f32>(at[(l.y*4u+2u)*16u+j]);d2_0=fma(vec4<f32>(a2.w),db0w,fma(vec4<f32>(a2.z),db0z,fma(vec4<f32>(a2.y),db0y,fma(vec4<f32>(a2.x),db0x,d2_0))));e2_0=fma(vec4<f32>(a2.w),eb0w,fma(vec4<f32>(a2.z),eb0z,fma(vec4<f32>(a2.y),eb0y,fma(vec4<f32>(a2.x),eb0x,e2_0))));
let a3=vec4<f32>(at[(l.y*4u+3u)*16u+j]);d3_0=fma(vec4<f32>(a3.w),db0w,fma(vec4<f32>(a3.z),db0z,fma(vec4<f32>(a3.y),db0y,fma(vec4<f32>(a3.x),db0x,d3_0))));e3_0=fma(vec4<f32>(a3.w),eb0w,fma(vec4<f32>(a3.z),eb0z,fma(vec4<f32>(a3.y),eb0y,fma(vec4<f32>(a3.x),eb0x,e3_0))));
}
}
let dsc0=vec4<f32>(sb[l.x+0u],sb[l.x+16u],sb[l.x+32u],sb[l.x+48u]);
t0_0=fma(d0_0,sa[l.y*4u+0u]*dsc0,t0_0);
t1_0=fma(d1_0,sa[l.y*4u+1u]*dsc0,t1_0);
t2_0=fma(d2_0,sa[l.y*4u+2u]*dsc0,t2_0);
t3_0=fma(d3_0,sa[l.y*4u+3u]*dsc0,t3_0);
let esc0=vec4<f32>(su[l.x+0u],su[l.x+16u],su[l.x+32u],su[l.x+48u]);
v0_0=fma(e0_0,sa[l.y*4u+0u]*esc0,v0_0);
v1_0=fma(e1_0,sa[l.y*4u+1u]*esc0,v1_0);
v2_0=fma(e2_0,sa[l.y*4u+2u]*esc0,v2_0);
v3_0=fma(e3_0,sa[l.y*4u+3u]*esc0,v3_0);
workgroupBarrier();}
var q0=vec4<f32>(0.0);if(row+0u<m){q0=vec4<f32>(silu(t0_0[0])*v0_0[0],silu(t0_0[1])*v0_0[1],silu(t0_0[2])*v0_0[2],silu(t0_0[3])*v0_0[3]);}
let x0=abs(q0);atomicMax(&rmax[l.y*4u+0u],bitcast<u32>(max(max(x0.x,x0.y),max(x0.z,x0.w))));
var q1=vec4<f32>(0.0);if(row+1u<m){q1=vec4<f32>(silu(t1_0[0])*v1_0[0],silu(t1_0[1])*v1_0[1],silu(t1_0[2])*v1_0[2],silu(t1_0[3])*v1_0[3]);}
let x1=abs(q1);atomicMax(&rmax[l.y*4u+1u],bitcast<u32>(max(max(x1.x,x1.y),max(x1.z,x1.w))));
var q2=vec4<f32>(0.0);if(row+2u<m){q2=vec4<f32>(silu(t2_0[0])*v2_0[0],silu(t2_0[1])*v2_0[1],silu(t2_0[2])*v2_0[2],silu(t2_0[3])*v2_0[3]);}
let x2=abs(q2);atomicMax(&rmax[l.y*4u+2u],bitcast<u32>(max(max(x2.x,x2.y),max(x2.z,x2.w))));
var q3=vec4<f32>(0.0);if(row+3u<m){q3=vec4<f32>(silu(t3_0[0])*v3_0[0],silu(t3_0[1])*v3_0[1],silu(t3_0[2])*v3_0[2],silu(t3_0[3])*v3_0[3]);}
let x3=abs(q3);atomicMax(&rmax[l.y*4u+3u],bitcast<u32>(max(max(x3.x,x3.y),max(x3.z,x3.w))));
workgroupBarrier();
{let scale=max(bitcast<f32>(atomicLoad(&rmax[l.y*4u+0u]))/127.0,1.1754943508222875e-38);let inverse=1.0/scale;
for(var c=0u;c<4u;c++){let q=clamp(even(q0[c]*inverse),-127,127);let gc=l.x+16u*c;atomicOr(&codes[(l.y*4u+0u)*16u+gc/4u],(u32(q)&255u)<<(8u*(gc%4u)));}
if(l.x==0u&&row+0u<m){y[((row+0u)*(n/64u)+g.x)*17u+16u]=bitcast<u32>(scale);}}
{let scale=max(bitcast<f32>(atomicLoad(&rmax[l.y*4u+1u]))/127.0,1.1754943508222875e-38);let inverse=1.0/scale;
for(var c=0u;c<4u;c++){let q=clamp(even(q1[c]*inverse),-127,127);let gc=l.x+16u*c;atomicOr(&codes[(l.y*4u+1u)*16u+gc/4u],(u32(q)&255u)<<(8u*(gc%4u)));}
if(l.x==0u&&row+1u<m){y[((row+1u)*(n/64u)+g.x)*17u+16u]=bitcast<u32>(scale);}}
{let scale=max(bitcast<f32>(atomicLoad(&rmax[l.y*4u+2u]))/127.0,1.1754943508222875e-38);let inverse=1.0/scale;
for(var c=0u;c<4u;c++){let q=clamp(even(q2[c]*inverse),-127,127);let gc=l.x+16u*c;atomicOr(&codes[(l.y*4u+2u)*16u+gc/4u],(u32(q)&255u)<<(8u*(gc%4u)));}
if(l.x==0u&&row+2u<m){y[((row+2u)*(n/64u)+g.x)*17u+16u]=bitcast<u32>(scale);}}
{let scale=max(bitcast<f32>(atomicLoad(&rmax[l.y*4u+3u]))/127.0,1.1754943508222875e-38);let inverse=1.0/scale;
for(var c=0u;c<4u;c++){let q=clamp(even(q3[c]*inverse),-127,127);let gc=l.x+16u*c;atomicOr(&codes[(l.y*4u+3u)*16u+gc/4u],(u32(q)&255u)<<(8u*(gc%4u)));}
if(l.x==0u&&row+3u<m){y[((row+3u)*(n/64u)+g.x)*17u+16u]=bitcast<u32>(scale);}}
workgroupBarrier();
for(var i=lane;i<1024u;i+=256u){let r=i/16u;if(g.y*64u+r<m){y[((g.y*64u+r)*(n/64u)+g.x)*17u+i%16u]=atomicLoad(&codes[i]);}}
}
