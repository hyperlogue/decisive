import assert from 'node:assert/strict';
import {readFile,writeFile,mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import vm from 'node:vm';

export async function checkDp4a(page) {
  const html=await readFile(page,'utf8');
  const context=vm.createContext({self:{postMessage(){}}});
  for(const match of html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)){
    const script=new vm.Script(match[2]);
    if(match[1].includes('id="probe"'))script.runInContext(context);
  }
  const p=context.PplxQuantizationProbe,d=context.PplxDp4aProbe;
  const input=Float32Array.from({length:3*64},(_,i)=>i<64?0:(i%255-127)/128);
  const packed=new Uint32Array(input.length/4+input.length/64);
  const scales=new Float32Array(packed.buffer,input.length,input.length/64);
  for(let g=0;g<input.length/64;g++){
    scales[g]=Math.max(...input.slice(g*64,g*64+64).map(Math.abs))/127;
    for(let j=0;j<64;j++){
      const i=g*64+j,q=scales[g]===0?0:Math.max(-127,Math.min(127,p.roundEven(Math.fround(input[i]/scales[g]))));
      packed[i>>>2]|=(q&255)<<((i&3)*8);
    }
  }
  assert.equal(d.code(Uint32Array.of(0x7f0180ff),0),-1);
  assert.equal(d.code(Uint32Array.of(0x7f0180ff),1),-128);
  assert.equal(d.code(Uint32Array.of(0x7f0180ff),2),1);
  assert.equal(d.code(Uint32Array.of(0x7f0180ff),3),127);
  assert.equal(d.checkActivation(input,packed).passed,true);
  assert.equal(d.checkActivation(input,packed).code_differences_from_cpu,0);
  const bad=packed.slice();bad[0]=127;
  bad[16]=0x80808080;
  assert.throws(()=>d.checkActivation(input,bad),/activation/);
  const weights=Uint16Array.from({length:34*64},(_,i)=>p.toBF16(((i*37)%251-125)/128));
  const w=p.pack(weights,64,8),actual=new Float32Array(3*34);
  for(let r=0;r<3;r++)for(let c=0;c<34;c++){
    let value=0;
    for(let k=0;k<64;k++)value+=d.code(packed,r*64+k)*scales[r]*p.unpack(w,c*64+k);
    actual[r*34+c]=value;
  }
  assert.equal(d.checkDot(actual,packed,w,3,64,34).passed,true);
  actual[0]=1;
  assert.throws(()=>d.checkDot(actual,packed,w,3,64,34),/CPU reference/);
  const core=context.PplxDp4aCore,primitive=context.PplxDp4aPrimitive;
  const k=68,m=3,n=34;
  const a=core.packRows(Float32Array.from({length:m*k},(_,i)=>(i%11-5)/5),k);
  const b=core.packRows(Float32Array.from({length:n*k},(_,i)=>(i%17-8)/8),k);
  const result=new Float32Array(m*n);
  for(let r=0;r<m;r++)for(let c=0;c<n;c++){
    let dot=0;for(let j=0;j<k;j++)dot+=d.code(a.data,r*k+j)*d.code(b.data,c*k+j);
    result[r*n+c]=Math.fround(Math.fround(Math.fround(dot)*a.scales[r])*b.scales[c]);
  }
  assert.equal(core.checkDot(result,a,b,m,k,n).passed,true);
  const au=a.data.map(v=>v^0x80808080),bu=b.data.map(v=>v^0x80808080);
  const asums=core.signedRowSums(a.data,k),bsums=core.signedRowSums(b.data,k);
  for(let r=0;r<m;r++)for(let c=0;c<n;c++){
    let u=0,signed=0;
    for(let j=0;j<k;j++){
      u+=((au[(r*k+j)>>>2]>>>(((r*k+j)&3)*8))&255)*((bu[(c*k+j)>>>2]>>>(((c*k+j)&3)*8))&255);
      signed+=d.code(a.data,r*k+j)*d.code(b.data,c*k+j);
    }
    assert.equal(u-128*(asums[r]+bsums[c])-k*16384,signed);
  }
  result[0]+=1;assert.throws(()=>core.checkDot(result,a,b,m,k,n),/CPU reference/);
  const zero=core.packRows(new Float32Array(8),4);
  assert.ok(zero.data.every(x=>x===0)&&zero.scales.every(x=>x===0));
  assert.throws(()=>core.packRows(Float32Array.of(Infinity,0,0,0),4),/Non-finite/);
  for(const Type of [Uint16Array,Uint32Array]){
    const original=Type.from({length:35},(_,i)=>i+17),transposed=core.transposeRows(original,5,7);
    assert.ok(transposed instanceof Type);
    for(let row=0;row<7;row++)for(let col=0;col<5;col++)assert.equal(transposed[col*7+row],original[row*5+col]);
    assert.deepEqual(core.transposeRows(transposed,7,5),original);
  }
  assert.throws(()=>core.transposeRows(new Uint32Array(4),3,2),/Transpose shape/);
  const pa=primitive.operands(35,137),pb=primitive.operands(35,1729);
  for(let index=0;index<35;index++)for(const steps of [1,2,17]){
    const av=Array.from({length:4},(_,j)=>d.code(pa.packed,index*4+j));
    const bv=Array.from({length:4},(_,j)=>d.code(pb.packed,index*4+j));let dot;
    for(let step=0;step<steps;step++){dot=av.reduce((sum,x,j)=>sum+x*bv[j],0);av.shift();av.push(dot>0?1:-1);}
    assert.equal(primitive.reference(pa.packed,pb.packed,index,steps),dot);
    for(let j=0;j<4;j++)assert.equal(p.fromF16(pa.half[index*4+j]),d.code(pa.packed,index*4+j));
  }
  const matrices=primitive.matrixOperands(35);
  for(let i=0;i<35;i++)for(const steps of [1,3,509]){
    let expected=Array.from(matrices.a.int32.slice(i*16,i*16+16));
    for(let step=0;step<steps;step++)expected=expected.map((_,at)=>{
      const c=at%4,r=Math.floor(at/4),column=matrices.b.int32.slice(i*16+c*4,i*16+c*4+4);
      const lane=Array.from(column).findIndex(x=>x!==0);return expected[r*4+lane]*column[lane]||0;
    });
    assert.deepEqual(Array.from(primitive.matrixReference(matrices.a.int32,matrices.b.int32,i,steps)),expected);
  }
  const directory=await mkdtemp(join(tmpdir(),'pplx-dp4a-shaders-'));
  await writeFile(join(directory,'quantize.wgsl'),d.QUANTIZE);
  await writeFile(join(directory,'dp4a.wgsl'),d.dotShader());
  for(const [name,bits,storage] of [['fp16',16,'fp16'],['q8',8,'q8']])
    await writeFile(join(directory,`${name}.wgsl`),vm.runInContext(`PplxQuantizationProbe.shader(KERNEL_SOURCE,${bits},'f16','${storage}')`,context));
  for(const tile of [32,64]){
    await writeFile(join(directory,`core-int8-${tile}.wgsl`),core.dotShader(tile));
    await writeFile(join(directory,`core-fp16-${tile}.wgsl`),vm.runInContext(`PplxDp4aCore.fp16Shader(KERNEL_SOURCE,${tile})`,context));
    for(const kind of ['int8','fp16'])for(const [transpose,pad] of [[true,0],[false,1],[true,1]]){
      const shader=vm.runInContext(`PplxDp4aCore.tunedShader(KERNEL_SOURCE,'${kind}',${tile},${transpose},${pad})`,context);
      await writeFile(join(directory,`layout-${kind}-${tile}-${transpose}-${pad}.wgsl`),shader);
    }
    for(const [transpose,pad] of [[false,0],[true,0],[false,1],[true,1]])await writeFile(join(directory,`raw-layout-${tile}-${transpose}-${pad}.wgsl`),vm.runInContext(`PplxDp4aCore.tunedShader(KERNEL_SOURCE,'int8',${tile},${transpose},${pad},true)`,context));
  }
  for(const kind of ['int8','fp16'])for(const chain of [false,true])await writeFile(join(directory,`primitive-${kind}-${chain}.wgsl`),primitive.shader(kind,chain));
  for(const arithmetic of ['builtin','reuse','outer','unsigned'])for(const rm of [4,8])for(const rn of [4,8]){
    await writeFile(join(directory,`register-${arithmetic}-${rm}-${rn}.wgsl`),core.registerShader({arithmetic,rm,rn}));
  }
  await writeFile(join(directory,'register-k128.wgsl'),core.registerShader({kw:32}));
  await writeFile(join(directory,'register-n128.wgsl'),core.registerShader({tn:128,rn:8}));
  for(const arithmetic of ['builtin','unsigned'])await writeFile(join(directory,`raw-${arithmetic}.wgsl`),core.registerShader({arithmetic,raw:true,rm:8}));
  for(const kind of ['fp16','int8','int32','int8-unpack-once','fp32'])await writeFile(join(directory,`arithmetic-${kind}.wgsl`),primitive.matrixShader(kind));
  return {transpose_roundtrip:true,prepacked_reference:true,primitive_reference:true,syntax:true,signed_packing:true,activation_quantization:true,integer_dot_reference:true,shaders:directory};
}
