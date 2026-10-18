globalThis.PplxDp4aProbe = (() => {
  const P = PplxQuantizationProbe;
  const require = (ok, message) => {if (!ok) throw Error(message);};
  const QUANTIZE = `
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> maxima:array<f32,16>;
fn even(v:f32)->i32 {let lo=floor(v);let f=v-lo;return i32(lo)+select(0,1,f>0.5||(f==0.5&&(i32(lo)&1)==1));}
@compute @workgroup_size(16)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) l:u32) {
 let group=g.x+g.y*p.d.w;
 if(group>=p.a.x*p.a.z/64u){return;}
 let at=group*64u+l*4u;
 let v=vec4<f32>(x[at],x[at+1u],x[at+2u],x[at+3u]);
 let ab=abs(v);maxima[l]=max(max(ab.x,ab.y),max(ab.z,ab.w));workgroupBarrier();
 for(var stride=8u;stride>0u;stride/=2u){if(l<stride){maxima[l]=max(maxima[l],maxima[l+stride]);}workgroupBarrier();}
 let scale=maxima[0]/127.0;
 if(l==0u){y[p.a.x*p.a.z/4u+group]=bitcast<u32>(scale);}
 var q=vec4<i32>(0);
 if(scale>0.0){q=clamp(vec4<i32>(even(v.x/scale),even(v.y/scale),even(v.z/scale),even(v.w/scale)),vec4<i32>(-127),vec4<i32>(127));}
 y[group*16u+l]=(u32(q.x)&255u)|((u32(q.y)&255u)<<8u)|((u32(q.z)&255u)<<16u)|((u32(q.w)&255u)<<24u);
}`;
  function dotShader() {
    let body=`requires packed_4x8_integer_dot_product;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(2) var<storage,read> scales:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<u32,512>;var<workgroup> bt:array<u32,512>;
var<workgroup> ascale:array<f32,32>;var<workgroup> bscale:array<f32,32>;
@compute @workgroup_size(8,8)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>) {
 let m=p.a.x;let n=p.a.y;let k=p.a.z;let groups=k/64u;let lane=l.y*8u+l.x;
 let row=g.y*32u+l.y*4u;let col=g.x*32u+l.x*4u;
 var s0=vec4<f32>(0.0);var s1=s0;var s2=s0;var s3=s0;
 for(var group=0u;group<groups;group++){
  for(var j=lane;j<512u;j+=64u){
   let ar=j/16u;let ac=j%16u;let br=j/32u;let bc=j%32u;
   at[j]=0u;bt[j]=0u;
   if(g.y*32u+ar<m){at[j]=x[(g.y*32u+ar)*(k/4u)+group*16u+ac];}
   if(g.x*32u+bc<n){bt[j]=w[(g.x*32u+bc)*(k/4u)+group*16u+br];}
  }
  if(lane<32u){
   ascale[lane]=0.0;bscale[lane]=0.0;
   if(g.y*32u+lane<m){ascale[lane]=bitcast<f32>(x[m*k/4u+(g.y*32u+lane)*groups+group]);}
   if(g.x*32u+lane<n){bscale[lane]=scales[(g.x*32u+lane)*groups+group];}
  }
  workgroupBarrier();
  var t0=vec4<i32>(0);var t1=t0;var t2=t0;var t3=t0;
  for(var j=0u;j<16u;j++){
   let i=j*32u+l.x*4u;let v=vec4<u32>(bt[i],bt[i+1u],bt[i+2u],bt[i+3u]);
`;
    for(let r=0;r<4;r++)body+=`let a${r}=at[(l.y*4u+${r}u)*16u+j];t${r}+=vec4<i32>(dot4I8Packed(a${r},v.x),dot4I8Packed(a${r},v.y),dot4I8Packed(a${r},v.z),dot4I8Packed(a${r},v.w));\n`;
    body+=`}let bs=vec4<f32>(bscale[l.x*4u],bscale[l.x*4u+1u],bscale[l.x*4u+2u],bscale[l.x*4u+3u]);\n`;
    for(let r=0;r<4;r++)body+=`s${r}=fma(vec4<f32>(t${r}),vec4<f32>(ascale[l.y*4u+${r}u])*bs,s${r});\n`;
    body+=`workgroupBarrier();}\nfor(var c=0u;c<4u;c++){if(col+c<n){`;
    for(let r=0;r<4;r++)body+=`if(row+${r}u<m){y[(row+${r}u)*n+col+c]=s${r}[c];}\n`;
    return body+`}}}`;
  }
  const code = (words,i) => (words[i>>>2] << (24-(i&3)*8)) >> 24;
  function checkActivation(input, words) {
    const groups=input.length/64, offset=input.length/4;
    const scales=new Float32Array(words.buffer,words.byteOffset+offset*4,groups);
    let maxError=0,mismatches=0;
    for(let group=0;group<groups;group++){
      let max=0;for(let j=0;j<64;j++)max=Math.max(max,Math.abs(input[group*64+j]));
      const scale=scales[group], expected=Math.fround(max/127);
      require(Number.isFinite(scale)&&Math.abs(scale-expected)<=1e-6*Math.max(expected,1e-30),'Invalid activation scale');
      for(let j=0;j<64;j++){
        const i=group*64+j,q=code(words,i),error=Math.abs(q*scale-input[i]);
        require(Math.abs(q)<=127&&error<=scale*.5001+1e-7,'Invalid activation quantization');
        const ref=expected===0?0:Math.max(-127,Math.min(127,P.roundEven(Math.fround(input[i]/expected))));
        if(q!==ref)mismatches++;
        maxError=Math.max(maxError,error);
      }
    }
    return {passed:true,values:input.length,max_absolute_error:maxError,code_differences_from_cpu:mismatches};
  }
  function checkDot(actual, words, weight, rows, k, n) {
    const groups=k/64, scales=new Float32Array(words.buffer,words.byteOffset+rows*k,rows*groups);
    let maxError=0;
    for(let s=0;s<32;s++){
      const row=s===31?rows-1:s*37%rows,col=s===31?n-1:s*1009%n;
      let expected=0;
      for(let group=0;group<groups;group++){
        let dot=0;
        for(let j=0;j<64;j++){
          const i=group*64+j, q=(weight.data[col*k+i]<<24)>>24;
          dot+=code(words,row*k+i)*q;
        }
        expected=Math.fround(expected+dot*Math.fround(scales[row*groups+group]*weight.scales[col*groups+group]));
      }
      const err=Math.abs(actual[row*n+col]-expected);maxError=Math.max(maxError,err);
      require(Number.isFinite(err)&&err<=1e-4+2e-5*Math.abs(expected),'Packed-dot CPU reference mismatch');
    }
    return {passed:true,samples:32,max_absolute_error:maxError,reference:'Exact INT8 dot per group64, FP32 scales and outer sums'};
  }
  async function run(source, provenance, progress=()=>{}) {
    const report={schema_version:1,experiment:'pplx-dp4a-v1',completed:false,started_at:new Date().toISOString(),source:provenance,cases:[],
      methodology:{baseline:'Native FP16 storage / FP16 tile FMA / FP32 outer sums',candidate:'W8A8 group64, symmetric [-127,127], FP32 scales, dot4I8Packed, INT32 dot sums and FP32 scaled outer sums',
        primary:'Resident weights; GPU span including activation quantization and GEMM; synchronized wall time also recorded',warmups:3,rounds:9,
        controls:'Q8 weights dequantized to FP16; cached-activation W8A8 GEMM excludes quantization and is diagnostic only',
        scope:'Synthetic actual-size MLP projections, not full model or trained-model quality; no checkpoint downloads'}};
    const started=performance.now();let device;
    try{
      require(globalThis.isSecureContext&&navigator.gpu,'WebGPU unavailable');
      report.wgsl_features=[...(navigator.gpu.wgslLanguageFeatures||[])].sort();
      require(report.wgsl_features.includes('packed_4x8_integer_dot_product'),'Packed integer dot products unavailable in this browser');
      const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});require(adapter,'No adapter');
      const info=adapter.info||{};
      report.adapter={vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,is_fallback_adapter:info.isFallbackAdapter??adapter.isFallbackAdapter??null};
      report.software_gpu=report.adapter.is_fallback_adapter===true||/swiftshader|llvmpipe|software/i.test(JSON.stringify(report.adapter));
      report.features=[...adapter.features].sort();
      require(adapter.features.has('shader-f16'),'FP16 baseline unavailable');
      const timestamp=adapter.features.has('timestamp-query');report.gpu_timestamps=timestamp;
      device=await adapter.requestDevice({requiredFeatures:['shader-f16',...(timestamp?['timestamp-query']:[])]});
      let deviceError;device.addEventListener('uncapturederror',e=>{deviceError=e.error.message;});
      device.lost.then(i=>{if(i.reason!=='destroyed')deviceError=i.message||'Device lost';});
      const layout=device.createBindGroupLayout({entries:Array.from({length:5},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===4?'uniform':binding===3?'storage':'read-only-storage'}}))});
      const pl=device.createPipelineLayout({bindGroupLayouts:[layout]});const pipelines={};
      for(const [id,shader] of Object.entries({quantize:QUANTIZE,dp4a:dotShader(),fp16:P.shader(source,16,'f16','fp16'),q8:P.shader(source,8,'f16')})){
        progress({message:`Compiling ${id}…`});const module=device.createShaderModule({code:shader});
        const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');require(!errors.length,errors.map(e=>e.message).join('\n'));
        pipelines[id]=await device.createComputePipelineAsync({layout:pl,compute:{module,entryPoint:'main'}});
      }
      const create=(data,usage)=>{const b=device.createBuffer({size:data.byteLength,usage,mappedAtCreation:true});new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();return b;};
      const query=timestamp?device.createQuerySet({type:'timestamp',count:4}):null;
      const resolved=timestamp?device.createBuffer({size:32,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}):null;
      const queryRead=timestamp?device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}):null;
      const read=async buffer=>{const stage=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,stage,0,buffer.size);device.queue.submit([e.finish()]);await stage.mapAsync(GPUMapMode.READ);return stage.getMappedRange().slice(0);}finally{stage.unmap();stage.destroy();}};
      for(const [name,k,n] of [['MLP gate/up',2560,9216],['MLP down',9216,2560]]){
        progress({message:`Preparing ${name} weights…`});const weights=P.weightsFor(k,n),half=P.packF16(weights),q8=P.pack(weights,k,8);
        const hw=create(half.data,GPUBufferUsage.STORAGE),qw=create(q8.data,GPUBufferUsage.STORAGE),qs=create(q8.scales,GPUBufferUsage.STORAGE),dummy=create(new Uint32Array(1),GPUBufferUsage.STORAGE);
        try{for(const rows of [117,229,1024,2048]){
          progress({message:`Checking ${name}, ${rows} tokens…`});const input=P.inputsFor(rows,k);input.fill(0,0,64);
          const xb=create(input,GPUBufferUsage.STORAGE);
          const packed=device.createBuffer({size:rows*k+rows*k/64*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
          const output=device.createBuffer({size:rows*n*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
          const params=new Uint32Array(16);params.set([rows,n,k]);params[15]=device.limits.maxComputeWorkgroupsPerDimension;
          const uniform=create(params,GPUBufferUsage.UNIFORM);
          const bind=(x,w,scale,y)=>device.createBindGroup({layout,entries:[x,w,scale,y,uniform].map((buffer,binding)=>({binding,resource:{buffer}}))});
          const groups={quantize:bind(xb,dummy,dummy,packed),fp16:bind(xb,hw,dummy,output),q8:bind(xb,qw,qs,output),dp4a:bind(packed,qw,qs,output)};
          const variants=[{id:'fp16',pipeline:'fp16'},{id:'q8-f16',pipeline:'q8'},{id:'w8a8-total',pipeline:'dp4a',quantize:true},{id:'w8a8-core',pipeline:'dp4a'}];
          const sample=async variant=>{
            const start=performance.now(),e=device.createCommandEncoder();
            const pass=(id,index,gx,gy)=>{const p=e.beginComputePass(timestamp?{timestampWrites:{querySet:query,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}}:{});p.setPipeline(pipelines[id]);p.setBindGroup(0,groups[id]);p.dispatchWorkgroups(gx,gy);p.end();};
            if(variant.quantize){const count=rows*k/64;pass('quantize',0,Math.min(count,params[15]),Math.ceil(count/params[15]));}
            const index=variant.quantize?2:0;pass(variant.pipeline,index,Math.ceil(n/32),Math.ceil(rows/32));
            if(timestamp){e.resolveQuerySet(query,0,index+2,resolved,0);e.copyBufferToBuffer(resolved,0,queryRead,0,(index+2)*8);}
            device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();const wall=performance.now()-start;require(!deviceError,deviceError);
            let gpu=null,quantize=null,gemm=null;
            if(timestamp){await queryRead.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(queryRead.getMappedRange());gpu=Number(t[index+1]-t[0])/1e6;gemm=Number(t[index+1]-t[index])/1e6;quantize=variant.quantize?Number(t[1]-t[0])/1e6:0;queryRead.unmap();require(gpu>0&&gemm>0,'Invalid GPU timestamp');}
            return {wall,gpu,quantize,gemm};
          };
          const record={name,rows,input:k,output:n,variants:[]};
          try{
            let baseline,packedWords,dp4aOutput;
            for(const v of variants){
              await sample(v);const actual=new Float32Array(await read(output));let check;
              if(v.quantize){packedWords=new Uint32Array(await read(packed));record.activation_check=checkActivation(input,packedWords);dp4aOutput=actual;}
              if(v.pipeline==='dp4a')check=checkDot(actual,packedWords,q8,rows,k,n);
              else check=P.checkSamples(actual,input,v.pipeline==='fp16'?half:q8,rows,k,n,'f16');
              if(v.id==='fp16')baseline=actual;
              if(v.id==='w8a8-core')require(P.compareOutput(actual,dp4aOutput).max_absolute_error===0,'Cached activation result differs');
              record.variants.push({id:v.id,kernel_check:check,synthetic_drift_from_fp16:P.compareOutput(actual,baseline),wall:[],gpu:[],quantize:[],gemm:[]});
            }
            progress({message:`Timing ${name}, ${rows} tokens: 3 warmups + 9 rounds…`});
            for(let round=-3;round<9;round++){
              const offset=(round+3)%variants.length,order=[...variants.slice(offset),...variants.slice(0,offset)];if(round%2===0)order.reverse();
              for(const v of order){const result=await sample(v);if(round>=0){const entry=record.variants.find(e=>e.id===v.id);for(const key of ['wall','gpu','quantize','gemm'])if(result[key]!==null)entry[key].push(result[key]);}}
            }
            for(const v of record.variants){for(const key of ['wall','gpu','quantize','gemm'])v[key]=v[key].length?(v[key].every(x=>x===0)?{samples_ms:v[key],median_ms:0,relative_mad:0}:P.statistics(v[key])):null;}
            const base=record.variants[0],candidate=record.variants.find(v=>v.id==='w8a8-total');
            record.dp4a_speedup_including_quantization={gpu:timestamp?base.gpu.median_ms/candidate.gpu.median_ms:null,wall:base.wall.median_ms/candidate.wall.median_ms};
            report.cases.push(record);progress({message:`Finished ${name}, ${rows} tokens`,record});
          }finally{for(const b of [xb,packed,output,uniform])b.destroy();}
        }}finally{for(const b of [hw,qw,qs,dummy])b.destroy();}
      }
      report.completed=true;
    }catch(e){report.error=String(e);}
    finally{device?.destroy();report.elapsed_ms=performance.now()-started;}
    return report;
  }
  return {QUANTIZE,dotShader,code,checkActivation,checkDot,run};
})();
