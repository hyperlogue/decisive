globalThis.PplxDp4aPrimitive = (() => {
  const P=PplxQuantizationProbe,D=PplxDp4aProbe;
  const require=(ok,message)=>{if(!ok)throw Error(message);};
  function shader(kind,chain,workgroup=64){
    const integer=kind==='int8',type=integer?'u32':'vec4<f16>';
    return `${integer?'requires packed_4x8_integer_dot_product;':'enable f16;'}
struct Params { count:u32, steps:u32, unused:vec2<u32> }
@group(0) @binding(0) var<storage,read> x:array<${type}>;
@group(0) @binding(1) var<storage,read> w:array<${type}>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(${workgroup})
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=p.count){return;}
 var a=x[i];let b=w[i];
 ${chain?`var result=${integer?'0':'0.0h'};
 for(var step=0u;step<p.steps;step++){
  result=${integer?'dot4I8Packed(a,b)':'dot(a,b)'};
  ${integer?'let next=select(255u,1u,result>0);a=(a>>8u)|(next<<24u);':'a=vec4<f16>(a.y,a.z,a.w,select(-1.0h,1.0h,result>0.0h));'}
 }
 y[i]=f32(result);`:`y[i]=f32(${integer?'dot4I8Packed(a,b)':'dot(a,b)'});`}
}`;
  }
  function operands(count,seed){
    const packed=new Uint32Array(count),half=new Uint16Array(count*4);
    for(let i=0;i<count*4;i++){
      let h=(i^seed)>>>0;h=Math.imul(h^(h>>>16),0x7feb352d);h=Math.imul(h^(h>>>15),0x846ca68b);h^=h>>>16;
      const v=(h&1)?1:-1;packed[i>>>2]|=(v&255)<<((i&3)*8);half[i]=(v===1?0x3c00:0xbc00);
    }
    return {packed,half};
  }
  function reference(a,b,index,steps){
    let x=a[index];const w=b[index];let result=0;
    for(let step=0;step<steps;step++){
      result=0;for(let j=0;j<4;j++)result+=((x<<(24-j*8))>>24)*((w<<(24-j*8))>>24);
      x=(x>>>8)|((result>0?1:255)<<24);
    }
    return result;
  }
  function matrixOperands(count){
    const a=new Int32Array(count*16),b=new Int32Array(count*16);
    const hash=x=>{let h=x>>>0;h=Math.imul(h^(h>>>16),0x7feb352d);h=Math.imul(h^(h>>>15),0x846ca68b);return (h^(h>>>16))>>>0;};
    for(let i=0;i<a.length;i++)a[i]=hash(i^137)%255-127;
    for(let i=0;i<count;i++){const h=hash(i^1729),shift=h&3;for(let c=0;c<4;c++)b[i*16+c*4+((c+shift)&3)]=(h&(1<<(c+2)))?1:-1;}
    const prepare=data=>{const packed=new Uint32Array(data.length/4);for(let j=0;j<data.length;j++)packed[j>>>2]|=(data[j]&255)<<((j&3)*8);return {int32:data,fp32:Float32Array.from(data),packed,half:Uint16Array.from(data,P.toF16)};};
    return {a:prepare(a),b:prepare(b)};
  }
  function matrixReference(a,b,index,steps){
    let state=Array.from(a.slice(index*16,index*16+16));
    for(let step=0;step<steps;step++){
      const next=new Array(16).fill(0);
      for(let r=0;r<4;r++)for(let c=0;c<4;c++)for(let k=0;k<4;k++)next[r*4+c]+=state[r*4+k]*b[index*16+c*4+k];
      state=next;
    }
    return state;
  }
  function matrixShader(kind,workgroup=64){
    const packed=kind==='int8',unpackOnce=kind==='int8-unpack-once',half=kind==='fp16';
    const floating=half||kind==='fp32';
    const scalar=half?'f16':kind==='fp32'?'f32':'i32',storage=packed||unpackOnce?'u32':`vec4<${scalar}>`;
    let s=`${half?'enable f16;':'requires packed_4x8_integer_dot_product;'}
struct Params { count:u32, steps:u32, unused:vec2<u32> }
@group(0) @binding(0) var<storage,read> x:array<${storage}>;
@group(0) @binding(1) var<storage,read> w:array<${storage}>;
@group(0) @binding(2) var<storage,read_write> y:array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p:Params;
fn idot(a:vec4<i32>,b:vec4<i32>)->i32{let q=a*b;return q.x+q.y+q.z+q.w;}
@compute @workgroup_size(${workgroup})
fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=p.count){return;}
`;
    for(let r=0;r<4;r++)for(const [name,source,mutable] of [['a','x',true],['b','w',false]])s+=`${mutable?'var':'let'} ${name}${r}=${unpackOnce?'unpack4xI8(':''}${source}[i*4u+${r}u]${unpackOnce?')':''};\n`;
    s+='for(var step=0u;step<p.steps;step++){\n';
    for(let r=0;r<4;r++){
      const dots=[0,1,2,3].map(c=>`${packed?'dot4I8Packed':floating?'dot':'idot'}(a${r},b${c})`).join(',');
      s+=`let n${r}=${packed?'pack4xI8(':''}vec4<${scalar}>(${dots})${packed?')':''};\n`;
    }
    for(let r=0;r<4;r++)s+=`a${r}=n${r};\n`;
    s+='}\n';for(let r=0;r<4;r++)s+=`y[i*4u+${r}u]=vec4<f32>(${packed?'unpack4xI8(':''}a${r}${packed?')':''});\n`;
    return s+'}';
  }
  async function run(source,provenance,progress=()=>{},options={}){
    const arithmetic=options.arithmetic===true,rounds=arithmetic?20:12;
    const report={schema_version:1,experiment:'pplx-dp4a-v1',mode:arithmetic?'register-arithmetic':'primitive',completed:false,started_at:new Date().toISOString(),source:provenance,cases:[],
      methodology:{warmups:3,rounds:12,baseline:'dot(vec4<f16>, vec4<f16>), FP16 result converted to FP32 output',candidate:'dot4I8Packed(u32,u32), signed INT8 lanes and INT32 result converted to FP32 output',
        inputs:'Identical deterministic {-1,+1} elements, exactly representable in both types; uploaded before timing',
        primary:'GPU timestamp around repeated untiled dispatches in one pass; repetition counts recorded per case to exceed timestamp granularity',
        independent:'One dot per invocation; includes global loads and stores, so this is throughput, not isolated instruction latency',
        dependent:'Each dot changes the next input through a sign test and shift/shuffle. No invariant dot can be hoisted; timing includes these dependency-maintenance operations.',
        excluded:'Quantization, packing, FP16 conversion, upload, output readback and CPU validation',
        interpretation:'Per-dot values are normalized kernel throughput. One-invocation chain gives time per dependent iteration, not pure hardware instruction latency.'}};
    if(arithmetic)report.methodology={warmups:3,rounds,inputs:'Identical signed INT8 integers [-127,127], exactly representable in FP16. Runtime-supplied signed-permutation B keeps the repeated A*B result in range.',primary:'4x4 matrix recurrence entirely in registers, 16 four-element dots per iteration. Every result feeds the next iteration; no global loads, stores or barriers in loop.',controls:'FP16, FP32 and unpacked INT32 have no packing or conversion in loop. int8-unpack-once uses packed input but expands before loop; int8 uses dot4I8Packed and repacks each new state.',excluded:'Initial conversion, packing, uploads and readback',interpretation:'Measures compiled arithmetic recurrence throughput including loop control. Not native instruction latency or proof about all INT8 hardware.'};
    const started=performance.now();let device;
    try{
      require(globalThis.isSecureContext&&navigator.gpu,'WebGPU unavailable');report.wgsl_features=[...(navigator.gpu.wgslLanguageFeatures||[])].sort();
      require(report.wgsl_features.includes('packed_4x8_integer_dot_product'),'Packed integer dot products unavailable');
      const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});require(adapter,'No adapter');
      const i=adapter.info||{};report.adapter={vendor:i.vendor,architecture:i.architecture,device:i.device,description:i.description,is_fallback_adapter:i.isFallbackAdapter??adapter.isFallbackAdapter??null};
      report.features=[...adapter.features].sort();require(adapter.features.has('shader-f16'),'FP16 unavailable');
      const timestamps=adapter.features.has('timestamp-query');report.gpu_timestamps=timestamps;
      device=await adapter.requestDevice({requiredFeatures:['shader-f16',...(timestamps?['timestamp-query']:[])]});
      let deviceError;device.addEventListener('uncapturederror',e=>deviceError=e.error.message);
      const layout=device.createBindGroupLayout({entries:[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===3?'uniform':binding===2?'storage':'read-only-storage'}}))});
      const pl=device.createPipelineLayout({bindGroupLayouts:[layout]});
      const create=(data,usage)=>{const b=device.createBuffer({size:data.byteLength,usage,mappedAtCreation:true});new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();return b;};
      const query=timestamps?device.createQuerySet({type:'timestamp',count:2}):null;
      const resolved=timestamps?device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}):null;
      const queryRead=timestamps?device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}):null;
      const specs=arithmetic?[
        {name:'Register matrix, one invocation',count:1,steps:32767,workgroup:1,repetitions:8},
        {name:'Register matrices, 509 steps',count:4096,steps:509,workgroup:64,repetitions:128},
        {name:'Register matrices, 2036 steps',count:4096,steps:2036,workgroup:64,repetitions:32},
      ]:[{name:'Independent dots',count:1048576,steps:1,chain:false,workgroup:64,repetitions:32},{name:'Dependent chain, one invocation',count:1,steps:65536,chain:true,workgroup:1,repetitions:4},{name:'Dependent chains, many invocations',count:4096,steps:1024,chain:true,workgroup:64,repetitions:32}];
      for(const spec of specs){
        progress({message:`Preparing ${spec.name}…`});const {a,b}=arithmetic?matrixOperands(spec.count):{a:operands(spec.count,137),b:operands(spec.count,1729)};
        const output=device.createBuffer({size:spec.count*(arithmetic?64:4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
        const stage=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const uniform=create(Uint32Array.of(spec.count,spec.steps,0,0),GPUBufferUsage.UNIFORM),variants=[];
        for(const kind of arithmetic?['fp16','int8','int32','int8-unpack-once','fp32']:['fp16','int8']){
          const data=kind==='fp16'?'half':kind==='int32'?'int32':kind==='fp32'?'fp32':'packed';
          const x=create(a[data],GPUBufferUsage.STORAGE),w=create(b[data],GPUBufferUsage.STORAGE);
          const module=device.createShaderModule({code:arithmetic?matrixShader(kind,spec.workgroup):shader(kind,spec.chain,spec.workgroup)});
          const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');require(!errors.length,errors.map(e=>e.message).join('\n'));
          variants.push({id:kind,x,w,pipeline:await device.createComputePipelineAsync({layout:pl,compute:{module,entryPoint:'main'}}),binding:device.createBindGroup({layout,entries:[x,w,output,uniform].map((buffer,binding)=>({binding,resource:{buffer}}))}),gpu:[],wall:[]});
        }
        const sample=async v=>{
          const start=performance.now(),e=device.createCommandEncoder(),pass=e.beginComputePass(timestamps?{timestampWrites:{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});
          pass.setPipeline(v.pipeline);pass.setBindGroup(0,v.binding);for(let repeat=0;repeat<spec.repetitions;repeat++)pass.dispatchWorkgroups(Math.ceil(spec.count/spec.workgroup));pass.end();
          if(timestamps){e.resolveQuerySet(query,0,2,resolved,0);e.copyBufferToBuffer(resolved,0,queryRead,0,16);}
          device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();const wall=performance.now()-start;require(!deviceError,deviceError);
          let gpu=null;if(timestamps){await queryRead.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(queryRead.getMappedRange());gpu=Number(t[1]-t[0])/1e6;queryRead.unmap();require(gpu>0,'Invalid GPU timestamp');}return {gpu,wall};
        };
        const record={...spec,rows:spec.count,input:4,output:arithmetic?16:1,logical_dots:spec.count*spec.steps*spec.repetitions*(arithmetic?16:1),variants:[]};
        try{
          let baseline;
          for(const v of variants){
            await sample(v);const e=device.createCommandEncoder();e.copyBufferToBuffer(output,0,stage,0,output.size);device.queue.submit([e.finish()]);await stage.mapAsync(GPUMapMode.READ);const values=new Float32Array(stage.getMappedRange().slice(0));stage.unmap();
            for(let j=0;j<Math.min(spec.count,32);j++){const at=j===31?spec.count-1:(j*1009)%spec.count;
              if(arithmetic){const expected=matrixReference(a.int32,b.int32,at,spec.steps);for(let lane=0;lane<16;lane++)require(values[at*16+lane]===expected[lane],'Register recurrence CPU reference mismatch');}
              else require(values[at]===reference(a.packed,b.packed,at,spec.steps),'Primitive CPU reference mismatch');}
            if(!baseline)baseline=values;
            const drift=P.compareOutput(values,baseline);require(drift.max_absolute_error===0,'INT8 and FP16 exact-input outputs differ');
            record.variants.push({id:v.id,kernel_check:{passed:true,reference:'Exact integer CPU dot and dependent recurrence'},synthetic_drift_from_fp16:drift});
          }
          progress({message:`Timing ${spec.name}: ${rounds} balanced rounds…`});
          for(let round=-3;round<rounds;round++){
            const first=((round%variants.length)+variants.length)%variants.length;
            for(const v of [...variants.slice(first),...variants.slice(0,first)]){const t=await sample(v);if(round>=0){v.wall.push(t.wall);if(t.gpu!==null)v.gpu.push(t.gpu);}}
          }
          for(const v of variants){const r=record.variants.find(r=>r.id===v.id);r.wall=P.statistics(v.wall);r.gpu=v.gpu.length?P.statistics(v.gpu):null;r.gemm=r.gpu;r.quantize={median_ms:0};r.normalized_ns_per_dot=(r.gpu??r.wall).median_ms*1e6/record.logical_dots;}
          const [fp,iq]=record.variants;record.int8_speedup=(fp.gpu??fp.wall).median_ms/(iq.gpu??iq.wall).median_ms;
          report.cases.push(record);progress({message:`Completed ${spec.name}`,record});
        }finally{for(const v of variants){v.x.destroy();v.w.destroy();}for(const x of [uniform,output,stage])x.destroy();}
      }
      report.completed=true;
    }catch(e){report.error=String(e);}finally{device?.destroy();report.elapsed_ms=performance.now()-started;}
    return report;
  }
  return {shader,operands,reference,matrixShader,matrixOperands,matrixReference,run};
})();
