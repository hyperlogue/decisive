globalThis.PplxSubgroupMatrix = (() => {
  const check=(condition,message)=>{if(!condition)throw Error(message);};
  function shader(type='f16',subgroupSize=32) {
    check(['f16','f32'].includes(type)&&subgroupSize===32,'Only the measured 32-lane 8x8 configuration is implemented');
    const half=type==='f16';
    return `${half?'enable f16;':''}
enable chromium_experimental_subgroup_matrix;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<${type}>;
@group(0) @binding(1) var<storage,read> w:array<${type}>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> partial:array<${type},256>;
alias Left=subgroup_matrix_left<${type},8,8>;
alias Right=subgroup_matrix_right<${type},8,8>;
alias Result=subgroup_matrix_result<${type},8,8>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(subgroup_id) sg:u32) {
 let row=g.y*16u+(sg/2u)*8u; let col=g.x*16u+(sg%2u)*8u;
 var total=vec2<f32>(0.0);
 for(var first=0u;first<p.a.z;first+=${half?'32u':'p.a.z'}) {
  var acc=Result();
  for(var j=first;j<first+${half?'32u':'p.a.z'};j+=8u) {
   let a=subgroupMatrixLoad<Left>(&x,row*p.a.z+j,false,p.a.z);
   let b=subgroupMatrixLoad<Right>(&w,col*p.a.z+j,true,p.a.z);
   acc=subgroupMatrixMultiplyAccumulate(a,b,acc);
  }
  subgroupMatrixStore(&partial,sg*64u,acc,false,8u);
  workgroupBarrier();
  total+=vec2<f32>(f32(partial[lane]),f32(partial[lane+128u]));
  workgroupBarrier();
 }
 for(var i=0u;i<2u;i++) {
  let at=lane+i*128u; let block=at/64u; let inner=at%64u;
  y[(g.y*16u+(block/2u)*8u+inner/8u)*p.a.y+g.x*16u+(block%2u)*8u+inner%8u]=total[i];
 }
}`;
  }
  async function run(source,provenance,progress=()=>{}) {
    const P=PplxQuantizationProbe,C=PplxDp4aCore;
    const report={schema_version:1,experiment:'pplx-dp4a-v1',mode:'subgroup-matrix',completed:false,source:provenance,cases:[],browser:navigator.userAgent,
      scope:'Resident synthetic GEMMs. Experimental subgroup matrices compared with padded FP16 workgroup tiling. No full-model or trained-quality claim.',
      methodology:{warmups:3,rounds:12,order:'Rotate variant order each round',inputs:'Seeded integers [-3,3], exactly represented in FP16/FP32; FP16 partial length32 keeps integer accumulation exact',padding:'Zero-pad M/N to16 and K to32 for every variant; allocation and padding excluded',timing:'GPU timestamps around four identical resident dispatches; normalized per GEMM; uploads/readback excluded'}};
    let device;
    try {
      const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
      check(adapter,'No WebGPU adapter');
      report.features=[...adapter.features].sort();
      report.matrix_configs=Array.from(adapter.info.subgroupMatrixConfigs||[],c=>({componentType:c.componentType,resultComponentType:c.resultComponentType,M:c.M,N:c.N,K:c.K}));
      report.subgroup_size=[adapter.info.subgroupMinSize,adapter.info.subgroupMaxSize];
      if(!adapter.features.has('chromium-experimental-subgroup-matrix')) {report.supported=false;report.reason='Experimental subgroup matrix feature is not exposed';report.completed=true;return report;}
      check(report.subgroup_size.every(x=>x===32),'This probe requires a fixed 32-lane subgroup');
      const types=['f16','f32'].filter(t=>report.matrix_configs.some(c=>c.componentType===t&&c.resultComponentType===t&&c.M===8&&c.N===8&&c.K===8));
      check(types.length,'No supported 8x8 floating-point matrix configuration');
      check(adapter.features.has('timestamp-query')&&adapter.features.has('shader-f16'),'Timestamp queries and FP16 required');
      device=await adapter.requestDevice({requiredFeatures:['shader-f16','timestamp-query','subgroups','chromium-experimental-subgroup-matrix']});
      report.supported=true;
      const layout=device.createBindGroupLayout({entries:[0,1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===4?'uniform':binding===3?'storage':'read-only-storage'}}))});
      const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
      const specs=[{id:'fp16-padded',type:'f16',tile:64,code:C.tunedShader(source,'fp16',64,false,1)},...types.map(type=>({id:`matrix-${type}`,type,tile:16,code:shader(type)}))];
      for(const spec of specs){progress(`Compiling ${spec.id}`);const module=device.createShaderModule({code:spec.code});const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');check(!errors.length,errors.map(e=>e.message).join('\n'));spec.pipeline=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:'main'}});}
      const create=(data,usage)=>{const b=device.createBuffer({size:data.byteLength,usage,mappedAtCreation:true});new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();return b;};
      const query=device.createQuerySet({type:'timestamp',count:2});
      const resolve=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      const timer=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const code=(index,seed)=>{let x=Math.imul(index^seed,0x45d9f3b);x=Math.imul(x^(x>>>16),0x45d9f3b);return ((x^(x>>>16))>>>0)%7-3;};
      for(const [m,k,n] of [[17,68,70],[117,2560,9216],[1024,2560,9216]]) {
        progress(`Preparing ${m} × ${k} × ${n}`);
        const mp=Math.ceil(m/16)*16,np=Math.ceil(n/16)*16,kp=Math.ceil(k/32)*32;
        const x=new Float32Array(mp*kp),w=new Float32Array(np*kp);
        for(let r=0;r<m;r++)for(let j=0;j<k;j++)x[r*kp+j]=code(r*k+j,137);
        for(let r=0;r<n;r++)for(let j=0;j<k;j++)w[r*kp+j]=code(r*k+j,1729);
        const buffers=[];const add=(data,usage=GPUBufferUsage.STORAGE)=>{const b=create(data,usage);buffers.push(b);return b;};
        const operands={f16:[add(Uint16Array.from(x,P.toF16)),add(Uint16Array.from(w,P.toF16))],f32:[add(x),add(w)]};
        const dummy=add(new Uint32Array(1)),params=new Uint32Array(16);params.set([mp,np,kp]);const uniform=add(params,GPUBufferUsage.UNIFORM);
        const out=device.createBuffer({size:mp*np*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});buffers.push(out);
        const read=device.createBuffer({size:out.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});buffers.push(read);
        const groups=Object.fromEntries(Object.entries(operands).map(([type,[a,b]])=>[type,device.createBindGroup({layout,entries:[a,b,dummy,out,uniform].map((buffer,binding)=>({binding,resource:{buffer}}))})]));
        const sample=async(spec,readback=false)=>{
          const e=device.createCommandEncoder(),pass=e.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}});
          pass.setPipeline(spec.pipeline);pass.setBindGroup(0,groups[spec.type]);for(let i=0;i<4;i++)pass.dispatchWorkgroups(Math.ceil(np/spec.tile),Math.ceil(mp/spec.tile));pass.end();
          e.resolveQuerySet(query,0,2,resolve,0);e.copyBufferToBuffer(resolve,0,timer,0,16);if(readback)e.copyBufferToBuffer(out,0,read,0,out.size);device.queue.submit([e.finish()]);
          await timer.mapAsync(GPUMapMode.READ);const ticks=new BigUint64Array(timer.getMappedRange()),ms=Number(ticks[1]-ticks[0])/4e6;timer.unmap();check(ms>0,'Invalid timestamp');
          if(!readback)return ms;await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange().slice(0));read.unmap();return values;
        };
        const record={rows:m,input:k,output:n,padded:[mp,kp,np],variants:[]};
        for(const spec of specs) {
          const values=await sample(spec,true);let samples=0;
          const indices=m<32?Array.from({length:mp*np},(_,i)=>i):Array.from({length:128},(_,i)=>i===127?(m-1)*np+n-1:(i*37%m)*np+(i*1009%n));
          for(const index of indices){const row=Math.floor(index/np),col=index%np;let dot=0;for(let j=0;j<kp;j++)dot+=x[row*kp+j]*w[col*kp+j];check(values[index]===dot,`${spec.id}: exact CPU mismatch at ${row},${col}: ${values[index]} != ${dot}`);samples++;}
          record.variants.push({id:spec.id,kernel_check:{passed:true,samples,reference:'Exact integer CPU dot'},gpu:[]});
        }
        progress(`Timing ${m} rows`);
        for(let round=-3;round<12;round++)for(let j=0;j<specs.length;j++){const at=((round+j)%specs.length+specs.length)%specs.length;const ms=await sample(specs[at]);if(round>=0)record.variants[at].gpu.push(ms);}
        for(const variant of record.variants)variant.gpu=P.statistics(variant.gpu);
        report.cases.push(record);for(const b of buffers)b.destroy();
      }
      resolve.destroy();timer.destroy();query.destroy();report.completed=true;
    }catch(error){report.error=String(error);}finally{device?.destroy();}
    return report;
  }
  return {shader,run};
})();
