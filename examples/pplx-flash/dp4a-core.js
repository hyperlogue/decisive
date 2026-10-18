globalThis.PplxDp4aCore = (() => {
  const P=PplxQuantizationProbe, D=PplxDp4aProbe;
  const require=(ok,message)=>{if(!ok)throw Error(message);};
  function packRows(values,k,decode=v=>v) {
    require(k>0&&k%4===0&&values.length%k===0,'Invalid INT8 matrix shape');
    const data=new Uint32Array(values.length/4),scales=new Float32Array(values.length/k);
    for(let row=0;row<scales.length;row++){
      let maximum=0;
      for(let j=0;j<k;j++){const v=decode(values[row*k+j]);require(Number.isFinite(v),'Non-finite operand');maximum=Math.max(maximum,Math.abs(v));}
      const scale=Math.fround(maximum/127);scales[row]=scale;
      for(let j=0;j<k;j++){
        const at=row*k+j,q=scale===0?0:Math.max(-127,Math.min(127,P.roundEven(decode(values[at])/scale)));
        data[at>>>2]|=(q&255)<<((at&3)*8);
      }
    }
    return {data,scales};
  }
  function dotShader(tile,raw=false) {
    const wg=tile/4,threads=wg*wg,size=tile*16;
    let s=`requires packed_4x8_integer_dot_product;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(2) var<storage,read> scales:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<${raw?'i32':'f32'}>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<u32,${size}>;var<workgroup> bt:array<u32,${size}>;
@compute @workgroup_size(${wg},${wg})
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
 let m=p.a.x;let n=p.a.y;let kw=p.a.z/4u;let lane=l.y*${wg}u+l.x;
 let row=g.y*${tile}u+l.y*4u;let col=g.x*${tile}u+l.x*4u;
 var s0=vec4<i32>(0);var s1=s0;var s2=s0;var s3=s0;
 for(var first=0u;first<kw;first+=16u){
  for(var i=lane;i<${size}u;i+=${threads}u){
   let r=i/16u;let c=i%16u;at[i]=0u;bt[c*${tile}u+r]=0u;
   if(g.y*${tile}u+r<m&&first+c<kw){at[i]=x[(g.y*${tile}u+r)*kw+first+c];}
   if(g.x*${tile}u+r<n&&first+c<kw){bt[c*${tile}u+r]=w[(g.x*${tile}u+r)*kw+first+c];}
  }
  workgroupBarrier();
  for(var j=0u;j<16u;j++){
   let i=j*${tile}u+l.x*4u;let v=vec4<u32>(bt[i],bt[i+1u],bt[i+2u],bt[i+3u]);
`;
    for(let r=0;r<4;r++)s+=`let a${r}=at[(l.y*4u+${r}u)*16u+j];s${r}+=vec4<i32>(dot4I8Packed(a${r},v.x),dot4I8Packed(a${r},v.y),dot4I8Packed(a${r},v.z),dot4I8Packed(a${r},v.w));\n`;
    s+=` }workgroupBarrier();\n }\n for(var c=0u;c<4u;c++){if(col+c<n){\n`;
    for(let r=0;r<4;r++)s+=`if(row+${r}u<m){y[(row+${r}u)*n+col+c]=${raw?`s${r}[c]`:`(f32(s${r}[c])*scales[row+${r}u])*scales[m+col+c]`};}\n`;
    return s+'}}}';
  }
  function fp16Shader(source,tile) {
    let shader=P.shader(source,16,'f16','fp16').replace('x: array<f32>','x: array<f16>');
    const start=shader.indexOf('        for(var j=l.y*8u+l.x;j<1024u;j+=64u) {');
    const end=shader.indexOf('        workgroupBarrier();',start);
    require(start>=0&&end>start,'FP16 tile source changed');
    const threads=tile*tile/16,size=tile*32;
    const loads=`let lane=l.y*${tile/4}u+l.x;
for(var i=lane;i<${size}u;i+=${threads}u){let r=i/32u;let c=i%32u;at[i]=0.0h;
if(g.y*${tile}u+r<m&&first+c<k){at[i]=x[(g.y*${tile}u+r)*k+first+c];}}
for(var i=lane;i<${size}u;i+=${threads}u){let r=i/32u;let c=i%32u;bt[c*${tile}u+r]=0.0h;
if(g.x*${tile}u+r<n&&first+c<k){bt[c*${tile}u+r]=w[(g.x*${tile}u+r)*k+first+c];}}
`;
    shader=shader.slice(0,start)+loads+shader.slice(end);
    return shader.replaceAll('array<f16,1024>',`array<f16,${size}>`)
      .replace('@workgroup_size(8,8)',`@workgroup_size(${tile/4},${tile/4})`)
      .replace('let row=g.y*32u+l.y*4u; let col=g.x*32u+l.x*4u;',`let row=g.y*${tile}u+l.y*4u; let col=g.x*${tile}u+l.x*4u;`)
      .replace('let i=j*32u+l.x*4u;',`let i=j*${tile}u+l.x*4u;`);
  }
  function transposeRows(data,k,n) {
    require(data.length===k*n,'Transpose shape mismatch');
    const out=new data.constructor(data.length);
    for(let row=0;row<n;row++)for(let col=0;col<k;col++)out[col*n+row]=data[row*k+col];
    return out;
  }
  function registerShader({tm=64,tn=64,rm=4,rn=4,kw=16,arithmetic='builtin',raw=false}={}) {
    require(tm%rm===0&&tn%rn===0&&rn%4===0,'Invalid register tile');
    const wx=tn/rn,wy=tm/rm,threads=wx*wy;
    require(threads<=256&&kw*(tm+tn)*4<=32768,'Register tile exceeds portable limits');
    const unsigned=arithmetic==='unsigned',builtin=arithmetic==='builtin'||unsigned;
    let s=`requires packed_4x8_integer_dot_product;
struct Params { a:vec4<u32>, b:vec4<u32>, c:vec4<u32>, d:vec4<u32> }
@group(0) @binding(0) var<storage,read> x:array<u32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
@group(0) @binding(2) var<storage,read> scales:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<${raw?'i32':'f32'}>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> at:array<u32,${tm*kw}>;var<workgroup> bt:array<u32,${tn*kw}>;
fn idot(a:vec4<i32>,b:vec4<i32>)->i32{return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;}
@compute @workgroup_size(${wx},${wy})
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
let m=p.a.x;let n=p.a.y;let k=p.a.z/4u;let lane=l.y*${wx}u+l.x;
let row=g.y*${tm}u+l.y*${rm}u;let col=g.x*${tn}u+l.x*${rn}u;
`;
    for(let r=0;r<rm;r++)for(let c=0;c<rn;c+=4)s+=`var s${r}_${c}=vec4<${unsigned?'u32':'i32'}>(0);\n`;
    s+=`for(var first=0u;first<k;first+=${kw}u){
for(var i=lane;i<${tm*kw}u;i+=${threads}u){let r=i/${kw}u;let c=i%${kw}u;at[i]=0u;
if(g.y*${tm}u+r<m&&first+c<k){at[i]=x[(g.y*${tm}u+r)*k+first+c];}}
for(var i=lane;i<${tn*kw}u;i+=${threads}u){let r=i/${kw}u;let c=i%${kw}u;bt[c*${tn}u+r]=0u;
if(g.x*${tn}u+r<n&&first+c<k){bt[c*${tn}u+r]=w[(g.x*${tn}u+r)*k+first+c];}}
workgroupBarrier();
for(var j=0u;j<${kw}u;j++){
`;
    for(let r=0;r<rm;r++)s+=`let a${r}=${builtin?'':'unpack4xI8('}at[(l.y*${rm}u+${r}u)*${kw}u+j]${builtin?'':')'};\n`;
    for(let c=0;c<rn;c++)s+=`let b${c}=${builtin?'':'unpack4xI8('}bt[j*${tn}u+l.x*${rn}u+${c}u]${builtin?'':')'};\n`;
    if(arithmetic==='outer')for(let c=0;c<rn;c+=4)for(const lane of ['x','y','z','w'])s+=`let bv${c}_${lane}=vec4<i32>(${[0,1,2,3].map(j=>`b${c+j}.${lane}`).join(',')});\n`;
    for(let r=0;r<rm;r++)for(let c=0;c<rn;c+=4){
      if(arithmetic==='outer')for(const lane of ['x','y','z','w'])s+=`s${r}_${c}+=a${r}.${lane}*bv${c}_${lane};\n`;
      else s+=`s${r}_${c}+=vec4<${unsigned?'u32':'i32'}>(${[0,1,2,3].map(j=>`${unsigned?'dot4U8Packed':builtin?'dot4I8Packed':'idot'}(a${r},b${c+j})`).join(',')});\n`;
    }
    s+='}workgroupBarrier();}\n';
    for(let r=0;r<rm;r++)for(let c=0;c<rn;c++){
      let value=`s${r}_${c-c%4}[${c%4}]`;
      if(unsigned)value=`i32(${value})-128*(i32(scales[m+n+row+${r}u])+i32(scales[2u*m+n+col+${c}u]))-i32(p.a.z)*16384`;
      const output=raw?`i32(${value})`:`(f32(${value})*scales[row+${r}u])*scales[m+col+${c}u]`;
      s+=`if(row+${r}u<m&&col+${c}u<n){y[(row+${r}u)*n+col+${c}u]=${output};}\n`;
    }
    return s+'}';
  }
  function signedRowSums(packed,k) {
    const sums=new Float32Array(packed.length*4/k);
    for(let row=0;row<sums.length;row++)for(let j=0;j<k;j++)sums[row]+=D.code(packed,row*k+j);
    return sums;
  }
  function tunedShader(source,kind,tile,transpose=false,pad=0,raw=false) {
    let shader=kind==='int8'?dotShader(tile,raw):fp16Shader(source,tile);
    const replace=(from,to)=>{require(shader.includes(from),'Tuned shader source changed: '+from);shader=shader.replaceAll(from,to);};
    if(kind==='int8'){
      if(transpose){
        replace(`bt[c*${tile}u+r]=0u;`,'bt[i]=0u;');
        replace(`if(g.x*${tile}u+r<n&&first+c<kw){bt[c*${tile}u+r]=w[(g.x*${tile}u+r)*kw+first+c];}`,
          `if(g.x*${tile}u+i%${tile}u<n&&first+i/${tile}u<kw){bt[i]=w[(first+i/${tile}u)*n+g.x*${tile}u+i%${tile}u];}`);
      }
      if(pad){
        replace(`bt:array<u32,${tile*16}>`,`bt:array<u32,${(tile+pad)*16}>`);
        if(transpose)replace('bt[i]=',`bt[(i/${tile}u)*${tile+pad}u+i%${tile}u]=`);
        else replace(`bt[c*${tile}u+r]`,`bt[c*${tile+pad}u+r]`);
        replace(`let i=j*${tile}u+l.x*4u;`,`let i=j*${tile+pad}u+l.x*4u;`);
      }
    }else{
      if(transpose){
        replace(`bt[c*${tile}u+r]=0.0h;`,'bt[i]=0.0h;');
        replace(`if(g.x*${tile}u+r<n&&first+c<k){bt[c*${tile}u+r]=w[(g.x*${tile}u+r)*k+first+c];}`,
          `if(g.x*${tile}u+i%${tile}u<n&&first+i/${tile}u<k){bt[i]=w[(first+i/${tile}u)*n+g.x*${tile}u+i%${tile}u];}`);
      }
      if(pad){
        replace(`bt: array<f16,${tile*32}>`,`bt: array<f16,${(tile+pad)*32}>`);
        if(transpose)replace('bt[i]=',`bt[(i/${tile}u)*${tile+pad}u+i%${tile}u]=`);
        else replace(`bt[c*${tile}u+r]`,`bt[c*${tile+pad}u+r]`);
        replace(`let i=j*${tile}u+l.x*4u;`,`let i=j*${tile+pad}u+l.x*4u;`);
      }
    }
    return shader;
  }
  function checkDot(actual,a,b,m,k,n) {
    require(k*127*127<2147483648,'INT32 accumulator could overflow');let maximum=0;
    for(let i=0;i<32;i++){
      const r=i===31?m-1:i*37%m,c=i===31?n-1:i*1009%n;
      let dot=0;for(let j=0;j<k;j++)dot+=D.code(a.data,r*k+j)*D.code(b.data,c*k+j);
      const expected=Math.fround(Math.fround(Math.fround(dot)*a.scales[r])*b.scales[c]);
      const error=Math.abs(actual[r*n+c]-expected);maximum=Math.max(maximum,error);
      require(Number.isFinite(error)&&error<=1e-5+2e-6*Math.abs(expected),'Prepacked INT8 dot disagrees with CPU reference');
    }
    return {passed:true,samples:32,max_absolute_error:maximum,reference:'Exact whole-K signed INT8 dot, INT32 accumulation, FP32 row/column scaling'};
  }
  async function run(source,provenance,progress=()=>{},options={}) {
    const unsigned=options.unsigned||options.specs?.some(s=>s.register?.arithmetic==='unsigned');
    const transposed=options.layouts||options.specs?.some(s=>s.transpose);
    const rawChecks=unsigned||options.confirmation;
    const rounds=options.rounds??(options.unsigned?18:options.tuning?24:options.layouts?16:12);
    const report={schema_version:1,experiment:'pplx-dp4a-v1',mode:options.confirmation?'confirmation':options.unsigned?'unsigned-correction':options.tuning?'register-tuning':options.layouts?'packed-layouts':'core-only',completed:false,started_at:new Date().toISOString(),source:provenance,cases:[],
      methodology:{warmups:3,rounds,baseline:'FP16 activation and weight storage; FP16 32-element FMA partials, FP32 outer sums and output',
        candidate:'Prepacked signed INT8 activations and weights; dot4I8Packed with INT32 accumulation across all K; FP32 scaled output',
        quantization:'Symmetric per-activation-row and per-weight-output-channel scales; zero point 0; [-127,127]',
        primary:'One resident GEMM dispatch; quantization, packing, conversion, upload and readback excluded from GPU timestamps',
        order:'Rotate first variant each round; measured rounds balance all variants',
        layouts:options.layouts?'Row-major versus pretransposed packed B, and one-element padding in workgroup B; preprocessing excluded':'Output-major packed rows',
        scope:'One actual MLP gate/up shape at two prefill lengths; synthetic operands; not full-model speed or quality'}};
    if(options.confirmation)report.methodology.primary='Resident GEMM; 117-token case batches eight identical dispatches to reduce short-run clock/scheduling noise; 1024-token case uses one dispatch. Times normalized per GEMM. Quantization, packing, upload and readback excluded.';
    if(unsigned){
      report.methodology.candidate='Signed packed dot versus unsigned packed dot with codes biased by 128; exact INT32 output correction; both represent the same quantized operands';
      report.methodology.correction='dot_signed = dot_unsigned - 128*(sum_signed_A + sum_signed_B) - K*16384. Biasing and row sums excluded; per-output correction included.';
      report.methodology.correctness='All raw integer outputs must exactly match signed dots. FP32 scaled outputs may differ by at most four float32 epsilons relative due to multiplication reassociation.';
    }
    const start=performance.now();let device;
    try{
      require(globalThis.isSecureContext&&navigator.gpu,'WebGPU unavailable');
      report.wgsl_features=[...(navigator.gpu.wgslLanguageFeatures||[])].sort();
      require(report.wgsl_features.includes('packed_4x8_integer_dot_product'),'Packed integer dot products unavailable');
      const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});require(adapter,'No adapter');
      const i=adapter.info||{};report.adapter={vendor:i.vendor,architecture:i.architecture,device:i.device,description:i.description,is_fallback_adapter:i.isFallbackAdapter??adapter.isFallbackAdapter??null};
      report.features=[...adapter.features].sort();require(adapter.features.has('shader-f16'),'FP16 unavailable');
      const timestamp=adapter.features.has('timestamp-query');report.gpu_timestamps=timestamp;
      device=await adapter.requestDevice({requiredFeatures:['shader-f16',...(timestamp?['timestamp-query']:[])]});
      let deviceError;device.addEventListener('uncapturederror',e=>deviceError=e.error.message);
      device.lost.then(i=>{if(i.reason!=='destroyed')deviceError=i.message;});
      const layout=device.createBindGroupLayout({entries:Array.from({length:5},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===4?'uniform':binding===3?'storage':'read-only-storage'}}))});
      const pl=device.createPipelineLayout({bindGroupLayouts:[layout]});
      const variants=[];
      const specs=options.specs??(options.unsigned?[
        {kind:'fp16',tile:64},{kind:'fp16',tile:64,pad:1},
        ...[{arithmetic:'builtin',rm:8},{arithmetic:'unsigned'},{arithmetic:'unsigned',rm:8},{arithmetic:'unsigned',rm:8,rn:8}].map(register=>({kind:'int8',tile:64,register:{tm:64,tn:64,rm:4,rn:4,kw:16,...register}})),
      ]:options.tuning?[
        {kind:'fp16',tile:64},{kind:'fp16',tile:64,pad:1},
        ...[
          {},{arithmetic:'reuse'},{arithmetic:'outer'},
          {rm:8,rn:8},{rm:8,rn:8,arithmetic:'reuse'},{rm:8,rn:8,arithmetic:'outer'},
          {rm:4,rn:8},{rm:8,rn:4},{kw:32},{tn:128,rn:8},
        ].map(register=>({kind:'int8',tile:64,register:{tm:64,tn:64,rm:4,rn:4,kw:16,arithmetic:'builtin',...register}})),
      ]:options.layouts?[
        {kind:'fp16',tile:64},{kind:'fp16',tile:64,transpose:true},{kind:'fp16',tile:64,pad:1},
        {kind:'int8',tile:32},{kind:'int8',tile:32,transpose:true},{kind:'int8',tile:32,pad:1},
        {kind:'int8',tile:64,transpose:true},{kind:'int8',tile:64,pad:1},
      ]:[32,64].flatMap(tile=>['fp16','int8'].map(kind=>({kind,tile}))));
      for(const {kind,tile,transpose=false,pad=0,register=null} of specs){
        const id=register?`int8-${register.arithmetic}-t${register.tm}x${register.tn}-r${register.rm}x${register.rn}-k${register.kw*4}`:`${kind}-tile${tile}${transpose?'-transposed':''}${pad?'-padded':''}`;
        const code=register?registerShader(register):tunedShader(source,kind,tile,transpose,pad);
        progress({message:`Compiling prepacked ${id}…`});const module=device.createShaderModule({code});
        const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');require(!errors.length,errors.map(e=>e.message).join('\n'));
        let integerPipeline=null;
        if(rawChecks&&kind==='int8'){
          const module=device.createShaderModule({code:register?registerShader({...register,raw:true}):tunedShader(source,kind,tile,transpose,pad,true)});
          integerPipeline=await device.createComputePipelineAsync({layout:pl,compute:{module,entryPoint:'main'}});
        }
        variants.push({id,kind,tile,transpose,pad,register,integerPipeline,pipeline:await device.createComputePipelineAsync({layout:pl,compute:{module,entryPoint:'main'}})});
      }
      const create=(data,usage)=>{const b=device.createBuffer({size:data.byteLength,usage,mappedAtCreation:true});new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();return b;};
      const read=async buffer=>{const stage=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,stage,0,buffer.size);device.queue.submit([e.finish()]);await stage.mapAsync(GPUMapMode.READ);return stage.getMappedRange().slice(0);}finally{stage.unmap();stage.destroy();}};
      const query=timestamp?device.createQuerySet({type:'timestamp',count:2}):null;
      const resolve=timestamp?device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}):null;
      const queryRead=timestamp?device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}):null;
      const k=2560,n=9216;progress({message:'Preparing resident FP16 and INT8 weights outside timing…'});
      const prep=performance.now(),weights=P.weightsFor(k,n),half=P.packF16(weights),q8=packRows(weights,k,P.fromBF16);
      const hw=create(half.data,GPUBufferUsage.STORAGE),qw=create(q8.data,GPUBufferUsage.STORAGE),dummy=create(new Uint32Array(1),GPUBufferUsage.STORAGE);
      const uw=unsigned?create(q8.data.map(v=>v^0x80808080),GPUBufferUsage.STORAGE):null;
      const weightSums=unsigned?signedRowSums(q8.data,k):null;
      const ht=transposed?create(transposeRows(half.data,k,n),GPUBufferUsage.STORAGE):null;
      const qt=transposed?create(transposeRows(q8.data,k/4,n),GPUBufferUsage.STORAGE):null;
      report.weight_preparation_ms=performance.now()-prep;
      for(const rows of [117,1024]){
        progress({message:`Preparing ${rows} activation rows outside timing…`});const prep=performance.now();
        const input=P.inputsFor(rows,k);input.fill(0,0,k);
        const inputHalf=Uint16Array.from(input,P.toF16),qa=packRows(input,k);
        const hinput=create(inputHalf,GPUBufferUsage.STORAGE),qinput=create(qa.data,GPUBufferUsage.STORAGE);
        const scales=new Float32Array((rows+n)*(unsigned?2:1));scales.set(qa.scales);scales.set(q8.scales,rows);
        const uinput=unsigned?create(qa.data.map(v=>v^0x80808080),GPUBufferUsage.STORAGE):null;
        if(unsigned){scales.set(signedRowSums(qa.data,k),rows+n);scales.set(weightSums,2*rows+n);}
        const scaleBuffer=create(scales,GPUBufferUsage.STORAGE),output=device.createBuffer({size:rows*n*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
        const params=new Uint32Array(16);params.set([rows,n,k]);const uniform=create(params,GPUBufferUsage.UNIFORM);
        const bind=(x,w,scale)=>device.createBindGroup({layout,entries:[x,w,scale,output,uniform].map((buffer,binding)=>({binding,resource:{buffer}}))});
        const bindings={fp16:bind(hinput,hw,dummy),int8:bind(qinput,qw,scaleBuffer)};
        if(unsigned)bindings['int8-unsigned']=bind(uinput,uw,scaleBuffer);
        if(transposed){bindings['fp16-transposed']=bind(hinput,ht,dummy);bindings['int8-transposed']=bind(qinput,qt,scaleBuffer);}
        const repetitions=options.confirmation&&rows===117?8:1;
        const record={name:'MLP gate/up',rows,input:k,output:n,repetitions,preparation_ms:performance.now()-prep,variants:[]};
        const sample=async (v,raw=false)=>{
          const start=performance.now(),e=device.createCommandEncoder();
          const pass=e.beginComputePass(timestamp?{timestampWrites:{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});
          pass.setPipeline(raw?v.integerPipeline:v.pipeline);pass.setBindGroup(0,bindings[v.kind+(v.register?.arithmetic==='unsigned'?'-unsigned':v.transpose?'-transposed':'')]);
          for(let repetition=0;repetition<repetitions;repetition++)pass.dispatchWorkgroups(Math.ceil(n/(v.register?.tn??v.tile)),Math.ceil(rows/(v.register?.tm??v.tile)));
          pass.end();
          if(timestamp){e.resolveQuerySet(query,0,2,resolve,0);e.copyBufferToBuffer(resolve,0,queryRead,0,16);}
          device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();const wall=performance.now()-start;require(!deviceError,deviceError);
          let gpu=null;if(timestamp){await queryRead.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(queryRead.getMappedRange());gpu=Number(t[1]-t[0])/1e6;queryRead.unmap();require(gpu>0,'Invalid timestamp');}
          return {wall:wall/repetitions,gpu:gpu===null?null:gpu/repetitions};
        };
        try{
          let baseline,int8Reference,rawReference;
          for(const v of variants){
            let rawCheck=null,scaledAgreement=null;
            if(v.integerPipeline){
              await sample(v,true);const integers=new Int32Array(await read(output));
              if(rawReference)require(integers.every((value,index)=>value===rawReference[index]),'Raw INT32 dots differ');
              else for(let j=0;j<32;j++){
                const row=j===31?rows-1:j*37%rows,col=j===31?n-1:j*1009%n;let dot=0;
                for(let t=0;t<k;t++)dot+=D.code(qa.data,row*k+t)*D.code(q8.data,col*k+t);
                require(integers[row*n+col]===dot,'Raw INT32 CPU reference mismatch');
              }
              rawReference=integers;rawCheck={passed:true,values:integers.length,reference:'Exact integer agreement with signed packed kernel; signed kernel sampled against CPU'};
            }
            await sample(v);const actual=new Float32Array(await read(output));
            const check=v.kind==='int8'?checkDot(actual,qa,q8,rows,k,n):P.checkSamples(actual,input,half,rows,k,n,'f16');
            if(!baseline)baseline=actual;
            if(v.kind==='int8'){
              if(int8Reference){
                const agreement=P.compareOutput(actual,int8Reference);
                scaledAgreement=agreement;
                const roundingOnly=rawCheck&&actual.every((value,index)=>Math.abs(value-int8Reference[index])<=4*2**-23*Math.max(Math.abs(value),Math.abs(int8Reference[index]),1e-20));
                if(agreement.max_absolute_error!==0&&!roundingOnly){
                  const differing=[];
                  for(let at=0;at<actual.length&&differing.length<8;at++)if(actual[at]!==int8Reference[at]){
                    const r=Math.floor(at/n),c=at%n;let dot=0;
                    for(let j=0;j<k;j++)dot+=D.code(qa.data,r*k+j)*D.code(q8.data,c*k+j);
                    differing.push({row:r,column:c,actual:actual[at],previous:int8Reference[at],cpu:Math.fround(Math.fround(Math.fround(dot)*qa.scales[r])*q8.scales[c]),integer_dot:dot});
                  }
                  report.verification_failure={variant:v.id,agreement,differing};
                  throw Error('INT8 tile outputs differ');
                }
              }
              int8Reference=actual;
            }
            else require(P.compareOutput(actual,baseline).max_absolute_error===0,'FP16 tile outputs differ');
            record.variants.push({id:v.id,configuration:{kind:v.kind,tile:v.tile,transpose:v.transpose,pad:v.pad,register:v.register},kernel_check:check,raw_integer_check:rawCheck,scaled_agreement_with_previous_int8:scaledAgreement,synthetic_drift_from_fp16:P.compareOutput(actual,baseline),gpu:[],wall:[]});
          }
          progress({message:`Timing resident ${rows} × ${k} × ${n}: 3 warmups + ${rounds} balanced rounds…`});
          for(let round=-3;round<rounds;round++){
            const offset=((round%variants.length)+variants.length)%variants.length;
            for(const v of [...variants.slice(offset),...variants.slice(0,offset)]){
              const t=await sample(v);if(round>=0){const r=record.variants.find(r=>r.id===v.id);r.wall.push(t.wall);if(t.gpu!==null)r.gpu.push(t.gpu);}
            }
          }
          for(const v of record.variants){v.wall=P.statistics(v.wall);v.gpu=v.gpu.length?P.statistics(v.gpu):null;v.gemm=v.gpu;v.quantize={median_ms:0};}
          const fp=record.variants.filter(v=>v.id.startsWith('fp16')).sort((a,b)=>(a.gpu??a.wall).median_ms-(b.gpu??b.wall).median_ms)[0];
          const iq=record.variants.filter(v=>v.id.startsWith('int8')).sort((a,b)=>(a.gpu??a.wall).median_ms-(b.gpu??b.wall).median_ms)[0];
          record.best_comparison={fp16:fp.id,int8:iq.id,speedup:(fp.gpu??fp.wall).median_ms/(iq.gpu??iq.wall).median_ms,
            paired_wins:(iq.gpu??iq.wall).samples_ms.filter((t,i)=>t<(fp.gpu??fp.wall).samples_ms[i]).length,rounds};
          report.cases.push(record);progress({message:`Completed prepacked matmul at ${rows} tokens`,record});
        }finally{for(const b of [hinput,qinput,uinput,scaleBuffer,output,uniform])b?.destroy();}
      }
      for(const b of [hw,qw,ht,qt,uw,dummy,resolve,queryRead])b?.destroy();query?.destroy();report.completed=true;
    }catch(e){report.error=String(e);}finally{device?.destroy();report.elapsed_ms=performance.now()-start;}
    return report;
  }
  return {packRows,dotShader,fp16Shader,tunedShader,transposeRows,registerShader,signedRowSums,checkDot,run};
})();
