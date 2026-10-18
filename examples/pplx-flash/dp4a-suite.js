globalThis.PplxDp4aSuite = (() => {
  const phases=[
    {id:'core',name:'Check real prefill matrices',options:{}},
    {id:'layouts',name:'Compare weight layouts',options:{layouts:true}},
    {id:'tuning',name:'Search INT8 register tiles',options:{tuning:true}},
    {id:'unsigned',name:'Try unsigned packed dots',options:{unsigned:true}},
    {id:'arithmetic',name:'Isolate arithmetic cost',options:{arithmetic:true},primitive:true},
    {id:'primitive',name:'Check four-element dots',options:{},primitive:true},
  ];
  const time=v=>v.gpu??v.wall;
  const verdict=(code,title,detail,extra={})=>({code,title,detail,...extra});
  function select(phases){
    const fp=new Map(),selected={};
    for(const phase of phases.filter(p=>['core','layouts','tuning','unsigned'].includes(p.id)))for(const c of phase.report.cases){
      const floats=c.variants.filter(v=>v.id.startsWith('fp16'));
      for(const v of floats)fp.set(v.id,v.configuration);
      const baseline=Math.min(...floats.map(v=>time(v).median_ms));
      for(const v of c.variants.filter(v=>v.id.startsWith('int8'))){
        const speedup=baseline/time(v).median_ms;
        if(!selected[c.rows]||speedup>selected[c.rows].exploratory_speedup)selected[c.rows]={id:v.id,configuration:v.configuration,exploratory_speedup:speedup,phase:phase.id};
      }
    }
    if(!selected[117]||!selected[1024])throw Error('Missing prefill cases for confirmation');
    const integers=new Map(Object.values(selected).map(v=>[v.id,v.configuration]));
    if([...integers.values()].some(c=>c.register?.arithmetic==='unsigned'))integers.set('int8-builtin-t64x64-r8x4-k64',{kind:'int8',tile:64,transpose:false,pad:0,register:{tm:64,tn:64,rm:8,rn:4,kw:16,arithmetic:'builtin'}});
    const specs=[...fp.values(),...integers.values()].filter((v,i,all)=>all.findIndex(x=>JSON.stringify(x)===JSON.stringify(v))===i);
    specs.sort((a,b)=>Number(a.register?.arithmetic==='unsigned')-Number(b.register?.arithmetic==='unsigned'));
    return {selected,specs,rounds:Math.max(4,Math.ceil(16/specs.length))*specs.length};
  }
  function summarize(report,hidden=false){
    const cap=report.capabilities??{};
    if(cap.webgpu===false||cap.adapter===false)return verdict('unavailable','WebGPU isn’t available here.','This browser cannot run the GPU comparison.',{supported:false});
    if(cap.packed_dot===false)return verdict('unsupported','INT8 isn’t supported here.','This browser does not expose packed 4×8 integer dots.',{supported:false});
    if(cap.fp16===false)return verdict('unavailable','This browser can’t compare INT8 with FP16.','FP16 support is missing.',{supported:cap.packed_dot??null});
    if(report.error)return verdict('failed','The test couldn’t validate INT8.',report.error,{supported:cap.packed_dot??null,correct:null});
    if(!report.completed)return verdict('incomplete','No result yet.','Keep this page visible until the test finishes.');
    if(hidden)return verdict('interrupted','No reliable timing result — run again.','The page was hidden during the test. Keep it visible and the screen awake.',{supported:true,correct:true});
    const confirmation=report.confirmation;
    if(!confirmation?.completed)return verdict('failed','The confirmation test did not finish.','No speed recommendation is available.');
    const comparisons=[];
    for(const c of confirmation.cases){
      const candidate=c.variants.find(v=>v.id===report.selection?.selected[c.rows]?.id);
      const baseline=c.variants.filter(v=>v.id.startsWith('fp16')).sort((a,b)=>time(a).median_ms-time(b).median_ms)[0];
      if(!candidate||!baseline||!candidate.kernel_check?.passed||!candidate.raw_integer_check?.passed||!baseline.kernel_check?.passed)return verdict('failed','The test couldn’t validate INT8.','A required correctness check or result is missing.');
      const iq=time(candidate),fp=time(baseline),rounds=iq.samples_ms.length;
      if(rounds<12||rounds!==fp.samples_ms.length||![iq.median_ms,fp.median_ms,iq.relative_mad,fp.relative_mad,...iq.samples_ms,...fp.samples_ms].every(Number.isFinite)||iq.median_ms<=0||fp.median_ms<=0)return verdict('failed','The timing results are incomplete.','Run the comparison again.');
      const speedup=fp.median_ms/iq.median_ms,wins=iq.samples_ms.filter((v,i)=>v<fp.samples_ms[i]).length;
      comparisons.push({tokens:c.rows,int8_ms:iq.median_ms,fp16_ms:fp.median_ms,speedup,int8_wins:wins,rounds,
        int8_kernel:candidate.id,fp16_kernel:baseline.id,timer:candidate.gpu&&baseline.gpu?'GPU timestamps':'Synchronized elapsed time',
        noisy:Math.max(iq.relative_mad,fp.relative_mad)>0.10,
        int8_faster:speedup>=1.10&&wins/rounds>=0.8,
        fp16_faster:speedup<=1/1.10&&(rounds-wins)/rounds>=0.8});
    }
    if(comparisons.length!==2||![117,1024].every(rows=>comparisons.some(c=>c.tokens===rows)))return verdict('failed','The prefill comparison is incomplete.','Both input lengths are required.');
    const extra={supported:true,correct:true,comparisons,policy:'Confirmation only; at least 10% faster, at least 80% paired wins, relative MAD at most 10%. Initial quantization and packing excluded.'};
    if(comparisons.some(c=>c.noisy))return verdict('noisy','INT8 works; speed is inconclusive.','Correctness passed, but some timings varied too much to recommend INT8 over FP16. See the result for each input length below.',extra);
    if(comparisons.every(c=>c.int8_faster))return verdict('faster','Yes — INT8 is faster.','Packed INT8 passed correctness and reliably beat FP16 at both tested input lengths.',{...extra,faster:true});
    if(comparisons.some(c=>c.int8_faster))return verdict('mixed','INT8 works; speed depends on input length.','INT8 only showed a reliable speed advantage at one tested length. Keep FP16 as the default.',{...extra,faster:null});
    return verdict('use-fp16','INT8 works, but use FP16.','Packed INT8 passed correctness but showed no reliable speed advantage on these prefill tests.',{...extra,faster:false});
  }
  async function run(source,provenance,progress=()=>{}){
    const report={schema_version:1,experiment:'pplx-dp4a-v1',mode:'one-click',completed:false,started_at:new Date().toISOString(),source:provenance,phases:[],
      scope:'Synthetic prepacked INT8 versus FP16 kernels at real MLP dimensions. No model-quality validation or full-model INT8 timing. Quantization, packing and uploads excluded.',capabilities:{}};
    const start=performance.now();
    try{
      const cap=report.capabilities;cap.webgpu=!!(globalThis.isSecureContext&&navigator.gpu);
      if(!cap.webgpu)return report;
      cap.packed_dot=navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product')??false;
      const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});cap.adapter=!!adapter;
      cap.fp16=adapter?.features.has('shader-f16')??false;
      if(!cap.packed_dot||!cap.adapter||!cap.fp16)return report;
      for(const [i,phase] of phases.entries()){
        const update=()=>progress({message:`${i+1}/7 · ${phase.name}…`,step:i+1,total:7});
        const runner=phase.primitive?PplxDp4aPrimitive:PplxDp4aCore;
        const result=await runner.run(source,provenance,update,phase.options);
        report.phases.push({id:phase.id,name:phase.name,report:result});
        progress({message:`${i+1}/7 · ${phase.name} complete`,step:i+1,total:7,phase_complete:report.phases.at(-1)});
        if(!result.completed)throw Error(`${phase.name}: ${result.error??'Incomplete result'}`);
      }
      report.selection=select(report.phases);
      for(let attempt=0;attempt<2;attempt++){
        progress({message:`7/7 · ${attempt?'Repeat noisy confirmation':'Confirm selected kernels'}…`,step:7,total:7});
        const result=await PplxDp4aCore.run(source,provenance,()=>progress({message:`7/7 · ${attempt?'Repeat noisy confirmation':'Confirm selected kernels'}…`,step:7,total:7}),{...report.selection,confirmation:true});
        report.confirmation=result;report.confirmation_attempts??=[];report.confirmation_attempts.push(result);
        if(!result.completed)throw Error(`Confirmation: ${result.error??'Incomplete result'}`);
        report.completed=true;
        if(summarize(report).code!=='noisy')break;
      }
    }catch(e){report.completed=false;report.error=String(e);}
    finally{report.elapsed_ms=performance.now()-start;report.verdict=summarize(report);}
    return report;
  }
  return {run,select,summarize};
})();
