import init, * as api from '/pkg/decisive.js';
import {loadFlash} from '/runtime/runtime.js';

self.postMessage({type:'ready'});
self.onmessage=async({data})=>{
  const resident=data.resident!==false;
  const report={schema_version:2,experiment:'pplx-checkpoint-phase-v1',completed:false,configuration:data,
    timing_scope:resident?'warm-request':'streamed-request-diagnostic',startup:[],runs:[]};
  const engines=[];
  const progress=message=>self.postMessage({type:'progress',message});
  const median=values=>{const s=[...values].sort((a,b)=>a-b);return(s[(s.length-1)>>1]+s[s.length>>1])/2;};
  const compare=(actual,expected)=>{
    let max=0,mismatches=0;
    const visit=(a,b)=>{if(typeof b==='number'){if(!Number.isFinite(a))throw Error('Nonfinite or missing response value');max=Math.max(max,Math.abs(a-b));}
      else if(b&&typeof b==='object'){if(!a||typeof a!=='object'||JSON.stringify(Object.keys(a).sort())!==JSON.stringify(Object.keys(b).sort()))throw Error('Response structure changed');for(const key of Object.keys(b))visit(a[key],b[key]);}
      else if(a!==b)mismatches++;};visit(actual,expected);return{max_absolute_difference:max,categorical_mismatches:mismatches};
  };
  const logits=(a,b)=>{
    const x=a.diagnostics.logits.flatMap(c=>c.values.flat()),y=b.diagnostics.logits.flatMap(c=>c.values.flat());
    if(!x.length||x.length!==y.length||![...x,...y].every(Number.isFinite))throw Error('Invalid checkpoint logits');
    let max=0,diff=0,norm=0;for(let i=0;i<x.length;i++){const d=x[i]-y[i];max=Math.max(max,Math.abs(d));diff+=d*d;norm+=y[i]*y[i];}
    return{max_absolute_error:max,relative_l2:Math.sqrt(diff/Math.max(norm,1e-30))};
  };
  try{
    if(!Array.isArray(data.modes)||data.modes.length<1||data.modes.length>2||new Set(data.modes).size!==data.modes.length||!Array.isArray(data.fixtures)||!data.fixtures.length||!Number.isInteger(data.rounds)||data.rounds<1||data.rounds>7||!Number.isInteger(data.warmups)||data.warmups<0||data.warmups>2)throw Error('Invalid checkpoint phase');
    const metadata=await(await fetch('/metadata')).json();report.metadata=metadata;
    let corpus;
    if(data.corpus){
      const bytes=await(await fetch('/quality-corpus.json')).arrayBuffer();
      const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
      if(hash!==metadata.corpus_sha256)throw Error('Quality corpus checksum mismatch');
      corpus=JSON.parse(new TextDecoder().decode(bytes));report.corpus_sha256=hash;report.purpose='quality-evaluation';
      if(new Set(corpus.cases.map(c=>c.id)).size!==corpus.cases.length)throw Error('Duplicate quality case IDs');
    }
    const bytes=await(await fetch('/pkg/decisive_bg.wasm')).arrayBuffer();
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
    if(hash!==metadata.build.wasm_sha256)throw Error('Wasm checksum mismatch');
    const wasm=await init({module_or_path:bytes});report.wasm_sha256=hash;
    if(data.q4_check){progress('Checking Q4 codes and raw dots against CPU');report.kernel_check=JSON.parse(await api.pplxFlashWebGpuQ4Check());if(!report.kernel_check.passed)throw Error('Q4 kernel check failed');}
    const reference=await(await fetch('/fixtures/reference.json')).json();
    for(const mode of data.modes){
      progress(`Initializing ${mode}`);
      const engine=await loadFlash(api,'/model',message=>{if(/(Layer |Preparing weights )(8|16|24|32)\//.test(message))progress(`${report.active??mode}: ${message}`);},'webgpu',{mode,profile:Boolean(data.profile),resident});
      engines.push({mode,engine});
      report.startup.push({mode,...engine.startup});
    }
    for(const fixture of data.fixtures){
      const qualityCase=corpus?.cases.find(c=>c.id===fixture);
      if(corpus?!qualityCase:!['request','multilingual','long'].includes(fixture))throw Error('Unknown checkpoint fixture');
      const request=qualityCase?qualityCase.request:await(await fetch(`/fixtures/${fixture==='long'?'request':fixture}.json`)).json();
      if(fixture==='long'){
        const words=Array(60).fill(request.state).join(' ').split(/\s+/),planner=engines[0].engine.model;
        const length=state=>JSON.parse(planner.prepare(JSON.stringify({...request,state}),2048)).reduce((n,c)=>n+c.ids.length,0);
        let lo=1,hi=words.length;
        while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(length(words.slice(0,mid).join(' '))<=1024)lo=mid;else hi=mid-1;}
        request.state=words.slice(0,lo).join(' ');
        while(length(request.state)<1024){const next=request.state+' x';if(length(next)>1024)break;request.state=next;}
      }
      for(let round=-data.warmups;round<data.rounds;round++){
        const order=round%2?[...engines].reverse():engines;
        for(const {mode,engine} of order){
          report.active=`${fixture} · ${mode} · ${round<0?'warmup':`run ${round+1}/${data.rounds}`}`;progress(report.active);
          const started_epoch_ms=performance.timeOrigin+performance.now();
          const result=await engine.predict(request,fixture==='long'?1024:2048,true);
          if(resident&&(result.request_ms!==result.elapsed_ms||result.decoder_weight_bytes_read!==0||result.layers.some(l=>l.pack_ms!==0||l.read_ms!==0)))throw Error('One-time decoder preparation leaked into warm request');
          if(result.layers.length!==32*result.response.usage.sequences)throw Error('Incomplete decoder request');
          if(['int8','int8-tuned','w8a8-f32','q4-fused','q4-buffered'].includes(mode.replace(/-reuse$/,''))&&result.layers.some(l=>l.profile?.dispatches.some(d=>['gemm','mlp_fused','gemm_small','mlp_small'].includes(d.kernel))))throw Error('All-INT8 projection configuration dispatched FP16');
          const run={mode,fixture,round,warmup:round<0,started_epoch_ms,finished_epoch_ms:performance.timeOrigin+performance.now(),...result,wasm_memory_bytes:wasm.memory.buffer.byteLength};
          if(!corpus&&mode.replace(/-reuse$/,'')==='bf16'&&fixture!=='long'){
            const expected=reference.cases.find(c=>JSON.stringify(c.request)===JSON.stringify(request));
            if(!expected)throw Error('Missing reference fixture');
            run.reference=compare(result.response,expected.response);
            if(run.reference.max_absolute_difference>1e-4||run.reference.categorical_mismatches)throw Error(`BF16 reference mismatch: ${JSON.stringify(run.reference)}`);
          }
          report.runs.push(run);progress(`${report.active}: ${(result.elapsed_ms/1000).toFixed(2)} s ${resident?'per request; model preparation excluded':'streaming diagnostic; not a warm-request benchmark'}`);
        }
      }
    }
    report.cases=data.fixtures.map(fixture=>{
      const runs=report.runs.filter(r=>r.fixture===fixture&&!r.warmup),baseline=runs.filter(r=>r.mode===data.modes[0]);
      const variants=Object.fromEntries(data.modes.map(mode=>{
        const samples=runs.filter(r=>r.mode===mode),times=samples.map(r=>r.elapsed_ms),center=median(times);
        return[mode,{samples_ms:times,median_ms:center,relative_mad:median(times.map(t=>Math.abs(t-center)))/center,median_compute_ms:median(samples.map(r=>r.layers.reduce((s,l)=>s+l.compute_ms,0))),median_weight_preparation_ms:median(samples.map(r=>r.layers.reduce((s,l)=>s+l.pack_ms,0))),input_tokens:samples[0].response.usage.input_tokens}];
      }));
      const paired=runs.filter(r=>r.mode===data.modes[1]).map(run=>{
        const ref=baseline.find(b=>b.round===run.round),difference=logits(run,ref);
        if(data.exact&&difference.max_absolute_error!==0)throw Error(`Same-precision checkpoint logits changed: ${difference.max_absolute_error}`);
        return{round:run.round,speedup:ref.elapsed_ms/run.elapsed_ms,logits:difference,response:compare(run.response,ref.response)};
      });
      return{fixture,variants,paired,speedup:data.modes.length===2?variants[data.modes[0]].median_ms/variants[data.modes[1]].median_ms:null,paired_wins:paired.filter(p=>p.speedup>1).length};
    });
    delete report.active;report.completed=true;
  }catch(error){report.error=String(error);}
  finally{for(const {engine} of engines)engine.dispose();}
  self.postMessage({type:'result',report});
};
