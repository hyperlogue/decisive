const q=new URLSearchParams(location.search);
let build={},lastPrediction,progress;
const status=s=>{progress=s;postMessage({progress:s});};
const best=p=>p.reduce((i,v,j)=>v>p[i]?j:i,0);
const raw=q.get('suite')!=='systemone';
const precision=q.get('precision')||'int8';
const bound=precision.startsWith('int8')?0.02:1e-4;
const audit=q.get('audit')==='quantization';
const profileFlags=new Set();
self.onmessage=({data})=>{if(data.flag)profileFlags.add(data.flag);};
const awaitProfile=async flag=>{while(!profileFlags.has(flag))await new Promise(r=>setTimeout(r,10));};
function compare(a,b,path='') {
 if(typeof a==='number'&&typeof b==='number') {
  if(!Number.isFinite(b)||(!audit&&Math.abs(a-b)>bound+1e-12))throw Error('Output drift at '+path);
  if(!audit&&path.endsWith('.noul')&&(a>=0.5)!==(b>=0.5))throw Error('Binary decision changed at '+path);
  return Math.abs(a-b);
 }
 if(a&&b&&typeof a==='object'&&typeof b==='object') {
  if(path.endsWith('.probabilities')&&!audit) {
   const winner=o=>Object.keys(o).reduce((i,j)=>o[j]>o[i]?j:i);
   if(winner(a)!==winner(b))throw Error('Probability argmax changed at '+path);
  }
  if(JSON.stringify(Object.keys(a).sort())!==JSON.stringify(Object.keys(b).sort()))throw Error('Output keys differ at '+path);
  return Math.max(0,...Object.keys(a).map(k=>compare(a[k],b[k],path+'.'+k)));
 }
 if(a!==b)throw Error('Output contract changed at '+path);
 return 0;
}
try {
 if(audit&&(!raw||!precision.startsWith('int8')))throw Error('Quantization audit requires raw quantized fixtures');
 const workers=Number(q.get('workers')||0),rounds=Number(q.get('rounds')||1),warmup=Number(q.get('warmup')||0),initialWarmup=Number(q.get('initial_warmup')||0);
 if(![workers,rounds,warmup,initialWarmup].every(Number.isSafeInteger)||workers<0||workers>64||rounds<1||warmup<0||initialWarmup<0)throw Error('Invalid workers/sample counts');
 const pkg=q.get('package')||'/artifacts/web/pkg-decima-threads';
 const checkpoint=q.get('checkpoint')||'/models/decima-small/pytorch';
 if(!pkg.startsWith('/artifacts/web/')||pkg.includes('..')||!checkpoint.startsWith('/models/')||checkpoint.includes('..'))throw Error('Package and checkpoint must be local');
 let moduleName, response;
 for(moduleName of ['decisive','disco','beam']){
  response=await fetch(pkg+'/'+moduleName+'_bg.wasm',{cache:'no-store'});
  if(response.status!==404)break;
 }
 if(!response.ok)throw Error('Missing Wasm: '+response.status);
 const wasmBytes=await response.arrayBuffer();
 const wasmHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',wasmBytes)),b=>b.toString(16).padStart(2,'0')).join('');
 const api=await import(pkg+'/'+moduleName+'.js');
 const start=performance.now();
 const wasm=await api.default({module_or_path:wasmBytes});
 const wasmInitMs=performance.now()-start;
 status('Initializing '+workers+' Decima workers');
 const poolStart=performance.now();
 if(workers)await api.initDecimaThreadPool(workers);
 const threadInitMs=performance.now()-poolStart;
 build={model:'decima',backend:'wasm',precision,package:pkg,module:moduleName,wasm_sha256:wasmHash,workers,coordinator:'dedicated Web Worker',shared_memory:typeof SharedArrayBuffer!=='undefined'&&wasm.memory.buffer instanceof SharedArrayBuffer,browser:navigator.userAgent,parameters:location.search};
 const checked=q.get('check')==='1'?JSON.parse(api.wasmKernelChecks()):null;
 const files=[];const downloadStart=performance.now();
 for(const file of ['decima.json','encoder/config.json','encoder/tokenizer.json','encoder/model.safetensors','head.safetensors']) {
  status('Downloading '+file);const r=await fetch(checkpoint+'/'+file,{cache:'no-store'});
  if(!r.ok)throw Error('Missing checkpoint file: '+file);
  files.push(new Uint8Array(await r.arrayBuffer()));
 }
 const downloadMs=performance.now()-downloadStart;
 status('Loading Decima');const loadStart=performance.now();
 const model=new api.DecimaModel(...files,precision);files.length=0;
 const loadMs=performance.now()-loadStart;
 const memoryAfterLoad=wasm.memory.buffer.byteLength;
 const segmentChecks=q.get('segments')==='1'?JSON.parse(model.checkEncoderSegments()):null;
 if(q.has('rules'))model.setLengthRules(q.get('rules'));
 const input=q.get('input')||(raw?'/benches/decima-reference.jsonl':'/benches/decima-systemone.jsonl');
 if(!(input.startsWith('/benches/')||input.startsWith('/artifacts/web/'))||input.includes('..'))throw Error('Input must be local fixtures');
 const fixtureResponse=await fetch(input,{cache:'no-store'});
 if(!fixtureResponse.ok)throw Error('Missing fixtures: '+input);
 const fixtures=(await fixtureResponse.text()).trim().split('\n').map(JSON.parse);
 const indices=q.get('indices')==='all'?fixtures.map((_,i)=>i):q.has('indices')?q.get('indices').split(',').map(Number):[0];
 const records=[];let changedDecisions=0;
 for(const [position,index] of indices.entries()) {
  const fixture=fixtures[index];if(!fixture)throw Error('Missing fixture '+index);
  const request=fixture.request_json??JSON.stringify(fixture.request);
  const times=[],warmTimes=[];let output,maxError=0,changedDecision=false;
  const warmRounds=position===0?Math.max(warmup,initialWarmup):warmup;
  for(let round=-warmRounds;round<rounds;round++) {
   if(q.has('profile')&&position===0&&round===0){postMessage({profileReady:true});await awaitProfile('profileStarted');}
   if(q.get('cache')==='off')model.clearCache();
   status(`Decima ${workers} workers: fixture ${index}, round ${round}`);
   const t=performance.now();
   output=JSON.parse(raw?model.decide(fixture.state,JSON.stringify(fixture.question)):model.predict(request));
   const ms=performance.now()-t;
   lastPrediction={fixture:index,reference:fixture.expected,output};
   maxError=Math.max(maxError,compare(fixture.expected,output));
   if(raw) {
    changedDecision=best(fixture.expected)!==best(output);
    if(!audit&&changedDecision)throw Error('Decision changed at fixture '+index);
   }
   (round>=0?times:warmTimes).push(ms);
   await new Promise(r=>setTimeout(r,0));
  }
  changedDecisions+=Number(changedDecision);
  records.push({fixture:index,output,reference:fixture.expected,gold:fixture.gold??null,max_error:maxError,changed_decision:changedDecision,warmup_latencies_ms:warmTimes,latencies_ms:times});
 }
 if(q.has('profile')){postMessage({profileDone:true});await awaitProfile('profileStopped');}
 let gold=null;
 if(raw) {
  const labeled=records.filter(r=>Number.isInteger(r.gold));
  const metrics=field=>{
   let correct=0,brier=0,nll=0;
   for(const r of labeled){const p=r[field];correct+=best(p)===r.gold;brier+=p.reduce((s,v,i)=>s+(v-(i===r.gold?1:0))**2,0);nll-=Math.log(Math.max(p[r.gold],1e-12));}
   return {labeled:labeled.length,correct,brier:brier/Math.max(1,labeled.length),nll:nll/Math.max(1,labeled.length)};
  };
  gold={reference:metrics('reference'),candidate:metrics('output')};
  if(!audit&&gold.candidate.correct<gold.reference.correct)throw Error('Gold accuracy regressed');
 }
 const memoryAfterInference=wasm.memory.buffer.byteLength;
 model.free();status('Complete');
 postMessage({result:{...build,checked,segmentChecks,audit,changed_decisions:changedDecisions,gates:{probability:records.every(r=>r.max_error<=bound),same_decisions:changedDecisions===0,gold_accuracy:gold?gold.candidate.correct>=gold.reference.correct:null},maximum_allowed_error:bound,wasm_init_ms:wasmInitMs,thread_init_ms:threadInitMs,download_ms:downloadMs,load_ms:loadMs,wasm_memory_after_load:memoryAfterLoad,wasm_memory_after_inference:memoryAfterInference,cache:q.get('cache')||'on',warmup,initial_warmup:initialWarmup,rounds,gold,records}});
}catch(e){postMessage({result:{...build,error:String(e),progress,last_prediction:lastPrediction,stack:e.stack}});}
