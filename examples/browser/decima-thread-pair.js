const q=new URLSearchParams(location.search);
const status=progress=>postMessage({progress});
const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
try {
 if(q.has('profile'))throw Error('Use decima-threads.html for worker profiles');
 const names=(q.get('variants')||'reference,candidate').split(',');
 const workerCounts=(q.get('workers')||'4').split(',').map(Number);
 const precision=q.get('precision')||'int8';
 const rounds=Number(q.get('rounds')||15),warmup=Number(q.get('warmup')||12);
 if(names.length<2||names.length>3||![rounds,warmup].every(Number.isSafeInteger)||rounds<1||warmup<0)throw Error('Invalid variants/sample counts');
 if(![1,names.length].includes(workerCounts.length)||workerCounts.some(n=>!Number.isSafeInteger(n)||n<0||n>64))throw Error('Invalid worker counts');
 if(names.some(n=>!/^[a-z0-9-]+$/.test(n)))throw Error('Invalid package name');
 const checkpoint=q.get('checkpoint')||'/models/decima-small/pytorch';
 if(!checkpoint.startsWith('/models/')||checkpoint.includes('..'))throw Error('Checkpoint must be local');
 const files=[];
 for(const name of ['decima.json','encoder/config.json','encoder/tokenizer.json','encoder/model.safetensors','head.safetensors']){
  status('Downloading '+name);const r=await fetch(checkpoint+'/'+name,{cache:'no-store'});
  if(!r.ok)throw Error('Missing checkpoint file '+name);files.push(new Uint8Array(await r.arrayBuffer()));
 }
 const packages=[];
 for(const [i,name] of names.entries()){
  const path='/artifacts/web/pkg-'+name,workers=workerCounts[i]??workerCounts[0];
  status('Loading '+name+', '+workers+' workers');
  let moduleName, r;
  for(moduleName of ['decisive','disco','beam']){
   r=await fetch(path+'/'+moduleName+'_bg.wasm',{cache:'no-store'});
   if(r.status!==404)break;
  }if(!r.ok)throw Error('Missing package '+path);
  const bytes=await r.arrayBuffer();const api=await import(path+'/'+moduleName+'.js');
  const wasm=await api.default({module_or_path:bytes});
  if(workers)await api.initDecimaThreadPool(workers);
  const model=new api.DecimaModel(...files,precision);
  if(q.has('rules'))model.setLengthRules(q.get('rules'));
  packages.push({name,path,module:moduleName,workers,model,wasm,wasm_sha256:await hash(bytes),wasm_memory_after_load:wasm.memory.buffer.byteLength});
 }
 files.length=0;
 const input=q.get('input')||'/benches/decima-lengths.jsonl';
 if(!(input.startsWith('/benches/')||input.startsWith('/artifacts/web/'))||input.includes('..'))throw Error('Input must be local');
 const response=await fetch(input,{cache:'no-store'});if(!response.ok)throw Error('Missing input');
 let fixtures=(await response.text()).trim().split('\n').map(JSON.parse);
 if(q.has('choice_counts')){
  const counts=q.get('choice_counts').split(',').map(Number);
  if(counts.some(n=>!Number.isSafeInteger(n)||n<2||n>255))throw Error('Invalid choice counts');
  const base=fixtures[0];
  fixtures=counts.map(count=>({...base,question:{...base.question,choices:Array.from({length:count},(_,i)=>'department '+i)}}));
 }
 const indices=q.get('indices')==='all'?fixtures.map((_,i)=>i):(q.get('indices')||'0,2,4').split(',').map(Number),records=[];
 for(const fixture of indices){
  const f=fixtures[fixture];if(!f)throw Error('Invalid fixture');
  const question=JSON.stringify(f.question),times=packages.map(()=>[]),outputs=packages.map(()=>null),orders=[];
  for(let round=-warmup;round<rounds;round++){
   const rotation=(round+warmup)%packages.length;
   const order=packages.map((_,i)=>(i+rotation)%packages.length);
   if(Math.floor((round+warmup)/packages.length)%2)order.reverse();
   status('Fixture '+fixture+', paired round '+round);
   for(const i of order){
    const p=packages[i];if(q.get('cache')==='off')p.model.clearCache();
    const start=performance.now();const output=JSON.parse(p.model.decide(f.state,question));
    const elapsed=performance.now()-start;
    if(!Array.isArray(output)||output.some(v=>!Number.isFinite(v)))throw Error('Invalid output');
    if(round>=0)times[i].push(elapsed);
    outputs[i]=output;
   }
   if(outputs.some(o=>JSON.stringify(o)!==JSON.stringify(outputs[0])))throw Error('Variants changed final output on fixture '+fixture);
   if(round>=0)orders.push(order);
   await new Promise(r=>setTimeout(r,0));
  }
  for(const [i,p] of packages.entries())records.push({fixture,choices:f.question.choices.length,variant:p.name,workers:p.workers,output:outputs[i],latencies_ms:times[i],orders});
 }
 const metadata=packages.map(({name,path,module,workers,wasm_sha256,wasm_memory_after_load,wasm})=>({name,path,module,workers,wasm_sha256,wasm_memory_after_load,wasm_memory_after_inference:wasm.memory.buffer.byteLength}));
 for(const p of packages)p.model.free();
 status('Complete');postMessage({result:{model:'decima',precision,packages:metadata,records,rounds,warmup,cache:q.get('cache')||'on',order:'rotated each round, reversed each full rotation cycle',coordinator:'dedicated Web Worker; isolated modules, sequential complete requests',browser:navigator.userAgent,parameters:location.search}});
}catch(e){postMessage({result:{error:String(e),stack:e.stack}});}
