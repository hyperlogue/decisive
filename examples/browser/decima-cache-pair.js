const q=new URLSearchParams(location.search);
const status=progress=>postMessage({progress});
const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
try {
 const names=(q.get('variants')||'reference,candidate').split(',');
 const workers=(q.get('workers')||'0,0').split(',').map(Number);
 if(names.length!==2||workers.length!==2||names.some(n=>!/^[a-z0-9-]+$/.test(n))||workers.some(n=>!Number.isSafeInteger(n)||n<0||n>64))throw Error('Invalid variants/workers');
 const precision=q.get('precision')||'int8',files=[];
 for(const name of ['decima.json','encoder/config.json','encoder/tokenizer.json','encoder/model.safetensors','head.safetensors']){
  status('Downloading '+name);const r=await fetch('/models/decima-small/pytorch/'+name,{cache:'no-store'});
  if(!r.ok)throw Error('Missing checkpoint '+name);files.push(new Uint8Array(await r.arrayBuffer()));
 }
 const packages=[];
 for(const [i,name] of names.entries()){
  status('Loading '+name);const path='/artifacts/web/pkg-'+name;
  let moduleName, r;
  for(moduleName of ['decisive','disco','beam']){
   r=await fetch(path+'/'+moduleName+'_bg.wasm',{cache:'no-store'});
   if(r.status!==404)break;
  }if(!r.ok)throw Error('Missing package');
  const bytes=await r.arrayBuffer(),api=await import(path+'/'+moduleName+'.js');await api.default({module_or_path:bytes});
  if(workers[i])await api.initDecimaThreadPool(workers[i]);
  packages.push({name,module:moduleName,workers:workers[i],wasm_sha256:await hash(bytes),model:new api.DecimaModel(...files,precision)});
 }
 files.length=0;
 const fixtures=(await (await fetch('/benches/decima-reference.jsonl')).text()).trim().split('\n').map(JSON.parse);
 const base=fixtures[0],question=base.question;
 const lengths=(await (await fetch('/benches/decima-lengths.jsonl')).text()).trim().split('\n').map(JSON.parse);
 const longest=lengths[4];
 const requests=[
  {name:'cold',state:base.state,question},
  {name:'repeat',state:base.state,question},
  {name:'changed-state',state:'The application cannot connect after the latest update.',question},
  {name:'changed-question',state:base.state,question:{...question,text:'Which department owns the next action?'}},
  {name:'mixed-cache',state:base.state,question:{...question,choices:['billing','customer success','shipping']}},
  {name:'reordered-choices',state:base.state,question:{...question,choices:['shipping','billing','technical support']}},
  {name:'duplicate-tokens',state:base.state,question:{...question,choices:['billing','billing','shipping']}},
  {name:'maximum-batch',state:longest.state,question:{...longest.question,choices:Array.from({length:8},(_,i)=>'category '+i+' '+('marker '.repeat(100)))}},
  {name:'nine-choices',state:base.state,question:{...question,choices:Array.from({length:9},(_,i)=>'department '+i)}},
  {name:'sixteen-short-choices',state:base.state,question:{...question,choices:Array.from({length:16},(_,i)=>'department '+i)}},
  {name:'thirty-two-short-choices',state:base.state,question:{...question,choices:Array.from({length:32},(_,i)=>'department '+i)}},
  {name:'spanish',...fixtures[3]},
  {name:'return-to-original',state:base.state,question},
  {name:'overlength-rejection',state:'sample '.repeat(1000),question},
  {name:'after-rejection',state:base.state,question},
 ];
 const records=[];
 const run=(model,r)=>{try{return {output:JSON.parse(model.decide(r.state,JSON.stringify(r.question)))}}catch(e){return {error:String(e)}}};
 for(const budget of [0,1,16*1024,64*1024,32*1024*1024]){
  for(const p of packages)p.model.setCacheBytes(budget);
  for(const r of requests){
   status('Cache '+budget+': '+r.name);const results=packages.map(p=>run(p.model,r));
   if(JSON.stringify(results[0])!==JSON.stringify(results[1]))throw Error('Cache contract differs: '+r.name+' at '+budget);
   if(r.name!=='overlength-rejection'&&results[0].error)throw Error('Unexpected rejection: '+r.name+': '+results[0].error);
   if(r.name==='overlength-rejection'&&!results[0].error?.includes('STATE_TRUNCATED'))throw Error('Missing overlength rejection');
   records.push({budget,case:r.name,...results[0]});await new Promise(r=>setTimeout(r,0));
  }
 }
 const settings=n=>({projection_rows:3,int8_projection_rows:n,attention_scores:1024});
 const policies=[{bands:[{through:32,settings:settings(1)}],fallback:settings(4)},{bands:[],fallback:settings(3)}];
 for(const [policy,rules] of policies.entries()){
  for(const p of packages)p.model.setLengthRules(JSON.stringify(rules));
  for(const r of requests){
   status('Length policy '+policy+': '+r.name);const results=packages.map(p=>run(p.model,r));
   if(JSON.stringify(results[0])!==JSON.stringify(results[1]))throw Error('Length policy contract differs: '+r.name);
   if(r.name!=='overlength-rejection'&&results[0].error)throw Error('Unexpected policy rejection: '+r.name+': '+results[0].error);
   if(r.name==='overlength-rejection'&&!results[0].error?.includes('STATE_TRUNCATED'))throw Error('Missing policy overlength rejection');
   records.push({policy,case:r.name,...results[0]});await new Promise(r=>setTimeout(r,0));
  }
 }
 for(const p of packages){p.model.setLengthRules('null');p.model.setCacheBytes(32*1024*1024);}
 const systemResponse=await fetch('/benches/decima-systemone.jsonl');
 if(!systemResponse.ok)throw Error('Missing System One fixtures');
 const systemFixtures=(await systemResponse.text()).trim().split('\n').map(JSON.parse);
 for(const [index,fixture] of systemFixtures.entries()){
  status('System One '+index);
  const request=fixture.request_json??JSON.stringify(fixture.request);
  const outputs=packages.map(p=>JSON.parse(p.model.predict(request)));
  if(JSON.stringify(outputs[0])!==JSON.stringify(outputs[1]))throw Error('System One response differs: '+index);
  records.push({case:'systemone-'+index,output:outputs[0]});
  await new Promise(r=>setTimeout(r,0));
 }
 for(const p of packages)p.model.free();
 status('Complete');postMessage({result:{precision,packages:packages.map(({model,...metadata})=>metadata),records,compared:records.length,all_exact:true,scope:'Output and rejection identity under cache hits, misses, eviction budgets, duplicates, changed state/question, and custom length policies and complete Jev-style responses. No timing claims.'}});
}catch(e){postMessage({result:{error:String(e),stack:e.stack}});}
