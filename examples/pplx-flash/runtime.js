import {createEmbeddingCache} from './embedding-cache.js';

export async function loadFlash(api, source, progress = () => {}, backend = 'wasm', configuration = null) {
  const initializationStart=performance.now(),resident=configuration?.resident===true;
  if (!["wasm", "webgpu"].includes(backend)) throw Error("Unknown Pplx backend");
  const files=['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'];
  let transferred=0;
  const read=async(name,start,end)=>{
    if(typeof source!=='string'){
      const file=source[name];
      if(!file)throw Error(`Missing model file: ${name}`);
      const bytes=new Uint8Array(await file.slice(start??0,end??file.size).arrayBuffer());
      if(start!==undefined&&bytes.length!==end-start)throw Error('Incomplete local byte range');
      transferred+=bytes.length;return bytes;
    }
    const headers=start===undefined?{}:{Range:`bytes=${start}-${end-1}`};
    const r=await fetch(`${source}/${name}`,{headers});
    if(start!==undefined && r.status!==206)throw Error(`Range support required: ${name} returned ${r.status}`);
    if(!r.ok)throw Error(`${name}: ${r.status}`);
    const bytes=new Uint8Array(await r.arrayBuffer());
    if(start!==undefined && bytes.length!==end-start)throw Error('Incomplete byte range');
    transferred+=bytes.length;return bytes;
  };
  const headers=[];
  for(const name of files){
    const prefix=await read(name,0,8);
    const n=Number(new DataView(prefix.buffer).getBigUint64(0,true));
    if(!Number.isSafeInteger(n)||n>16*1024*1024)throw Error('Invalid safetensors header');
    headers.push(new TextDecoder().decode(await read(name,8,8+n)));
  }
  const data=await Promise.all(['config.json','decision_config.json','joint_config.json','tokenizer.json','readout.safetensors'].map(n=>read(n)));
  if (backend === "webgpu" && !api.PplxFlashWebGpu) throw Error("Build this package with the webgpu feature");
  if(configuration && (backend!=="webgpu" || !api.PplxFlashWebGpu?.createConfigured))throw Error('Configured checkpoint checks require a kernel-checks WebGPU build');
  const model=configuration ? await api.PplxFlashWebGpu.createConfigured(...data,JSON.stringify(headers),configuration.mode,Boolean(configuration.profile)) : backend === "webgpu" ? await api.PplxFlashWebGpu.create(...data,JSON.stringify(headers)) : new api.PplxFlashModel(...data,JSON.stringify(headers));
  let adapter=backend === "webgpu" ? JSON.parse(model.adapter()) : null;
  const tensor=async name=>{const r=JSON.parse(model.tensorRange(name));return read(files[r.file],r.start,r.end);};
  const embedding=JSON.parse(model.tensorRange('language_model.embed_tokens.weight'));
  const embeddingCache=createEmbeddingCache(id=>read(files[embedding.file],embedding.start+id*5120,embedding.start+(id+1)*5120));
  const norm=await tensor('language_model.norm.weight');
  let worker,sequence=0,busy=false;
  const pendingReads=new Map();
  function loadLayer(layer,prefetch) {
    const ranges=JSON.parse(model.ranges(layer));
    const groups=[...new Set(ranges.map(r=>r.file))].map(file=>{
      const group=ranges.filter(r=>r.file===file);
      return {group,file,lo:Math.min(...group.map(r=>r.start)),hi:Math.max(...group.map(r=>r.end))};
    });
    if(!prefetch)return Promise.all(groups.map(async g=>({...g,bytes:await read(files[g.file],g.lo,g.hi)})));
    if(!worker){
      worker=new Worker(new URL('./read-worker.js',import.meta.url),{type:'module'});
      worker.onmessage=({data})=>{
        const waiting=pendingReads.get(data.id);pendingReads.delete(data.id);
        if(!waiting)return;
        if(data.error)waiting.reject(Error(data.error));
        else{transferred+=data.buffers.reduce((n,b)=>n+b.byteLength,0);waiting.resolve(data.buffers);}
      };
      worker.onerror=event=>{for(const p of pendingReads.values())p.reject(Error(event.message));pendingReads.clear();};
    }
    const id=sequence++;
    const promise=new Promise((resolve,reject)=>pendingReads.set(id,{resolve,reject}));
    worker.postMessage({id,requests:groups.map(g=>({...(typeof source==='string'?{url:new URL(`${source}/${files[g.file]}`,location.href).href}:{file:source[files[g.file]]}),start:g.lo,end:g.hi}))});
    const result=promise.then(buffers=>groups.map((g,i)=>({...g,bytes:new Uint8Array(buffers[i])})));
    result.catch(()=>{});return result;
  }
  if(resident){
    try{
      if(typeof model.stageTensor!=='function'||typeof model.cacheLayer!=='function')throw Error('Resident weights require an updated kernel-checks build');
      let pending=loadLayer(0,true);
      for(let layer=0;layer<32;layer++){
        progress(`Preparing weights ${layer+1}/32`);
        for(const {group,lo,bytes} of await pending){
          for(const r of group)model.stageTensor(layer,r.name,bytes.subarray(r.start-lo,r.end-lo));
        }
        if(layer<31)pending=loadLayer(layer+1,true);
        await model.cacheLayer(layer);
      }
      adapter=JSON.parse(model.adapter());
      if(adapter.weight_residency!=='all-decoder-layers')throw Error('Resident weight preparation did not complete');
    }catch(error){worker?.terminate();embeddingCache.clear();model.free();throw error;}
  }
  const startup={elapsed_ms:performance.now()-initializationStart,bytes_read:transferred,
    weight_residency:resident?'all-decoder-layers':'one-layer',
    resident_decoder_weight_bytes:adapter?.resident_decoder_weight_bytes??0};
  return {model,startup,dispose(){if(busy)throw Error("Inference is still running");worker?.terminate();embeddingCache.clear();model.free();},async predict(request,maxLength=2048,prefetch=true){
    if(busy)throw Error("One request per model instance may run at a time");
    busy=true;
    try{
    const start=performance.now(),chunks=JSON.parse(model.prepare(JSON.stringify(request),maxLength)),timings=[];
    const prepare_ms=performance.now()-start;let embedding_ms=0,readout_ms=0,decoder_weight_bytes_read=0;
    const embedding_cache={hits:0,misses:0,retained_bytes:0,capacity_bytes:0};
    const before=transferred;
    for(const chunk of chunks){
      const embeddingStart=performance.now();
      const cached=await embeddingCache.load(chunk.ids);model.begin(cached.rows);
      embedding_cache.hits+=cached.hits;embedding_cache.misses+=cached.misses;
      embedding_cache.retained_bytes=cached.retained_bytes;embedding_cache.capacity_bytes=cached.capacity_bytes;
      embedding_ms+=performance.now()-embeddingStart;
      let pending=resident?null:loadLayer(0,prefetch);
      for(let layer=0;layer<32;layer++){
        progress(`Layer ${layer+1}/32`);
        const loadStart=performance.now();
        for(const {group,lo,bytes} of resident?[]:await pending){
          const data=await bytes;
          decoder_weight_bytes_read+=data.length;
          for(const r of group)model.tensor(r.name,data.subarray(r.start-lo,r.end-lo));
        }
        const loaded=performance.now();
        if(!resident&&prefetch&&layer<31){pending=loadLayer(layer+1,prefetch);}
        const compute=performance.now();const phases=JSON.parse(await model.runLayer());
        timings.push({chunk:timings.length/32|0,layer,...phases,read_ms:resident?0:loaded-loadStart,prefetch_schedule_ms:resident?0:compute-loaded,pack_compute_ms:performance.now()-compute});
        if(!resident&&!prefetch&&layer<31)pending=loadLayer(layer+1,prefetch);
        await new Promise(resolve=>setTimeout(resolve,0));
      }
      const readoutStart=performance.now();await model.finishChunk(norm);readout_ms+=performance.now()-readoutStart;
    }
    const response=JSON.parse(model.result());
    const elapsed_ms=performance.now()-start;
    return {response,backend,adapter,elapsed_ms,request_ms:resident?elapsed_ms:null,
      timing_scope:resident?'warm-request':'streamed-request-diagnostic',decoder_weight_bytes_read,
      prepare_ms,embedding_ms,embedding_cache,readout_ms,bytes_read:transferred-before,prefetch:!resident&&prefetch,fetch_worker:!resident&&prefetch,layers:timings,...(configuration?{configuration,diagnostics:JSON.parse(model.diagnostics())}:{})};
    }finally{busy=false;}
  }};
}
