import {createEmbeddingCache} from './embedding-cache.js';

export async function loadFlash(api, source, progress = () => {}, backend = 'wasm', configuration = null) {
  const initializationStart=performance.now(),resident=configuration?.resident===true;
  const packed=configuration?.mode?.startsWith('q4-e4')===true;
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
  let packedManifest;
  if(packed){
    packedManifest=JSON.parse(new TextDecoder().decode(await read('packed-q4-e4.json')));
    if(packedManifest.schema_version!==1||packedManifest.scheme!=='q4-w4a8-e4-r4-g64'||packedManifest.revision!=='6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d'||packedManifest.layers.length!==32||packedManifest.embedding_row_bytes!==1440)throw Error('Invalid packed Q4-E4 manifest');
    headers.push(...packedManifest.headers);
    for(const layer of packedManifest.layers){if(!/^q4-e4-layer-\d{2}\.bin$/.test(layer.file))throw Error('Invalid packed layer path');files.push(layer.file);}
  }else{
  for(const name of files){
    const prefix=await read(name,0,8);
    const n=Number(new DataView(prefix.buffer).getBigUint64(0,true));
    if(!Number.isSafeInteger(n)||n>16*1024*1024)throw Error('Invalid safetensors header');
    headers.push(new TextDecoder().decode(await read(name,8,8+n)));
  }
  }
  const data=await Promise.all(['config.json','decision_config.json','joint_config.json','tokenizer.json',packed?packedManifest.readout:'readout.safetensors'].map(n=>read(n)));
  if (backend === "webgpu" && !api.PplxFlashWebGpu) throw Error("Build this package with the webgpu feature");
  if(configuration && (backend!=="webgpu" || !api.PplxFlashWebGpu?.createConfigured))throw Error('Configured checkpoint checks require a kernel-checks WebGPU build');
  const model=configuration ? await api.PplxFlashWebGpu.createConfigured(...data,JSON.stringify(headers),configuration.mode,Boolean(configuration.profile)) : backend === "webgpu" ? await api.PplxFlashWebGpu.create(...data,JSON.stringify(headers)) : new api.PplxFlashModel(...data,JSON.stringify(headers));
  let weightBuffers=null;
  if(backend==='webgpu'&&!resident&&configuration?.doubleBuffering!==false&&configuration?.mode!=='q4-e4-serial'){
    try{
      if(typeof model.weightBuffers!=='function')throw Error('GPU double buffering requires an updated Wasm build');
      weightBuffers=model.weightBuffers();
    }catch(error){model.free();throw error;}
  }
  let adapter=backend === "webgpu" ? JSON.parse(model.adapter()) : null;
  const tensor=async name=>{const r=JSON.parse(model.tensorRange(name));return read(files[r.file],r.start,r.end);};
  const embedding=JSON.parse(model.tensorRange('language_model.embed_tokens.weight'));
  const embeddingCache=packed?createEmbeddingCache(id=>read(packedManifest.embedding,id*1440,(id+1)*1440),{rowBytes:1440}):createEmbeddingCache(id=>read(files[embedding.file],embedding.start+id*5120,embedding.start+(id+1)*5120));
  const norm=packed?await read(packedManifest.norm):await tensor('language_model.norm.weight');
  let worker,sequence=0,busy=false;
  const pendingReads=new Map();
  function loadLayer(layer,prefetch) {
    const ranges=packed?packedManifest.layers[layer].tensors.map(t=>({...t,file:layer+2})):JSON.parse(model.ranges(layer));
    const groups=[];
    for(const file of new Set(ranges.map(r=>r.file))){
      let current=null;
      for(const r of ranges.filter(r=>r.file===file).sort((a,b)=>a.start-b.start)){
        if(!current||current.hi-current.lo>=4<<20)groups.push(current={group:[],file,lo:r.start,hi:r.end});
        current.group.push(r);current.hi=Math.max(current.hi,r.end);
      }
    }
    if(!prefetch)return Promise.all(groups.map(async g=>({...g,bytes:await read(files[g.file],g.lo,g.hi)})));
    if(!worker){
      worker=new Worker(new URL('./read-worker.js',import.meta.url),{type:'module'});
      worker.onmessage=({data})=>{
        const waiting=pendingReads.get(data.id);pendingReads.delete(data.id);
        if(!waiting)return;
        if(data.error)waiting.reject(Error(data.error));
        else{transferred+=data.buffers.reduce((n,b)=>n+b.byteLength,0);waiting.resolve(data.buffers);}
      };
      worker.onerror=event=>{for(const p of pendingReads.values())p.reject(Error(event.message));pendingReads.clear();worker.terminate();worker=undefined;};
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
          for(const r of group)await model.stageTensor(layer,r.name,bytes.subarray(r.start-lo,r.end-lo));
        }
        if(layer<31)pending=loadLayer(layer+1,true);
        await model.cacheLayer(layer);
      }
      adapter=JSON.parse(model.adapter());
      if(adapter.weight_residency!=='all-decoder-layers')throw Error('Resident weight preparation did not complete');
      if(packed&&adapter.resident_decoder_weight_bytes!==packedManifest.decoder_bytes)throw Error('Resident packed weight allocation mismatch');
    }catch(error){worker?.terminate();embeddingCache.clear();model.free();throw error;}
  }
  const startup={elapsed_ms:performance.now()-initializationStart,bytes_read:transferred,
    prepacked_decoder_bytes:packedManifest?.decoder_bytes??0,
    weight_residency:resident?'all-decoder-layers':weightBuffers?'two-layer-banks':'one-layer',
    resident_decoder_weight_bytes:adapter?.resident_decoder_weight_bytes??0};
  return {model,startup,dispose(){if(busy)throw Error("Inference is still running");worker?.terminate();embeddingCache.clear();weightBuffers?.free();model.free();},async predict(request,maxLength=2048,prefetch=true){
    if(busy)throw Error("One request per model instance may run at a time");
    busy=true;
    try{
    const queued=Boolean((resident||weightBuffers&&prefetch)&&configuration?.queued!==false&&model.supportsQueued?.());
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
      if(queued&&resident){
        let pending=0;
        const first=timings.length;
        const collect=async()=>{
          const completed=JSON.parse(await model.waitLayer());
          Object.assign(timings[first+completed.layer],completed);
          pending--;
        };
        try{
          for(let layer=0;layer<32;layer++){
            progress(`Layer ${layer+1}/32`);
            if(pending===2)await collect();
            const phases=JSON.parse(model.queueLayer());
            pending++;
            timings.push({chunk:first/32,layer,...phases,pack_ms:0,read_ms:0,
              read_wait_ms:0,upload_enqueue_ms:0,queue_wait_ms:0});
          }
          while(pending)await collect();
        }catch(error){
          try{await model.abortQueued();}
          catch(cleanup){throw new AggregateError([error,cleanup],'Resident inference and cleanup failed');}
          throw error;
        }
      }else if(queued){
        let reading=loadLayer(0,true),lookahead=null,pending=0;
        const first=timings.length;
        const collect=async()=>{
          const completed=JSON.parse(await model.waitLayer());
          Object.assign(timings[first+completed.layer],completed);
          pending--;
        };
        try{
          for(let layer=0;layer<32;layer++){
            progress(`Layer ${layer+1}/32`);
            const readStart=performance.now(),groups=await reading,loaded=performance.now();
            lookahead=layer<31?loadLayer(layer+1,true):null;
            if(pending===2)await collect();
            const uploadStart=performance.now();
            for(const {group,lo,bytes} of groups){
              decoder_weight_bytes_read+=bytes.length;
              for(const r of group)weightBuffers.stageTensor(layer,r.name,bytes.subarray(r.start-lo,r.end-lo));
            }
            weightBuffers.finishUpload(layer);
            const uploaded=performance.now();
            const phases=JSON.parse(model.queueLayer());
            pending++;
            timings.push({chunk:first/32,layer,...phases,read_ms:loaded-readStart+uploaded-uploadStart,
              read_wait_ms:loaded-readStart,upload_enqueue_ms:uploaded-uploadStart,
              weight_bank:layer%2,gpu_upload_prefetch:layer<31,queue_wait_ms:0});
            reading=lookahead;lookahead=null;
          }
          while(pending)await collect();
        }catch(error){
          const drained=await Promise.allSettled([reading,lookahead,model.abortQueued()]);
          if(drained[2].status==='rejected')throw new AggregateError([error,drained[2].reason],'Queued inference and cleanup failed');
          throw error;
        }
      }else if(weightBuffers){
        const upload=async(layer,reading)=>{
          const start=performance.now(),groups=await reading,loaded=performance.now();
          for(const {group,lo,bytes} of groups){
            decoder_weight_bytes_read+=bytes.length;
            for(const r of group)weightBuffers.stageTensor(layer,r.name,bytes.subarray(r.start-lo,r.end-lo));
          }
          weightBuffers.finishUpload(layer);
          const done=performance.now();
          return {read_ms:done-start,read_wait_ms:loaded-start,upload_enqueue_ms:done-loaded};
        };
        let uploaded=await upload(0,loadLayer(0,prefetch));
        for(let layer=0;layer<32;layer++){
          progress(`Layer ${layer+1}/32`);
          const start=performance.now();
          let next=null;
          if(prefetch&&layer<31){
            const reading=loadLayer(layer+1,true);
            const submitted=weightBuffers.whenSubmitted(layer);
            next=(async()=>{
              try{await submitted;return await upload(layer+1,reading);}
              finally{await Promise.allSettled([reading]);}
            })();
          }
          let running;
          try{
            running=Promise.resolve(model.runLayer()).then(value=>({
              phases:JSON.parse(value),pack_compute_ms:performance.now()-start
            }));
          }catch(error){
            weightBuffers.reset();
            if(next)await Promise.allSettled([next]);
            throw error;
          }
          const [computed,prefetched]=await Promise.allSettled([running,next]);
          if(computed.status==='rejected')throw computed.reason;
          if(prefetched.status==='rejected')throw prefetched.reason;
          const pipeline_ms=performance.now()-start;
          timings.push({chunk:timings.length/32|0,layer,...computed.value.phases,...uploaded,
            pack_compute_ms:computed.value.pack_compute_ms,pipeline_ms,
            weight_bank:layer%2,prefetch_schedule_ms:0,gpu_upload_prefetch:prefetch&&layer<31});
          if(layer<31){
            uploaded=next?prefetched.value:await upload(layer+1,loadLayer(layer+1,false));
            // Keep read_ms additive with compute_ms for existing reports. The
            // next bank only stalls the pipeline if its upload finishes later
            // than this layer. Detailed read/upload intervals can overlap.
            if(next)uploaded.read_ms=Math.max(0,pipeline_ms-computed.value.pack_compute_ms);
          }
          await new Promise(resolve=>setTimeout(resolve,0));
        }
      }else{
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
      }
      const readoutStart=performance.now();await model.finishChunk(norm);readout_ms+=performance.now()-readoutStart;
    }
    const response=JSON.parse(model.result());
    const elapsed_ms=performance.now()-start;
    if(packed&&(decoder_weight_bytes_read!==(resident?0:packedManifest.decoder_bytes*chunks.length)||timings.some(l=>l.pack_ms!=null&&l.pack_ms!==0)))throw Error('Packed request accounting mismatch');
    return {response,backend,adapter:queued?{...adapter,timing:'queued execution: host encoding and completion collection; per-layer GPU compute time unavailable'}:adapter,elapsed_ms,request_ms:resident||packed?elapsed_ms:null,
      timing_scope:resident?'warm-request':packed?'prepacked-streamed-request':'streamed-request-diagnostic',decoder_weight_bytes_read,
      prepare_ms,embedding_ms,embedding_cache,readout_ms,bytes_read:transferred-before,prefetch:!resident&&prefetch,fetch_worker:!resident&&prefetch,
      double_buffered:Boolean(weightBuffers&&prefetch),weight_buffers:weightBuffers?JSON.parse(weightBuffers.diagnostics()):null,
      queued_execution:queued,max_in_flight_layers:queued?2:1,
      layers:timings,...(configuration?{configuration,diagnostics:JSON.parse(model.diagnostics())}:{})};
    }catch(error){weightBuffers?.reset();throw error;}finally{busy=false;}
  }};
}
