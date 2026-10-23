import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {selectModel,requirePublished,emptyModelState,validateModelState} from '../lib/model-catalog.js';
import {loadDecima} from '../lib/model-runtime.js';
import {loadPplxFlash} from '../lib/pplx-runtime.js';

test('both models pin their release and exact download size; unpublished entries stay blocked',async()=>{
  const catalog=JSON.parse(await readFile(new URL('../models.json',import.meta.url)));
  const decima=requirePublished(selectModel(catalog));
  assert.equal(decima.id,'decima-small');
  assert.match(decima.revision,/^[a-f0-9]{40}$/);
  assert.equal(emptyModelState(decima).total,506679429);
  assert.ok(Object.values(decima.files).every(f=>f.bytes>0&&/^[a-f0-9]{64}$/.test(f.sha256)));
  const q4=selectModel(catalog,'pplx-flash-q4');
  assert.equal(requirePublished(q4),q4);
  assert.equal(q4.revision,'a264abdd91f7f002d929bb038acffd51c453115d');
  assert.equal(emptyModelState(q4).total,2387682594);
  assert.equal(Object.keys(q4.files).length,42);
  assert.equal(Object.keys(q4.files).filter(name=>/^q4-e4-layer-/.test(name)).length,32);
  assert.ok(Object.values(q4.files).every(f=>f.bytes>0&&/^[a-f0-9]{64}$/.test(f.sha256)));
  assert.deepEqual(q4.requiredFeatures,['shader-f16']);
  assert.ok(!Object.keys(q4.files).some(name=>name.endsWith('.safetensors')));
  assert.equal(emptyModelState(q4).ready,false);
  assert.throws(()=>requirePublished({...q4,available:false}),/modelComingSoon/);
  assert.throws(()=>requirePublished({...q4,repository:null}),/modelComingSoon/);
  assert.notEqual(q4.storeName,decima.storeName);
});

test('Q4 uses packed local files, streaming WebGPU, the 1024-token cap, and actual output metrics',async()=>{
  const api={},files={'q4-e4-layer-00.bin':new Blob(['packed'])},progress=[];
  const request={state:'complete draft',questions:{}};
  let freed=false,seenRequest,seenLimit;
  const result={response:{answers:{},usage:{input_tokens:42}},elapsed_ms:123,backend:'webgpu'};
  const runtime=await loadPplxFlash(api,files,m=>progress.push(m),async(a,f,onProgress,backend,configuration)=>{
    assert.equal(a,api);assert.equal(f,files);
    assert.equal(backend,'webgpu');
    assert.deepEqual(configuration,{mode:'q4-e4',resident:false,profile:false});
    return {dispose(){freed=true;},async predict(r,limit){seenRequest=r;seenLimit=limit;onProgress('Layer 1/32');return result;}};
  });
  const output=await runtime.predict(request);
  assert.equal(seenRequest,request);assert.equal(seenLimit,1024);
  assert.equal(output.response,result.response);assert.equal(output.elapsed_ms,123);
  assert.equal(output.model,'Pplx Flash · Q4');assert.deepEqual(progress,['Layer 1/32']);
  runtime.dispose();assert.equal(freed,true);
});

test('Q4 cancellation stops at a layer boundary and releases the signal for another request',async()=>{
  const controller=new AbortController();let calls=0;
  const runtime=await loadPplxFlash({}, {}, ()=>{}, async(a,f,progress)=>({dispose(){},async predict(){
    if(++calls===1)controller.abort();
    progress('Layer 2/32');return {response:{},elapsed_ms:1,backend:'webgpu'};
  }}));
  await assert.rejects(runtime.predict({},controller.signal),{name:'AbortError'});
  await runtime.predict({});assert.equal(calls,2);
  await assert.rejects(runtime.predict({},controller.signal),{name:'AbortError'});
  assert.equal(calls,2,'already-cancelled requests never reach inference');
});
test('Decima receives the encoder File without buffering it into WASM memory',async()=>{
  const encoder={arrayBuffer(){assert.fail('Encoder must stay a Blob/File');}};
  const files=Object.fromEntries(['pytorch/decima.json','pytorch/encoder/config.json','pytorch/encoder/tokenizer.json','pytorch/head.safetensors'].map(name=>[name,new Blob(['{}'])]));
  files['pytorch/encoder/model.safetensors']=encoder;
  let freed=false,received;
  const api={DecimaWebGpuModel:{async create(a,b,c,weights,d){
    assert.equal(weights,encoder);
    assert.ok([a,b,c,d].every(bytes=>bytes instanceof Uint8Array));
    return {async predict(json){received=JSON.parse(json);return '{"answers":{}}';},free(){freed=true;}};
  }}};
  const runtime=await loadDecima(api,files);
  const request={state:'private draft',questions:{}};
  const output=await runtime.predict(request);
  assert.deepEqual(received,request);assert.equal(output.backend,'webgpu');
  assert.deepEqual(output.response,{answers:{}});
  runtime.dispose();assert.equal(freed,true);
});

test('a stale worker cannot attach Pplx size or files to Decima',async()=>{
  const catalog=JSON.parse(await readFile(new URL('../models.json',import.meta.url)));
  const model=selectModel(catalog),state=emptyModelState(model);
  assert.equal(validateModelState(model,state),state);
  for(const stale of [
    {...state,modelId:undefined,revision:undefined,total:9099931557},
    {...state,modelId:'pplx-flash-q4'},
    {...state,revision:'previous-release'},
    {...state,total:9099931557},
    {...state,files:state.files.map(f=>({...f,expected:1}))},
  ])assert.throws(()=>validateModelState(model,stale),/versionMismatch/);
});
