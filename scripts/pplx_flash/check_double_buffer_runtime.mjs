import assert from 'node:assert/strict';
import {loadFlash} from '../../examples/pplx-flash/runtime.js';

const files={};
for(const name of ['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']){
  const bytes=new Uint8Array(20000),view=new DataView(bytes.buffer);
  view.setBigUint64(0,2n,true);bytes.set([123,125],8);
  for(let i=0;i<32;i++)view.setUint32(100+i*8,i+1,true);
  files[name]=new Blob([bytes]);
}
for(const name of ['config.json','decision_config.json','joint_config.json','tokenizer.json','readout.safetensors'])files[name]=new Blob(['{}']);
let readFailure=-1,computeFailure=-1,failBeforeSubmit=false,slowReads=false,lastModel,activeReads=0;
globalThis.Worker=class{
  async postMessage({id,requests}){
    activeReads++;
    try{
    if(slowReads)await new Promise(resolve=>setTimeout(resolve,18));
    if(requests.some(r=>r.start===100+readFailure*8)){
      this.onmessage({data:{id,error:'Injected read failure'}});return;
    }
    const buffers=await Promise.all(requests.map(r=>r.file.slice(r.start,r.end).arrayBuffer()));
    this.onmessage({data:{id,buffers}});
    }finally{activeReads--;}
  }
  terminate(){}
};
class Buffers{
  constructor(model){this.model=model;this.reset();this.events=[];}
  reset(){assert.equal(this.model.computing,null,'reset raced GPU completion');this.banks=[null,null];this.waiter?.reject(Error('cancelled'));this.waiter=null;}
  stageTensor(layer,name,bytes){
    assert.notEqual(this.model.computing,layer,'active bank overwritten');
    assert.equal(this.banks[layer%2],null,'more than two layers are live');
    assert.equal(name,`layer${layer}`);
    assert.equal(new DataView(bytes.buffer,bytes.byteOffset).getUint32(0,true),layer+1);
    this.banks[layer%2]={layer,value:layer+1};
    this.events.push(['upload',layer,this.model.computing]);
  }
  finishUpload(layer){assert.equal(this.banks[layer%2].layer,layer);}
  whenSubmitted(layer){assert.equal(this.waiter,null);return new Promise((resolve,reject)=>{this.waiter={layer,resolve,reject};});}
  submitted(layer){if(this.waiter){assert.equal(this.waiter.layer,layer);this.waiter.resolve();this.waiter=null;}this.events.push(['submit',layer]);}
  failed(){this.waiter?.reject(Error('Injected compute failure'));this.waiter=null;}
  diagnostics(){return JSON.stringify({banks:2});}
  free(){}
}
class Model{
  static async create(){return lastModel=new Model();}
  constructor(){this.computing=null;this.borrowed=false;this.buffers=new Buffers(this);}
  checkBorrow(){assert.equal(this.borrowed,false,'runtime borrowed the model during asynchronous computation');}
  weightBuffers(){return this.buffers;}
  adapter(){this.checkBorrow();return JSON.stringify({weight_residency:'two-layer-banks'});}
  tensorRange(name){this.checkBorrow();return JSON.stringify({file:0,start:10,end:name.includes('embed_tokens')?15370:12});}
  ranges(layer){this.checkBorrow();return JSON.stringify([{name:`layer${layer}`,file:1,start:100+layer*8,end:104+layer*8}]);}
  prepare(request){this.checkBorrow();this.buffers.reset();this.chunks=JSON.parse(request).chunks??1;this.total=0;return JSON.stringify(Array.from({length:this.chunks},()=>({ids:[0,1]})));}
  begin(){this.checkBorrow();this.layer=0;}
  tensor(){throw Error('buffered runtime used the legacy upload API');}
  async runLayer(){
    this.checkBorrow();this.borrowed=true;const layer=this.layer;
    const active=this.buffers.banks[layer%2];assert.equal(active.layer,layer);
    this.computing=layer;
    try{
      if(computeFailure===layer&&failBeforeSubmit)throw Error('Injected compute failure');
      this.buffers.submitted(layer);
      await new Promise(resolve=>setTimeout(resolve,8));
      if(computeFailure===layer)throw Error('Injected compute failure');
      assert.equal(active.value,layer+1,'lookahead changed active weights');
      this.total+=active.value;this.layer++;
      this.buffers.banks[layer%2]=null;
      this.buffers.events.push(['complete',layer]);
      return JSON.stringify({compute_ms:8,pack_ms:0});
    }catch(error){this.buffers.failed();throw error;}
    finally{this.computing=null;this.borrowed=false;}
  }
  finishChunk(){this.checkBorrow();assert.equal(this.layer,32);}
  result(){this.checkBorrow();assert.equal(this.total,this.chunks*528);return JSON.stringify({total:this.total});}
  free(){this.checkBorrow();}
}
async function engine(){return loadFlash({PplxFlashWebGpu:Model},files,()=>{},'webgpu');}
const e=await engine();
assert.equal(e.startup.weight_residency,'two-layer-banks');
const first=await e.predict({});
assert.equal(first.double_buffered,true);assert.equal(first.layers.length,32);
assert.equal(first.decoder_weight_bytes_read,128);
assert.ok(first.layers.every((l,i)=>l.weight_bank===i%2&&l.upload_enqueue_ms>=0&&l.read_wait_ms>=0&&l.read_ms>=0));
assert.ok(first.layers.reduce((sum,l)=>sum+l.read_ms+l.pack_ms+l.compute_ms,0)<=first.elapsed_ms,
  'overlapped upload time was counted twice in the phase summary');
for(let i=1;i<32;i++){
  const events=lastModel.buffers.events;
  const upload=events.findIndex(v=>v[0]==='upload'&&v[1]===i);
  const submit=events.findIndex(v=>v[0]==='submit'&&v[1]===i-1);
  const complete=events.findIndex(v=>v[0]==='complete'&&v[1]===i-1);
  assert.ok(submit<upload&&upload<complete,'GPU upload did not overlap the compute wait');
}
const serial=await e.predict({},2048,false);
assert.deepEqual(serial.response,first.response);assert.equal(serial.double_buffered,false);
const multi=await e.predict({chunks:2});assert.equal(multi.layers.length,64);
slowReads=true;assert.deepEqual((await e.predict({})).response,first.response);slowReads=false;
for(const failure of ['read','before-submit','after-submit']){
  readFailure=failure==='read'?1:-1;computeFailure=failure==='read'?-1:0;failBeforeSubmit=failure==='before-submit';
  slowReads=failBeforeSubmit;
  const run=e.predict({});
  await assert.rejects(e.predict({}),/One request/);
  assert.throws(()=>e.dispose(),/still running/);
  await assert.rejects(run,/Injected/);
  assert.equal(activeReads,0,'failed request left an unfinished CPU read');
  assert.equal(lastModel.computing,null);assert.ok(lastModel.buffers.banks.every(v=>v===null));
  readFailure=computeFailure=-1;slowReads=false;
  assert.deepEqual((await e.predict({})).response,first.response,'failed request poisoned the next request');
}
e.dispose();
console.log('Double-buffer runtime: upload/compute overlap, Wasm borrow isolation, serial parity, multiple chunks, delayed reads, failure draining and retries passed');

class QueuedBuffers extends Buffers{
  reset(){assert.equal(this.model.inFlight?.length??0,0,'reset raced queued GPU work');super.reset();}
  diagnostics(){return JSON.stringify({banks:2,peak_in_flight_layers:this.model.peak});}
}
class QueuedModel extends Model{
  static async create(){return lastModel=new QueuedModel();}
  constructor(){super();this.inFlight=[];this.tail=Promise.resolve();this.peak=0;this.buffers=new QueuedBuffers(this);}
  supportsQueued(){return true;}
  queueLayer(){
    this.checkBorrow();assert.ok(this.inFlight.length<2,'GPU queue exceeded its window');
    const layer=this.layer,active=this.buffers.banks[layer%2];assert.equal(active.layer,layer);
    if(computeFailure===layer&&failBeforeSubmit)throw Error('Injected encoding failure');
    this.buffers.events.push(['queue',layer]);
    const work=this.tail.then(async()=>{
      this.computing=layer;
      try{
        await new Promise(resolve=>setTimeout(resolve,8));
        if(computeFailure===layer)throw Error('Injected queued compute failure');
        assert.equal(this.buffers.banks[layer%2],active,'queued weights changed before completion');
        this.total+=active.value;this.buffers.events.push(['complete',layer]);
      }finally{this.computing=null;}
    });
    this.tail=work.catch(()=>{});
    this.inFlight.push({layer,work});this.peak=Math.max(this.peak,this.inFlight.length);this.layer++;
    return JSON.stringify({layer,encode_enqueue_ms:0,pack_ms:null,compute_ms:null,in_flight_layers:this.inFlight.length});
  }
  async waitLayer(){
    this.checkBorrow();this.borrowed=true;
    const first=this.inFlight[0];assert.ok(first);this.buffers.events.push(['collect',first.layer]);
    const start=performance.now();
    try{await first.work;return JSON.stringify({layer:first.layer,queue_wait_ms:performance.now()-start});}
    finally{this.inFlight.shift();this.buffers.banks[first.layer%2]=null;this.borrowed=false;}
  }
  async abortQueued(){
    this.checkBorrow();this.borrowed=true;
    await Promise.allSettled(this.inFlight.map(v=>v.work));
    this.inFlight=[];this.borrowed=false;this.buffers.reset();
  }
  free(){super.free();assert.equal(this.inFlight.length,0,'dispose raced queued GPU work');}
}
const queued=await loadFlash({PplxFlashWebGpu:QueuedModel},files,()=>{},'webgpu');
const queuedFirst=await queued.predict({});
assert.equal(queuedFirst.queued_execution,true);assert.equal(queuedFirst.weight_buffers.peak_in_flight_layers,2);
assert.match(queuedFirst.adapter.timing,/queued execution/);
assert.deepEqual(queuedFirst.response,first.response);
const events=lastModel.buffers.events;
assert.ok(events.findIndex(e=>e[0]==='queue'&&e[1]===1)<events.findIndex(e=>e[0]==='collect'&&e[1]===0),
  'second layer was not submitted before waiting for the first');
assert.ok(queuedFirst.layers.every(l=>l.compute_ms===null&&l.pack_ms===null&&l.queue_wait_ms>=0));
assert.ok(queuedFirst.layers.reduce((sum,l)=>sum+l.read_ms+l.encode_enqueue_ms+l.queue_wait_ms,0)<=queuedFirst.elapsed_ms+1);
assert.deepEqual((await queued.predict({},2048,false)).response,first.response);
assert.equal((await queued.predict({chunks:2})).layers.length,64);
slowReads=true;assert.deepEqual((await queued.predict({})).response,first.response);slowReads=false;
for(const failure of ['read','before-submit','after-submit']){
  readFailure=failure==='read'?2:-1;computeFailure=failure==='read'?-1:1;failBeforeSubmit=failure==='before-submit';slowReads=failBeforeSubmit;
  const run=queued.predict({});
  await assert.rejects(queued.predict({}),/One request/);assert.throws(()=>queued.dispose(),/still running/);
  await assert.rejects(run,/Injected/);
  assert.equal(activeReads,0);assert.equal(lastModel.inFlight.length,0);assert.equal(lastModel.computing,null);
  assert.ok(lastModel.buffers.banks.every(b=>b===null));
  readFailure=computeFailure=-1;slowReads=false;
  assert.deepEqual((await queued.predict({})).response,first.response);
}
queued.dispose();
console.log('Queued runtime: two submissions before collection, bounded banks, serial parity, timing accounting, delayed reads, failure draining and retries passed');
