import assert from 'node:assert/strict';
import {loadFlash} from '../../examples/pplx-flash/runtime.js';

const files={};
for(const name of ['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']){
  const bytes=new Uint8Array(20000);new DataView(bytes.buffer).setBigUint64(0,2n,true);bytes.set([123,125],8);
  files[name]=new Blob([bytes]);
}
for(const name of ['config.json','decision_config.json','joint_config.json','tokenizer.json','readout.safetensors'])files[name]=new Blob(['{}']);
let staged=0,prepared=0,executed=0,freed=0,terminated=0,fail=false;
globalThis.Worker=class{
  async postMessage({id,requests}){
    const buffers=await Promise.all(requests.map(r=>r.file.slice(r.start,r.end).arrayBuffer()));
    this.onmessage({data:{id,buffers}});
  }
  terminate(){terminated++;}
};
class Model{
  static async createConfigured(){return new Model();}
  tensorRange(name){return JSON.stringify({file:0,start:10,end:name.includes('embed_tokens')?15370:12});}
  ranges(layer){return JSON.stringify([{name:`layer${layer}`,file:1,start:100+layer*4,end:104+layer*4}]);}
  stageTensor(layer,name,bytes){assert.equal(layer,prepared);assert.equal(name,`layer${layer}`);assert.equal(bytes.length,4);staged++;}
  cacheLayer(layer){if(fail)throw Error('Preparation failed');assert.equal(layer,prepared);prepared++;}
  adapter(){return JSON.stringify({weight_residency:prepared===32?'all-decoder-layers':'one-layer',resident_decoder_weight_bytes:128});}
  prepare(request){return JSON.stringify([{ids:JSON.parse(request).ids}]);}
  begin(){}
  tensor(){throw Error('Warm request attempted a weight upload');}
  runLayer(){assert.equal(prepared,32);executed++;return JSON.stringify({compute_ms:0,pack_ms:0});}
  finishChunk(){}
  result(){return JSON.stringify({answers:{choice:'a'},usage:{input_tokens:2,sequences:1}});}
  diagnostics(){return JSON.stringify({logits:[{keys:['choice'],values:[[1,2]]}]});}
  free(){freed++;}
}
const engine=await loadFlash({PplxFlashWebGpu:Model},files,()=>{},'webgpu',{mode:'int8-tuned',resident:true});
assert.equal(prepared,32);assert.equal(staged,32);assert.equal(executed,0);
assert.equal(engine.startup.weight_residency,'all-decoder-layers');
const first=await engine.predict({ids:[0,1]});
const repeated=await engine.predict({ids:[0,1]});
const different=await engine.predict({ids:[0,2]});
for(const run of [first,repeated,different]){
  assert.equal(run.timing_scope,'warm-request');assert.equal(run.request_ms,run.elapsed_ms);
  assert.equal(run.decoder_weight_bytes_read,0);assert.equal(run.layers.length,32);
  assert.ok(run.layers.every(l=>l.pack_ms===0&&l.read_ms===0));
}
assert.equal(first.bytes_read,10240);assert.equal(repeated.bytes_read,0);assert.equal(different.bytes_read,5120);
assert.equal(prepared,32);assert.equal(staged,32);assert.equal(executed,96);
engine.dispose();assert.equal(freed,1);assert.equal(terminated,1);
prepared=0;fail=true;
await assert.rejects(loadFlash({PplxFlashWebGpu:Model},files,()=>{},'webgpu',{mode:'int8',resident:true}),/Preparation failed/);
assert.equal(freed,2);assert.equal(terminated,2);
console.log('Resident runtime: one-time loading, repeated/changing inputs, request accounting and failure cleanup passed');
