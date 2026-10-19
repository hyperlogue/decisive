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

const packedFiles={};
for(const name of ['config.json','decision_config.json','joint_config.json','tokenizer.json'])packedFiles[name]=new Blob(['{}']);
const layers=Array.from({length:32},(_,i)=>({file:`q4-e4-layer-${String(i).padStart(2,'0')}.bin`,bytes:4,tensors:[{name:`layer${i}`,start:0,end:4}]}));
for(const layer of layers)packedFiles[layer.file]=new Blob([new Uint8Array(4)]);
packedFiles['q4-e4-embedding.bin']=new Blob([new Uint8Array(3*1440)]);
packedFiles['q4-e4-readout.bin']=new Blob([new Uint8Array(4)]);
packedFiles['q4-e4-norm.bin']=new Blob([new Uint8Array(4)]);
packedFiles['packed-q4-e4.json']=new Blob([JSON.stringify({schema_version:1,scheme:'q4-w4a8-e4-r4-g64',revision:'6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d',embedding_row_bytes:1440,headers:['{}','{}'],layers,decoder_bytes:128,embedding:'q4-e4-embedding.bin',readout:'q4-e4-readout.bin',norm:'q4-e4-norm.bin'})]);
let packedUploads=0;
class Packed extends Model{
  static async createConfigured(){return new Packed();}
  ranges(){throw Error('Packed mode used BF16 offsets');}
  tensor(name,bytes){assert.equal(name,`layer${packedUploads%32}`);assert.equal(bytes.length,4);packedUploads++;}
  runLayer(){return JSON.stringify({compute_ms:0,pack_ms:0});}
  begin(bytes){assert.equal(bytes.length,2880);}
}
const packed=await loadFlash({PplxFlashWebGpu:Packed},packedFiles,()=>{},'webgpu',{mode:'q4-e4-serial',resident:false});
for(const ids of [[0,1],[0,1],[0,2]]){
  const result=await packed.predict({ids});
  assert.equal(result.timing_scope,'prepacked-streamed-request');
  assert.equal(result.request_ms,result.elapsed_ms);assert.equal(result.decoder_weight_bytes_read,128);
  assert.ok(result.layers.every(l=>l.pack_ms===0));
}
assert.equal(packedUploads,96);packed.dispose();
console.log('Packed Q4-E4: packed embedding rows, no BF16 source files, repeated layer reads/uploads, and request accounting passed');

let residentPackedModel,failResidentLayer=-1;
class ResidentPacked extends Packed{
  constructor(){super();this.cached=0;this.staged=0;this.executed=0;this.pending=[];this.highWater=0;this.aborts=0;this.uploads=0;this.freed=false;}
  static async createConfigured(){residentPackedModel=new ResidentPacked();return residentPackedModel;}
  async stageTensor(layer,name,bytes){
    assert.equal(layer,this.cached);assert.equal(name,`layer${layer}`);assert.equal(bytes.length,4);
    await Promise.resolve();
    if(layer===failResidentLayer)throw Error('Resident allocation failed');
    this.staged++;this.uploads++;
  }
  async cacheLayer(layer){assert.equal(layer,this.cached);assert.equal(this.staged,this.cached+1);this.cached++;}
  adapter(){return JSON.stringify({weight_residency:this.cached===32?'all-decoder-layers':'one-layer',resident_decoder_weight_bytes:this.cached*4});}
  prepare(request){assert.equal(this.pending.length,0);this.next=0;return super.prepare(request);}
  supportsQueued(){return true;}
  queueLayer(){
    assert.equal(this.cached,32);assert.ok(this.pending.length<2);assert.ok(this.next<32);
    const layer=this.next++;this.pending.push(layer);this.highWater=Math.max(this.highWater,this.pending.length);this.executed++;
    return JSON.stringify({layer,pack_ms:null,compute_ms:null,in_flight_layers:this.pending.length});
  }
  async waitLayer(){
    await Promise.resolve();assert.ok(this.pending.length>0);
    if(this.failCollection){this.failCollection=false;throw Error('GPU completion failed');}
    return JSON.stringify({layer:this.pending.shift(),queue_wait_ms:0});
  }
  async abortQueued(){this.aborts++;this.pending=[];}
  tensor(){throw Error('Resident request attempted a packed weight upload');}
  runLayer(){assert.equal(this.cached,32);this.executed++;return JSON.stringify({compute_ms:0,pack_ms:0});}
  free(){assert.equal(this.pending.length,0);this.freed=true;}
}
const residentPacked=await loadFlash({PplxFlashWebGpu:ResidentPacked},packedFiles,()=>{},'webgpu',{mode:'q4-e4',resident:true});
assert.equal(residentPacked.startup.resident_decoder_weight_bytes,128);
assert.equal(residentPackedModel.uploads,32);assert.equal(residentPackedModel.executed,0);
const residentResults=[];
for(const ids of [[0,1],[0,1],[0,2]])residentResults.push(await residentPacked.predict({ids}));
for(const result of residentResults){
  assert.equal(result.timing_scope,'warm-request');assert.equal(result.decoder_weight_bytes_read,0);
  assert.equal(result.double_buffered,false);assert.equal(result.weight_buffers,null);
  assert.equal(result.queued_execution,true);assert.equal(result.max_in_flight_layers,2);
  assert.equal(result.layers.length,32);assert.ok(result.layers.every(l=>l.pack_ms===0&&l.read_ms===0&&l.upload_enqueue_ms===0));
}
assert.deepEqual(residentResults.map(r=>r.bytes_read),[2880,0,1440]);
assert.equal(residentPackedModel.uploads,32);assert.equal(residentPackedModel.executed,96);assert.equal(residentPackedModel.highWater,2);
residentPackedModel.failCollection=true;
await assert.rejects(residentPacked.predict({ids:[0,1]}),/GPU completion failed/);
assert.equal(residentPackedModel.aborts,1);assert.equal(residentPackedModel.pending.length,0);
await residentPacked.predict({ids:[0,1]});
residentPacked.dispose();assert.equal(residentPackedModel.freed,true);
const residentSerial=await loadFlash({PplxFlashWebGpu:ResidentPacked},packedFiles,()=>{},'webgpu',{mode:'q4-e4',resident:true,queued:false});
const serialResult=await residentSerial.predict({ids:[0,1]});
assert.equal(serialResult.queued_execution,false);assert.equal(serialResult.decoder_weight_bytes_read,0);assert.equal(residentPackedModel.executed,32);
residentSerial.dispose();
failResidentLayer=5;
await assert.rejects(loadFlash({PplxFlashWebGpu:ResidentPacked},packedFiles,()=>{},'webgpu',{mode:'q4-e4',resident:true}),/Resident allocation failed/);
assert.equal(residentPackedModel.cached,5);assert.equal(residentPackedModel.freed,true);
console.log('Resident packed Q4-E4: awaited one-time uploads, queued backpressure, serial fallback, zero request weight I/O, completion failure recovery, and allocation failure cleanup passed');
