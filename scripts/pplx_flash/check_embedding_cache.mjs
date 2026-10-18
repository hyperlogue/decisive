import assert from 'node:assert/strict';
import {Blob} from 'node:buffer';
import {pathToFileURL} from 'node:url';

export async function checkEmbeddingCache(root) {
  const {createEmbeddingCache} = await import(pathToFileURL(`${root}/examples/pplx-flash/embedding-cache.js`));
  const reads = [];
  const row = id => Uint8Array.of(id, id + 1, id + 2, id + 3);
  const cache = createEmbeddingCache(async id => { reads.push(id); return row(id); }, {rowBytes: 4, capacityBytes: 8});
  const first = await cache.load([1, 2, 1]);
  assert.deepEqual([...first.rows], [...row(1), ...row(2), ...row(1)]);
  assert.equal(first.misses, 2); assert.equal(first.hits, 0); assert.equal(first.retained_bytes, 8);
  first.rows.fill(0);
  const repeat = await cache.load([2, 1]);
  assert.deepEqual([...repeat.rows], [...row(2), ...row(1)]);
  assert.equal(repeat.hits, 2); assert.equal(repeat.misses, 0); assert.equal(reads.length, 2);
  await cache.load([3]);
  assert.equal((await cache.load([1])).hits, 1);
  assert.equal((await cache.load([2])).misses, 1);
  const oversized = await cache.load([4, 5, 6, 7]);
  assert.deepEqual([...oversized.rows], [4, 5, 6, 7].flatMap(id => [...row(id)]));
  assert.equal(oversized.retained_bytes, 8);
  cache.clear(); assert.equal((await cache.load([7])).misses, 1);
  const disabled = createEmbeddingCache(async id => row(id), {rowBytes: 4, capacityBytes: 0});
  await disabled.load([1]); assert.equal((await disabled.load([1])).misses, 1);
  let broken = true;
  const recovering = createEmbeddingCache(async id => broken ? new Uint8Array(2) : row(id), {rowBytes: 4, capacityBytes: 8});
  await assert.rejects(recovering.load([1]), /row length/);
  broken = false; assert.equal((await recovering.load([1])).misses, 1);
  await assert.rejects(cache.load([-1]), /token ID/);
  const other = createEmbeddingCache(async () => Uint8Array.of(9,9,9,9), {rowBytes: 4});
  assert.deepEqual([...(await other.load([1])).rows], [9,9,9,9]);
  const {loadFlash}=await import(pathToFileURL(`${root}/examples/pplx-flash/runtime.js`));
  const makeFile=(count)=>{const bytes=new Uint8Array(10+count*5120);new DataView(bytes.buffer).setBigUint64(0,2n,true);bytes.set([123,125],8);for(let i=0;i<count;i++)bytes.fill(i,10+i*5120,10+(i+1)*5120);return new Blob([bytes]);};
  const files={'model-00001-of-00002.safetensors':makeFile(5),'model-00002-of-00002.safetensors':makeFile(1)};
  for(const name of ['config.json','decision_config.json','joint_config.json','tokenizer.json','readout.safetensors'])files[name]=new Blob(['{}']);
  const inputs=[];
  class Model {
    tensorRange(name){return JSON.stringify({file:name.includes('embed_tokens')?0:1,start:10,end:5130});}
    prepare(request){return JSON.stringify([{ids:JSON.parse(request).ids}]);}
    begin(bytes){inputs.push(bytes.slice());}
    ranges(){return '[]';}
    tensor(){throw Error('Unexpected weight upload in mock');}
    runLayer(){return '{}';}
    finishChunk(){}
    result(){return '{}';}
    free(){}
  }
  const runtime=await loadFlash({PplxFlashModel:Model},files);
  const a=await runtime.predict({ids:[1,2,1]},2048,false);
  const b=await runtime.predict({ids:[2,3,1]},2048,false);
  const c=await runtime.predict({ids:[1,2,1]},2048,false);
  assert.deepEqual(inputs[0],inputs[2]);
  assert.equal(a.bytes_read,2*5120);assert.equal(b.bytes_read,5120);assert.equal(c.bytes_read,0);
  assert.equal(b.embedding_cache.hits,2);assert.equal(c.embedding_cache.misses,0);
  runtime.dispose();
  return {byte_identity:true,cross_request_hits:true,bounded_eviction:true,request_larger_than_cache:true,failed_read_recovery:true,checkpoint_isolation:true,runtime_reads:[a.bytes_read,b.bytes_read,c.bytes_read]};
}
