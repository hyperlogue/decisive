import test from 'node:test';
import assert from 'node:assert/strict';
import {loadModelInWorker} from '../dist/local.js';

const request={state:'shared context',questions:{q:{type:'noul'}}};
function fixture(predict) {
  const stats={caches:0,released:0,controls:0,models:0,calls:[]};
  const api={
    InferenceControl:class {
      constructor(progress){this.progress=progress;stats.controls++;}
      cancel(){this.cancelled=true;}
      free(){stats.controls--;}
    },
    DecisiveModel:{async load(config,files,control){
      assert.equal(JSON.parse(config).model,'decima');
      return {
        info:()=>JSON.stringify({model:'decima',backend:'webgpu',caches:['choice'],parameters:{},tuning:{}}),
        createCache(json){stats.caches++;return {options:JSON.parse(json),clear(){this.cleared=true;},dispose(){},free(){stats.released++;}};},
        async predict(json,parameters,cache,control,batch){
          stats.calls.push({request:JSON.parse(json),parameters:JSON.parse(parameters),cache,batch});
          await predict?.(control,cache);
          if(control.cancelled)throw Object.assign(new Error('Cancelled'),{code:'CANCELLED'});
          const result={response:{answers:{q:{type:'noul',noul:.4}},usage:{input_tokens:3}},model:'decima',backend:'webgpu',elapsed_ms:2};
          return JSON.stringify(batch?JSON.parse(json).map(()=>({status:'success',result})):result);
        },
        dispose(){stats.models++;},free(){},
      };
    }},
  };
  return {api,stats,load:()=>loadModelInWorker(api,{model:'decima',backend:'webgpu',source:{files:{}},parameters:{long:true}})};
}

test('single and batch predictions share one explicit opaque cache',async()=>{
  const f=fixture(),model=await f.load(),cache=await model.createCache({maxBytes:100});
  await model.predict(request,{cache,parameters:{window:32}});
  const result=await model.predictBatch([request,{...request,state:'another context'}],{cache});
  assert.equal(result.length,2);assert.equal(f.stats.caches,1);
  assert.equal(f.stats.calls[0].cache,f.stats.calls[1].cache);
  assert.deepEqual(f.stats.calls[0].parameters,{window:32});
  await cache.clear();assert.equal(f.stats.calls[0].cache.cleared,true);
  await model.dispose();assert.equal(f.stats.released,1);assert.equal(f.stats.controls,0);
  await assert.rejects(cache.clear(),{code:'DISPOSED'});
  await assert.rejects(model.predict(request),{code:'DISPOSED'});
  await model.dispose();assert.equal(f.stats.models,1);
});

test('omitting a cache uses an ephemeral zero-retention cache',async()=>{
  const f=fixture(),model=await f.load();
  await model.predict(request);await model.predict(request);
  assert.equal(f.stats.caches,2);assert.equal(f.stats.released,2);
  assert.deepEqual(f.stats.calls.map(c=>c.cache.options),[{maxBytes:0},{maxBytes:0}]);
  await model.dispose();
});

test('cache ownership, exclusive use, and disposal are enforced while inference runs',async()=>{
  let finish;const f=fixture(()=>new Promise(resolve=>finish=resolve));
  const first=await f.load(),second=await f.load(),cache=await first.createCache();
  await assert.rejects(second.predict(request,{cache}),{code:'CACHE_OWNER'});
  const running=first.predict(request,{cache});
  await assert.rejects(first.predictBatch([request],{cache}),{code:'CACHE_BUSY'});
  await assert.rejects(cache.clear(),{code:'CACHE_BUSY'});
  await assert.rejects(cache.dispose(),{code:'CACHE_BUSY'});
  await assert.rejects(first.dispose(),{code:'MODEL_BUSY'});
  finish();await running;await first.dispose();await second.dispose();
});

test('cancellation reaches Rust and releases controls and temporary caches after failure',async()=>{
  const abort=new AbortController();
  const f=fixture(control=>{abort.abort();assert.equal(control.cancelled,true);});
  const model=await f.load();
  await assert.rejects(model.predict(request,{signal:abort.signal}),{code:'CANCELLED'});
  assert.equal(f.stats.controls,0);assert.equal(f.stats.released,1);
  await assert.rejects(model.predict(request,{signal:abort.signal}),{name:'AbortError'});
  assert.equal(f.stats.calls.length,1);
  await model.dispose();
});
