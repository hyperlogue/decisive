import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {selectModel,requirePublished,emptyModelState,validateModelState} from '../lib/model-catalog.js';

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
  const q8=requirePublished(selectModel(catalog,'pplx-flash-q8'));
  assert.match(q8.revision,/^[a-f0-9]{40}$/);
  assert.equal(q8.engine,'pplx-q8');
  assert.equal(emptyModelState(q8).total,4490404034);
  assert.equal(Object.keys(q8.files).length,42);
  assert.equal(Object.keys(q8.files).filter(name=>/^q8-e8-layer-/.test(name)).length,32);
  assert.ok(Object.values(q8.files).every(f=>f.bytes>0&&/^[a-f0-9]{64}$/.test(f.sha256)));
  assert.deepEqual(q8.requiredFeatures,['shader-f16']);
  assert.equal(new Set(catalog.models.map(m=>m.storeName)).size,catalog.models.length);
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
