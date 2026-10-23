import test from 'node:test';
import assert from 'node:assert/strict';
import {checkWebGPU} from '../../browser/dist/probe.js';

test('WebGPU must provide a usable device, not just expose an API',async()=>{
  assert.equal(await checkWebGPU(null),false);
  assert.equal(await checkWebGPU({requestAdapter:async()=>null}),false);
  assert.equal(await checkWebGPU({requestAdapter:async()=>{throw Error('blocked');}}),false);
  assert.equal(await checkWebGPU({requestAdapter:async()=>({requestDevice:async()=>{throw Error('device unavailable');}})}),false);
});
test('the successful probe releases its temporary GPU device',async()=>{
  let destroyed=false;
  const gpu={requestAdapter:async()=>({requestDevice:async()=>({destroy(){destroyed=true;}})})};
  assert.equal(await checkWebGPU(gpu),true);
  assert.equal(destroyed,true);
});
test('Q4 requires a working shader-f16 device while Decima needs only baseline WebGPU',async()=>{
  let descriptors=[],destroyed=0;
  const baseline={requestAdapter:async()=>({features:new Set(),requestDevice:async options=>{
    descriptors.push(options);return {destroy(){destroyed++;}};
  }})};
  assert.equal(await checkWebGPU(baseline),true);
  assert.equal(await checkWebGPU(baseline,['shader-f16']),false);
  assert.equal(descriptors.length,1,'unsupported precision must not request a device');
  const supported={requestAdapter:async()=>({features:new Set(['shader-f16']),requestDevice:async options=>{
    descriptors.push(options);return {destroy(){destroyed++;}};
  }})};
  assert.equal(await checkWebGPU(supported,['shader-f16']),true);
  assert.deepEqual(descriptors[1],{requiredFeatures:['shader-f16']});
  assert.equal(destroyed,2);
});
