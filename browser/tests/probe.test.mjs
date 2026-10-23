import test from 'node:test';
import assert from 'node:assert/strict';
import {probe} from '../dist/probe.js';

test('capability probes acquire a fresh adapter for each device request',async()=>{
  let adapters=0,destroyed=0;
  const old=Object.getOwnPropertyDescriptor(navigator,'gpu');
  Object.defineProperty(navigator,'gpu',{configurable:true,value:{async requestAdapter(){
    adapters++;let consumed=false;
    return {features:new Set(['shader-f16']),async requestDevice(){
      if(consumed)throw new DOMException('Adapter was consumed','OperationError');
      consumed=true;return {destroy(){destroyed++;}};
    }};
  }}});
  try{
    assert.deepEqual(await probe(),{webgpu:true,shaderF16:true});
    assert.equal(adapters,2);assert.equal(destroyed,2);
  }finally{
    if(old)Object.defineProperty(navigator,'gpu',old);else delete navigator.gpu;
  }
});
