import test from 'node:test';
import assert from 'node:assert/strict';
import {inferenceErrorKey} from '../.compiled/lib/inference-errors.js';

test('unsupported selections and build features keep actionable messages',()=>{
  for(const code of ['UNSUPPORTED_MODEL','UNSUPPORTED_BACKEND','UNSUPPORTED_FORMAT','MODEL_NOT_COMPILED','BACKEND_NOT_COMPILED','FORMAT_NOT_COMPILED']){
    assert.equal(inferenceErrorKey(code),'modelUnsupported',code);
  }
  assert.equal(inferenceErrorKey('MISSING_GPU_FEATURE'),'gpuPrecisionRequired');
  assert.equal(inferenceErrorKey('GPU_UNAVAILABLE'),'webgpuUnavailable');
  assert.equal(inferenceErrorKey('RESOURCE_LIMIT'),'modelMemoryLimit');
});
test('cancellation and context errors stay distinct from unknown failures',()=>{
  assert.equal(inferenceErrorKey('CANCELLED'),'paused');
  assert.equal(inferenceErrorKey('INPUT_TOO_LONG','pplx-q4'),'pplxInputTooLong');
  assert.equal(inferenceErrorKey('INPUT_TOO_LONG','pplx-q8'),'pplxInputTooLong');
  assert.equal(inferenceErrorKey('INPUT_TOO_LONG','decima'),'inputTooLong');
  assert.equal(inferenceErrorKey('INFERENCE_FAILED'),undefined);
  assert.equal(inferenceErrorKey(undefined),undefined);
});
test('each Pplx catalog engine selects its packed weight format',async()=>{
  const {inferenceConfig}=await import('../.compiled/lib/inference-config.js');
  assert.equal(inferenceConfig({engine:'pplx-q4'},{}).format,'q4-e4');
  assert.equal(inferenceConfig({engine:'pplx-q8'},{}).format,'q8-e8');
  assert.equal(inferenceConfig({engine:'pplx-q8'},{}).parameters.maxSequenceTokens,1024);
});
