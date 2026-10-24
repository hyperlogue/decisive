import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {performance} from 'node:perf_hooks';
import {fileURLToPath, pathToFileURL} from 'node:url';
import vm from 'node:vm';

export async function checkSynthetic(root, page) {
  const javascript=await readFile(`${root}/examples/pplx-flash/synthetic-full.js`,'utf8');
  const html=await readFile(page,'utf8');
  for(const match of html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)) {
    if(!match[1].includes('application/octet-stream'))new vm.Script(match[2]);
  }
  const messages=[],trace=[];
  let freed=0;
  class Mock {
    static async create(half){return new Mock(half);}
    static async createProfiled(){return new Mock(true);}
    constructor(half){this.half=half;this.layer=32;}
    adapter(){return JSON.stringify({backend:'mock'});}
    configure(block,reuse){assert.ok([16,128].includes(block));assert.equal(typeof reuse,'boolean');}
    async configureGemm(m,n,unroll){this.tiled=true;assert.ok([32,64].includes(m)&&[32,64].includes(n));assert.equal(typeof unroll,'boolean');}
    async configureGemmSubtile(tile,load){assert.ok([32,64].includes(tile)&&[8,16].includes(load));}
    async configureGemmK64(tile){assert.ok([32,64].includes(tile));}
    configureInputChange(first,last,seed){this.change=seed===0?0:first===0?1:2;}
    async configureMlp(tile){assert.ok([32,64].includes(tile));}
    async configurePadding(){}
    configurePureInt8(){this.pure=true;}
    async configureInt8FloatPartials(rows){assert.ok([4,8].includes(rows));}
    async configureDeltaScan(width){assert.ok([4,16,128].includes(width));}
    async configureInt8(fused,fp16Mlp){this.int8=true;assert.equal(typeof fused,'boolean');assert.equal(typeof fp16Mlp,'boolean');assert.ok(!fused||!fp16Mlp);}
    async checkInt8(){return JSON.stringify({passed:true});}
    async configureAttention(){}
    async configurePackedInputs(){}
    async configureNorm(){}
    async configureSmallRows(tile){assert.ok([4,8].includes(tile));}
    configureBatchedPasses(){}
    async checkAttention(){return JSON.stringify({passed:true});}
    async checkMlp(){return JSON.stringify({passed:true});}
    async checkProjection(){return JSON.stringify({passed:true});}
    prepare(rows){assert.equal(this.layer,32);this.layer=0;this.rows=rows;trace.push({half:this.half,rows,tiled:Boolean(this.tiled)});return JSON.stringify({embedding_ms:1});}
    async prepareLayer(){return JSON.stringify({generation_ms:100,weight_bytes:1024});}
    async runLayer(){const layer=this.layer++;return JSON.stringify({layer,finite:true,compute_ms:this.half?1:2,audit_ms:50,hidden_max_abs:1,dependency:{prefix_hash:this.change===1||(this.change===2&&layer>=3)?'changed':'original',suffix_hash:this.change?'changed':'original'}});}
    async finish(){assert.equal(this.layer,32);return JSON.stringify({completed_layers:32,finite:true,readout_ms:1,logits:[[this.int8&&(this.rows>8||this.pure)?1.1:1,2],[3,4]],probabilities:[[0.25,0.75],[0.25,0.75]]});}
    free(){freed++;}
  }
  const bind=async()=>({memory:{buffer:new ArrayBuffer(8)}});bind.PplxFlashSyntheticGpu=Mock;
  const self={postMessage:message=>messages.push(message)};
  const navigator={gpu:{requestAdapter:async()=>({info:{vendor:'mock'},features:new Set(['shader-f16'])})}};
  new vm.Script(javascript).runInContext(vm.createContext({self,performance,wasm_bindgen:bind,navigator}));
  assert.equal(messages[0].type,'ready');
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[117,229],seed:137,warmups:1,rounds:3}});
  const report=messages.at(-1).report;
  assert.equal(report.completed,true);
  assert.equal(report.runs.length,16);
  assert.equal(freed,2);
  assert.deepEqual(trace.slice(0,4).map(x=>x.half),[true,false,false,true]);
  for(const c of report.cases){
    assert.equal(c.variants['bf16-f32'].median_execution_ms,66);
    assert.equal(c.variants['fp16-f16'].median_execution_ms,34);
    assert.equal(c.variants['bf16-f32'].median_generation_ms,3200);
    assert.equal(c.speedup,66/34);
    assert.equal(c.paired.length,3);
    assert.ok(c.paired.every(p=>p.logits.max_absolute_error===0&&p.argmax_changes===0));
  }
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[117],seed:137,warmups:1,rounds:3,optimization:true}});
  const optimized=messages.at(-1).report;
  assert.equal(optimized.completed,true);
  assert.equal(optimized.baseline_id,'fp16-baseline');
  assert.equal(optimized.runs.length,16);
  assert.equal(optimized.cases[0].paired.length,9);
  assert.ok(optimized.cases[0].paired.every(p=>p.logits.max_absolute_error===0));
  assert.equal(freed,6);
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[117],seed:137,warmups:0,rounds:1,optimization:true,profile:true}});
  assert.equal(messages.at(-1).report.completed,true);
  assert.equal(messages.at(-1).report.profile_enabled,true);
  assert.equal(messages.at(-1).report.runs.length,1);
  assert.equal(freed,7);
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[117],seed:137,warmups:0,rounds:1,optimization:true,tiles:true}});
  assert.equal(messages.at(-1).report.completed,true);
  assert.equal(messages.at(-1).report.runs.length,4);
  assert.equal(freed,11);
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[117],seed:137,warmups:0,rounds:1,optimization:true,tiles:true,unroll:true}});
  assert.equal(messages.at(-1).report.completed,true);
  assert.ok(messages.at(-1).report.runs.some(r=>r.id==='fp16-unroll64'));
  assert.equal(freed,15);
  for(const mode of ['fusion','attention','k64','combo','subtile','operands','norm','smallrows']){
    await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[117],seed:137,warmups:0,rounds:1,optimization:true,[mode]:true,audit:true}});
    assert.equal(messages.at(-1).report.completed,true);
    assert.equal(messages.at(-1).report.runs.length,4);
  }
  assert.equal(freed,47);
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[17,65],seed:1729,warmups:0,rounds:1,optimization:true,selected:true,audit:true}});
  assert.equal(messages.at(-1).report.completed,true);
  assert.equal(messages.at(-1).report.configuration_mode,'selected');
  assert.equal(messages.at(-1).report.engines[1].configuration.mlpTile,64);
  assert.equal(messages.at(-1).report.runs.length,4);
  assert.equal(freed,49);
  const pairedStart=trace.length;
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[17],seed:137,warmups:1,rounds:3,optimization:true,selected:true,audit:true}});
  assert.equal(messages.at(-1).report.completed,true);
  assert.deepEqual(trace.slice(pairedStart+2).map(x=>x.tiled),[false,true,true,false,false,true], 'two-engine measured order must alternate');
  assert.equal(freed,51);
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[117],seed:137,warmups:0,rounds:1,optimization:true,cache:true}});
  assert.equal(messages.at(-1).report.completed,true);
  assert.equal(messages.at(-1).report.runs.length,3);
  assert.equal(messages.at(-1).report.cases[0].dependency[3].prefix_unchanged_when_questions_change,false);
  assert.equal(freed,54);
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[17],seed:137,warmups:0,rounds:1,optimization:true,safari:true,audit:true}});
  const safari=messages.at(-1).report;
  assert.equal(safari.completed,true);
  assert.equal(safari.engines[1].int8_check.passed,true);
  assert.ok(safari.cases[0].paired[0].logits.max_absolute_error>0, 'INT8 drift must be reported without applying FP16 equivalence');
  assert.equal(freed,56);
  for(const comparison of ['padding','hybrid']){
    await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[17],seed:137,warmups:0,rounds:1,optimization:true,safari:true,comparison,audit:true}});
    const paired=messages.at(-1).report;
    assert.equal(paired.completed,true);
    assert.equal(paired.engines.length,2);
    assert.equal(paired.engines[0].configuration.pad,comparison!=='padding');
    assert.equal(Boolean(paired.engines[1].configuration.fp16Mlp),comparison==='hybrid');
  }
  assert.equal(freed,60);
  for(const candidate of ['scan4','i8f32r8']){
    await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[17],seed:137,warmups:0,rounds:1,optimization:true,safari:true,safariTune:true,path:'hybrid',candidate,audit:true}});
    const tuned=messages.at(-1).report;
    assert.equal(tuned.completed,true);
    assert.equal(tuned.baseline_id,'tuning-baseline');
    assert.equal(tuned.runs.length,2);
    assert.equal(tuned.cases[0].paired[0].logits.max_absolute_error,0);
  }
  assert.equal(freed,64);
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[3],seed:137,warmups:0,rounds:1,optimization:true,safari:true,comparison:'pure',audit:true}});
  assert.equal(messages.at(-1).report.completed,true);
  assert.ok(messages.at(-1).report.cases[0].paired[0].logits.max_absolute_error>0);
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[3],seed:137,warmups:0,rounds:1,optimization:true,safari:true,safariTune:true,path:'pure-int8',candidate:'combined',audit:true}});
  assert.equal(messages.at(-1).report.completed,true);
  assert.equal(messages.at(-1).report.cases[0].paired[0].logits.max_absolute_error,0);
  assert.equal(freed,68);
  for(const [path,candidate] of [['fp16','combined'],['hybrid','combined-f32r8'],['pure-int8','combined'],['pure-int8','combined-f32r8']]){
    await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[3,65],seed:1729,warmups:0,rounds:1,optimization:true,safari:true,safariTune:true,path,candidate,audit:true}});
    const result=messages.at(-1).report;
    assert.equal(result.completed,true);
    if(!candidate.startsWith('gemm'))assert.equal(result.engines[1].configuration.scanUnroll,4);
    assert.ok(result.cases.every(c=>c.paired[0].logits.max_absolute_error===0));
  }
  const freedBeforeFailure=freed;
  Mock.prototype.runLayer=async()=>{throw Error('test device failure');};
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[117],seed:137,warmups:0,rounds:1}});
  assert.equal(messages.at(-1).report.completed,false);
  assert.match(messages.at(-1).report.error,/test device failure/);
  assert.equal(freed,freedBeforeFailure+2);
  Mock.prototype.checkProjection=async()=>{throw Error('test initialization failure');};
  await self.onmessage({data:{bytes:new ArrayBuffer(0),rows:[117],seed:137,warmups:0,rounds:1,optimization:true}});
  assert.equal(messages.at(-1).report.completed,false);
  assert.match(messages.at(-1).report.error,/initialization failure/);
  assert.equal(freed,freedBeforeFailure+3);
  return {script_syntax:true,warmup_exclusion:true,alternating_order:true,generation_excluded:true,failure_cleanup:true};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.argv[2]) throw Error('Usage: node scripts/pplx_flash/check_synthetic.mjs packaged.html');
  const root = fileURLToPath(new URL('../..', import.meta.url));
  console.log(JSON.stringify(await checkSynthetic(root, process.argv[2]), null, 2));
}
