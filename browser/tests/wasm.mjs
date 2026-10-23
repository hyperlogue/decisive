import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=fileURLToPath(new URL('../../',import.meta.url));
const base=process.env.PRIVACY_TEST_URL || 'http://127.0.0.1:8768/site/dist/';
const catalog=JSON.parse(await fs.readFile(path.join(root,'site/models.json')));
const names=Object.keys(catalog.models.find(m=>m.id==='decima-small').files);
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,
  args:['--no-sandbox','--disable-dev-shm-usage',...(process.env.PRIVACY_SOFTWARE_GPU==='1'?['--enable-unsafe-webgpu','--use-angle=swiftshader']:[])]});
try{
  const page=await browser.newPage();
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  const fixture=new URL('sdk-smoke.html',base).href;
  await page.route(fixture,route=>route.fulfill({contentType:'text/html',body:'<!doctype html><input id="files" type="file" multiple>'}));
  await page.goto(fixture);
  await page.locator('#files').setInputFiles(names.map(name=>path.join(process.env.PRIVACY_MODEL_DIR || path.join(root,'models/decima-small'),name)));
  const report=await page.evaluate(async({names})=>{
    const files=Object.fromEntries(names.map((name,index)=>[name,document.querySelector('#files').files[index]]));
    const api=await import('./pkg/decisive.js');await api.default();
    const request={state:'My name is Alex Rivera.',questions:{name:{type:'noul',instructions:'Does the message disclose a real personal name?'}}};
    const bytes=async name=>new Uint8Array(await files[name].arrayBuffer());
    const legacy=await api.DecimaWebGpuModel.create(await bytes('pytorch/decima.json'),await bytes('pytorch/encoder/config.json'),
      await bytes('pytorch/encoder/tokenizer.json'),files['pytorch/encoder/model.safetensors'],await bytes('pytorch/head.safetensors'));
    let reference;
    try{reference=JSON.parse(await legacy.predict(JSON.stringify(request)));}finally{legacy.free();}
    const {loadModel}=await import('./sdk/index.js');
    const model=await loadModel({model:'decima',backend:'webgpu',source:{files}});
    const cache=await model.createCache({maxBytes:1024*1024});
    const first=await model.predict(request,{cache});
    const batch=await model.predictBatch([request,{state:null,questions:{}},request],{cache});
    await cache.clear();
    let wrongParameters;
    try{await model.predict(request,{cache,parameters:{maxSequenceTokens:1024}});}catch(error){wrongParameters=error.code;}
    let unsupportedCache;
    try{await model.createCache({requirePrefix:true});}catch(error){unsupportedCache=error.code;}
    await model.dispose();
    let disposed;
    try{await model.predict(request,{cache});}catch(error){disposed=error.code;}
    return {reference,first,batch,wrongParameters,unsupportedCache,disposed};
  },{names});
  assert.deepEqual(report.first.response,report.reference);
  assert.deepEqual(report.batch.map(item=>item.status),['success','error','success']);
  assert.deepEqual(report.batch[0].result.response,report.reference);
  assert.deepEqual(report.batch[2].result.response,report.reference);
  assert.equal(report.batch[1].error.code,'INVALID_REQUEST');
  assert.equal(report.wrongParameters,'INVALID_PARAMETERS');
  assert.equal(report.unsupportedCache,'UNSUPPORTED_CACHE');
  assert.equal(report.disposed,'DISPOSED');
  assert.deepEqual(errors,[]);
  console.log('Real WASM SDK: legacy Decima output parity, Worker transport, shared batch cache, per-item failure, parameter validation, cache capabilities and disposal: PASS');
  if(process.env.PPLX_MODEL_DIR){
    const files=['config.json','decision_config.json','joint_config.json','tokenizer.json','readout.safetensors',
      'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'];
    const sizes=await Promise.all(files.map(name=>fs.stat(path.join(process.env.PPLX_MODEL_DIR,name)).then(s=>s.size)));
    assert.ok(sizes.some(size=>size>2**32),'large-file regression requires a shard above 4 GiB');
    await page.locator('#files').setInputFiles(files.map(name=>path.join(process.env.PPLX_MODEL_DIR,name)));
    const info=await page.evaluate(async names=>{
      const files=Object.fromEntries(names.map((name,index)=>[name,document.querySelector('#files').files[index]]));
      const {loadModel}=await import('./sdk/index.js');
      const model=await loadModel({model:'pplx-flash',backend:'webgpu',format:'bf16',source:{files},tuning:{mode:'off'}});
      try{return model.info;}finally{await model.dispose();}
    },files);
    assert.equal(info.model,'pplx-flash');
    assert.equal(info.backend,'webgpu');
    console.log('BF16 checkpoint with >4 GiB shards: bounded metadata/range loading: PASS (no inference)');
  }
}finally{await browser.close();}
