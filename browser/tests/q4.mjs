import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=fileURLToPath(new URL('../../',import.meta.url));
const base=process.env.PRIVACY_TEST_URL || 'http://127.0.0.1:8768/site/dist/';
const directory=process.env.PPLX_Q4_MODEL_DIR || path.join(root,'artifacts/huggingface/pplx-decider-flash-v1.2-q4-e4');
const catalog=JSON.parse(await fs.readFile(path.join(root,'site/models.json')));
const names=Object.keys(catalog.models.find(m=>m.id==='pplx-flash-q4').files);
const request=JSON.parse(await fs.readFile(path.join(root,'tests/fixtures/pplx_flash/request.json')));
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,
  args:['--no-sandbox','--disable-dev-shm-usage',...(process.env.PRIVACY_SOFTWARE_GPU==='1'?['--enable-unsafe-webgpu','--use-angle=swiftshader']:[])]});
try{
  const page=await browser.newPage();
  const fixture=new URL('q4-sdk-smoke.html',base).href;
  await page.route(fixture,route=>route.fulfill({contentType:'text/html',body:'<!doctype html><input id="files" type="file" multiple>'}));
  await page.goto(fixture);
  const capabilities=await page.evaluate(async()=> (await import('./sdk/index.js')).probe());
  assert.ok(capabilities.shaderF16,'Q4 parity check requires a working shader-f16 GPU; inference was not run');
  await page.locator('#files').setInputFiles(names.map(name=>path.join(directory,name)));
  const report=await page.evaluate(async({names,request})=>{
    const files=Object.fromEntries(names.map((name,index)=>[name,document.querySelector('#files').files[index]]));
    const api=await import('./pkg/decisive.js');await api.default();
    const {loadFlash}=await import('../../examples/pplx-flash/runtime.js');
    const legacy=await loadFlash(api,files,()=>{},'webgpu',{mode:'q4-e4',resident:false,profile:false});
    let reference;
    try{reference=await legacy.predict(request,1024);}finally{legacy.dispose();}
    const {loadModel}=await import('./sdk/index.js');
    const model=await loadModel({model:'pplx-flash',backend:'webgpu',format:'q4-e4',source:{files}});
    const cache=await model.createCache({maxBytes:16*1024*1024});
    try{
      const first=await model.predict(request,{cache});
      const controller=new AbortController();let cancelled;
      try{await model.predict(request,{cache,signal:controller.signal,onProgress:event=>{
        if(event.phase==='inference'&&event.completed>=1)controller.abort();
      }});}catch(error){cancelled=error.code;}
      const recovered=await model.predict(request,{cache});
      return {reference:reference.response,first:first.response,recovered:recovered.response,cancelled,
        legacy_ms:reference.elapsed_ms,sdk_ms:first.elapsed_ms,recovered_ms:recovered.elapsed_ms,tuning:model.info.tuning};
    }finally{await model.dispose();}
  },{names,request});
  assert.deepEqual(report.first,report.reference);
  assert.deepEqual(report.recovered,report.reference);
  assert.equal(report.cancelled,'CANCELLED');
  assert.equal(report.first.answers.team.choice,'support');
  if(process.env.PRIVACY_Q4_OUTPUT)await fs.writeFile(process.env.PRIVACY_Q4_OUTPUT,JSON.stringify(report,null,2)+'\n');
  console.log('Real Q4: legacy scheduler parity, cancellation and cache/model reuse after cancellation: PASS');
  console.log(JSON.stringify({legacy_ms:report.legacy_ms,sdk_ms:report.sdk_ms,recovered_ms:report.recovered_ms,scope:'Smoke timings; not a performance benchmark'}));
}finally{await browser.close();}
