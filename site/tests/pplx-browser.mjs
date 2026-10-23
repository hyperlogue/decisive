import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.PRIVACY_TEST_URL||'http://127.0.0.1:8768/site/dist/';
const catalog=JSON.parse(await fs.readFile(new URL('../models.json',import.meta.url)));
const original=catalog.models.find(m=>m.id==='pplx-flash-q4');
const payload={'packed-q4-e4.json':Buffer.from('{"test":true}'),'q4-e4-layer-00.bin':Buffer.from('Q4 browser storage fixture')};
const fixture={...original,repository:'example/q4-browser-test',revision:'fixture',files:Object.fromEntries(Object.entries(payload).map(([name,data])=>[name,{bytes:data.length,sha256:createHash('sha256').update(data).digest('hex')}]))};
const recorded=JSON.parse(await fs.readFile(new URL('./fixtures/english-result.json',import.meta.url)));
for(const answer of Object.values(recorded.output.response.answers)){delete answer.max_probabilities;delete answer.chunk;}
const double=`export async function loadModelInWorker(api,options,controls){
  if(options.backend!=='webgpu'||options.format!=='q4-e4')throw Error('Wrong Q4 configuration');
  if(!options.source.files['q4-e4-layer-00.bin'])throw Error('Wrong checkpoint');
  return {info:{},async createCache(){return {async dispose(){}};},async dispose(){},async predict(request,{signal,cache}){
    if(options.parameters.maxSequenceTokens!==1024||request.long!==undefined||!cache)throw Error('Wrong prediction configuration');
    const draft=request.state.context[1].content;
    if(draft==='too long fixture')throw Object.assign(Error('Input too long'),{code:'INPUT_TOO_LONG'});
    if(draft==='memory limit fixture')throw Object.assign(Error('pplx-flash / webgpu / q4-e4 exceeds device limits'),{code:'RESOURCE_LIMIT'});
    if(draft==='unsupported fixture')throw Object.assign(Error('Selected weight format is unavailable in this build'),{code:'FORMAT_NOT_COMPILED'});
    if(draft==='missing feature fixture')throw Object.assign(Error('pplx-flash / webgpu / q4-e4 requires shader-f16'),{code:'MISSING_GPU_FEATURE'});
    for(let i=1;i<=32;i++){await new Promise(r=>setTimeout(r,draft==='cancel this inference'?200:30));signal.throwIfAborted();controls.onProgress({phase:'inference',completed:i-1,total:32});}
    return {response:${JSON.stringify(recorded.output.response)},backend:'webgpu',elapsed_ms:123};
  }};
}`;
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
try{
  const context=await browser.newContext({serviceWorkers:'block',viewport:{width:390,height:844}});
  await context.route('**/models.json',r=>r.fulfill({json:{...catalog,models:catalog.models.map(m=>m.id===fixture.id?fixture:m)}}));
  await context.route('**/lib/webgpu.js',r=>r.fulfill({contentType:'text/javascript',body:'export async function checkWebGPU(){return true;}'}));
  await context.route('**/sdk/local.js',r=>r.fulfill({contentType:'text/javascript',body:double}));
  const requests=[],errors=[];
  await context.route('https://huggingface.co/example/q4-browser-test/**',r=>{
    requests.push(r.request().url());const body=payload[r.request().url().split('/').at(-1)];
    assert.ok(body,'only packed fixture files requested');
    return r.fulfill({body,headers:{'access-control-allow-origin':'*'}});
  });
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base);await page.waitForSelector('#setup-dialog[open]');
  await page.evaluate(async()=>{
    const dir=await(await navigator.storage.getDirectory()).getDirectoryHandle('decisive-privacy-decima-small-v1',{create:true});
    const writer=await(await dir.getFileHandle('preserved.txt',{create:true})).createWritable();await writer.write('keep Decima');await writer.close();
  });
  await page.selectOption('#setup-model','pplx-flash-q4');
  await page.waitForFunction(()=>!document.getElementById('start-analysis').disabled);
  await page.click('#start-analysis');
  await page.waitForFunction(()=>document.getElementById('composer-status').textContent==='Updates as you type');
  await page.fill('#message-input',recorded.draft);await page.waitForSelector('#signal-contact');
  const raw=JSON.parse(await page.locator('#raw-output').textContent());
  assert.equal(raw.input.state.context[1].content,recorded.draft);assert.equal(Object.hasOwn(raw.input,'long'),false);
  assert.equal(raw.elapsed_ms,123);assert.equal(await page.locator('#runtime-model').innerText(),fixture.name);
  assert.equal(requests.length,2);
  await page.reload();await page.waitForSelector('#setup-dialog[open]');
  await page.selectOption('#setup-model','pplx-flash-q4');
  await page.waitForFunction(()=>document.getElementById('start-analysis').textContent==='Use model');
  await page.click('#start-analysis');
  await page.waitForFunction(()=>document.getElementById('composer-status').textContent==='Updates as you type');
  await page.fill('#message-input','too long fixture');
  await page.waitForFunction(()=>document.getElementById('analysis-error').textContent.includes('1,024-token'));
  assert.equal(await page.locator('#share-result').isDisabled(),true);
  for(const [draft,message,detail] of [
    ['memory limit fixture','memory budget','exceeds device limits'],
    ['unsupported fixture','cannot run the selected model','unavailable in this build'],
    ['missing feature fixture','shader-f16','requires shader-f16'],
  ]){
    await page.fill('#message-input',draft);await page.click('#open-setup');await page.click('#start-analysis');
    await page.waitForFunction(message=>document.getElementById('analysis-error').textContent.includes(message),message);
    assert.ok((await page.locator('#raw-output').textContent()).includes(detail));
    assert.equal(await page.locator('#share-result').isDisabled(),true);
  }
  await page.fill('#message-input','cancel this inference');await page.click('#open-setup');await page.click('#start-analysis');
  await page.waitForFunction(()=>document.getElementById('composer-status').textContent.includes('Layer'));
  await page.click('#open-setup');await page.click('#manage-models');await page.click('#stop-model');
  await page.waitForFunction(()=>document.getElementById('stop-model').hidden&&!document.getElementById('delete-model').disabled);
  assert.equal(await page.locator('#share-result').isDisabled(),true);
  await page.click('#delete-model');await page.click('#confirm-delete');
  await page.waitForFunction(()=>document.getElementById('model-badge').textContent==='Not downloaded');
  const kept=await page.evaluate(async()=>{
    const root=await navigator.storage.getDirectory();
    const dir=await root.getDirectoryHandle('decisive-privacy-decima-small-v1');
    return (await(await dir.getFileHandle('preserved.txt')).getFile()).text();
  });
  assert.equal(kept,'keep Decima');assert.deepEqual(errors,[]);
  console.log('Q4 supported-device flow: verified download, adapter, 1024 limit, capability/memory/build errors, OPFS reuse, cancellation, and scoped deletion: PASS (inference mocked)');
}finally{await browser.close();}
