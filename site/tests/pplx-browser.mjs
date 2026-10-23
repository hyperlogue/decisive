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
const source=await fs.readFile(new URL('../lib/pplx-runtime.js',import.meta.url),'utf8');
const double=`const importRuntime=async(api,files,progress,backend,config)=>{
  if(backend!=='webgpu'||config.mode!=='q4-e4'||config.resident!==false)throw Error('Wrong Q4 configuration');
  if(!files['q4-e4-layer-00.bin']||files['model-00001-of-00002.safetensors'])throw Error('Wrong checkpoint');
  return {dispose(){},async predict(request,limit){
    if(limit!==1024||request.long!==undefined)throw Error('Wrong token limit/request');
    const draft=request.state.context[1].content;
    if(draft==='too long fixture')throw Error('single question exceeds token limit: 1200 > 1024; inputs are never truncated');
    for(let i=1;i<=32;i++){await new Promise(r=>setTimeout(r,draft==='cancel this inference'?200:30));progress('Layer '+i+'/32');}
    return {response:${JSON.stringify(recorded.output.response)},backend:'webgpu',elapsed_ms:123};
  }};
};`;
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
try{
  const context=await browser.newContext({serviceWorkers:'block',viewport:{width:390,height:844}});
  await context.route('**/models.json',r=>r.fulfill({json:{...catalog,models:catalog.models.map(m=>m.id===fixture.id?fixture:m)}}));
  await context.route('**/lib/webgpu.js',r=>r.fulfill({contentType:'text/javascript',body:'export async function checkWebGPU(){return true;}'}));
  await context.route('**/lib/pplx-runtime.js',r=>r.fulfill({contentType:'text/javascript',body:source.replace(/^const importRuntime = .*;$/m,double)}));
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
  console.log('Q4 supported-device flow: verified download, adapter, 1024 limit, OPFS reuse, cancellation, and scoped deletion: PASS (inference mocked)');
}finally{await browser.close();}
