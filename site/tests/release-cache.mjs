import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const dist=path.resolve(process.env.PRIVACY_TEST_DIST || fileURLToPath(new URL('../dist',import.meta.url)));
const oldWorker=await fs.readFile(new URL('./fixtures/legacy-worker.js',import.meta.url),'utf8');
const oldModel=await fs.readFile(new URL('./fixtures/legacy-model.json',import.meta.url),'utf8');
const oldSW=`const CACHE='privacy-upgrade-regression';
self.addEventListener('install',e=>e.waitUntil((async()=>{
 const cache=await caches.open(CACHE);
 for(const [target,fixture] of [['workers/model-worker.js','old-worker'],['model.json','old-model']]){
  const response=await fetch('./'+fixture);await cache.put(new URL('./'+target,self.location.href),response);
 }
 await self.skipWaiting();
})()));
self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));
self.addEventListener('fetch',e=>{
 if(['workers/model-worker.js','model.json'].some(p=>e.request.url===new URL('./'+p,self.location.href).href))
  e.respondWith(caches.open(CACHE).then(c=>c.match(e.request)));
});`;
const types={'.js':'text/javascript','.json':'application/json','.html':'text/html','.css':'text/css','.wasm':'application/wasm','.woff2':'font/woff2'};
let deploy=0;
const server=http.createServer(async(req,res)=>{
 try{
  const pathname=new URL(req.url,'http://localhost').pathname;
  const special={'/seed':'<!doctype html><title>Cache migration test</title>','/old-sw.js':oldSW,'/old-worker':oldWorker,'/old-model':oldModel};
  let data,type;
  if(pathname in special){data=special[pathname];type=pathname==='/seed'?'text/html':pathname==='/old-model'?'application/json':'text/javascript';}
  else{
   const file=path.resolve(dist,'.'+(pathname==='/'?'/index.html':pathname));
   if(!file.startsWith(dist+path.sep))throw Error('outside test root');
   data=await fs.readFile(file);type=types[path.extname(file)]||'application/octet-stream';
   if(deploy&&pathname==='/sw.js')data=`${data}\n// deploy ${deploy}`;
   if(deploy&&pathname==='/release.json')data=JSON.stringify({version:`deploy${deploy}`});
   if(deploy&&type==='text/html')data=String(data).replace('<head>',`<head><meta name="deploy" content="${deploy}">`);
  }
  res.writeHead(200,{'Content-Type':type,'Cache-Control':type==='text/html'?'max-age=600':'no-store'});res.end(data);
 }catch{res.writeHead(404);res.end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const base=`http://127.0.0.1:${server.address().port}/`;
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-webgpu','--use-angle=swiftshader']});
try{
 const context=await browser.newContext();const page=await context.newPage();
 await page.goto(base+'seed');
 await page.evaluate(async()=>{
  await navigator.serviceWorker.register('./old-sw.js',{scope:'./'});await navigator.serviceWorker.ready;
  const dir=await (await navigator.storage.getDirectory()).getDirectoryHandle('decisive-privacy-decima-small-v1',{create:true});
  const writer=await (await dir.getFileHandle('preserved-test-file',{create:true})).createWritable();
  await writer.write('preserved');await writer.close();
 });
 await page.waitForFunction(()=>!!navigator.serviceWorker.controller);
 await page.goto(base);await page.waitForSelector('#intro-dialog[open]');await page.click('#intro-continue');await page.waitForSelector('#setup-dialog[open]');
 const shown={model:await page.locator('#setup-models .model-option.selected .model-option-name').textContent(),size:await page.locator('#setup-models [data-model="decima-small"] .model-option-size').textContent()};
 console.log(JSON.stringify(shown));
 assert.equal(shown.model,'Decima Small');assert.match(shown.size,/506.7 MB/);
 assert.equal(await page.evaluate(async()=>{
  const dir=await (await navigator.storage.getDirectory()).getDirectoryHandle('decisive-privacy-decima-small-v1');
  return (await (await dir.getFileHandle('preserved-test-file')).getFile()).text();
 }),'preserved');
 await context.close();
 const mismatch=await browser.newContext({serviceWorkers:'block'});
 await mismatch.route('**/workers/model-worker.js',r=>r.fulfill({contentType:'text/javascript',body:oldWorker}));
 await mismatch.route('**/model.json',r=>r.fulfill({contentType:'application/json',body:oldModel}));
 const blocked=await mismatch.newPage();await blocked.goto(base);await blocked.waitForSelector('#intro-dialog[open]');await blocked.click('#intro-continue');await blocked.waitForSelector('#setup-dialog[open]');
 assert.match(await blocked.locator('#setup-models [data-model="decima-small"] .model-option-size').textContent(),/506.7 MB/);
 assert.equal(await blocked.locator('#start-analysis').isDisabled(),true);
 assert.match(await blocked.locator('#setup-error').textContent(),/older version/);
 await mismatch.close();
 console.log('Old cache bypassed; mismatched worker blocked; model files preserved: PASS');
 const live=await browser.newContext();const tab=await live.newPage();
 await tab.goto(base);await tab.waitForSelector('#intro-dialog[open]');
 await tab.waitForFunction(()=>!!navigator.serviceWorker.controller);
 assert.equal(await tab.locator('#update-banner').isVisible(),false);
 deploy=1;
 await tab.evaluate(async()=>(await navigator.serviceWorker.getRegistration()).update());
 await tab.waitForSelector('#update-banner[open]');
 await tab.click('#reload-update');
 await tab.waitForSelector('meta[name="deploy"][content="1"]',{state:'attached'});
 assert.equal(await tab.locator('#update-banner').isVisible(),false);
 await live.close();
 console.log('New deploy activates in open tabs, prompts reload, and serves fresh HTML: PASS');
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
