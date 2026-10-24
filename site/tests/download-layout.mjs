import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {emptyModelState} from '../lib/model-catalog.js';

const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.PRIVACY_TEST_URL||'http://127.0.0.1:8768/site/dist/';
const catalog=JSON.parse(await fs.readFile(new URL('../models.json',import.meta.url)));
const states=Object.fromEntries(catalog.models.map(model=>[model.id,emptyModelState(model)]));
// Keep downloads pending; inject progress through the real Worker's message handler.
const worker=`const states=${JSON.stringify(states)};
self.onmessage=({data:{id,method,args}})=>{
  if(method==='download')return;
  self.postMessage({id,result:method==='capabilities'?{sync:true,gpu:true,shaderF16:true}:states[args.modelId]});
};`;
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
try{
  for(const model of catalog.models){
    const context=await browser.newContext({serviceWorkers:'block',reducedMotion:'reduce'});
    await context.route('**/workers/model-worker.js',route=>route.fulfill({contentType:'text/javascript',body:worker}));
    await context.addInitScript(()=>{
      window.Worker=class extends Worker {
        constructor(...args){super(...args);window.downloadWorker=this;}
      };
    });
    const page=await context.newPage(),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(base);await page.click('#intro-continue');
    const selector=`#setup-models [data-model="${model.id}"]`;
    await page.click(`${selector} .model-download`);
    await page.waitForSelector(`${selector} progress:not([hidden])`);
    for(const width of [320,390,600,1100]){
      await page.setViewportSize({width,height:844});
      for(const locale of ['en','es','zh','ar']){
        await page.click('#close-setup');await page.selectOption('#language-picker',locale);await page.click('#open-setup');
        await page.evaluate(()=>document.fonts.ready);
        const measurements=await page.evaluate(({selector,state})=>{
          const card=document.querySelector(selector),status=card.querySelector('.model-option-size');
          const samples=[0,999,1000,9900,10000,999900,1000000,9900000,10000000,10100000,99900000,100000000,100100000,999900000,1000000000,1010000000,state.total-1];
          return samples.filter(bytes=>bytes<state.total).map(bytes=>{
            let remaining=bytes;
            for(const file of state.files){
              const downloaded=Math.min(remaining,file.expected);remaining-=downloaded;
              window.downloadWorker.dispatchEvent(new MessageEvent('message',{data:{event:'progress',phase:'downloading',name:file.name,bytes:downloaded}}));
            }
            const box=card.getBoundingClientRect(),statusBox=status.getBoundingClientRect();
            const bar=card.querySelector('progress');
            return {bytes,text:status.textContent,height:box.height,statusHeight:statusBox.height,
              lineHeight:parseFloat(getComputedStyle(status).lineHeight),progress:bar.value,
              barTop:bar.getBoundingClientRect().top-box.top,
              fits:statusBox.left>=box.left&&statusBox.right<=box.right&&box.left>=0&&box.right<=innerWidth};
          });
        },{selector,state:states[model.id]});
        const label=`${model.id}, ${locale}, ${width}px`;
        assert.equal(new Set(measurements.map(m=>m.height)).size,1,`${label}: card height changes during download`);
        assert.equal(new Set(measurements.map(m=>m.barTop)).size,1,`${label}: progress bar moves during download`);
        for(const m of measurements){
          assert.ok(m.statusHeight<=m.lineHeight+1,`${label}: status wraps: ${m.text}`);
          assert.ok(m.fits,`${label}: status overflows card or viewport`);
          assert.equal(m.progress,m.bytes,`${label}: progress updates still reach the card`);
        }
      }
    }
    assert.deepEqual(errors,[]);await context.close();
  }
  console.log('Download status stays on one line with stable card height: both models, four locales, mobile and desktop: PASS');
}finally{await browser.close();}
