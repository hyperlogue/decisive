import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const base=process.env.PRIVACY_TEST_URL || 'http://127.0.0.1:8768/site/dist/';
const original=await fs.readFile(new URL('../dist/workers/model-worker.js',import.meta.url),'utf8');
const observations=[];
try{
  for(const failure of ['adapter-null','device-rejected']){
    const context=await browser.newContext({serviceWorkers:'block'});
    let workers=0;
    await context.route('**/workers/model-worker.js',async route=>{
      const firstWorker=++workers===1;
      const platform=`let attempt=0;
        Object.defineProperty(navigator,'gpu',{value:{async requestAdapter(){
          const first=${firstWorker} && ++attempt===1;
          if(first && ${JSON.stringify(failure)}==='adapter-null')return null;
          return {async requestDevice(){
            if(first && ${JSON.stringify(failure)}==='device-rejected')throw new DOMException('GPU is starting','OperationError');
            return {destroy(){}};
          }};
        }}});`;
      await route.fulfill({contentType:'text/javascript',body:platform+'\n'+original});
    });
    const page=await context.newPage();
    await page.goto(base);await page.waitForSelector('#setup-dialog[open]');
    const first=await page.locator('#cap-gpu').textContent();
    await page.reload();await page.waitForSelector('#setup-dialog[open]');
    const afterRefresh=await page.locator('#cap-gpu').textContent();
    observations.push({failure,first,afterRefresh});
    await context.close();
  }
  console.log(JSON.stringify(observations,null,2));
  for(const observation of observations){
    assert.equal(observation.afterRefresh,'Available');
    assert.equal(observation.first,'Available',`${observation.failure}: GPU recovers immediately, but first load reports unavailable until refresh`);
  }
}finally{await browser.close();}
