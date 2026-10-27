import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage',...(process.env.PRIVACY_SOFTWARE_GPU==='1'?['--enable-unsafe-webgpu','--use-angle=swiftshader']:[])]});
try{
  const page=await browser.newPage();const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.PRIVACY_TEST_URL||'http://127.0.0.1:8768/site/dist/');
  await page.waitForSelector('#intro-dialog[open]');await page.click('#intro-continue');await page.waitForSelector('#setup-dialog[open]');
  const decima='#setup-models [data-model="decima-small"]';
  assert.equal(await page.locator(`${decima}.selected`).count(),1);
  await page.click(`${decima} .model-download`);
  const start=Date.now();let previous=-1;
  for(;;){
    const state=await page.evaluate(sel=>({ready:document.querySelector(sel).classList.contains('ready'),error:document.querySelector('#setup-error').hidden?'':document.querySelector('#setup-error').textContent,progress:document.querySelector(`${sel} progress`).value/document.querySelector(`${sel} progress`).max}),decima);
    if(state.error)throw Error(state.error);
    const step=Math.floor(state.progress*10);if(step!==previous){console.log('Download',step*10+'%');previous=step;}
    if(state.ready)break;if(Date.now()-start>600000)throw Error('Download timeout');
    await new Promise(resolve=>setTimeout(resolve,2000));
  }
  await page.click('#start-analysis');await page.waitForFunction(()=>document.querySelector('#composer-status').textContent==='Updates as you type',null,{timeout:180000});
  await page.fill('#message-input','My name is Alex Rivera. My email is alex@example.com.');
  await page.waitForFunction(()=>!document.querySelector('#share-result').disabled||!document.querySelector('#analysis-error').hidden,null,{timeout:300000});
  assert.equal(await page.locator('#analysis-error').isVisible(),false,await page.locator('#analysis-error').textContent());
  const output=JSON.parse(await page.locator('#raw-output').textContent());
  assert.equal(output.backend,'webgpu');assert.equal(Object.keys(output.output.answers).length,3);
  await fs.writeFile('/tmp/decima-browser-download.json',JSON.stringify(output,null,2));
  await page.screenshot({path:'/tmp/decima-real-analysis.png'});
  console.log('Real Decima analysis completed',Math.round(output.elapsed_ms)+' ms');
  await page.click('#open-setup');await page.click(`${decima} .model-delete`);await page.click('#confirm-delete');
  await page.waitForFunction(sel=>!document.querySelector(sel).hidden,`${decima} .model-download`);
  assert.equal(await page.locator(`${decima} .model-delete`).isVisible(),false);
  assert.deepEqual(errors,[]);console.log('Hugging Face → OPFS → WebGPU → analysis → deletion: PASS');
}finally{await browser.close();}
