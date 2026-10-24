import {fileURLToPath} from 'node:url';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
import fs from 'node:fs/promises';
import path from 'node:path';
const root=fileURLToPath(new URL('../../',import.meta.url));
const modelDirectory=process.env.PRIVACY_MODEL_DIR || path.join(root,'models/decima-small');
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage',...(process.env.PRIVACY_SOFTWARE_GPU==='1'?['--enable-unsafe-webgpu','--use-angle=swiftshader']:[])]});
try{
const page=await browser.newPage();page.on('pageerror',e=>console.log('PAGEERROR',e.message));
await page.goto(process.env.PRIVACY_REAL_TEST_URL || 'http://127.0.0.1:8768/site/tests/real-model.html');
const catalog=JSON.parse(await fs.readFile(path.join(root,'site/models.json')));
const manifest=catalog.models.find(m=>m.id===catalog.default);
await page.locator('#files').setInputFiles(Object.keys(manifest.files).map(name=>path.join(modelDirectory,name)));
let last='';const start=Date.now();
for(;;){
const state=await page.evaluate(()=>({latest:window.progress.at(-1),results:window.results,failure:window.failure}));
const marker=JSON.stringify({latest:state.latest,count:state.results.length});if(marker!==last){console.log(marker);last=marker;}
await fs.writeFile(process.env.PRIVACY_REAL_TEST_OUTPUT || '/tmp/privacy-real-smoke.json',JSON.stringify({...state,elapsed_wall_ms:Date.now()-start},null,2));
if(state.failure)throw Error(state.failure);if(state.results.length===4){
 const actual=state.results.map(r=>r.scores.filter(s=>s.detected).map(s=>s.id));
 console.log('Observed categories:',JSON.stringify(actual));
 const long=state.results[3].response;
 console.log('Long draft:',long.usage.input_tokens,'tokens in',long.usage.chunks,'chunks,',Math.round(state.results[3].elapsed_ms),'ms');
 if(!(long.usage.input_tokens>512&&long.usage.chunks>1))throw Error('Long draft was not chunked');
 if(!actual[3].includes('credential'))throw Error('Password in the long draft was missed');
 break;
}
if(Date.now()-start>20*60*1000)throw Error('Smoke test timed out');
await new Promise(r=>setTimeout(r,2000));
}
console.log('REAL_MODEL_SMOKE_PASSED');
}finally{await browser.close();}
