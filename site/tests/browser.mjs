import fs from 'node:fs/promises';
import path from 'node:path';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const screenshots=process.env.PRIVACY_SCREENSHOTS || '/tmp/decisive-privacy-qa';
await fs.mkdir(screenshots,{recursive:true});
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-webgpu','--use-angle=swiftshader']});
const base=process.env.PRIVACY_TEST_URL || 'http://127.0.0.1:8768/site/dist/';
try{
 const context=await browser.newContext({viewport:{width:390,height:844},deviceScaleFactor:1});
 const page=await context.newPage(),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto(base);await page.waitForSelector('#intro-dialog[open]');
 assert.equal(await page.locator('#setup-dialog[open]').count(),0);
 assert.equal(await page.locator('#intro-dialog .intro-steps li').count(),3);
 await page.screenshot({path:path.join(screenshots,'intro-mobile.png')});
 await page.click('#intro-continue');await page.waitForSelector('#setup-dialog[open]');
 assert.equal(await page.locator('#cap-gpu, .webgpu-status, #manage-models, #models-dialog').count(),0);
 assert.equal(await page.locator('#backend, #browser-capabilities, .compatibility-details').count(),0);
 assert.doesNotMatch(await page.locator('#setup-dialog').textContent(),/This page requires WebGPU to work/);
 assert.match(await page.locator('#setup-models [data-model="pplx-flash-q4"] .model-link').getAttribute('href'),/^https:\/\/huggingface\.co\/rickye26\/pplx-decider-flash-v1\.2-q4\/tree\//);
 assert.match(await page.locator('#setup-models [data-model="decima-small"] .model-option-size').textContent(),/506.7 MB/);
 assert.equal(await page.locator('#setup-models .model-option').count(),3);
 assert.equal(await page.locator('#setup-dialog select, #skip-setup').count(),0);
 assert.equal(await page.locator('#start-analysis').isDisabled(),true);
 assert.equal(await page.locator('#setup-models [data-model="decima-small"] .model-download').isVisible(),true);
 assert.equal(await page.locator('#setup-models [data-model="decima-small"] .model-download').innerText(),'Download');
 assert.equal(await page.locator('#setup-models .model-option:not(.blocked) .model-badge:visible').count(),0,'no Not downloaded pill');
 assert.equal(await page.locator('#setup-models [data-model="pplx-flash-q4"] .model-option-name').textContent(),'pplx-decider-flash-v1.2-q4');
 assert.equal(await page.locator('#setup-download, #setup-pause').count(),0);
 const q4Card='#setup-models [data-model="pplx-flash-q4"]';
await page.waitForSelector(`${q4Card}.blocked`);
assert.equal(await page.locator('#setup-models .model-option.blocked').count(),2);
assert.equal(await page.locator('#setup-models [data-model="pplx-flash-q8"].blocked .model-option-name').textContent(),'pplx-decider-flash-v1.2-q8');
assert.match(await page.locator('#setup-models [data-model="pplx-flash-q8"] .model-link').getAttribute('href'),/^https:\/\/huggingface\.co\/rickye26\/pplx-decider-flash-v1\.2-q8\/tree\/10e640e/);
assert.equal(await page.locator(`${q4Card} .model-badge`).innerText(),'Disabled');
assert.equal(await page.locator(`${q4Card} .model-tip`).isVisible(),false);
await page.hover(`${q4Card} .model-badge`);
await page.waitForSelector(`${q4Card} .model-tip`,{state:'visible'});
assert.match(await page.locator(`${q4Card} .model-tip`).textContent(),/shader-f16\), so pplx-decider-flash-v1\.2-q4/);
assert.equal(await page.locator(`${q4Card} .model-tip`).isVisible(),true);
assert.equal(await page.evaluate(()=>document.getElementById('setup-error').previousElementSibling===document.querySelector('#setup-dialog .modal-heading')),true);
 assert.doesNotMatch(await page.locator('#setup-dialog .setup-note').textContent(),/one-time/);
 assert.match(await page.locator('#setup-dialog .setup-note').textContent(),/Analysis runs locally on your GPU\. Your messages stay in this browser\.$/);
 assert.equal(await page.locator(`${q4Card} .model-pick`).isDisabled(),true);
 assert.equal(await page.locator(`${q4Card} .model-download`).isVisible(),false);
 assert.equal(await page.locator('#setup-error').isVisible(),false);
 await page.screenshot({path:path.join(screenshots,'setup-mobile.png')});
 await page.click('#close-setup');await page.fill('#message-input','Private test text: alex@example.com');
 await page.locator('#message-input').blur();
 assert.equal(await page.locator('#enable-analysis').count(),0);
 assert.match(await page.locator('#page-description').textContent(),/Decisive.*WebGPU/);
 assert.equal(await page.evaluate(()=>{
   const box=id=>document.getElementById(id).getBoundingClientRect();
   return Math.abs(box('draft-region').top-box('site-header').bottom)<1&&
     Math.abs(box('analysis-scroll').top-box('draft-region').bottom)<1&&
     Math.abs(box('analysis-scroll').bottom-innerHeight)<1;
 }),true,'draft below header and analysis fills remaining height');
 assert.equal(await page.locator('#share-result').isDisabled(),true);
 for(const locale of ['en','es','zh','ar']){
   await page.selectOption('#language-picker',locale);
   assert.equal(await page.locator('#page-description #project-link').getAttribute('href'),'https://github.com/hyperlogue/decisive');
   assert.equal(await page.locator('#project-link svg').count(),1);
   assert.equal(await page.locator('#project-link').innerText(),'Decisive');
   assert.equal(await page.evaluate(()=>document.documentElement.dir),locale==='ar'?'rtl':'ltr');
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 }
 await page.screenshot({path:path.join(screenshots,'editor-ar.png')});
 await page.selectOption('#language-picker','en');
 const blockedQ4=await page.evaluate(()=>new Promise(resolve=>{
   const worker=new Worker(new URL('workers/model-worker.js',location.href),{type:'module'});
   worker.onmessage=({data})=>{worker.terminate();resolve(data.error);};
   worker.postMessage({id:1,method:'download',args:{modelId:'pplx-flash-q4'}});
 }));
 assert.equal(blockedQ4,'gpuPrecisionRequired');
 await page.evaluate(()=>{location.hash='#models';});await page.waitForSelector('#setup-dialog[open]');
 assert.equal(await page.evaluate(()=>location.hash),'');
 await page.click('#close-setup');assert.equal(await page.locator('#message-input').inputValue(),'Private test text: alex@example.com');
 await page.click('#open-setup');await page.waitForSelector('#setup-dialog[open]');await page.keyboard.press('Escape');
 assert.equal(await page.locator('#setup-dialog').getAttribute('open'),null);
 assert.ok(requests.every(url=>new URL(url).origin===new URL(base).origin),'typing must not upload or fetch external data');
 await page.evaluate(()=>navigator.serviceWorker.ready);await page.waitForFunction(()=>!!navigator.serviceWorker.controller);
 await context.setOffline(true);await page.reload();await page.waitForSelector('#intro-dialog[open]');await page.click('#intro-continue');await page.waitForSelector('#setup-dialog[open]');
 assert.match(await page.locator('#setup-models [data-model="decima-small"] .model-option-size').textContent(),/506.7 MB/);
 await context.setOffline(false);
 assert.deepEqual(errors,[]);await context.close();
 console.log('Mobile, locales, initial compatibility, no uploads, offline shell: PASS');

 const fixtures={'config.json':Buffer.from('{"fixture":true}'),'weights.bin':Buffer.from('small fixture for real browser OPFS and SHA256 verification')};
 const manifest={id:'decima-small',name:'Decima Small',engine:'decima',available:true,storeName:'decisive-privacy-decima-small-v1',repository:'example/privacy-test',revision:'test-revision',files:Object.fromEntries(Object.entries(fixtures).map(([name,body])=>[name,{bytes:body.length,sha256:createHash('sha256').update(body).digest('hex')}]))};
 const c=await browser.newContext({viewport:{width:1100,height:850},serviceWorkers:'block'});
 const catalog=JSON.parse(await fs.readFile(new URL('../models.json',import.meta.url)));
 await c.route('**/models.json',r=>r.fulfill({json:{...catalog,models:[manifest,...catalog.models.filter(m=>m.id!==manifest.id)]}}));
 await c.route('https://huggingface.co/example/privacy-test/**',async r=>{await new Promise(resolve=>setTimeout(resolve,250));await r.fulfill({status:200,body:fixtures[r.request().url().split('/').at(-1)],headers:{'access-control-allow-origin':'*'}});});
 const p=await c.newPage();p.on('pageerror',e=>errors.push(e.message));p.on('console',m=>console.log('CONSOLE',m.type(),m.text()));
 const decima='#setup-models [data-model="decima-small"]';
 await p.goto(base);await p.waitForSelector('#intro-dialog[open]');await p.click('#intro-continue');await p.waitForSelector('#setup-dialog[open]');
 await p.click(`${decima} .model-download`);
 await p.waitForFunction(sel=>document.querySelector(sel).innerText==='Pause',`${decima} .model-download`);
 assert.equal(await p.locator(`${decima} progress`).isVisible(),true);
 assert.equal(await p.locator('#setup-status').textContent(),'');
 await p.click('#close-setup');
 await p.click('#open-setup');
 await p.waitForFunction(sel=>document.querySelector(sel).classList.contains('ready'),decima,{timeout:30000}).catch(async e=>{console.log('DOWNLOAD STATE',await p.locator('#setup-dialog').innerText());throw e;});
 assert.equal(await p.locator(`${decima} .model-badge`).innerText(),'Ready');
 assert.equal(await p.locator(`${decima} .model-delete`).isEnabled(),true);
 assert.equal(await p.locator('#setup-status').textContent(),'');
 await p.reload();await p.waitForSelector('#intro-dialog[open]');await p.click('#intro-continue');await p.waitForSelector('#setup-dialog[open]');
 await p.waitForFunction(()=>!document.getElementById('start-analysis').disabled);
 assert.equal(await p.locator(`${decima} .model-download`).isVisible(),false);
 assert.equal(await p.locator(`${decima} .model-delete`).isVisible(),true);
 assert.equal(await p.locator('#setup-models [data-model="pplx-flash-q4"] .model-delete').isVisible(),false);
 await p.screenshot({path:path.join(screenshots,'setup-desktop.png')});
 await p.evaluate(async()=>{
   const root=await navigator.storage.getDirectory();const dir=await root.getDirectoryHandle('decisive-privacy-pplx-v1',{create:true});
   const writer=await (await dir.getFileHandle('old-weights.bin',{create:true})).createWritable();await writer.write(new Uint8Array(100));await writer.close();
 });
 await p.click('#close-setup');await p.click('#open-setup');
 await p.waitForSelector('#legacy-model:not([hidden])');
 await p.click('#delete-legacy');await p.click('#confirm-delete');
 await p.waitForSelector('#legacy-model[hidden]',{state:'attached'});
 assert.equal(await p.locator(`${decima}.ready`).count(),1);
 const other=await c.newPage();await other.goto(base);await other.waitForSelector('#intro-dialog[open]');await other.click('#intro-continue');await other.waitForSelector('#setup-dialog[open]');
 await other.evaluate(()=>{window.held=false;navigator.locks.request('decisive-privacy-model-files',{mode:'shared'},()=>new Promise(resolve=>{window.release=resolve;window.held=true;}));});
 await other.waitForFunction(()=>window.held);
 await p.click(`${decima} .model-delete`);await p.waitForSelector('#delete-dialog[open]');
 assert.equal(await p.locator(`${decima}.ready`).count(),1,'delete waits for confirmation');
 await p.click('#cancel-delete');assert.equal(await p.locator(`${decima}.ready`).count(),1);
 await p.click(`${decima} .model-delete`);await p.click('#confirm-delete');await p.waitForFunction(()=>!document.querySelector('#setup-error').hidden);
 assert.match(await p.locator('#setup-error').textContent(),/Another tab/);assert.equal(await p.locator(`${decima}.ready`).count(),1);
 await other.evaluate(()=>window.release());await other.close();
 await p.click(`${decima} .model-delete`);await p.click('#confirm-delete');
 await p.waitForFunction(sel=>!document.querySelector(sel).hidden,`${decima} .model-download`);
 assert.equal(await p.locator(`${decima} .model-delete`).isVisible(),false);
 assert.equal(await p.locator('#start-analysis').isDisabled(),true);
 await p.click(`${decima} .model-download`);await p.waitForFunction(()=>!document.getElementById('start-analysis').disabled,null,{timeout:30000});
 await p.click(`${decima} .model-delete`);await p.click('#confirm-delete');
 await p.waitForFunction(sel=>document.getElementById('start-analysis').disabled&&!document.querySelector(sel).hidden,`${decima} .model-download`);
 assert.equal(await p.locator('#setup-dialog').getAttribute('open'),'');
 assert.deepEqual(errors,[]);await c.close();
 console.log('Real WASM hashing, browser OPFS persistence, cross-tab lock, scoped deletion: PASS');

 const recorded=JSON.parse(await fs.readFile(new URL('./fixtures/english-result.json',import.meta.url)));
 const locales=JSON.parse(await fs.readFile(new URL('../lib/locales.json',import.meta.url)));
 const credentialDraft='My password is Birch!42 and the verification code is 739204.';
 const privateKeyDraft='My wallet recovery phrase is: abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about.';
 const credentialOutput=structuredClone(recorded.output);
 for(const answer of Object.values(credentialOutput.response.answers)){
   if(answer.type==='noul')answer.noul=0;
   else{
     answer.probabilities=Object.fromEntries(Object.keys(answer.probabilities).map(k=>[k,k==='none'?1:0]));
     answer.max_probabilities={...answer.probabilities};
   }
 }
 const privateKeyOutput=structuredClone(credentialOutput);
 credentialOutput.elapsed_ms=245;credentialOutput.response.usage.input_tokens=41;
 privateKeyOutput.elapsed_ms=310;privateKeyOutput.response.usage.input_tokens=62;
 privateKeyOutput.response.answers.secret.max_probabilities.privatekey=.94;
 Object.assign(credentialOutput.response.answers.secret.max_probabilities,{password:.61,token:.72,otp:.83});
 const allOutput=structuredClone(credentialOutput);
 Object.assign(allOutput.response.answers.personal.max_probabilities,{contact:.9,precise:.9,identity:.9});
 allOutput.response.answers.financial.noul=.9;
 const productionCatalog=await fetch(new URL('models.json',base)).then(r=>r.json());
 const productionManifest=productionCatalog.models.find(m=>m.id===productionCatalog.default);
 const files=Object.entries(productionManifest.files).map(([name,f])=>({name,expected:f.bytes,bytes:f.bytes,verified:true,state:'verified'}));
 const total=files.reduce((n,f)=>n+f.bytes,0);
 const state={modelId:productionManifest.id,revision:productionManifest.revision,files,total,stored:total,verified:total,remaining:0,ready:true,legacyStored:0};
 const recordedWorker=`const state=${JSON.stringify(state)}, output=${JSON.stringify(recorded.output)}, credentialOutput=${JSON.stringify(credentialOutput)}, privateKeyOutput=${JSON.stringify(privateKeyOutput)}, allOutput=${JSON.stringify(allOutput)};
 self.onmessage=({data:{id,method,args}})=>{
   if(method==='load' && args?.backend && args.backend!=='webgpu'){self.postMessage({id,error:'webgpuRequired'});return;}
   if(method==='predict' && (!args.request.questions.secret.criteria.otp || args.request.questions.secret.criteria.credential)){self.postMessage({id,error:'invalidOutput'});return;}
   const result=method==='inspect'?state:method==='capabilities'?{sync:true,gpu:true}:method==='predict'?
     args.request.state.context[1].content==='all five fixture'?allOutput:
     args.request.state.context[1].content===${JSON.stringify(privateKeyDraft)}?privateKeyOutput:
     args.request.state.context[1].content===${JSON.stringify(credentialDraft)}?credentialOutput:output:{};
   setTimeout(()=>self.postMessage({id,result}),method==='predict'?600:0);
 };`;
 const ui=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,serviceWorkers:'block'});
 await ui.route('**/workers/model-worker.js',r=>r.fulfill({contentType:'text/javascript',body:recordedWorker}));
 const view=await ui.newPage();view.on('pageerror',e=>errors.push(e.message));
 await view.goto(base);await view.waitForSelector('#intro-dialog[open]');await view.click('#intro-continue');await view.waitForSelector('#setup-dialog[open]');await view.click('#start-analysis');
 assert.equal(await view.locator('.score-row').count(),5);
 assert.equal(await view.locator('#score-name, #score-birthday, #score-privatekey, #score-password, #score-token, #score-otp').count(),0);
 await view.fill('#message-input',recorded.draft);await view.waitForSelector('#signal-contact');
 assert.equal(await view.locator('.signal-card').count(),1);
 assert.equal(await view.locator('#runtime-tokens').innerText(),'74');
 assert.equal(await view.locator('#runtime-time').innerText(),'33.19 s');
 await view.locator('#message-input').blur();
 const analysisBox=await view.locator('#analysis-scroll').boundingBox();
 const draftBox=await view.locator('#draft-region').boundingBox();
 await view.click('#signal-contact');assert.equal(await view.locator('#analysis-details').getAttribute('open'),'');
 await view.locator('#analysis-scroll').evaluate(el=>el.scrollTop=el.scrollHeight);
 assert.deepEqual(await view.locator('#analysis-scroll').boundingBox(),analysisBox,'analysis viewport stays fixed as details expand');
 assert.deepEqual(await view.locator('#draft-region').boundingBox(),draftBox,'analysis scrolling does not move draft');
 const raw=JSON.parse(await view.locator('#raw-output').textContent());
 assert.deepEqual(Object.keys(raw.input.questions),['personal','financial','secret']);
 assert.equal(Object.hasOwn(raw.input.questions.personal.criteria,'birthday'),false);
 assert.equal(raw.output.answers.personal.probabilities.contact,recorded.output.response.answers.personal.probabilities.contact);
 await view.screenshot({path:path.join(screenshots,'analysis-mobile.png')});
 await view.evaluate(()=>{window.previousSignal=document.getElementById('signal-contact');});
 await view.fill('#message-input','draft changing');
 await view.waitForFunction(()=>document.getElementById('composer-status').textContent==='Analyzing…');
 assert.deepEqual(await view.evaluate(()=>({
   samePill:window.previousSignal===document.getElementById('signal-contact'),
   tokens:document.getElementById('runtime-tokens').textContent,
   time:document.getElementById('runtime-time').textContent,
   score:document.querySelector('#score-contact .score-value').textContent,
 })),{samePill:true,tokens:'74',time:'33.19 s',score:'0.778'});
 assert.equal(await view.locator('#share-result').isDisabled(),true);
 await view.fill('#message-input',credentialDraft);await view.waitForSelector('#signal-credential');
 assert.equal(await view.locator('.signal-card').count(),1);
 assert.equal(await view.locator('#score-credential .score-value').innerText(),'0.830');
 assert.equal(await view.locator('#runtime-time').innerText(),'245 ms');
 assert.equal(await view.locator('#runtime-tokens').innerText(),'41');
 await view.screenshot({path:path.join(screenshots,'credentials-mobile.png')});
 await view.fill('#message-input',privateKeyDraft);await view.waitForFunction(()=>!document.getElementById('share-result').disabled);
 assert.equal(await view.locator('.signal-card').count(),1);
 assert.equal(await view.locator('#signal-privatekey').count(),0);
 assert.equal(await view.locator('#score-credential .score-value').innerText(),'0.940');
 await view.locator('#message-input').blur();
 await view.evaluate(()=>{
   const fillText=CanvasRenderingContext2D.prototype.fillText;
   CanvasRenderingContext2D.prototype.fillText=function(text,...args){window.sharedText.push(text);return fillText.call(this,text,...args);};
 });
 for(const locale of ['en','es','zh','ar']){
   await view.selectOption('#language-picker',locale);
   assert.equal(await view.locator('#signal-credential .signal-label').innerText(),locales[locale].short_credential);
   await view.evaluate(()=>{window.sharedText=[];});await view.click('#share-result');
   await view.waitForSelector('#share-dialog[open]');
   assert.equal(await view.locator('#share-image').evaluate(img=>img.naturalWidth),1080);
   assert.equal(await view.evaluate(label=>window.sharedText.filter(text=>text===label).length,locales[locale].credential),1);
   await view.screenshot({path:path.join(screenshots,`share-${locale}.png`)});
   const saved=view.waitForEvent('download');await view.click('#download-image');
   const download=await saved;assert.equal(download.suggestedFilename(),'privacy-analysis.png');
   await view.click('#close-share');
 }
 await view.selectOption('#language-picker','en');
 await view.setViewportSize({width:390,height:400});
 await view.fill('#message-input','all five fixture');await view.waitForFunction(()=>!document.getElementById('share-result').disabled);
 assert.equal(await view.locator('.signal-card').count(),5);
 assert.equal(await view.locator('#page-description').isVisible(),false);
 assert.equal(await view.locator('.header-secondary').isVisible(),false);
 assert.equal(await view.locator('#analysis-title').isVisible(),false);
 const pillsFit=()=>{
   const input=document.getElementById('draft-region').getBoundingClientRect();
   const pills=[...document.querySelectorAll('.signal-card')].map(el=>el.getBoundingClientRect());
   const metrics=document.getElementById('inference-metrics').getBoundingClientRect();
   return pills.every(box=>box.top>=input.bottom&&box.bottom<=innerHeight&&box.height<=40)&&metrics.top>=pills.at(-1).bottom;
 };
 await view.screenshot({path:path.join(screenshots,'typing-mobile.png'),animations:'disabled'});
 assert.equal(await view.evaluate(pillsFit),true,'all five pills fit below the focused input at keyboard height');
 await view.tap('#signal-credential');
 assert.equal(await view.locator('#analysis-details').getAttribute('open'),'','tapping a pill works as the keyboard layout closes');
 assert.equal(await view.locator('#signal-credential').getAttribute('aria-pressed'),'true');
 await view.locator('#message-input').focus();
 await view.emulateMedia({colorScheme:'dark'});
 await view.screenshot({path:path.join(screenshots,'typing-mobile-dark.png'),animations:'disabled'});
 await view.locator('#message-input').blur();await view.selectOption('#language-picker','ar');await view.locator('#message-input').focus();
 await view.screenshot({path:path.join(screenshots,'typing-mobile-ar.png'),animations:'disabled'});
 assert.equal(await view.evaluate(pillsFit),true,'Arabic pills fit at keyboard height');
 await view.evaluate(()=>{
   Object.defineProperties(visualViewport,{height:{value:360,configurable:true},offsetTop:{value:40,configurable:true}});
   visualViewport.dispatchEvent(new Event('resize'));
 });
 assert.deepEqual(await view.locator('#app-shell').evaluate(el=>{const box=el.getBoundingClientRect();return {top:box.top,height:box.height};}),{top:40,height:360},'follow an offset visual viewport');
 await view.evaluate(()=>{delete visualViewport.height;delete visualViewport.offsetTop;visualViewport.dispatchEvent(new Event('resize'));});
 await view.setViewportSize({width:320,height:480});
 assert.equal(await view.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 assert.equal(await view.locator('#message-input').isVisible(),true);
 await view.emulateMedia({reducedMotion:'reduce'});
 await view.fill('#message-input','A new draft');
 await view.waitForFunction(()=>!document.getElementById('share-result').disabled);
 assert.equal(await view.locator('#live-analysis').evaluate(el=>el.getAnimations({subtree:true}).length),0);
 await view.fill('#message-input','');
 assert.equal(await view.locator('.signal-card').count(),0);
 assert.equal(await view.locator('#runtime-time').innerText(),'—');
 assert.equal(await view.locator('#runtime-tokens').innerText(),'—');
 await view.setViewportSize({width:1100,height:850});
 await view.fill('#message-input','all five fixture');await view.waitForFunction(()=>!document.getElementById('share-result').disabled);
 assert.equal(await view.locator('#page-description').isVisible(),true,'desktop header stays visible while typing');
 await view.screenshot({path:path.join(screenshots,'pills-desktop-dark.png'),animations:'disabled'});
 await view.emulateMedia({colorScheme:'light'});
 await view.screenshot({path:path.join(screenshots,'pills-desktop.png'),animations:'disabled'});
 await view.fill('#message-input','x'.repeat(1001));
 assert.equal((await view.locator('#message-input').inputValue()).length,1000);
 await view.selectOption('#language-picker','en');
 assert.equal(await view.locator('#character-count').textContent(),'1,000 / 1,000');
 assert.deepEqual(errors,[]);await ui.close();
 console.log('Recorded and synthetic output: pills, retained results, mobile typing, character limit, four-language PNG export: PASS');

 const unavailable=await browser.newContext({viewport:{width:390,height:844},serviceWorkers:'block'});
 await unavailable.route('**/lib/webgpu.js',r=>r.fulfill({contentType:'text/javascript',body:'export async function checkWebGPU(){return false;}'}));
 const blocked=await unavailable.newPage();blocked.on('pageerror',e=>errors.push(e.message));
 await blocked.goto(base);await blocked.waitForSelector('#intro-dialog[open]');await blocked.click('#intro-continue');await blocked.waitForSelector('#setup-dialog[open]');
 assert.equal(await blocked.locator('#start-analysis').isDisabled(),true);
 assert.match(await blocked.locator('#setup-error').textContent(),/WebGPU is not available/);
 await blocked.screenshot({path:path.join(screenshots,'webgpu-unavailable.png')});
 assert.equal(await blocked.locator('#setup-error').isVisible(),true);
 assert.equal(await blocked.locator('#setup-models .model-download:visible').count(),0);
 assert.equal(await blocked.locator('#setup-models .model-option:not(.blocked)').count(),0,'every card disabled without WebGPU');
 assert.equal(await blocked.locator('#setup-models .model-pick:enabled').count(),0);
 assert.match(await blocked.locator('#setup-models [data-model="decima-small"] .model-tip').textContent(),/WebGPU is not available/);
 const refusal=await blocked.evaluate(()=>new Promise(resolve=>{
   const worker=new Worker(new URL('workers/model-worker.js',location.href),{type:'module'});
   worker.onmessage=({data})=>{worker.terminate();resolve(data.error);};
   worker.postMessage({id:1,method:'load',args:{backend:'wasm'}});
 }));
 assert.equal(refusal,'webgpuRequired');
 await unavailable.close();
 console.log('WebGPU unavailable blocks download/analysis; CPU requests rejected: PASS');

}finally{await browser.close();}
