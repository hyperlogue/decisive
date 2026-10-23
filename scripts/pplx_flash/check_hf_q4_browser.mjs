import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base=process.argv[2];
assert.ok(base,'Pass the serve_phone.py version URL (ending in /v/current/)');
const request=JSON.parse(await fs.readFile(new URL('../../tests/fixtures/pplx_flash/request.json',import.meta.url)));
const software=process.env.PRIVACY_SOFTWARE_GPU==='1';
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,
  args:['--no-sandbox','--disable-dev-shm-usage',...(software?['--enable-unsafe-webgpu','--use-angle=swiftshader']:[])]});
try {
  const page=await browser.newPage();
  await page.route(base,route=>route.fulfill({contentType:'text/html',body:'<!doctype html><title>Q4 release smoke check</title>'}));
  await page.goto(base);
  await page.evaluate(({request})=>{
    window.smoke={progress:'Loading runtime'};
    (async()=>{
      let engine;
      try {
        const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});
        window.smoke.features=adapter?[...adapter.features]:[];
        if(!adapter?.features.has('shader-f16'))throw Error('The browser adapter lacks shader-f16; Q4 inference was not run.');
        const api=await import(new URL('./pkg/decisive.js',location.href));
        await api.default();
        const {loadFlash}=await import(new URL('./runtime.js',location.href));
        engine=await loadFlash(api,new URL('./model',location.href).href,
          progress=>window.smoke.progress=progress,'webgpu',{mode:'q4-e4',resident:false,profile:false});
        const output=await engine.predict(request,1024);
        window.smoke.result={request,response:output.response,backend:output.backend,
          elapsed_ms:output.elapsed_ms,decoder_weight_bytes_read:output.decoder_weight_bytes_read,
          layers:output.layers.length,timing_scope:output.timing_scope,
          startup:engine.startup,adapter:output.adapter};
      }catch(error){window.smoke.error=String(error?.stack||error);}
      finally{engine?.dispose();window.smoke.done=true;}
    })();
  },{request});
  const start=Date.now();let last;
  for(;;) {
    const state=await page.evaluate(()=>window.smoke);
    if(state.progress!==last){console.log(state.progress);last=state.progress;}
    if(state.error){
      await fs.writeFile(process.env.Q4_SMOKE_OUTPUT || '/tmp/pplx-q4-release-smoke.json',JSON.stringify({
        passed:false,device:software?'Chromium SwiftShader (correctness only)':'Browser WebGPU',
        features:state.features,error:state.error,
      },null,2)+'\n');
      throw Error(state.error);
    }
    if(state.done){
      assert.equal(state.result.backend,'webgpu');
      assert.equal(state.result.response.answers.team.choice,'support');
      assert.equal(state.result.layers,32);
      assert.ok(state.result.decoder_weight_bytes_read>0);
      assert.equal(state.result.timing_scope,'prepacked-streamed-request');
      const report={passed:true,device:software?'Chromium SwiftShader (correctness only)':'Browser WebGPU',...state.result};
      const output=process.env.Q4_SMOKE_OUTPUT || '/tmp/pplx-q4-release-smoke.json';
      await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
      console.log(JSON.stringify({passed:true,choice:report.response.answers.team.choice,input_tokens:report.response.usage.input_tokens,elapsed_ms:report.elapsed_ms}));
      break;
    }
    if(Date.now()-start>15*60*1000)throw Error('Q4 smoke test timed out');
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
}finally{await browser.close();}
