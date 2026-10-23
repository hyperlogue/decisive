import assert from 'node:assert/strict';
import path from 'node:path';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base=process.env.PRIVACY_TEST_URL || 'http://127.0.0.1:8768/site/dist/';
const browser=await chromium.launch({executablePath:process.env.PRIVACY_BROWSER_EXECUTABLE,headless:true,
  args:['--no-sandbox','--disable-dev-shm-usage',...(process.env.PRIVACY_SOFTWARE_GPU==='1'?['--enable-unsafe-webgpu','--use-angle=swiftshader']:[])]});
try{
  const page=await browser.newPage(),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  const fixture=new URL('sdk-errors.html',base).href;
  await page.route(fixture,route=>route.fulfill({contentType:'text/html',body:'<!doctype html><input id="files" type="file" multiple>'}));
  await page.goto(fixture);
  const report=await page.evaluate(async()=>{
    const api=await import('./pkg/decisive.js');await api.default();
    const build=await fetch('./pkg/build.json').then(r=>r.json());
    const features=new Set(build.features.split(','));
    const cases=[
      [{model:'decima',backend:'webgpu',format:'int8'},'UNSUPPORTED_FORMAT','f32'],
      [{model:'pplx-flash',backend:'wasm',format:'q4-e4'},'UNSUPPORTED_FORMAT','bf16'],
      [{model:'kev',backend:'webgpu'},'UNSUPPORTED_BACKEND','native CPU'],
      [{model:'jev-style',backend:'webgpu'},'UNSUPPORTED_BACKEND','native CPU'],
      [{model:'unknown',backend:'webgpu'},'UNSUPPORTED_MODEL','decima'],
      [{model:'decima',backend:'cuda'},'UNSUPPORTED_BACKEND','wasm'],
    ];
    if(!features.has('laya')&&!features.has('all-models'))cases.push([{model:'laya',backend:'wasm'},'MODEL_NOT_COMPILED','laya']);
    if(features.has('decima')&&!features.has('int8'))cases.push([{model:'decima',backend:'wasm',format:'int8'},'FORMAT_NOT_COMPILED','int8']);
    let reads=0;
    const files=new Proxy({}, {get(){reads++;throw Error('Checkpoint read before validation');}});
    const capture=async task=>{try{const model=await task();await model.dispose();return {code:'UNEXPECTED_SUCCESS'};}catch(error){return {code:error.code,message:error.message};}};
    const control=new api.InferenceControl();
    const direct=[];
    try{
      for(const [selection,code,alternative] of cases){
        const error=await capture(()=>api.DecisiveModel.load(JSON.stringify(selection),files,control));
        direct.push({selection,code,alternative,error});
      }
    }finally{control.free();}
    const {loadModel}=await import('./sdk/index.js');
    const worker=await capture(()=>loadModel({model:'pplx-flash',backend:'wasm',format:'q4-e4',source:{files:{}}}));
    return {direct,reads,worker};
  });
  assert.equal(report.reads,0,'unsupported selections must fail before reading checkpoint files');
  for(const {selection,code,alternative,error} of report.direct){
    assert.equal(error.code,code,JSON.stringify(selection));
    assert.ok(error.message.includes(selection.model),error.message);
    assert.ok(error.message.includes(alternative),error.message);
  }
  assert.equal(report.worker.code,'UNSUPPORTED_FORMAT');
  assert.match(report.worker.message,/pplx-flash.*wasm.*q4-e4/);
  assert.match(report.worker.message,/bf16/);
  console.log('Real WASM: unsupported selections rejected before checkpoint reads; build-feature errors and Worker error transport: PASS');
  if(process.env.PPLX_Q4_MODEL_DIR){
    const hasF16=await page.evaluate(async()=>{
      const adapter=await navigator.gpu?.requestAdapter();
      return adapter ? adapter.features.has('shader-f16') : null;
    });
    assert.notEqual(hasF16,null,'Q4 missing-feature check requires a WebGPU adapter');
    if(!hasF16){
      const names=['packed-q4-e4.json','config.json','decision_config.json','joint_config.json','tokenizer.json','q4-e4-readout.bin'];
      await page.locator('#files').setInputFiles(names.map(name=>path.join(process.env.PPLX_Q4_MODEL_DIR,name)));
      const error=await page.evaluate(async names=>{
        const files=Object.fromEntries(names.map((name,index)=>[name,document.querySelector('#files').files[index]]));
        const {loadModel}=await import('./sdk/index.js');
        try{const model=await loadModel({model:'pplx-flash',backend:'webgpu',format:'q4-e4',source:{files},tuning:{mode:'off'}});await model.dispose();return {};}
        catch(error){return {code:error.code,message:error.message};}
      },names);
      assert.equal(error.code,'MISSING_GPU_FEATURE',error.message);
      assert.match(error.message,/shader-f16/);
      assert.match(error.message,/Decima/);
      console.log('Real Q4 metadata on a GPU without shader-f16: specific capability error and supported alternative: PASS (no inference)');
    }else console.log('Q4 missing-feature check skipped: GPU provides shader-f16');
  }
  assert.deepEqual(errors,[]);
}finally{await browser.close();}
