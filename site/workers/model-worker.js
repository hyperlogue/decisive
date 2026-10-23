import {ModelStore, LOCK_NAME} from '../lib/model-store.js';
import {loadDecima} from '../lib/model-runtime.js';
import {loadPplxFlash} from '../lib/pplx-runtime.js';
import {selectModel, requirePublished} from '../lib/model-catalog.js';
import {checkWebGPU} from '../lib/webgpu.js';

let catalogPromise, apiPromise, engine = null, activeModel = null, busy = false, cancellation = null;
const stores = new Map();
const notify = value => self.postMessage({event:'progress', ...value});
const broadcast = typeof BroadcastChannel === 'function' ? new BroadcastChannel('decisive-privacy-storage') : null;

async function api() {
  if (!apiPromise) apiPromise = (async () => {
    const [module, build] = await Promise.all([
      import('../pkg/decisive.js'),
      fetch(new URL('../pkg/build.json', import.meta.url)).then(r=>r.json()),
    ]);
    const response = await fetch(new URL('../pkg/decisive_bg.wasm', import.meta.url));
    if (!response.ok) throw new Error('runtimeFailed');
    const bytes = await response.arrayBuffer();
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))]
      .map(n=>n.toString(16).padStart(2,'0')).join('');
    if (digest !== build.wasm_sha256) throw new Error('runtimeFailed');
    await module.default({module_or_path:bytes});
    return module;
  })().catch(error => { apiPromise = null; throw error; });
  return apiPromise;
}
async function definition(id) {
  catalogPromise ??= fetch(new URL('../models.json', import.meta.url)).then(r=>r.json());
  return selectModel(await catalogPromise,id);
}
async function store(id) {
  const model = id === 'legacy' ? {id:'legacy',storeName:'decisive-privacy-pplx-v1',revision:'retired',files:{}} : await definition(id);
  if (!stores.has(model.id)) stores.set(model.id,(async()=>new ModelStore({
    manifest:model,storeName:model.storeName,root:await navigator.storage.getDirectory(),
    createHasher:()=>new loadedApi.PplxFlashHasher(),
  }))());
  return stores.get(model.id);
}
async function inspect(id) {
  const model=await definition(id);
  const state=await (await store(id)).inspect();
  state.modelId=model.id;state.revision=model.revision;
  state.legacyStored=(await (await store('legacy')).inspect()).stored;
  return state;
}
let loadedApi;
async function locked(mode, task) {
  if (!navigator.locks) throw new Error('storageUnsupported');
  return navigator.locks.request(LOCK_NAME, {mode, ifAvailable:true}, lock => {
    if (!lock) throw new Error('anotherTab');
    return task();
  });
}
function unload() {
  engine?.dispose();
  engine = null; activeModel = null;
}
async function requireModelGPU(model) {
  if (!(await checkWebGPU())) throw new Error('webgpuUnavailable');
  if (model.requiredFeatures?.length && !(await checkWebGPU(undefined, model.requiredFeatures))) {
    throw new Error('gpuPrecisionRequired');
  }
}
const methods = {
  async capabilities() {
    const gpu=await checkWebGPU();
    return {gpu, shaderF16:gpu&&await checkWebGPU(undefined,['shader-f16']), sync:typeof FileSystemFileHandle !== 'undefined' &&
      typeof FileSystemFileHandle.prototype.createSyncAccessHandle === 'function'};
  },
  async inspect({modelId} = {}) { return locked('shared',()=>inspect(modelId)); },
  async download({modelId} = {}) {
    const model=requirePublished(await definition(modelId));
    await requireModelGPU(model);
    return locked('exclusive', async () => {
      unload();
      loadedApi = await api();
      const cache = await store(modelId), state = await cache.inspect();
      const estimate = await navigator.storage.estimate();
      if (estimate.quota && estimate.usage != null && estimate.quota-estimate.usage < state.remaining + 16*1024*1024) {
        throw new Error('insufficientSpace');
      }
      try { await cache.download({signal:cancellation.signal, progress:notify}); return await inspect(modelId); }
      finally { broadcast?.postMessage('changed'); }
    });
  },
  async remove({modelId} = {}) {
    return locked('exclusive',async()=>{
      unload();
      await (await store(modelId)).remove();
      const result=await inspect(modelId);
      broadcast?.postMessage('changed');
      return result;
    });
  },
  async removeLegacy() {
    return locked('exclusive',async()=>{
      await (await store('legacy')).remove();
      broadcast?.postMessage('changed');
      return true;
    });
  },
  async unload() { unload(); return true; },
  async load({backend:requested,modelId} = {}) {
    return locked('shared',async()=>{
      if (requested && requested !== 'webgpu') throw new Error('webgpuRequired');
      const model=requirePublished(await definition(modelId));
      if (engine && activeModel===model.id) return {backend:'webgpu',model:model.name};
      unload();
      await requireModelGPU(model);
      notify({phase:'loading'});
      loadedApi=await api();
      const files=await (await store(model.id)).files();
      const load=model.engine==='decima'?loadDecima:model.engine==='pplx-q4'?loadPplxFlash:null;
      if(!load)throw new Error('modelComingSoon');
      engine=await load(loadedApi,files,message=>notify({phase:busy&&activeModel?'analyzing':'loading',message}));
      if(cancellation.signal.aborted){unload();throw new DOMException('Cancelled','AbortError');}
      activeModel=model.id;
      return {backend:'webgpu',model:model.name};
    });
  },
  async predict({request,modelId}) {
    return locked('shared',async()=>{
      if (!engine || activeModel!==(await definition(modelId)).id) throw new Error('modelMissing');
      if (!(await (await store(activeModel)).inspect()).ready) { unload(); throw new Error('modelMissing'); }
      notify({phase:'analyzing'});
      const result=await engine.predict(request,cancellation.signal);
      if(cancellation.signal.aborted)throw new DOMException('Cancelled','AbortError');
      return result;
    });
  },
};
self.onmessage = async ({data:{id, method, args}}) => {
  if (method === 'pause') {
    cancellation?.abort();
    self.postMessage({id,result:true}); return;
  }
  if (busy) { self.postMessage({id,error:'busy'}); return; }
  if (!methods[method]) { self.postMessage({id,error:'unknownCommand'}); return; }
  busy = true; cancellation = new AbortController();
  try { self.postMessage({id,result:await methods[method](args)}); }
  catch(error) {
    let code = error.name === 'AbortError' ? 'paused' :
      error.name === 'QuotaExceededError' ? 'insufficientSpace' :
      error.name === 'NoModificationAllowedError' ? 'anotherTab' :
      error.name === 'TypeError' && method === 'download' ? 'downloadFailed' : error.message;
    if (method === 'predict' || method === 'load') {
      try { unload(); } catch {}
      const known = ['paused','modelMissing','anotherTab','storageUnsupported','runtimeFailed','webgpuRequired','webgpuUnavailable','gpuPrecisionRequired','modelComingSoon','inputTooLong','pplxInputTooLong'];
      if(/512|maximum state|state.*too long/i.test(String(error)))code='inputTooLong';
      if(/single question exceeds token limit/i.test(String(error)))code='pplxInputTooLong';
      if (!known.includes(code)) code = 'inferenceFailed';
    }
    self.postMessage({id,error:code,detail:method==='predict'||method==='load'?String(error):undefined});
  } finally { busy = false; cancellation = null; }
};
