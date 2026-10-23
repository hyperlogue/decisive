import {ModelStore, LOCK_NAME} from '../lib/model-store.js';
import {loadFlash} from '../runtime/runtime.js';
import {checkWebGPU} from '../lib/webgpu.js';

let storePromise, apiPromise, engine = null, backend = null, busy = false, cancellation = null;
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
async function store() {
  if (!storePromise) storePromise = (async () => {
    const [manifest, root] = await Promise.all([
      fetch(new URL('../model.json', import.meta.url)).then(r=>r.json()),
      navigator.storage.getDirectory(),
    ]);
    return new ModelStore({manifest, root, createHasher:() => new loadedApi.PplxFlashHasher()});
  })().catch(error=>{storePromise=null;throw error;});
  return storePromise;
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
  engine = null; backend = null;
}
const methods = {
  async capabilities() {
    return {gpu:await checkWebGPU(), sync:typeof FileSystemFileHandle !== 'undefined' &&
      typeof FileSystemFileHandle.prototype.createSyncAccessHandle === 'function'};
  },
  async inspect() { return locked('shared',async()=> (await store()).inspect()); },
  async download() {
    return locked('exclusive', async () => {
      unload();
      loadedApi = await api();
      const cache = await store(), state = await cache.inspect();
      const estimate = await navigator.storage.estimate();
      if (estimate.quota && estimate.usage != null && estimate.quota-estimate.usage < state.remaining + 16*1024*1024) {
        throw new Error('insufficientSpace');
      }
      try { return await cache.download({signal:cancellation.signal, progress:notify}); }
      finally { broadcast?.postMessage('changed'); }
    });
  },
  async remove() {
    return locked('exclusive',async()=>{
      unload();
      const result = await (await store()).remove();
      broadcast?.postMessage('changed');
      return result;
    });
  },
  async unload() { unload(); return true; },
  async load({backend:requested} = {}) {
    return locked('shared',async()=>{
      if (requested && requested !== 'webgpu') throw new Error('webgpuRequired');
      if (engine) return {backend};
      unload();
      if (!(await checkWebGPU())) throw new Error('webgpuUnavailable');
      notify({phase:'loading'});
      loadedApi = await api();
      const files = await (await store()).files();
      engine = await loadFlash(loadedApi, files, message=>{
        if (cancellation?.signal.aborted) throw new DOMException('Cancelled','AbortError');
        notify({phase:busy && backend ? 'analyzing' : 'loading', message});
      }, 'webgpu');
      backend = 'webgpu';
      return {backend, startup:engine.startup};
    });
  },
  async predict({request}) {
    return locked('shared',async()=>{
      if (!engine) throw new Error('modelMissing');
      if (!(await (await store()).inspect()).ready) { unload(); throw new Error('modelMissing'); }
      notify({phase:'analyzing'});
      return engine.predict(request, 2048, true);
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
      const known = ['paused','modelMissing','anotherTab','storageUnsupported','runtimeFailed','webgpuRequired','webgpuUnavailable'];
      if (!known.includes(code)) code = 'inferenceFailed';
    }
    self.postMessage({id,error:code,detail:method==='predict'||method==='load'?String(error):undefined});
  } finally { busy = false; cancellation = null; }
};
