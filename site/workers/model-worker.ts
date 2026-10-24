import type {StorageDirectory} from '../lib/model-store.js';
import {ModelStore, LOCK_NAME} from '../lib/model-store.js';
import {loadModelInWorker} from '@decisive/browser/local.js';
import type {WasmModule} from '@decisive/browser/local.js';
import type {Predictor, OpaqueCache, Request, Progress} from '@decisive/browser/types.js';
import {inferenceConfig} from '../lib/inference-config.js';
import {inferenceErrorKey} from '../lib/inference-errors.js';
import {selectModel, requirePublished} from '../lib/model-catalog.js';
import {checkWebGPU} from '../lib/webgpu.js';

interface CatalogModel { id: string; name: string; engine: string; storeName: string; revision: string; repository: string; available: boolean; files: Record<string, {bytes:number;sha256:string}>; requiredFeatures?: string[] }
interface Hasher { update(bytes: Uint8Array): void; finish(): string; free(): void }
type API = WasmModule & {CheckpointHasher: new () => Hasher};
interface Args { modelId?: string; backend?: string; request?: Request }
let catalogPromise: Promise<{default:string;models:CatalogModel[]}> | undefined;
let apiPromise: Promise<API> | null = null;
let engine: Predictor | null = null, inferenceCache: OpaqueCache | undefined;
let activeModel: string | null = null, busy = false, cancellation: AbortController | null = null;
let closeReaders = () => {};
const stores = new Map<string, Promise<ModelStore>>();
const notify = (value: Record<string, unknown> = {}) => self.postMessage({event:'progress', ...value});
const broadcast = typeof BroadcastChannel === 'function' ? new BroadcastChannel('decisive-privacy-storage') : null;

async function api() {
  if (!apiPromise) apiPromise = (async () => {
    const [module, build] = await Promise.all([
      import(new URL('../pkg/decisive.js', import.meta.url).href) as Promise<API>,
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
async function definition(id?: string) {
  catalogPromise ??= fetch(new URL('../models.json', import.meta.url)).then(r=>r.json());
  return selectModel(await catalogPromise,id);
}
async function store(id?: string) {
  const model = id === 'legacy' ? {id:'legacy',name:'Legacy',engine:'retired',available:false,repository:'',storeName:'decisive-privacy-pplx-v1',revision:'retired',files:{}} : await definition(id);
  if (!stores.has(model.id)) stores.set(model.id,(async()=>new ModelStore({
    manifest:model,storeName:model.storeName,root:await navigator.storage.getDirectory() as unknown as StorageDirectory,
    createHasher:()=>new loadedApi!.CheckpointHasher(),
  }))());
  return stores.get(model.id)!;
}
async function inspect(id?: string) {
  const model=await definition(id);
  const state = await (await store(id)).inspect() as Awaited<ReturnType<ModelStore['inspect']>> & {modelId:string;revision:string;legacyStored:number};
  state.modelId=model.id;state.revision=model.revision;
  state.legacyStored=(await (await store('legacy')).inspect()).stored;
  return state;
}
let loadedApi: API | undefined;
async function locked<T>(mode: LockMode, task: () => Promise<T>): Promise<T> {
  if (!navigator.locks) throw new Error('storageUnsupported');
  return navigator.locks.request(LOCK_NAME, {mode, ifAvailable:true}, lock => {
    if (!lock) throw new Error('anotherTab');
    return task();
  });
}
async function unload() {
  await engine?.dispose();
  closeReaders(); closeReaders = () => {};
  inferenceCache = undefined;
  engine = null; activeModel = null;
}
async function requireModelGPU(model: CatalogModel) {
  if (!(await checkWebGPU())) throw new Error('webgpuUnavailable');
  if (model.requiredFeatures?.length && !(await checkWebGPU(undefined, model.requiredFeatures))) {
    throw new Error('gpuPrecisionRequired');
  }
}
const methods: Record<string, (args: Args) => Promise<unknown>> = {
  async capabilities() {
    const gpu=await checkWebGPU();
    return {gpu, shaderF16:gpu&&await checkWebGPU(undefined,['shader-f16']), sync:typeof FileSystemFileHandle !== 'undefined' &&
      typeof (FileSystemFileHandle.prototype as FileSystemFileHandle & {createSyncAccessHandle?:unknown}).createSyncAccessHandle === 'function'};
  },
  async inspect({modelId} = {}) { return locked('shared',()=>inspect(modelId)); },
  async download({modelId} = {}) {
    const model=requirePublished(await definition(modelId));
    await requireModelGPU(model);
    return locked('exclusive', async () => {
      await unload();
      loadedApi = await api();
      const cache = await store(modelId), state = await cache.inspect();
      const estimate = await navigator.storage.estimate();
      if (estimate.quota && estimate.usage != null && estimate.quota-estimate.usage < state.remaining + 16*1024*1024) {
        throw new Error('insufficientSpace');
      }
      try { await cache.download({signal:cancellation!.signal, progress:notify}); return await inspect(modelId); }
      finally { broadcast?.postMessage('changed'); }
    });
  },
  async remove({modelId} = {}) {
    return locked('exclusive',async()=>{
      await unload();
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
  async unload() { await unload(); return true; },
  async load({backend:requested,modelId} = {}) {
    return locked('shared',async()=>{
      if (requested && requested !== 'webgpu') throw new Error('webgpuRequired');
      const model=requirePublished(await definition(modelId));
      if (engine && activeModel===model.id) return {backend:'webgpu',model:model.name};
      await unload();
      await requireModelGPU(model);
      notify({phase:'loading'});
      loadedApi=await api();
      const cache=await store(model.id);
      // Pplx streams its layers on every request, so it reads OPFS through sync handles when available.
      let files: Awaited<ReturnType<ModelStore['readers']>>['files'];
      if (model.engine.startsWith('pplx-')) { const readers=await cache.readers(); files=readers.files; closeReaders=readers.close; }
      else files=await cache.files();
      engine = await loadModelInWorker(loadedApi, inferenceConfig(model, files), {
        signal: cancellation!.signal,
        onProgress: (event: Progress) => notify({phase: activeModel ? 'analyzing' : 'loading',
          message: event.phase === 'inference' ? `Layer ${event.completed % 32 + 1}/32` : event.phase}),
      });
      inferenceCache = await engine.createCache();
      if(cancellation!.signal.aborted){await unload();throw new DOMException('Cancelled','AbortError');}
      activeModel=model.id;
      return {backend:'webgpu',model:model.name};
    });
  },
  async predict({request,modelId}) {
    return locked('shared',async()=>{
      if (!engine || activeModel!==(await definition(modelId)).id) throw new Error('modelMissing');
      if (!(await (await store(activeModel!)).inspect()).ready) { await unload(); throw new Error('modelMissing'); }
      notify({phase:'analyzing'});
      const result=await engine.predict(request!,{cache:inferenceCache!,signal:cancellation!.signal});
      if(cancellation!.signal.aborted)throw new DOMException('Cancelled','AbortError');
      return result;
    });
  },
};
self.onmessage = async ({data:{id, method, args = {}}}: MessageEvent<{id:number;method:string;args?:Args}>) => {
  if (method === 'pause') {
    cancellation?.abort();
    self.postMessage({id,result:true}); return;
  }
  if (busy) { self.postMessage({id,error:'busy'}); return; }
  if (!methods[method]) { self.postMessage({id,error:'unknownCommand'}); return; }
  busy = true; cancellation = new AbortController();
  try { self.postMessage({id,result:await methods[method](args)}); }
  catch(caught) {
    const error = caught as Error & {code?:string};
    let code = error.name === 'AbortError' ? 'paused' :
      error.name === 'QuotaExceededError' ? 'insufficientSpace' :
      error.name === 'NoModificationAllowedError' ? 'anotherTab' :
      error.name === 'TypeError' && method === 'download' ? 'downloadFailed' : error.message;
    if (method === 'predict' || method === 'load') {
      try { await unload(); } catch {}
      const known = ['paused','insufficientSpace','modelMissing','anotherTab','storageUnsupported','runtimeFailed','webgpuRequired','webgpuUnavailable','gpuPrecisionRequired','modelUnsupported','modelMemoryLimit','modelComingSoon','inputTooLong','pplxInputTooLong'];
      code = inferenceErrorKey(error.code, error.code === 'INPUT_TOO_LONG' ? (await definition(args.modelId)).engine : undefined) ?? code;
      if (!known.includes(code)) code = 'inferenceFailed';
    }
    self.postMessage({id,error:code,detail:method==='predict'||method==='load'?String(error):undefined});
  } finally { busy = false; cancellation = null; }
};
