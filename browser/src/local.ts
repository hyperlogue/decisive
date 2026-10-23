import { InferenceError } from './types.js';
import type { Backend, BatchResult, CacheOptions, Controls, Family, InferenceResult, LoadOptions, Model, ModelInfo, OpaqueCache, PredictOptions, Predictor, Progress, Registry, Request, Selection } from './types.js';

interface WasmControl { cancel(): void; free(): void }
interface WasmCache { clear(): void; dispose(): void; free(): void }
interface WasmModel {
  createCache(options: string): WasmCache;
  info(): string;
  predict(requests: string, parameters: string, cache: WasmCache, control: WasmControl, batch: boolean): Promise<string>;
  dispose(): void;
  free(): void;
}
export interface WasmModule {
  default(options?: { module_or_path: BufferSource }): Promise<unknown>;
  InferenceControl: new (progress?: (json: string) => void) => WasmControl;
  DecisiveModel: { load(configuration: string, files: Record<string, Blob | Uint8Array>, control: WasmControl): Promise<WasmModel> };
}

class Cache implements OpaqueCache {
  closed = false;
  busy = false;
  constructor(readonly owner: object, readonly value: WasmCache, readonly release: () => void) {}
  async clear(): Promise<void> { this.check(); this.value.clear(); }
  check(): void {
    if (this.closed) throw new InferenceError('DISPOSED', 'Cache is disposed');
    if (this.busy) throw new InferenceError('CACHE_BUSY', 'Cache is in use');
  }
  async dispose(): Promise<void> {
    if (this.closed) return;
    this.check(); this.value.dispose(); this.value.free(); this.closed = true; this.release();
  }
}
function control(api: WasmModule, options: Controls): { value: WasmControl; dispose(): void } {
  options.signal?.throwIfAborted();
  const value = new api.InferenceControl(options.onProgress ? json => options.onProgress?.(JSON.parse(json) as Progress) : undefined);
  const cancel = (): void => value.cancel();
  options.signal?.addEventListener('abort', cancel, { once: true });
  return { value, dispose() { options.signal?.removeEventListener('abort', cancel); value.free(); } };
}

export function loadModelInWorker<M extends Family, B extends Backend<M>>(api: WasmModule, options: LoadOptions<M, B>, controls?: Controls): Promise<Model<Registry[M][B]>>;
export function loadModelInWorker(api: WasmModule, options: Selection, controls?: Controls): Promise<Predictor>;
export async function loadModelInWorker(api: WasmModule, options: Selection, controls: Controls = {}): Promise<Model<unknown>> {
  const cancellation = control(api, controls);
  let runtime: WasmModel;
  try {
    const { source, ...configuration } = options;
    runtime = await api.DecisiveModel.load(JSON.stringify(configuration), source.files, cancellation.value);
  } finally { cancellation.dispose(); }
  const owner = {}, caches = new Set<Cache>();
  let disposed = false, busy = false;
  const check = (): void => {
    if (disposed) throw new InferenceError('DISPOSED', 'Model is disposed');
    if (busy) throw new InferenceError('MODEL_BUSY', 'Model is in use');
  };
  const createCache = (options: CacheOptions = {}): Cache => {
    check();
    const value = runtime.createCache(JSON.stringify(options));
    const cache = new Cache(owner, value, () => caches.delete(cache));
    caches.add(cache); return cache;
  };
  async function predict(request: Request | readonly Request[], options: PredictOptions<unknown>, batch: boolean): Promise<InferenceResult | BatchResult> {
    if (disposed) check();
    if (options.cache && (!(options.cache instanceof Cache) || options.cache.owner !== owner)) throw new InferenceError('CACHE_OWNER', 'Cache belongs to another model');
    const cache = options.cache as Cache | undefined;
    cache?.check(); check();
    options.signal?.throwIfAborted();
    const cancellation = control(api, { onProgress: controls.onProgress, ...options } as Controls);
    let active: Cache;
    try { active = cache ?? createCache({ maxBytes: 0 }); } catch (error) { cancellation.dispose(); throw error; }
    busy = true; active.busy = true;
    try {
      return JSON.parse(await runtime.predict(JSON.stringify(request), JSON.stringify(options.parameters ?? {}), active.value, cancellation.value, batch)) as InferenceResult | BatchResult;
    } finally {
      active.busy = false; busy = false; cancellation.dispose();
      if (!cache) await active.dispose();
    }
  }
  return {
    info: JSON.parse(runtime.info()) as ModelInfo,
    createCache: async options => createCache(options),
    predict: (request, options = {}) => predict(request, options, false) as Promise<InferenceResult>,
    predictBatch: (requests, options = {}) => predict(requests, options, true) as Promise<BatchResult>,
    async dispose() {
      if (disposed) return;
      check(); disposed = true;
      for (const cache of [...caches]) await cache.dispose();
      runtime.dispose(); runtime.free(); disposed = true;
    },
  };
}
