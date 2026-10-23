import { InferenceError } from './types.js';
import type { Backend, BatchResult, CacheOptions, Controls, Family, InferenceResult, LoadOptions, Model, ModelInfo, OpaqueCache, PredictOptions, Predictor, Registry, Request, Selection } from './types.js';
import type { Command, Reply } from './protocol.js';
export * from './types.js';
export { probe } from './probe.js';

export function loadModel<M extends Family, B extends Backend<M>>(options: LoadOptions<M, B>, controls?: Controls): Promise<Model<Registry[M][B]>>;
export function loadModel(options: Selection, controls?: Controls): Promise<Predictor>;
export async function loadModel(options: Selection, controls: Controls = {}): Promise<Model<unknown>> {
  controls.signal?.throwIfAborted();
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  let sequence = 0, closed = false;
  const pending = new Map<number, { resolve(value: unknown): void; reject(error: unknown): void; controls: Controls; cleanup(): void }>();
  const rejectAll = (error: Error): void => {
    closed = true; worker.terminate();
    for (const call of pending.values()) { call.cleanup(); call.reject(error); }
    pending.clear();
  };
  worker.onmessage = ({ data }: MessageEvent<Reply>) => {
    const call = pending.get(data.id); if (!call) return;
    if ('progress' in data) { call.controls.onProgress?.(data.progress); return; }
    pending.delete(data.id); call.cleanup();
    if ('error' in data) call.reject(new InferenceError(data.error.code, data.error.message));
    else call.resolve(data.result);
  };
  worker.onerror = () => rejectAll(new InferenceError('WORKER_FAILED', 'Inference worker failed'));
  const call = (method: Command['method'], args: unknown, callbacks: Controls = {}): Promise<unknown> => {
    if (closed) return Promise.reject(new InferenceError('DISPOSED', 'Model is disposed'));
    try { callbacks.signal?.throwIfAborted(); } catch (error) { return Promise.reject(error); }
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      const cancel = (): void => worker.postMessage({ id, method: 'cancel' } satisfies Command);
      callbacks.signal?.addEventListener('abort', cancel, { once: true });
      pending.set(id, { resolve, reject, controls: callbacks, cleanup() { callbacks.signal?.removeEventListener('abort', cancel); } });
      try { worker.postMessage({ id, method, args } satisfies Command); }
      catch (error) { pending.get(id)?.cleanup(); pending.delete(id); reject(error); }
    });
  };
  let info: ModelInfo;
  try { info = await call('load', options, controls) as ModelInfo; }
  catch (error) { rejectAll(new InferenceError('LOAD_FAILED', 'Model loading failed')); throw error; }
  const cacheIds = new WeakMap<OpaqueCache, number>();
  async function predict(requests: Request | readonly Request[], options: PredictOptions<unknown>, batch: boolean): Promise<unknown> {
    const cacheId = options.cache ? cacheIds.get(options.cache) : undefined;
    if (options.cache && cacheId === undefined) throw new InferenceError('CACHE_OWNER', 'Cache belongs to another model');
    return call(batch ? 'predictBatch' : 'predict', { requests, parameters: options.parameters, cacheId }, { onProgress: controls.onProgress, ...options } as Controls);
  }
  return {
    info,
    async createCache(options: CacheOptions = {}) {
      const id = await call('createCache', options) as number;
      let cacheClosed = false;
      const cache: OpaqueCache = {
        async clear() { if(cacheClosed)throw new InferenceError('DISPOSED','Cache is disposed');await call('clearCache', { id }); },
        async dispose() { if(cacheClosed||closed)return;await call('disposeCache', { id });cacheClosed=true; },
      };
      cacheIds.set(cache, id); return cache;
    },
    predict: (request, options = {}) => predict(request, options, false) as Promise<InferenceResult>,
    predictBatch: (requests, options = {}) => predict(requests, options, true) as Promise<BatchResult>,
    async dispose() { if (!closed) { await call('dispose', {}); closed = true; worker.terminate(); } },
  };
}
