import { loadModelInWorker } from './local.js';
import type { WasmModule } from './local.js';
import { failure, InferenceError } from './types.js';
import type { CacheOptions, Model, OpaqueCache, PredictOptions, Request, Selection } from './types.js';
import type { Command, Reply } from './protocol.js';

const host = globalThis as unknown as { onmessage: ((event: MessageEvent<Command>) => void) | null; postMessage(value: Reply): void };
let model: Model<unknown> | undefined, nextCache = 0;
const caches = new Map<number, OpaqueCache>(), controllers = new Map<number, AbortController>();
host.onmessage = async ({ data }) => {
  if (data.method === 'cancel') { controllers.get(data.id)?.abort(); return; }
  const controller = new AbortController(); controllers.set(data.id, controller);
  const controls = { signal: controller.signal, onProgress: (progress: import('./types.js').Progress) => host.postMessage({ id: data.id, progress }) };
  try {
    let result: unknown;
    if (data.method === 'load') {
      if (model) throw new InferenceError('MODEL_BUSY', 'Model is already loaded');
      const url = new URL('../pkg/decisive.js', import.meta.url).href;
      const api = await import(url) as WasmModule;
      await api.default();
      model = await loadModelInWorker(api, data.args as Selection, controls);
      result = model.info;
    } else {
      if (!model) throw new InferenceError('DISPOSED', 'Load a model first');
      const args = data.args as { id?: number; cacheId?: number; requests?: Request | Request[]; parameters?: unknown };
      if (data.method === 'createCache') { result = ++nextCache; caches.set(nextCache, await model.createCache(data.args as CacheOptions)); }
      else if (data.method === 'clearCache' || data.method === 'disposeCache') {
        const cache = caches.get(args.id ?? -1);
        if (!cache) throw new InferenceError('DISPOSED', 'Cache is disposed');
        if (data.method === 'clearCache') await cache.clear();
        else { await cache.dispose(); caches.delete(args.id!); }
        result = null;
      } else if (data.method === 'dispose') { await model.dispose(); model = undefined; caches.clear(); result = null; }
      else {
        const cache = args.cacheId === undefined ? undefined : caches.get(args.cacheId);
        if (args.cacheId !== undefined && !cache) throw new InferenceError('DISPOSED', 'Cache is disposed');
        const options: PredictOptions<unknown> = { ...controls, parameters: args.parameters, ...(cache ? { cache } : {}) };
        result = data.method === 'predictBatch' ? await model.predictBatch(args.requests as Request[], options) : await model.predict(args.requests as Request, options);
      }
    }
    host.postMessage({ id: data.id, result });
  } catch (error) { host.postMessage({ id: data.id, error: failure(error) }); }
  finally { controllers.delete(data.id); }
};
