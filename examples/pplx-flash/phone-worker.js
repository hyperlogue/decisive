import init, * as api from './pkg/decisive.js';
import {loadFlash} from './runtime.js';
import {cachePackedModel} from './packed-cache.js';

let engine, requests, metadata, busy = false, controller, initialized = false;
const emit = data => self.postMessage(data);
const progress = data => emit({type: 'progress', ...data});
const json = async path => {
  const response = await fetch(new URL(path, import.meta.url));
  if (!response.ok) throw Error(`${path}: HTTP ${response.status}`);
  return response.json();
};

function exactInput(base, model, target) {
  const words = Array(100).fill(base.state).join(' ').split(/\s+/);
  const chunks = state => JSON.parse(model.prepare(JSON.stringify({...base, state}), 2048));
  const length = state => chunks(state).reduce((sum, chunk) => sum + chunk.ids.length, 0);
  let lo = 0, hi = words.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (length(words.slice(0, mid).join(' ')) <= target) lo = mid; else hi = mid - 1;
  }
  let state = words.slice(0, lo).join(' '), tokens = length(state);
  while (tokens < target) {
    const next = state + ' x', count = length(next);
    if (count <= tokens || count > target) throw Error(`Cannot construct ${target} tokens`);
    state = next; tokens = count;
  }
  const plan = chunks(state);
  if (plan.length !== 1 || plan[0].ids.length !== target) throw Error('Input shape mismatch');
  return {...base, state};
}

self.onmessage = async ({data}) => {
  if (data.type === 'cancel') { controller?.abort(); return; }
  if (busy) return;
  busy = true;
  try {
    if (data.type === 'prepare') {
      if (engine) throw Error('Already prepared');
      controller = new AbortController();
      if (!navigator.gpu) throw Error('WebGPU is unavailable. Open this page in Safari over HTTPS.');
      const start = performance.now();
      const manifest = await json('download.json');
      metadata = await json('metadata.json');
      if (!initialized) {
        progress({message: 'Loading WebGPU runtime…'});
        const response = await fetch(new URL('./pkg/decisive_bg.wasm', import.meta.url));
        if (!response.ok) throw Error('Cannot load Wasm runtime');
        const bytes = await response.arrayBuffer();
        const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
        if (hash !== metadata.build.wasm_sha256) throw Error('Wasm build checksum mismatch');
        await init({module_or_path: bytes}); initialized = true;
      }
      const cached = await cachePackedModel(manifest, api.PplxFlashHasher, progress, controller.signal);
      progress({message: 'Preparing model and shaders…'});
      engine = await loadFlash(api, cached.files, message => progress({message}), 'webgpu', {mode: 'q4-e4', resident: data.resident === true, profile: false});
      const base = await json('request.json');
      requests = Object.fromEntries([256, 512, 1024].map(n => [n, exactInput(base, engine.model, n)]));
      emit({type: 'prepared', metadata, preparation: {elapsed_ms: performance.now() - start,
        network_bytes: cached.network_bytes, model_bytes: cached.total_bytes, startup: engine.startup}, requests});
    } else if (data.type === 'run') {
      if (!engine || !requests?.[data.tokens]) throw Error('Prepare the model first');
      const start = performance.now();
      const result = await engine.predict(requests[data.tokens], data.tokens, true);
      const wall = performance.now() - start;
      if (result.response.usage.input_tokens !== data.tokens || result.response.usage.sequences !== 1 || result.layers.length !== 32) throw Error('Forward-pass shape mismatch');
      if (!result.queued_execution) throw Error('Expected queued execution');
      if (result.configuration.resident ? result.decoder_weight_bytes_read !== 0 || result.double_buffered : !result.double_buffered) throw Error('Weight residency mismatch');
      emit({type: 'result', tokens: data.tokens, warmup: data.warmup === true, request_wall_ms: wall, result});
    } else throw Error('Unknown command');
  } catch (error) {
    if (data.type === 'prepare') { engine?.dispose(); engine = null; requests = null; }
    emit({type: 'error', message: String(error), prepared: Boolean(engine && requests)});
  } finally { busy = false; }
};
