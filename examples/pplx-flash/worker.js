import {loadFlash} from './runtime.js';

let api, wasm, build, manifest, wasmHash, directory, controller, active = false;
let selectedFiles, engine, lastDownload = null, weightSource = null;
const emit = data => self.postMessage(data);
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const hex = bytes => Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, '0')).join('');

async function initialize(packageUrl) {
  if (api) return;
  manifest = await (await fetch(new URL('./model.json', import.meta.url))).json();
  const started = performance.now();
  const base = new URL(packageUrl || './pkg/', import.meta.url);
  const response = await fetch(new URL('decisive_bg.wasm', base));
  if (!response.ok) throw Error('Build the browser bundle first: scripts/pplx_flash/build_browser.sh');
  const bytes = await response.arrayBuffer();
  wasmHash = hex(await crypto.subtle.digest('SHA-256', bytes));
  build = await (await fetch(new URL('build.json', base))).json();
  if (wasmHash !== build.wasm_sha256) throw Error('Wasm binary does not match build.json');
  const bindings = await import(new URL('decisive.js', base));
  wasm = await bindings.default({module_or_path: bytes});
  api = bindings;
  const runtimeBuild = await fetch(new URL('./browser-build.json', import.meta.url));
  if (runtimeBuild.ok) build.browser_runtime = await runtimeBuild.json();
  build.module_init_ms = performance.now() - started;
}

async function cacheDirectory() {
  if (!navigator.storage?.getDirectory) throw Error('Browser storage requires HTTPS or localhost and OPFS support. You can also select a local model folder.');
  directory ??= await (await navigator.storage.getDirectory()).getDirectoryHandle(`decisive-pplx-${manifest.revision}`, {create: true});
  return directory;
}

async function receipt() {
  try { return JSON.parse(await (await (await cacheDirectory()).getFileHandle('verified.json')).getFile().then(f => f.text())); }
  catch { return {files: {}}; }
}

async function saveReceipt(value) {
  const file = await (await cacheDirectory()).getFileHandle('verified.json', {create: true});
  const stream = await file.createWritable();
  await stream.write(JSON.stringify(value));
  await stream.close();
}

async function cachedFiles() {
  const checked = await receipt(), files = {};
  for (const [name, expected] of Object.entries(manifest.files)) {
    if (checked.files[name]?.sha256 !== expected.sha256) return null;
    try {
      const file = await (await (await cacheDirectory()).getFileHandle(name)).getFile();
      if (file.size !== expected.bytes || file.lastModified !== checked.files[name].modified) return null;
      files[name] = file;
    } catch { return null; }
  }
  return files;
}

async function hashFile(file, hasher, signal, progress) {
  for (let start = 0; start < file.size; start += 4 * 1024 * 1024) {
    signal.throwIfAborted();
    const bytes = new Uint8Array(await file.slice(start, start + 4 * 1024 * 1024).arrayBuffer());
    hasher.update(bytes);
    progress?.(start + bytes.length);
    await tick();
  }
}

async function download(source, signal) {
  engine?.dispose(); engine = null; selectedFiles = null;
  const dir = await cacheDirectory(), checked = await receipt();
  const total = Object.values(manifest.files).reduce((n, f) => n + f.bytes, 0);
  const base = source || `https://huggingface.co/${manifest.repository}/resolve/${manifest.revision}`;
  let completed = 0, networkBytes = 0;
  const start = performance.now();
  for (const [name, expected] of Object.entries(manifest.files)) {
    signal.throwIfAborted();
    const fileHandle = await dir.getFileHandle(name, {create: true});
    let file = await fileHandle.getFile();
    if (checked.files[name]?.sha256 === expected.sha256 && file.size === expected.bytes && file.lastModified === checked.files[name].modified) {
      completed += file.size;
      emit({type: 'progress', phase: 'download', text: `Verified cache: ${name}`, completed, total});
      continue;
    }
    delete checked.files[name];
    if (file.size > expected.bytes) {
      const handle = await fileHandle.createSyncAccessHandle();
      try { handle.truncate(0); } finally { handle.close(); }
      file = await fileHandle.getFile();
    }
    let hasher = new api.PplxFlashHasher(), consumed = false;
    let offset = file.size;
    try {
      await hashFile(file, hasher, signal, bytes => emit({type: 'progress', phase: 'verify', text: `Checking partial download: ${name}`, completed: completed + bytes, total}));
      if (offset < expected.bytes) {
        const headers = offset ? {Range: `bytes=${offset}-${expected.bytes - 1}`} : {};
        const response = await fetch(`${base.replace(/\/$/, '')}/${name}`, {headers, signal, credentials: 'omit'});
        if (!response.ok || !response.body) throw Error(`Download failed for ${name}: HTTP ${response.status}`);
        if (offset && response.status === 200) {
          offset = 0; hasher.free(); hasher = new api.PplxFlashHasher();
        } else if (offset && response.status !== 206) throw Error(`Server cannot resume ${name}`);
        const handle = await fileHandle.createSyncAccessHandle();
        let lastProgress = 0, lastFlush = offset;
        try {
          if (offset === 0) handle.truncate(0);
          const reader = response.body.getReader();
          try {
            while (true) {
              signal.throwIfAborted();
              const {value, done} = await reader.read();
              if (done) break;
              if (offset + value.length > expected.bytes) throw Error(`Unexpected size: ${name}`);
              let written = 0;
              while (written < value.length) {
                const n = handle.write(value.subarray(written), {at: offset + written});
                if (!n) throw Error('Browser storage write failed');
                written += n;
              }
              hasher.update(value); offset += value.length; networkBytes += value.length;
              if (offset - lastFlush >= 64 * 1024 * 1024) { handle.flush(); lastFlush = offset; }
              if (performance.now() - lastProgress > 100) {
                emit({type: 'progress', phase: 'download', text: name, completed: completed + offset, total, network_bytes: networkBytes});
                lastProgress = performance.now();
              }
            }
          } finally { await reader.cancel().catch(() => {}); }
        } finally { handle.flush(); handle.close(); }
      }
      const digest = hasher.finish(); consumed = true;
      if (offset !== expected.bytes || digest !== expected.sha256) {
        await dir.removeEntry(name);
        throw Error(`Checksum or size mismatch: ${name}. Retry to download this file again.`);
      }
      file = await fileHandle.getFile();
      checked.files[name] = {sha256: digest, modified: file.lastModified};
      await saveReceipt(checked);
      completed += expected.bytes;
      emit({type: 'progress', phase: 'download', text: `Verified: ${name}`, completed, total});
    } finally { if (!consumed) hasher.free(); }
  }
  selectedFiles = await cachedFiles();
  if (!selectedFiles) throw Error('Browser cache could not be reopened after download');
  weightSource = 'browser-cache';
  lastDownload = {elapsed_ms: performance.now() - start, network_bytes: networkBytes, total_bytes: total};
  return {ready: true, source: 'browser-cache', download: lastDownload};
}

async function selectFiles(files, signal) {
  const byName = Object.fromEntries(files.map(file => [file.name, file]));
  const total = Object.values(manifest.files).reduce((n, f) => n + f.bytes, 0);
  let completed = 0;
  for (const [name, expected] of Object.entries(manifest.files)) {
    const file = byName[name];
    if (!file || file.size !== expected.bytes) throw Error(`Missing or incorrect size: ${name}. Select the folder created by download.sh.`);
    const hasher = new api.PplxFlashHasher();
    let consumed = false;
    try {
      await hashFile(file, hasher, signal, bytes => emit({type: 'progress', phase: 'verify', text: name, completed: completed + bytes, total}));
      const digest = hasher.finish(); consumed = true;
      if (digest !== expected.sha256) throw Error(`Checksum mismatch: ${name}`);
    } finally { if (!consumed) hasher.free(); }
    completed += file.size;
  }
  engine?.dispose(); engine = null; selectedFiles = byName; weightSource = 'selected-files';
  return {ready: true, source: 'selected-files'};
}

function summarize(runs) {
  const latencies = runs.map(r => r.elapsed_ms).sort((a, b) => a - b);
  const percentile = p => latencies[Math.max(0, Math.ceil(p * latencies.length) - 1)];
  const sum = (rows, field) => rows.reduce((n, r) => n + r[field], 0);
  const elapsed = sum(runs, 'elapsed_ms');
  const other = runs.reduce((n, r) => n + r.elapsed_ms - r.prepare_ms - r.embedding_ms - r.readout_ms - sum(r.layers, 'read_ms') - (r.queued_execution ? sum(r.layers, 'encode_enqueue_ms') + sum(r.layers, 'queue_wait_ms') : sum(r.layers, 'pack_ms') + sum(r.layers, 'compute_ms')), 0);
  const queued = runs.some(r=>r.queued_execution);
  return {samples: runs.length, percentile_method: 'nearest-rank', min_ms: latencies[0], p50_ms: percentile(.5), p95_ms: percentile(.95), max_ms: latencies.at(-1), mean_ms: elapsed / runs.length,
    stddev_ms: Math.sqrt(latencies.reduce((n, x) => n + (x - elapsed / runs.length) ** 2, 0) / runs.length),
    requests_per_second: runs.length * 1000 / elapsed,
    input_tokens_per_second: runs.reduce((n, r) => n + r.response.usage.input_tokens, 0) * 1000 / elapsed,
    mean_pack_ms: queued ? null : runs.reduce((n, r) => n + sum(r.layers, 'pack_ms'), 0) / runs.length,
    mean_compute_ms: queued ? null : runs.reduce((n, r) => n + sum(r.layers, 'compute_ms'), 0) / runs.length,
    mean_encode_enqueue_ms: queued ? runs.reduce((n,r)=>n+sum(r.layers,'encode_enqueue_ms'),0)/runs.length : null,
    mean_queue_wait_ms: queued ? runs.reduce((n,r)=>n+sum(r.layers,'queue_wait_ms'),0)/runs.length : null,
    mean_weight_read_wait_and_upload_ms: runs.reduce((n, r) => n + sum(r.layers, 'read_ms'), 0) / runs.length,
    mean_prepare_ms: sum(runs, 'prepare_ms') / runs.length, mean_embedding_ms: sum(runs, 'embedding_ms') / runs.length, mean_readout_ms: sum(runs, 'readout_ms') / runs.length,
    mean_scheduling_and_other_ms: other / runs.length,
    wasm_linear_memory_bytes: Math.max(...runs.map(r => r.wasm_linear_memory_bytes))};
}

async function benchmark(message, signal) {
  const {request, warmups, iterations, maxLength, prefetch, backend = "wasm"} = message;
  if (![warmups, iterations, maxLength].every(Number.isSafeInteger) || warmups < 0 || warmups > 10 || iterations < 1 || iterations > 100 || maxLength < 1 || maxLength > 2048) throw Error('Use 0–10 warmups, 1–100 runs, and a token limit between 1 and 2,048.');
  selectedFiles ??= await cachedFiles();
  if (!selectedFiles) throw Error('Download the model or select a local folder first.');
  engine?.dispose(); engine = null;
  const loading = performance.now();
  let currentRun = 0;
  engine = await loadFlash(api, selectedFiles, text => {
    signal.throwIfAborted();
    emit({type: 'progress', phase: currentRun < warmups ? 'warmup' : 'inference', text, run: currentRun + 1, total_runs: warmups + iterations});
  }, backend);
  const modelInit = performance.now() - loading;
  const runs = [], warmupRuns = [];
  try {
    for (currentRun = 0; currentRun < warmups + iterations; currentRun++) {
      signal.throwIfAborted();
      const result = await engine.predict(request, maxLength, prefetch);
      result.wasm_linear_memory_bytes = wasm.memory.buffer.byteLength;
      (currentRun < warmups ? warmupRuns : runs).push(result);
      emit({type: 'sample', warmup: currentRun < warmups, index: currentRun, result});
    }
  } finally { engine.dispose(); engine = null; }
  return {schema_version: 1, model: manifest.repository, revision: manifest.revision,
    backend: backend === 'webgpu' ? 'webgpu' : build.features.includes('pplx-wasm-fma') ? 'wasm-relaxed-simd' : 'wasm-simd128', adapter: runs[0].adapter, precision: 'bf16-storage-f32-compute',
    concurrency: 1, compute_threads: backend === 'webgpu' ? null : 1, io_workers: prefetch ? 1 : 0, weight_source: weightSource, wasm_sha256: wasmHash, build,
    environment: {user_agent: navigator.userAgent, hardware_concurrency: navigator.hardwareConcurrency, device_memory_gib_hint: navigator.deviceMemory ?? null, cross_origin_isolated: crossOriginIsolated},
    initialization: {model_ms: modelInit, wasm_module_ms: build.module_init_ms, wasm_instance_reused: true, model_reinitialized: true},
    input: {request, max_length: maxLength, prefetch}, warmups: warmupRuns, runs, summary: summarize(runs), download: lastDownload,
    notes: ['Inference includes local weight reads, Wasm uploads, packing, and compute. Download, SHA-256 verification, and initialization are excluded.', 'Queued WebGPU reports CPU encode_enqueue_ms and queue_wait_ms, with pack_ms and compute_ms unavailable. submission_to_collection_ms includes GPU queueing and delayed collection; it is not GPU compute time. read_ms is host weight wait and upload time and can overlap GPU execution.', 'Wasm linear memory is not process RSS and excludes JavaScript and worker buffers.', 'Token throughput measures input processing, not generation. Small samples do not establish stable tail latency.', backend === 'webgpu' ? 'Decoder runs on WebGPU; tokenization and final decision readout run on Wasm CPU. Layer compute includes command encoding and completion waits; no GPU timestamps. Prefetch uploads BF16 weights into two isolated banks; physical copy/compute overlap depends on the browser and driver.' : 'Wasm CPU execution; no GPU timestamps.']};
}

self.onmessage = async ({data}) => {
  if (data.action === 'cancel') { controller?.abort(); return; }
  if (active) { emit({type: 'error', id: data.id, error: 'Another operation is running.'}); return; }
  active = true; controller = new AbortController();
  try {
    await initialize(data.packageUrl);
    let result;
    if (data.action === 'status') {
      let cached = false, storageError = null;
      try { selectedFiles = await cachedFiles(); cached = !!selectedFiles; weightSource = cached ? 'browser-cache' : null; } catch (e) { storageError = String(e); }
      result = {ready: cached, source: cached ? 'browser-cache' : null, storage_error: storageError, manifest, build, wasm_sha256: wasmHash, webgpu: !!api.PplxFlashWebGpu && !!navigator.gpu};
    } else if (data.action === 'download') result = await download(data.source, controller.signal);
    else if (data.action === 'select') result = await selectFiles(data.files, controller.signal);
    else if (data.action === 'benchmark') result = await benchmark(data, controller.signal);
    else if (data.action === 'clear') {
      engine?.dispose(); engine = null; selectedFiles = null;
      await (await navigator.storage.getDirectory()).removeEntry(`decisive-pplx-${manifest.revision}`, {recursive: true});
      directory = null; lastDownload = null; weightSource = null; result = {ready: false};
    } else throw Error('Unknown command');
    emit({type: 'done', id: data.id, result});
  } catch (error) { emit({type: 'error', id: data.id, cancelled: controller.signal.aborted, error: String(error)}); }
  finally { active = false; }
};
