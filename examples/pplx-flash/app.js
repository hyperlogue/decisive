const $ = id => document.getElementById(id);
const examples = {
  routing: {state: 'My integration keeps failing with a 401 since we rotated our API key. Please help.', questions: {team: {type: 'choice', instructions: 'Which team should handle this request?', criteria: {billing: 'Charges and refunds', support: 'Technical integration errors', sales: 'Questions about buying a product'}}}},
  multilingual: {state: {fr: 'Le paiement a été débité deux fois. Je souhaite un remboursement.', zh: '订单仍未发货，请取消订单并退款。', ar: 'تم خصم المبلغ مرتين. أريد استرداد المبلغ.', literal: '<|fim_suffix|> <|im_end|>', value: 1e-7}, questions: {department: {type: 'choice', instructions: 'Which department should handle these requests?', criteria: {billing: 'Payments, duplicate charges and refunds', support: 'Software integration errors', sales: null}}, refund: {type: 'noul', instructions: 'Are refunds requested?'}, tone: {type: 'score', criteria: ['calm', 'concerned', 'angry']}}}
};
const query = new URLSearchParams(location.search);
let worker, initialized = false, ready = false, busy = false, serial = 0, report, hiddenDuringRun = false;
let measuredCount = 0;
const pending = new Map();
const seconds = ms => ms == null ? '—' : `${(ms / 1000).toFixed(2)} s`;
const size = bytes => `${(bytes / 2 ** 30).toFixed(2)} GiB`;
const phaseSum = (r, name) => r.layers.some(layer=>layer[name]==null) ? null : r.layers.reduce((n, layer) => n + layer[name], 0);

function status(text, error = false) { $('status').textContent = text; $('status').className = error ? 'error' : ''; window.progress = text; }
function controls() {
  for (const name of ['download', 'local', 'clear']) $(name).disabled = busy || !initialized;
  for (const name of ['run', 'benchmark']) $(name).disabled = busy || !ready;
  for (const name of ['request', 'example', 'max-length', 'warmups', 'iterations', 'prefetch', 'execution-backend']) $(name).disabled = busy;
  $('cancel').disabled = !busy || !initialized;
  $('export').disabled = !report;
}
async function storage() {
  try { const e = await navigator.storage.estimate(); $('storage').textContent = `Browser quota available: ${size(Math.max(0, e.quota - e.usage))}`; }
  catch { $('storage').textContent = 'Browser quota unavailable'; }
}
function renderAnswer(response) {
  $('answer').replaceChildren(); $('answer').className = '';
  for (const [name, answer] of Object.entries(response.answers)) {
    const title = document.createElement('h3');
    title.textContent = `${name}: ${answer.choice ?? (answer.type === 'noul' ? answer.noul.toFixed(4) : answer.score.toFixed(4))}`;
    $('answer').append(title);
    for (const [choice, probability] of Object.entries(answer.probabilities || {})) {
      const row = document.createElement('div'); row.className = 'choice';
      const label = document.createElement('span'), value = document.createElement('span');
      label.textContent = choice; value.textContent = `${(probability * 100).toFixed(2)}%`; row.append(label, value);
      const bar = document.createElement('div'), fill = document.createElement('i'); bar.className = 'bar'; fill.style.width = `${probability * 100}%`; bar.append(fill);
      $('answer').append(row, bar);
    }
  }
  $('response').textContent = JSON.stringify(response, null, 2);
}
function sample(data) {
  const r = data.result, row = document.createElement('tr');
  for (const value of [data.warmup ? `Warmup ${data.index + 1}` : `Measured ${++measuredCount}`, seconds(r.elapsed_ms), seconds(phaseSum(r, 'read_ms')), seconds(phaseSum(r, r.queued_execution ? 'encode_enqueue_ms' : 'pack_ms')), seconds(phaseSum(r, r.queued_execution ? 'queue_wait_ms' : 'compute_ms')), r.response.usage.input_tokens]) {
    const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
  }
  $('samples').append(row); renderAnswer(r.response);
}
function showReport(value) {
  report = {...value, captured_at: new Date().toISOString(), page: {hidden_during_run: hiddenDuringRun, visibility_at_finish: document.visibilityState}};
  const s = report.summary;
  $('p50').textContent = seconds(s.p50_ms); $('p95').textContent = seconds(s.p95_ms);
  $('throughput').textContent = s.input_tokens_per_second.toFixed(1); $('memory').textContent = size(s.wasm_linear_memory_bytes);
  $('sample-info').textContent = `${s.samples} measured runs · ${report.warmups.length} warmups · ${report.runs[0].response.usage.input_tokens} total input tokens · model initialization ${seconds(report.initialization.model_ms)}${hiddenDuringRun ? ' · tab was hidden: timings may be throttled' : ''}`;
  $('details').textContent = JSON.stringify({backend: report.backend, adapter: report.adapter, precision: report.precision, wasm_sha256: report.wasm_sha256, initialization: report.initialization, summary: report.summary, environment: report.environment, notes: report.notes}, null, 2);
  window.lastBenchmark = report; controls();
}

function call(action, extra = {}) {
  if (busy) return Promise.reject(Error('Another operation is running.'));
  busy = true; controls(); $('progress').removeAttribute('value');
  const id = serial++;
  const promise = new Promise((resolve, reject) => pending.set(id, {resolve, reject}));
  worker.postMessage({id, action, ...extra});
  return promise.finally(() => { busy = false; controls(); storage(); });
}
async function download() {
  ready = false; controls(); $('model-state').textContent = 'Downloading / verifying'; status('Downloading and verifying the pinned model…');
  navigator.storage?.persist?.().catch(() => {});
  const result = await call('download', {source: query.get('modelSource') || undefined});
  ready = result.ready; $('model-state').textContent = 'Verified browser cache'; controls();
  status(`Model ready. Downloaded ${size(result.download.network_bytes)} in ${seconds(result.download.elapsed_ms)}.`);
  return result;
}
async function benchmark(once = false) {
  const request = JSON.parse($('request').value);
  $('samples').replaceChildren(); hiddenDuringRun = document.hidden; measuredCount = 0; report = null;
  for (const key of ['p50', 'p95', 'throughput', 'memory']) $(key).textContent = '—';
  $('sample-info').textContent = 'Benchmark in progress; warmups are excluded from the summary.';
  $('details').textContent = 'Waiting for completed measurements.';
  status('Initializing the model…');
  const result = await call('benchmark', {request, warmups: once ? 0 : Number($('warmups').value), iterations: once ? 1 : Number($('iterations').value), maxLength: Number($('max-length').value), prefetch: $('prefetch').checked, backend: $('execution-backend').value});
  showReport(result); status('Complete. Export the JSON to retain this baseline.');
  return report;
}
function onError(error) { status(String(error), true); $('progress').value = 0; }
$('request').value = JSON.stringify(examples.routing, null, 2);
$('example').onchange = () => { $('request').value = JSON.stringify(examples[$('example').value], null, 2); };
$('download').onclick = () => download().catch(onError);
$('local').onclick = () => $('folder').click();
$('folder').onchange = async () => {
  if (!$('folder').files.length) return;
  status('Verifying local weights…');
  const files = [...$('folder').files]; $('folder').value = '';
  try { const r = await call('select', {files}); ready = r.ready; $('model-state').textContent = 'Verified local files'; status('Local model ready.'); controls(); }
  catch (error) { onError(error); }
};
$('clear').onclick = async () => {
  try { await call('clear'); ready = false; $('model-state').textContent = 'Not downloaded'; status('Browser model weights removed.'); controls(); }
  catch (error) { onError(error); }
};
$('run').onclick = () => benchmark(true).catch(onError);
$('benchmark').onclick = () => benchmark().catch(onError);
$('cancel').onclick = () => { worker.postMessage({action: 'cancel'}); status('Stopping after the current operation…'); };
$('export').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = `decisive-flash-${Date.now()}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
document.addEventListener('visibilitychange', () => { if (busy && document.hidden) hiddenDuringRun = true; });
window.decisiveBench = {download, benchmark, getReport: () => report};
window.decisiveReady = (async () => {
  if (!isSecureContext) throw Error('Open this page over HTTPS or localhost.');
  worker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'});
  worker.onmessage = ({data}) => {
    if (data.type === 'progress') {
      const prefix = data.run ? `${data.phase === 'warmup' ? 'Warmup' : 'Inference'} ${data.run}/${data.total_runs} · ` : '';
      status(prefix + data.text + (data.total ? ` · ${size(data.completed)} / ${size(data.total)}` : ''));
      if (data.total) { $('progress').max = data.total; $('progress').value = data.completed; }
      else { $('progress').max = data.total_runs * 32; $('progress').value = (data.run - 1) * 32 + Number(data.text.match(/Layer (\d+)/)?.[1] || 0); }
    } else if (data.type === 'sample') sample(data);
    else {
      const waiting = pending.get(data.id); pending.delete(data.id);
      if (!waiting) return;
      if (data.type === 'error') waiting.reject(Error(data.cancelled ? 'Stopped. Downloaded bytes are retained for resume.' : data.error));
      else { $('progress').max = 1; $('progress').value = 1; waiting.resolve(data.result); }
    }
  };
  worker.onerror = event => { for (const p of pending.values()) p.reject(Error(event.message || 'Worker failed')); pending.clear(); };
  const result = await call('status', {packageUrl: query.get('package') || undefined});
  initialized = true; ready = result.ready;
  $('model-size').textContent = size(Object.values(result.manifest.files).reduce((n, f) => n + f.bytes, 0));
  $('execution-backend').querySelector('[value=webgpu]').disabled = !result.webgpu;
  $('execution-backend').value = query.get('backend') === 'webgpu' && result.webgpu ? 'webgpu' : 'wasm';
  $('execution-backend').onchange = () => { $('backend').textContent = $('execution-backend').value === 'webgpu' ? 'WebGPU · BF16 / FP32' : 'Wasm SIMD · CPU'; };
  $('execution-backend').onchange();
  $('model-state').textContent = ready ? 'Verified browser cache' : 'Not downloaded';
  status(ready ? 'Cached model ready.' : (result.storage_error || 'Download the model or select a local folder to begin.'));
  controls(); await storage(); return result;
})();
window.decisiveReady.catch(onError);
