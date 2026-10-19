const $ = id => document.getElementById(id);
const buttons = [...document.querySelectorAll('[data-tokens]')];
const worker = new Worker(new URL('./phone-worker.js', import.meta.url), {type: 'module'});
const report = {schema_version: 1, experiment: 'pplx-phone-forward-v1', started_at: new Date().toISOString(),
  browser: navigator.userAgent, weight_source: 'OPFS local files', runs: [],
  methodology: {tokens: [256, 512, 1024], sequences: 1, mode: 'q4-e4', double_buffering: true,
    primary: 'Request wall time, including local weight reads, uploads, tokenization, all 32 layers and decision readout',
    excluded: 'Download, checksum verification, model and shader initialization',
    warmups: 'Explicitly marked by the user; first run at each length is identified separately',
    median: 'Measured foreground runs only; cold and warm runs identified in raw results', gpu_timestamps: false}};
let ready = false, busy = false, runActive = false, hidden = false, wakeLock;
const setBusy = value => {
  busy = value;
  $('prepare').disabled = value || ready;
  $('residency').disabled = value || ready;
  $('cancel').disabled = !value || runActive;
  for (const button of buttons) button.disabled = value || !ready;
};
async function holdScreen() {
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {}
}
async function releaseScreen() {
  try { await wakeLock?.release(); } catch {}
  wakeLock = null;
}
document.addEventListener('visibilitychange', () => { if (runActive && document.hidden) hidden = true; });
function render() {
  $('rows').replaceChildren();
  for (const tokens of [256, 512, 1024]) {
    const all = report.runs.filter(r => r.tokens === tokens);
    const measured = all.filter(r => !r.warmup && !r.page_hidden);
    const times = measured.map(r => r.request_wall_ms).sort((a, b) => a - b);
    const median = times.length ? (times[Math.floor((times.length - 1) / 2)] + times[Math.floor(times.length / 2)]) / 2 : null;
    const row = document.createElement('tr');
    for (const value of [tokens, measured.length, all.length ? `${(all.at(-1).request_wall_ms / 1000).toFixed(3)} s${all.at(-1).warmup ? ' warmup' : all.at(-1).page_hidden ? ' hidden' : ''}` : '—', median ? `${(median / 1000).toFixed(3)} s` : '—', median ? (tokens * 1000 / median).toFixed(1) : '—']) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    $('rows').append(row);
  }
  $('details').textContent = JSON.stringify(report, null, 2);
  $('download').disabled = $('copy').disabled = report.runs.length === 0;
}
worker.onmessage = ({data}) => {
  if (data.type === 'progress') {
    $('status').textContent = data.message + (data.total ? ` · ${(100 * data.completed / data.total).toFixed(1)}%` : '');
    if (data.total) $('progress').value = data.completed / data.total;
  } else if (data.type === 'prepared') {
    ready = true; report.metadata = data.metadata; report.preparation = data.preparation; report.requests = data.requests;
    const resident = data.preparation.startup.weight_residency === 'all-decoder-layers';
    report.methodology.double_buffering = !resident;
    report.methodology.resident = resident;
    report.methodology.primary = resident
      ? 'Request wall time with all decoder weights on GPU, including tokenization, embedding row reads, activation upload, all 32 layers and CPU decision readout'
      : 'Request wall time, including local weight reads, uploads, tokenization, all 32 layers and decision readout';
    report.methodology.excluded = 'Download, checksum verification, model and shader initialization, one-time resident decoder uploads';
    $('progress').value = 1; $('status').textContent = 'Ready. Choose 256, 512 or 1024 tokens.';
    setBusy(false); releaseScreen(); render();
  } else if (data.type === 'result') {
    const first = !report.runs.some(r => r.tokens === data.tokens);
    report.runs.push({...data, type: undefined, page_hidden: hidden, first_run_at_length: first, completed_at: new Date().toISOString()});
    runActive = false; setBusy(false); releaseScreen();
    $('status').textContent = `${data.tokens} tokens: ${(data.request_wall_ms / 1000).toFixed(3)} s.${hidden ? ' Page was hidden; excluded from median.' : data.warmup ? ' Warmup; excluded from median.' : ' Saved in the report.'}`;
    $('warmup').checked = false; render();
    window.lastRun = report.runs.at(-1); window.benchmarkReport = report;
    fetch(new URL('./results', import.meta.url), {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(report)}).catch(() => {});
  } else if (data.type === 'error') {
    ready = data.prepared; runActive = false; setBusy(false); releaseScreen();
    $('status').textContent = data.message;
  }
};
worker.onerror = event => {
  ready = false; runActive = false; setBusy(false); releaseScreen();
  $('prepare').disabled = true;
  $('status').textContent = `${event.message || 'Worker stopped'}. Reload to retry; verified downloads remain cached.`;
};
$('prepare').onclick = async () => {
  if (busy) return;
  setBusy(true); $('status').textContent = 'Preparing…'; await holdScreen();
  worker.postMessage({type: 'prepare', resident: $('residency').value === 'resident'});
};
$('cancel').onclick = () => worker.postMessage({type: 'cancel'});
for (const button of buttons) button.onclick = async () => {
  if (busy || !ready) return;
  runActive = true; hidden = document.hidden; setBusy(true);
  const tokens = Number(button.dataset.tokens);
  $('status').textContent = `Running ${tokens} tokens…`; await holdScreen();
  worker.postMessage({type: 'run', tokens, warmup: $('warmup').checked});
};
$('download').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = `pplx-phone-${Date.now()}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
};
$('copy').onclick = async () => {
  try { await navigator.clipboard.writeText(JSON.stringify(report, null, 2)); $('export-status').textContent = 'Report copied.'; }
  catch { $('export-status').textContent = 'Copy unavailable. Use Save JSON report.'; }
};
render();
if (!isSecureContext || !navigator.gpu) {
  $('prepare').disabled = true;
  $('status').textContent = 'WebGPU is unavailable. Open this page in an up-to-date Safari using HTTPS.';
}
