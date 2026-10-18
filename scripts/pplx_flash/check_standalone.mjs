import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {performance} from 'node:perf_hooks';
import {clearTimeout, setTimeout} from 'node:timers';
import {pathToFileURL} from 'node:url';
import {TextDecoder, TextEncoder} from 'node:util';
import vm from 'node:vm';

export async function checkStandalone(path) {
  const html = await readFile(path, 'utf8');
  const script = id => {
    const match = html.match(new RegExp(`<script id="${id}"[^>]*>([\\s\\S]*?)<\\/script>`));
    assert.ok(match, `Missing ${id}`);
    return match[1];
  };
  const source = script('runner');
  const bytes = Buffer.from(script('wasm-bytes'), 'base64');
  const build = JSON.parse(script('build-data'));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), build.wasm_sha256);
  const messages = [];
  const self = {postMessage: message => messages.push(message), isSecureContext: true};
  const context = vm.createContext({
    self, navigator: {}, TextEncoder, TextDecoder, console,
    setTimeout, clearTimeout, performance, bytes,
  });
  new vm.Script(source, {filename: 'packaged-worker.js'}).runInContext(context);
  assert.equal(messages[0]?.type, 'ready', 'Worker must announce startup before receiving Wasm');
  assert.equal(messages[0].webgpu, false);
  await self.onmessage({data: bytes});
  assert.equal(messages.at(-1).type, 'error');
  assert.equal(messages.at(-1).stage, 'capabilities');
  assert.match(messages.at(-1).error, /WebGPU is unavailable/);
  await vm.runInContext('wasm_bindgen({module_or_path: bytes})', context);
  const simd = JSON.parse(vm.runInContext('wasm_bindgen.wasmKernelChecks()', context));
  assert.equal(simd['pplx-flash'].max_absolute_error, 0);
  return {classic_worker_startup: true, missing_gpu_reported: true, wasm_integrity: true, simd};
}

if (typeof process !== 'undefined' && process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.argv[2]) throw Error('Usage: node scripts/pplx_flash/check_standalone.mjs packaged.html');
  console.log(JSON.stringify(await checkStandalone(process.argv[2]), null, 2));
}
