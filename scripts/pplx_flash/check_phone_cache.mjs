import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../../examples/pplx-flash/packed-cache.js', import.meta.url), 'utf8');
const payload = new TextEncoder().encode('packed checkpoint test bytes');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const manifest = {revision: 'test', files: {'weights.bin': {bytes: payload.length, sha256: digest(payload)}}};
class Hasher {
  hash = createHash('sha256');
  update(bytes) { this.hash.update(bytes); }
  finish() { return this.hash.digest('hex'); }
  free() {}
}
async function scenario(initial = new Uint8Array(), behavior = 'range') {
  const entries = new Map(), requests = [];
  let clock = 0, locks = 0;
  function entry(bytes = new Uint8Array()) {
    const e = {bytes, modified: ++clock, locked: false};
    e.getFile = async () => ({size: e.bytes.length, lastModified: e.modified, text: async () => new TextDecoder().decode(e.bytes)});
    e.createSyncAccessHandle = async () => {
      assert.equal(e.locked, false); e.locked = true; locks++;
      return {
        getSize: () => e.bytes.length,
        truncate(n) { const next = new Uint8Array(n); next.set(e.bytes.subarray(0, n)); e.bytes = next; e.modified = ++clock; },
        read(out, {at}) { const bytes = e.bytes.subarray(at, at + out.length); out.set(bytes); return bytes.length; },
        write(bytes, {at}) {
          const n = Math.min(bytes.length, 7), next = new Uint8Array(Math.max(e.bytes.length, at + n));
          next.set(e.bytes); next.set(bytes.subarray(0, n), at); e.bytes = next; e.modified = ++clock; return n;
        },
        flush() {}, close() { assert.equal(e.locked, true); e.locked = false; locks--; },
      };
    };
    return e;
  }
  entries.set('weights.bin', entry(initial));
  const dir = {getFileHandle: async (name, options) => {
    if (!entries.has(name)) {
      if (!options?.create) throw Error('Missing');
      entries.set(name, entry());
    }
    return entries.get(name);
  }};
  const fetch = async (url, options) => {
    requests.push({url: String(url), range: options.headers.Range});
    const offset = behavior === 'ignore-range' ? 0 : Number(options.headers.Range?.match(/bytes=(\d+)/)?.[1] || 0);
    const body = behavior === 'corrupt' ? new Uint8Array(payload.length).fill(42) : payload.subarray(offset);
    return new Response(body, {status: offset ? 206 : 200, headers: offset ? {'Content-Range': `bytes ${offset}-${payload.length - 1}/${payload.length}`} : {}});
  };
  const context = vm.createContext({navigator: {storage: {getDirectory: async () => ({getDirectoryHandle: async () => dir})}}, fetch,
    URL, TextEncoder, Uint8Array, performance, setTimeout, Error});
  const module = new vm.SourceTextModule(source, {context, initializeImportMeta: meta => { meta.url = 'https://test.invalid/probe/packed-cache.js'; }});
  await module.link(() => { throw Error('Unexpected dependency'); }); await module.evaluate();
  const run = () => module.namespace.cachePackedModel(manifest, Hasher, () => {}, new AbortController().signal);
  return {run, entries, requests, locks: () => locks};
}

for (const [initial, behavior] of [[new Uint8Array(), 'range'], [payload.slice(0, 9), 'range'], [payload.slice(0, 9), 'ignore-range'], [payload, 'range']]) {
  const s = await scenario(initial, behavior), result = await s.run();
  assert.equal(result.total_bytes, payload.length);
  assert.deepEqual(s.entries.get('weights.bin').bytes, payload);
  assert.equal(s.locks(), 0);
  if (initial.length === 9) assert.equal(s.requests[0].range, `bytes=9-${payload.length - 1}`);
  const count = s.requests.length, again = await s.run();
  assert.equal(again.network_bytes, 0); assert.equal(s.requests.length, count);
}
const corrupt = await scenario(new Uint8Array(), 'corrupt');
await assert.rejects(corrupt.run, /Checksum or size mismatch/);
assert.equal(corrupt.locks(), 0); assert.equal(corrupt.entries.get('weights.bin').bytes.length, 0);
console.log('Phone OPFS cache: fresh download, resume, ignored Range, verified reuse and corrupt-file retry passed');
