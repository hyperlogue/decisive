import assert from 'node:assert/strict';
import {readFile, mkdtemp, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {performance} from 'node:perf_hooks';
import {setTimeout, clearTimeout} from 'node:timers';
import {pathToFileURL} from 'node:url';
import vm from 'node:vm';

export async function checkQuantization(path) {
  const html = await readFile(path, 'utf8');
  const source = html.match(/<script id="probe">([\s\S]*?)<\/script>/)[1];
  const messages = [];
  const self = {postMessage: message => messages.push(message)};
  const context = vm.createContext({self, navigator: {}, isSecureContext: true, performance, setTimeout, clearTimeout, console});
  new vm.Script(source).runInContext(context);
  assert.equal(messages[0].type, 'ready');
  const probe = context.PplxQuantizationProbe;
  assert.deepEqual([0.5, 1.5, 2.5, -0.5, -1.5, -2.5].map(probe.roundEven), [0, 2, 2, 0, -2, -2]);
  assert.equal(probe.roundF16(1 + 2 ** -11), 1);
  assert.equal(probe.roundF16(1 + 3 * 2 ** -11), 1 + 2 ** -9);
  assert.equal(probe.roundF16(2 ** -25), 0);
  assert.equal(probe.roundF16(65504), 65504);
  assert.equal(probe.roundF16(65520), Infinity);
  for (let bits = 0; bits < 65536; bits++) {
    const value = probe.fromF16(bits);
    if (Number.isNaN(value)) assert.ok(Number.isNaN(probe.fromF16(probe.toF16(value))));
    else assert.equal(probe.toF16(value), bits);
    const bf = probe.fromBF16(bits);
    if (!Number.isNaN(bf)) assert.equal(probe.fromF16(probe.toF16(bf)), probe.roundF16(bf));
  }
  for (const [value, bits] of [[1 + 2 ** -11, 0x3c00], [1 + 3 * 2 ** -11, 0x3c02],
    [2 ** -25, 0], [3 * 2 ** -25, 2], [65504, 0x7bff], [65520, 0x7c00], [-0, 0x8000]]) {
    assert.equal(probe.toF16(value), bits);
  }
  const tiny = Uint16Array.from([0, 1, 2 ** -14, 2 ** -24, 2 ** -25, 3 * 2 ** -25, -(2 ** -24)].map(probe.toBF16));
  const half = probe.packF16(tiny);
  assert.equal(half.data.byteLength, tiny.byteLength);
  assert.equal(half.conversion.overflow, 0);
  assert.equal(half.conversion.underflow_to_zero, 1);
  assert.equal(half.conversion.subnormal, 3);
  assert.equal(half.conversion.rounded, 2);
  for (let i = 0; i < tiny.length; i++) assert.equal(probe.unpack(half, i), probe.roundF16(probe.fromBF16(tiny[i])));
  assert.throws(() => probe.packF16(Uint16Array.of(probe.toBF16(65536))), /overflows/);
  assert.throws(() => probe.packF16(Uint16Array.of(0x7f80)), /Non-finite/);
  const weights = Uint16Array.from({length: 3 * 128}, (_, i) => probe.toBF16(
    i < 64 ? 0 : i < 128 ? -0.25 : ((i * 97) % 251 - 125) / 17));
  for (const bits of [8, 4]) {
    const packed = probe.pack(weights, 128, bits);
    assert.equal(packed.data.byteLength, weights.length * bits / 8);
    assert.equal(packed.scales.byteLength, weights.length / 64 * 4);
    assert.equal(packed.scales[0], 0);
    const words = new Uint32Array(packed.data.buffer);
    for (let i = 0; i < weights.length; i++) {
      const shift = 32 - bits;
      const q = (words[Math.floor(i / (32 / bits))] << (shift - i % (32 / bits) * bits)) >> shift;
      const reconstructed = probe.unpack(packed, i);
      assert.equal(reconstructed, Math.fround(q * packed.scales[Math.floor(i / 64)]));
      assert.ok(Math.abs(reconstructed - probe.fromBF16(weights[i])) <= packed.scales[Math.floor(i / 64)] / 2 + 1e-6);
    }
  }
  assert.throws(() => probe.pack(weights, 63, 8), /shape/);
  assert.throws(() => probe.pack(weights, 128, 3), /precision/);
  const invalid = weights.slice(); invalid[0] = 0x7f80;
  assert.throws(() => probe.pack(invalid, 128, 8), /Non-finite/);

  const cases = ['MLP gate/up', 'MLP down'].flatMap(name => [117, 229, 1024, 2048].map(rows => ({name, rows,
    variants: ['bf16-f16', 'q8-f32', 'q4-f32'].map(id => ({id, kernel_check: {passed: true}, speedup: {
      compute: {ratio: 1.2, paired_wins: 7, rounds: 7, noisy: false}, streamed: {ratio: 2, paired_wins: 7, rounds: 7, noisy: false}, resident: {ratio: 1.02, noisy: false},
    }}))})));
  assert.ok(probe.summarize(cases).scopes.every(s => s.prefill_candidate === 'bf16-f16'));
  cases[0].variants[0].speedup.resident.ratio = 0.9;
  assert.equal(probe.summarize(cases).scopes[0].prefill_candidate, 'q8-f32');
  cases[0].variants[1].speedup.compute.noisy = true;
  assert.equal(probe.summarize(cases).scopes[0].prefill_candidate, 'q4-f32');
  cases[0].variants[2].speedup.compute.ratio = 1.01;
  assert.equal(probe.summarize(cases).scopes[0].prefill_candidate, null);
  assert.ok(probe.summarize(cases.slice(0, 2)).scopes.every(s => s.prefill_candidate === null));

  await self.onmessage();
  assert.equal(messages.at(-1).type, 'result');
  assert.equal(messages.at(-1).report.completed, false);
  assert.match(messages.at(-1).report.error, /WebGPU unavailable/);
  const shaders = await mkdtemp(join(tmpdir(), 'decisive-quant-shaders-'));
  for (const compute of ['f32', 'f16']) for (const bits of [16, 8, 4]) {
    await writeFile(join(shaders, `q${bits}-${compute}.wgsl`), vm.runInContext(`PplxQuantizationProbe.shader(KERNEL_SOURCE, ${bits}, '${compute}')`, context));
  }
  const nativeHalf = vm.runInContext("PplxQuantizationProbe.shader(KERNEL_SOURCE, 16, 'f16', 'fp16')", context);
  assert.match(nativeHalf, /var<storage, read> w: array<f16>/);
  assert.match(nativeHalf, /fn bf\(i:u32\)->f16 \{ return w\[i\]; \}/);
  await writeFile(join(shaders, 'fp16-f16.wgsl'), nativeHalf);
  return {packing_and_reconstruction: true, fp16_conversion: true, classic_worker_startup: true, speed_gate: true,
    missing_gpu_reported: true, shaders_for_validation: shaders};
}

if (typeof process !== 'undefined' && process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.argv[2]) throw Error('Usage: node scripts/pplx_flash/check_quantization.mjs packaged.html');
  console.log(JSON.stringify(await checkQuantization(process.argv[2]), null, 2));
}
