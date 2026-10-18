globalThis.PplxQuantizationProbe = (() => {
  const GROUP = 64;
  const BITS = [16, 8, 4];
  const VARIANTS = ['f32', 'f16'].flatMap(compute => BITS.map(bits => ({
    id: `${bits === 16 ? 'bf16' : `q${bits}`}-${compute}`, bits, compute, storage: bits === 16 ? 'bf16' : `q${bits}`,
    name: `${bits === 16 ? 'BF16' : `Q${bits}`} / ${compute.toUpperCase()}`,
  }))).concat({id: 'fp16-f16', bits: 16, compute: 'f16', storage: 'fp16', name: 'FP16 / F16'});
  const ROWS = [117, 229, 1024, 2048];
  const SHAPES = [
    {name: 'MLP gate/up', input: 2560, output: 9216, matrices: 64},
    {name: 'MLP down', input: 9216, output: 2560, matrices: 32},
  ];
  const WARMUPS = 3;
  const ROUNDS = 11;
  const float = new Float32Array(1);
  const word = new Uint32Array(float.buffer);
  const assert = (condition, message) => { if (!condition) throw Error(message); };
  const fromBF16 = bits => { word[0] = bits << 16; return float[0]; };
  const toBF16 = value => {
    float[0] = value;
    return ((word[0] + 0x7fff + ((word[0] >>> 16) & 1)) >>> 16) & 0xffff;
  };
  const roundEven = value => {
    const low = Math.floor(value), fraction = value - low;
    return fraction === 0.5 ? low + (low & 1) : Math.round(value);
  };
  const roundF16 = value => {
    if (!Number.isFinite(value) || value === 0) return value;
    const magnitude = Math.abs(value);
    if (magnitude >= 65520) return Math.sign(value) * Infinity;
    const step = 2 ** Math.max(-24, Math.floor(Math.log2(magnitude)) - 10);
    return Math.sign(value) * roundEven(magnitude / step) * step;
  };
  const fromF16 = bits => {
    const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >>> 10) & 31, mantissa = bits & 1023;
    return sign * (exponent === 31 ? (mantissa ? NaN : Infinity) : exponent === 0 ? mantissa * 2 ** -24 : (1 + mantissa / 1024) * 2 ** (exponent - 15));
  };
  const toF16 = value => {
    float[0] = roundF16(value);
    const bits = word[0], sign = (bits >>> 16) & 0x8000;
    const exponent = ((bits >>> 23) & 255) - 127, mantissa = bits & 0x7fffff;
    if (exponent === 128) return sign | (mantissa ? 0x7e00 : 0x7c00);
    if (float[0] === 0) return sign;
    return sign | (exponent >= -14 ? ((exponent + 15) << 10) | (mantissa >>> 13) : (mantissa | 0x800000) >>> (-exponent - 1));
  };
  function packF16(weights) {
    const data = new Uint16Array(weights.length);
    const conversion = {values: weights.length, rounded: 0, underflow_to_zero: 0, subnormal: 0, overflow: 0};
    for (let i = 0; i < weights.length; i++) {
      const value = fromBF16(weights[i]);
      assert(Number.isFinite(value), 'Non-finite weight');
      data[i] = toF16(value);
      const converted = fromF16(data[i]);
      if (!Number.isFinite(converted)) conversion.overflow++;
      if (converted !== value) conversion.rounded++;
      if (value !== 0 && converted === 0) conversion.underflow_to_zero++;
      if ((data[i] & 0x7c00) === 0 && (data[i] & 1023) !== 0) conversion.subnormal++;
    }
    assert(conversion.overflow === 0, `FP16 weight conversion overflows: ${conversion.overflow} values`);
    return {bits: 16, format: 'fp16', data, scales: new Float32Array(1), conversion};
  }
  function pack(weights, input, bits) {
    assert(input > 0 && input % GROUP === 0 && weights.length % input === 0, 'Invalid grouped matrix shape');
    assert(BITS.includes(bits), 'Unsupported precision');
    if (bits === 16) return {bits, data: weights, scales: new Float32Array(1)};
    const data = new Uint8Array(weights.length * bits / 8);
    const scales = new Float32Array(weights.length / GROUP);
    const values = new Float32Array(GROUP);
    const limit = (1 << (bits - 1)) - 1;
    for (let base = 0; base < weights.length; base += GROUP) {
      let maximum = 0;
      for (let i = 0; i < GROUP; i++) {
        const value = fromBF16(weights[base + i]);
        assert(Number.isFinite(value), 'Non-finite weight');
        values[i] = value;
        maximum = Math.max(maximum, Math.abs(value));
      }
      const scale = Math.fround(maximum / limit);
      scales[base / GROUP] = scale;
      for (let i = 0; i < GROUP; i++) {
        const q = scale === 0 ? 0 : Math.max(-limit, Math.min(limit, roundEven(Math.fround(values[i] / scale))));
        const at = base + i;
        if (bits === 8) data[at] = q & 255;
        else data[at >> 1] |= (q & 15) << ((at & 1) * 4);
      }
    }
    return {bits, data, scales};
  }
  function unpack(packed, index) {
    if (packed.format === 'fp16') return fromF16(packed.data[index]);
    if (packed.bits === 16) return fromBF16(packed.data[index]);
    let q = packed.bits === 8 ? (packed.data[index] << 24) >> 24 :
      (((packed.data[index >> 1] >>> ((index & 1) * 4)) & 15) << 28) >> 28;
    return Math.fround(q * packed.scales[index >> 6]);
  }
  function shader(source, bits, compute = 'f32', storage = bits === 16 ? 'bf16' : `q${bits}`) {
    let header = source.header;
    if (storage === 'fp16') {
      assert(bits === 16 && compute === 'f16', 'Native FP16 storage requires FP16 compute');
      for (const [from, to] of [
        ['w: array<u32>', 'w: array<f16>'],
        ['fn wf(i:u32)->f32 { return bitcast<f32>(w[i]); }', 'fn wf(i:u32)->f32 { return f32(w[i]); }'],
      ]) {
        assert(header.includes(from), 'Production weight storage changed; update the FP16 experiment');
        header = header.replace(from, to);
      }
      const pattern = /fn bf\(i:u32\)->f32 \{[^\n]+\}/;
      assert(pattern.test(header), 'Unsupported baseline weight decoder');
      header = header.replace(pattern, 'fn bf(i:u32)->f16 { return w[i]; }');
    }
    if (bits !== 16) {
      const perWord = 32 / bits, shift = 32 - bits;
      const replacement = `fn bf(i:u32)->f32 {
        let q=bitcast<i32>(w[i/${perWord}u] << (${shift}u-(i%${perWord}u)*${bits}u)) >> ${shift}u;
        return f32(q)*bitcast<f32>(aux[i/64u]);
      }`;
      const pattern = /fn bf\(i:u32\)->f32 \{[^\n]+\}/;
      assert(pattern.test(header), 'Unsupported baseline weight decoder');
      header = header.replace(pattern, replacement);
    }
    let gemm = source.gemm;
    if (compute === 'f16') {
      const replace = (from, to) => {
        assert(gemm.includes(from), 'Production GEMM changed; update the FP16 experiment');
        gemm = gemm.replaceAll(from, to);
      };
      replace('array<f32,1024>', 'array<f16,1024>');
      replace('at[j]=x[(g.y*32u+r)*k+first+c];', 'at[j]=f16(x[(g.y*32u+r)*k+first+c]);');
      replace('bt[j]=bf((g.x*32u+c)*k+first+r);', 'bt[j]=f16(bf((g.x*32u+c)*k+first+r));');
      replace('for(var j=0u;j<32u;j++) {', 'var t0=vec4<f16>(0.0h);var t1=t0;var t2=t0;var t3=t0;\n        for(var j=0u;j<32u;j++) {');
      replace('let v=vec4<f32>', 'let v=vec4<f16>');
      for (let i = 0; i < 4; i++) {
        replace(`s${i}=fma(vec4<f32>`, `t${i}=fma(vec4<f16>`);
        replace(`v,s${i});`, `v,t${i});`);
      }
      replace('        workgroupBarrier();\n    }', '        s0+=vec4<f32>(t0);s1+=vec4<f32>(t1);s2+=vec4<f32>(t2);s3+=vec4<f32>(t3);\n        workgroupBarrier();\n    }');
      header = `enable f16;\n${header}`;
    }
    return `${header}\n${gemm}`;
  }
  function random(seed) {
    let state = seed >>> 0;
    return () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; };
  }
  function weightsFor(input, output) {
    const next = random(137);
    const weights = new Uint16Array(input * output);
    const scale = 1 / Math.sqrt(input);
    for (let i = 0; i < weights.length; i++) {
      const value = (next() * 2 - 1) * scale * (i % 257 === 0 ? 8 : 1);
      weights[i] = toBF16(value);
    }
    return weights;
  }
  function inputsFor(rows, input) {
    const next = random(293);
    return Float32Array.from({length: rows * input}, () => next() * 2 - 1);
  }
  function median(values) {
    const sorted = [...values].sort((a, b) => a - b), n = sorted.length;
    return (sorted[(n - 1) >> 1] + sorted[n >> 1]) / 2;
  }
  function statistics(samples) {
    const center = median(samples);
    return {samples_ms: samples, median_ms: center, relative_mad: median(samples.map(x => Math.abs(x - center))) / center};
  }
  function speedup(baseline, candidate) {
    return {
      ratio: baseline.median_ms / candidate.median_ms,
      paired_wins: baseline.samples_ms.filter((v, i) => candidate.samples_ms[i] < v).length,
      rounds: baseline.samples_ms.length,
      noisy: Math.max(baseline.relative_mad, candidate.relative_mad) > 0.1,
    };
  }
  function summarize(cases) {
    const scopes = SHAPES.map(shape => {
      const rows = cases.filter(c => c.name === shape.name);
      const eligible = ['fp16-f16', 'bf16-f16', 'q8-f16', 'q8-f32', 'q4-f16', 'q4-f32'].filter(id => rows.length === ROWS.length && rows.every(c => {
        const variant = c.variants.find(v => v.id === id);
        const compute = variant?.speedup?.compute;
        const resident = variant?.speedup?.resident;
        return compute && resident && compute.ratio >= 1.1 && compute.rounds >= 7 && compute.paired_wins >= Math.ceil(compute.rounds * 0.85) &&
          !compute.noisy && !resident.noisy && resident.ratio >= 0.95 && variant.kernel_check.passed;
      }));
      return {name: shape.name, prefill_candidate: eligible[0] ?? null,
        full_model_quality_validated: false};
    });
    return {scopes, gate: 'Each tested length: >=10% faster resident GPU compute, >=85% paired wins over >=7 rounds, <=5% resident wall regression, <=10% relative MAD; kernel checks pass. Upload savings do not qualify.',
      interpretation: 'A prefill compute candidate only. Synthetic weights cannot establish model accuracy or full-request speed.'};
  }
  function compareOutput(actual, baseline) {
    let maximum = 0, squared = 0, reference = 0;
    assert(actual.length === baseline.length, 'Output shape differs');
    for (let i = 0; i < actual.length; i++) {
      assert(Number.isFinite(actual[i]), 'Non-finite GPU output');
      const delta = actual[i] - baseline[i];
      maximum = Math.max(maximum, Math.abs(delta));
      squared += delta * delta;
      reference += baseline[i] * baseline[i];
    }
    return {max_absolute_error: maximum, rmse: Math.sqrt(squared / actual.length), relative_l2: Math.sqrt(squared / Math.max(reference, 1e-30))};
  }
  function checkSamples(actual, input, packed, rows, k, n, compute) {
    let maximum = 0, normalized = 0;
    for (let i = 0; i < 32; i++) {
      const row = i === 31 ? rows - 1 : (i * 37) % rows;
      const col = i === 31 ? n - 1 : (i * 1009) % n;
      let expected = 0;
      if (compute === 'f16') {
        for (let first = 0; first < k; first += 32) {
          let partial = 0;
          for (let j = first; j < Math.min(k, first + 32); j++) {
            partial = roundF16(roundF16(input[row * k + j]) * roundF16(unpack(packed, col * k + j)) + partial);
          }
          expected = Math.fround(expected + partial);
        }
      } else {
        for (let j = 0; j < k; j++) expected += input[row * k + j] * unpack(packed, col * k + j);
      }
      const error = Math.abs(actual[row * n + col] - expected);
      maximum = Math.max(maximum, error);
      const tolerance = compute === 'f16' ? 1e-3 + 1e-3 * Math.abs(expected) : 1e-4 + 2e-5 * Math.abs(expected);
      normalized = Math.max(normalized, error / tolerance);
    }
    assert(Number.isFinite(normalized) && normalized <= 1, `Kernel disagrees with reconstructed-weight CPU reference: ${maximum}`);
    return {passed: true, samples: 32, max_absolute_error: maximum,
      tolerance: compute === 'f16' ? '1e-3 + 1e-3 * abs(reference)' : '1e-4 + 2e-5 * abs(reference)',
      reference: compute === 'f16' ? 'CPU F16 operands/FMA, 32-element partial sums, FP32 cross-tile sum' : 'CPU FP32 operands with reconstructed weights; not BF16 parity'};
  }

  async function run(source, provenance, progress = () => {}) {
    const started = performance.now();
    const report = {schema_version: 1, experiment: 'pplx-weight-only-quantization-v1', completed: false,
      benchmark_version: 3, primary_metric: 'resident GPU prefill compute time',
      started_at: new Date().toISOString(), source: provenance, cases: [],
      scheme: {group_size: GROUP, scale: 'fp32 max-absolute / (2^(bits-1)-1)', rounding: 'nearest ties-to-even', zero_point: 0, layout: 'output-major rows, input groups, signed codes packed in u32',
        arithmetic: {f32: 'FP32 operands/FMA/accumulation', f16: 'F16 operands/FMA, F16 partial sums over 32 elements, FP32 cross-tile accumulation'}, activation_storage: 'FP32 for all variants; F16 conversion happens inside the tile load'},
      methodology: {warmups: WARMUPS, rounds: ROUNDS, order: 'rotating and reversing all weight/compute variants within each round',
        resident: 'encode + submit + synchronized completion; preuploaded weights',
        streamed: 'fresh mapped weight/scale buffers + CPU copy + encode + submit + synchronized completion',
        excluded: 'checkpoint I/O, CPU quantization, JS-to-Wasm copying, other layers, readout and shader compilation',
        data: 'deterministic synthetic BF16 weights with outliers; actual MLP matrix dimensions',
        quality: 'kernel parity against reconstructed weights plus synthetic drift; no full-model accuracy claim',
        preconversion: 'BF16 converted once on CPU to native array<f16> storage before timing; same 2 bytes per weight; conversion counts cover synthetic weights only'},
    };
    let device;
    try {
      assert(globalThis.isSecureContext && navigator.gpu, 'WebGPU unavailable. Open this file in Chrome.');
      const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
      assert(adapter, 'No WebGPU adapter');
      const info = adapter.info || {};
      report.adapter = {vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
        is_fallback_adapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null};
      report.features = [...adapter.features].sort();
      report.wgsl_features = [...(navigator.gpu.wgslLanguageFeatures || [])].sort();
      report.software_gpu = report.adapter.is_fallback_adapter === true || /swiftshader|llvmpipe|software/i.test(JSON.stringify(report.adapter));
      const timestamp = adapter.features.has('timestamp-query');
      const half = adapter.features.has('shader-f16');
      const activeVariants = VARIANTS.filter(v => v.compute === 'f32' || half);
      device = await adapter.requestDevice({requiredFeatures: [...(timestamp ? ['timestamp-query'] : []), ...(half ? ['shader-f16'] : [])]});
      report.gpu_timestamps = timestamp;
      let deviceError;
      device.addEventListener('uncapturederror', event => { deviceError = event.error.message; });
      device.lost.then(info => { if (info.reason !== 'destroyed') deviceError = info.message || 'GPU device lost'; });
      const layout = device.createBindGroupLayout({entries: Array.from({length: 5}, (_, binding) => ({
        binding, visibility: GPUShaderStage.COMPUTE, buffer: {type: binding === 4 ? 'uniform' : binding === 3 ? 'storage' : 'read-only-storage'},
      }))});
      const pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [layout]});
      const pipelines = new Map();
      for (const variant of activeVariants) {
        progress({message: `Compiling ${variant.name} shader…`});
        const module = device.createShaderModule({code: shader(source, variant.bits, variant.compute, variant.storage)});
        const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
        assert(!errors.length, errors.map(e => e.message).join('\n'));
        pipelines.set(variant.id, await device.createComputePipelineAsync({layout: pipelineLayout, compute: {module, entryPoint: 'main'}}));
      }
      const create = (data, usage) => {
        const buffer = device.createBuffer({size: data.byteLength, usage, mappedAtCreation: true});
        new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        buffer.unmap();
        return buffer;
      };
      const query = timestamp ? device.createQuerySet({type: 'timestamp', count: 2}) : null;
      const resolved = timestamp ? device.createBuffer({size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC}) : null;
      const queryRead = timestamp ? device.createBuffer({size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}) : null;
      for (const shape of SHAPES) {
        progress({message: `Preparing ${shape.name} weights (${shape.output} × ${shape.input})…`});
        const weights = weightsFor(shape.input, shape.output);
        const packedVariants = [...new Set(activeVariants.map(v => v.storage))].map(storage => {
          const bits = activeVariants.find(v => v.storage === storage).bits;
          const start = performance.now();
          const packed = storage === 'fp16' ? packF16(weights) : pack(weights, shape.input, bits);
          return {bits, storage, packed, packing_ms: performance.now() - start,
            weight: create(packed.data, GPUBufferUsage.STORAGE), scales: create(packed.scales, GPUBufferUsage.STORAGE)};
        });
        const variants = activeVariants.map(v => ({...packedVariants.find(p => p.storage === v.storage), ...v}));
        try {
          for (const rows of ROWS) {
            const {input: k, output: n} = shape;
            const input = inputsFor(rows, k);
            const xb = create(input, GPUBufferUsage.STORAGE);
            const params = new Uint32Array(16); params.set([rows, n, k]);
            const uniform = create(params, GPUBufferUsage.UNIFORM);
            const output = device.createBuffer({size: rows * n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
            const staging = device.createBuffer({size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
            const bindings = (weight, scales) => device.createBindGroup({layout, entries: [xb, weight, scales, output, uniform].map((buffer, binding) => ({binding, resource: {buffer}}))});
            const groups = new Map(packedVariants.map(v => [v.storage, bindings(v.weight, v.scales)]));
            const record = {name: shape.name, rows, input: k, output: n, flops: 2 * rows * k * n, variants: []};
            const sample = async (variant, mode) => {
              const start = performance.now();
              let weight, scales;
              try {
                if (mode === 'streamed') {
                  weight = create(variant.packed.data, GPUBufferUsage.STORAGE);
                  scales = create(variant.packed.scales, GPUBufferUsage.STORAGE);
                }
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginComputePass(timestamp ? {timestampWrites: {querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1}} : {});
                pass.setPipeline(pipelines.get(variant.id));
                pass.setBindGroup(0, weight ? bindings(weight, scales) : groups.get(variant.storage));
                pass.dispatchWorkgroups(Math.ceil(n / 32), Math.ceil(rows / 32));
                pass.end();
                if (timestamp) { encoder.resolveQuerySet(query, 0, 2, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, queryRead, 0, 16); }
                device.queue.submit([encoder.finish()]);
                await device.queue.onSubmittedWorkDone();
                const wall = performance.now() - start;
                assert(!deviceError, deviceError);
                let gpu = null;
                if (timestamp) {
                  await queryRead.mapAsync(GPUMapMode.READ);
                  const times = new BigUint64Array(queryRead.getMappedRange());
                  gpu = Number(times[1] - times[0]) / 1e6;
                  queryRead.unmap();
                  assert(Number.isFinite(gpu) && gpu >= 0, 'Invalid GPU timestamps');
                }
                return {wall, gpu};
              } finally { weight?.destroy(); scales?.destroy(); }
            };
            try {
              let baseline, halfBaseline;
              for (const variant of variants) {
                progress({message: `Checking ${shape.name}, ${rows} tokens, ${variant.name}…`});
                await sample(variant, 'resident');
                const encoder = device.createCommandEncoder();
                encoder.copyBufferToBuffer(output, 0, staging, 0, output.size);
                device.queue.submit([encoder.finish()]);
                await staging.mapAsync(GPUMapMode.READ);
                const actual = new Float32Array(staging.getMappedRange()).slice();
                staging.unmap();
                const kernel = checkSamples(actual, input, variant.packed, rows, k, n, variant.compute);
                if (variant.id === 'bf16-f32') baseline = actual;
                if (variant.id === 'bf16-f16') halfBaseline = actual;
                record.variants.push({id: variant.id, bits: variant.bits, compute: variant.compute, storage: variant.storage, name: variant.name,
                  storage_bytes: variant.packed.data.byteLength + (variant.bits === 16 ? 0 : variant.packed.scales.byteLength),
                  packing_ms: variant.packing_ms, kernel_check: kernel, weight_conversion: variant.packed.conversion,
                  preconversion_drift: variant.storage === 'fp16' ? compareOutput(actual, halfBaseline) : undefined,
                  synthetic_drift_from_bf16: compareOutput(actual, baseline), resident: [], streamed: [], gpu_ms: {resident: [], streamed: []}});
              }
              for (const mode of ['resident', 'streamed']) {
                progress({message: `Timing ${shape.name}, ${rows} tokens: ${mode} (${ROUNDS} rounds)…`});
                for (let round = -WARMUPS; round < ROUNDS; round++) {
                  const offset = (round + WARMUPS) % variants.length;
                  const order = [...variants.slice(offset), ...variants.slice(0, offset)];
                  if (round % 2 === 0) order.reverse();
                  for (const variant of order) {
                    const result = await sample(variant, mode);
                    if (round >= 0) {
                      const entry = record.variants.find(v => v.id === variant.id);
                      entry[mode].push(result.wall);
                      if (timestamp) entry.gpu_ms[mode].push(result.gpu);
                    }
                  }
                }
              }
              for (const entry of record.variants) {
                entry.resident = statistics(entry.resident);
                entry.streamed = statistics(entry.streamed);
                entry.gpu_ms = timestamp ? {resident: statistics(entry.gpu_ms.resident), streamed: statistics(entry.gpu_ms.streamed)} : null;
                entry.effective_tflops = timestamp ? record.flops / (entry.gpu_ms.resident.median_ms * 1e9) : null;
                if (entry.id !== 'bf16-f32') entry.speedup = {
                  compute: timestamp ? speedup(record.variants[0].gpu_ms.resident, entry.gpu_ms.resident) : null,
                  resident: speedup(record.variants[0].resident, entry.resident), streamed: speedup(record.variants[0].streamed, entry.streamed)};
              }
              record.q4_vs_q8 = Object.fromEntries(['f32', 'f16'].filter(compute => compute === 'f32' || half).map(compute => {
                const q8 = record.variants.find(v => v.id === `q8-${compute}`);
                const q4 = record.variants.find(v => v.id === `q4-${compute}`);
                return [compute, {compute: timestamp ? speedup(q8.gpu_ms.resident, q4.gpu_ms.resident) : null,
                  resident: speedup(q8.resident, q4.resident), streamed: speedup(q8.streamed, q4.streamed)}];
              }));
              if (half) {
                const before = record.variants.find(v => v.id === 'bf16-f16');
                const after = record.variants.find(v => v.id === 'fp16-f16');
                record.fp16_preconversion = {compute: timestamp ? speedup(before.gpu_ms.resident, after.gpu_ms.resident) : null,
                  resident: speedup(before.resident, after.resident), streamed: speedup(before.streamed, after.streamed)};
              }
              report.cases.push(record);
              progress({message: `Finished ${shape.name}, ${rows} tokens.`, record});
            } finally { xb.destroy(); uniform.destroy(); output.destroy(); staging.destroy(); }
          }
        } finally { for (const v of packedVariants) { v.weight.destroy(); v.scales.destroy(); } }
      }
      report.completed = true;
      report.summary = summarize(report.cases);
    } catch (error) {
      report.error = String(error);
    } finally {
      report.elapsed_ms = performance.now() - started;
      device?.destroy();
    }
    return report;
  }
  return {pack, packF16, unpack, toBF16, fromBF16, toF16, fromF16, roundEven, roundF16, shader, summarize, run,
    weightsFor, inputsFor, statistics, compareOutput, checkSamples};
})();
