# Decisive

An inference engine for decision models in browsers and edge devices.

## Shared inference interface

`decisive::inference::Model<B>` wraps a concrete Rust backend. `Backend` has
associated `Parameters` and `Cache` types; Decima, Laya, Kev, Jev-style, and
native Pplx implement it. The existing Decima and Laya WebGPU models also
implement it. Requests contain `state` and `questions`; model-specific
prediction parameters are separate from the input and cancellation handle.

```rust
let mut model = decisive::inference::Model::new(backend);
let mut cache = model.create_cache(Default::default())?;
let cancellation = decisive::inference::Cancellation::default();
let result = model.predict(&request, None, Some(&mut cache), &cancellation).await?;
let results = model.predict_batch(&requests, None, Some(&mut cache), &cancellation).await?;
```

A batch borrows one cache for the entire call and returns ordered per-request
outcomes. Different requests can reuse the same cache. Separate model instances
cannot exchange cache objects. Omitting a cache uses an ephemeral cache with no
cross-request retention. A failed request invalidates its reusable cache state.
The byte budget covers reusable entries, not model weights or inference scratch memory.
The cache representation is opaque; serialization is reserved for a future
version and is not currently implemented.

## Browser SDK

The handwritten SDK is strict TypeScript in `browser/`. Build it with
`npm ci --prefix browser --ignore-scripts` and `npm run build --prefix browser`.
Deploy its JavaScript under `sdk/` alongside a wasm-bindgen package under `pkg/`.
The SDK creates a Worker by default; applications already running inside a Worker
can use `loadModelInWorker` from `sdk/local.js` with an initialized WASM module.

```typescript
import {loadModel} from './sdk/index.js';

const model = await loadModel({
  model: 'decima',
  backend: 'webgpu',
  source: {files},
  parameters: {long: true},
});
const cache = await model.createCache({maxBytes: 64 * 1024 * 1024});
try {
  const result = await model.predict(request, {cache, signal});
  const batch = await model.predictBatch(requests, {cache, signal});
} finally {
  await model.dispose();
}
```

The source is a filename-to-`File`, `Blob`, or `Uint8Array` map. Prefer files for
large checkpoints; Rust reads the streamed Pplx weights and Decima GPU vocabulary
in ranges. Downloads, OPFS permissions, and download persistence remain separate
from inference. The demo provides verified files through its TypeScript storage
module.

`loadModel` infers prediction parameters from the model/backend. Decima supports
`long`, `window`, `overlap`, `lang`, and `minConfidence`; Laya has `stride` instead
of `overlap`; Pplx has `maxSequenceTokens` (default 1,024). Kev and Jev-style
currently accept an empty parameter object. Rust rejects unknown parameters.
Call-time parameters override the configured defaults. A runtime-selected model
can be used as a `Predictor`, whose defaults are already bound and whose callers
do not need model-specific configuration.

`model.info` reports the backend, cache capabilities, and tuning information.
Decima retains choice encodings, Pplx retains embedding rows, and Kev/Jev-style
retain prefix execution state. These are different capabilities; requiring
prefix caching on an unsupported model returns `UNSUPPORTED_CACHE`. Laya's new
interface currently retains scratch allocations without enabling its legacy
token/state caches. Model disposal releases all caches created through the SDK.
Concurrent use of the same cache returns `CACHE_BUSY`; another concurrent call
on the same model returns `MODEL_BUSY`. Batches currently execute sequentially.

Pplx layer reads, embedding caching, prefetching, GPU uploads, queue backpressure,
and failure cleanup run in Rust. Loading accepts `tuning: {mode: 'auto' | 'off',
budgetMs: 2000}`. The budget is cooperative between kernel-selection stages;
an already-started stage finishes. Candidates retain the existing correctness
gates. Tuning results stay with the loaded model and are not persisted.
Cancellation is cooperative: Pplx checks between layers and embedding reads;
other backends finish their active prediction before acknowledging cancellation.

Available backends depend on Cargo features. The demo compiles Decima and Pplx and exposes
WebGPU only in its interface. Full builds also expose Laya, Kev, Jev-style, and WASM execution;
Kev/Jev-style have no WebGPU kernels. Their current resident weight conversion
has a conservative browser memory check, so a large checkpoint may require the
native backend. The demo does not expose a CPU fallback.

Browser loading rejects unsupported combinations before reading checkpoint files.
Errors have a stable `code` and an explanatory `message` with the selected model,
backend, or format and a suggested next step. `UNSUPPORTED_MODEL`,
`UNSUPPORTED_BACKEND`, and `UNSUPPORTED_FORMAT` identify unsupported selections;
`MODEL_NOT_COMPILED`, `BACKEND_NOT_COMPILED`, and `FORMAT_NOT_COMPILED` distinguish
missing Cargo features. Pplx Q4 requires `shader-f16`; a device without it receives
`MISSING_GPU_FEATURE`. `GPU_UNAVAILABLE` identifies WebGPU initialization failures.
`RESOURCE_LIMIT` covers GPU limits and the Kev/Jev-style loader's conservative
3 GiB resident-conversion budget. This budget is a preflight estimate, not a
guarantee that every browser can allocate that much memory. These errors do not
trigger automatic format conversion or backend fallback.

Run `npm test --prefix browser` for SDK lifecycle and type-contract tests, and
the Rust library tests for the native interface. Real browser checks and the
Pages build are described in [site/README.md](site/README.md).
