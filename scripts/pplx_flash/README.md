# Pplx Flash tools

Run commands from the repository root. This directory contains release tooling,
CPU evaluations, and browser experiments. The application now uses the
[TypeScript SDK](../../browser/) with Rust orchestration; the experimental pages
still use [runtime.js](../../examples/pplx-flash/runtime.js). Their results do
not validate the SDK scheduler. See [site/README.md](../../site/README.md) and
`browser/tests/q4.mjs` for the current application and scheduler checks.

## Release and evaluation tools

| Files | Purpose |
| --- | --- |
| `download.sh` | Fetch the pinned BF16 checkpoint and verify its hashes. |
| `prepare_hf_q4.py`, `hf_q4_model_card.md`, `hf_q4_upload.md` | Prepare and verify the packed Q4 release; see the [release instructions](hf_q4_upload.md). |
| `serve_phone.py`, `check_hf_q4_browser.mjs` | Serve packed checkpoints with byte ranges and smoke-test Q4 inference through the legacy JavaScript runtime. The smoke test requires Playwright, Chromium, and `shader-f16`. |
| `prepare_quality.py`, `cpu_bench.py` | Prepare a frozen corpus, run native BF16/Q8/Q4 evaluations, and summarize quality and performance. `cpu_bench.py --help` includes reproduction commands. |
| `reference.py` | Independent FP32 parity oracle using the publisher's protocol and Transformers layers. Requires the pinned publisher source, PyTorch, Transformers, safetensors, and model files. |
| `bench.sh` | Native and optional legacy browser reference-fixture parity. |

The CPU benchmark commands are referenced by checked-in results in `benches/`.
Keep their input/output formats reproducible when changing them.

## Browser experiments and legacy comparisons

These tools support retained experiments. They are not all interchangeable, and
removing a workflow also requires reviewing its pages under `examples/browser/`,
JavaScript under `examples/pplx-flash/`, and any results under `benches/`.

| Files | Purpose |
| --- | --- |
| `build_browser.sh` | Build the legacy Pplx demo and WASM package. Set `DECISIVE_PPLX_CHECKS=1` for kernel checks and classic worker bindings used by standalone packages. Application builds use `scripts/build_decima_wasm.sh` and `site/build.py`. |
| `bench_webgpu.sh`, `verify_webgpu.py`, `package_webgpu.py`, `check_standalone.mjs` | GPU kernel checks, full-model reference parity, and a standalone diagnostic page. |
| `serve_checkpoint.py`, `analyze_checkpoint.py`, `check_checkpoint_runtime.mjs`, `check_double_buffer_runtime.mjs`, `check_embedding_cache.mjs`, `check_phone_cache.mjs` | Checkpoint comparisons and regression checks for legacy loading, streaming, and caches. The new SDK still compares against this runtime. |
| `package_quantization.py`, `check_quantization.mjs`, `check_dp4a*.mjs`, `package_subgroup_matrix.py`, `check_subgroup_matrix.mjs` | Isolated weight-quantization, packed integer dot-product, and subgroup matrix probes. |
| `package_synthetic.py`, `check_synthetic*.mjs`, `analyze_synthetic.py`, `export_synthetic_trace.py` | Full-shape synthetic model comparisons, queued runs, reports, and traces. These do not measure checkpoint task quality. |
| `package_phone.py` | Combine synthetic and matrix probes into a phone bundle. Distinct from `serve_phone.py`, which serves real packed checkpoints. |
| `serve_quantization.py`, `run_safari_queue.py`, `sample_gpu.py` | Collect experiment reports; automate Safari on macOS; sample Apple GPU counters. Safari automation takes a JSON file containing the collector `url` and final report `output` path. |
| `build_opportunities.py`, `serve_opportunities.py` | Regenerate the opportunity probe's embedded shaders from Rust sources and collect its reports. |
| `analyze_quality.py`, `render_quality.py`, `check_quantized_rows.mjs` | Analyze browser quality reports and check WASM row quantization. `cpu_bench.py` also imports the shared quality metrics. |
| `int8_bench.py`, `quant_probe.py` | Earlier CPU INT8 and Python weight-reconstruction experiments. The legacy `int8` mode retains floating-point heads and differs from `cpu_bench.py`'s `q8`; the Python probe is not the packed Rust quantizer. |

## Local checks

Python tests use the standard library. One test starts a loopback HTTP server.
The following Node checks use mocks and do not require a GPU or model download:

```sh
python3 -m unittest discover -s scripts/pplx_flash -p 'test_*.py'
node scripts/pplx_flash/check_checkpoint_runtime.mjs
node scripts/pplx_flash/check_double_buffer_runtime.mjs
node scripts/pplx_flash/check_embedding_cache.mjs
node --experimental-vm-modules scripts/pplx_flash/check_phone_cache.mjs
```

Checks for packaged experiments take the generated HTML file as their argument:

```sh
python3 scripts/pplx_flash/package_quantization.py artifacts/web/quantization.html
node scripts/pplx_flash/check_quantization.mjs artifacts/web/quantization.html
python3 scripts/pplx_flash/package_quantization.py --dp4a artifacts/web/dp4a.html
node scripts/pplx_flash/check_dp4a.mjs artifacts/web/dp4a.html
python3 scripts/pplx_flash/package_quantization.py --suite artifacts/web/int8-suite.html
node scripts/pplx_flash/check_dp4a_suite.mjs artifacts/web/int8-suite.html
python3 scripts/pplx_flash/package_subgroup_matrix.py artifacts/web/subgroup.html
node scripts/pplx_flash/check_subgroup_matrix.mjs artifacts/web/subgroup.html
```

Packagers refuse to overwrite existing outputs. `check_synthetic.mjs` and
`check_synthetic_queue.mjs` likewise accept HTML generated by
`package_synthetic.py` (use `--queue` for the latter). `check_standalone.mjs`
accepts the output of `package_webgpu.py` and also executes WASM SIMD checks.
`check_quantized_rows.mjs` takes a WASM package directory and an optional JSON
file of golden rows. Package checks do not replace real GPU inference checks.

Generated bundles, benchmark reports, downloaded checkpoints, and `__pycache__`
are ignored by Git. Keep them out of this source directory.
