---
license: apache-2.0
base_model: perplexity-ai/pplx-decider-flash-v1.2
base_model_relation: quantized
tags:
  - decisive
  - webgpu
  - decision-model
  - quantized
  - q4
---

# Pplx Decider Flash v1.2 — Decisive Q4-E4

Packed Q4 weights for running Pplx Decider Flash v1.2 with the
[Decisive browser runtime](https://github.com/hyperlogue/dicisive).
This is a format conversion of the upstream checkpoint, with no additional training.

The browser runtime files total **{{TOTAL_GB}} GB ({{TOTAL_GIB}} GiB;
{{TOTAL_BYTES}} bytes)**, including tokenizer, configuration, packed index,
license and notice. This is download/storage size, not peak memory usage.
The model card and release metadata add a small amount to a complete repository download.

## Source and attribution

- Source: [perplexity-ai/pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2), revision `{{SOURCE_REVISION}}`.
- Upstream base: Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, as recorded in the unchanged upstream `NOTICE`.
- Exporter/runtime: Decisive commit `{{RUNTIME_REVISION}}`.
- The upstream `LICENSE` and `NOTICE` are preserved. Changes here are quantization,
  packing, and release documentation. Configuration and tokenizer files are unchanged.

## Format

This is Decisive's custom `q4-w4a8-e4-r4-g64` binary format. It is **not GGUF,
GPTQ, AWQ, or a Transformers-loadable safetensors checkpoint**. The original
BF16 `config.json` describes the source architecture; the actual storage format
is specified by `packed-q4-e4.json`.

| Component | Storage |
| --- | --- |
| Decoder matrices | Signed Q4, groups of 64, one FP32 scale per group |
| Token embeddings | Same Q4 layout; 1,440 bytes per 2,560-value row |
| Decision readout | Same Q4 layout, 255 × 2,560 values |
| Other decoder tensors and final norm | Original BF16 bytes |
| Projection activations | Quantized to 8 bits by the runtime |

Each group stores 32 bytes of codes followed by one little-endian FP32 scale
(36 bytes total, or 4.5 effective bits per matrix weight including scales).
The exporter uses symmetric maximum-absolute scaling, ties-to-even rounding,
and integer values from −7 to 7, stored with a bias of 8. The first value occupies
the low nibble of each byte. No calibration data is used.

There are 32 layer files, one embedding file, one readout file, and one norm
file. The index contains tensor shapes, byte ranges, and per-file SHA-256 hashes.
Its `revision` identifies the **upstream source**, not this repository's release
commit. Consumers should additionally pin this repository's own commit SHA.

## Browser use

Use the unchanged Decisive `examples/pplx-flash/runtime.js` and its companion
modules with a WASM build containing `pplx-flash,wasm,webgpu,kernel-checks`.
The selected Q4 mode currently uses the `createConfigured` entry point supplied
by `kernel-checks`.

```js
// api: initialized Decisive WASM module.
// files: filename -> File/Blob map downloaded and verified against the release manifest.
const engine = await loadFlash(api, files, onProgress, 'webgpu', {
  mode: 'q4-e4',
  resident: false,
  profile: false,
});
const result = await engine.predict(request, 1024);
engine.dispose();
```

The default streaming mode uses two layer buffers. A source URL can also be used
instead of local files if its server supports HTTP byte ranges. A secure browser
context, WebGPU, `shader-f16`, and sufficient device resources are required.
The default `q4-e4` mode computes group dot products with FP32 partials; the
separate `q4-e4-int8` and `q4-e4-signed` variants require packed integer dot support.
Do not assume that ordinary WebGPU availability guarantees the model will fit
or run well on every phone.

`release-manifest.json` provides the exact runtime download file list, sizes,
SHA-256 digests, source revision, and exporter provenance. `SHA256SUMS` covers
every upload file except itself.

## Validation and limitations

The preparation process verified the pinned source hashes, every packed file
hash, tensor shapes and byte ranges, and {{SAMPLE_CHECKS}} independent scalar
sample or unchanged-tensor checks against the BF16 source. These checks establish
packaging and quantization consistency. They are not a full inference or model
quality evaluation of this release.

Quantization can change scores and decisions. No new accuracy or speed claim is
made here. The model's original intended uses and limitations remain applicable;
see the upstream model card. In particular, a privacy-demo score is not a
guarantee that a message is safe to share.

## Reproduce

From the pinned Decisive checkout, with the pinned original model downloaded:

```sh
cargo build --locked --release --no-default-features --features cli,pplx-flash
python3 scripts/pplx_flash/prepare_hf_q4.py prepare \
  --exporter target/release/decisive \
  --source models/pplx-decider-flash-v1.2 \
  --output artifacts/huggingface/pplx-decider-flash-v1.2-q4-e4
```

The preparation command refuses to overwrite an existing output directory and
does not upload anything. The packaging helper is a companion to this release;
the exporter and inference code come from the runtime commit above.

The packed weights themselves can be reproduced using only that runtime commit:

```sh
target/release/decisive pplx-flash \
  --model models/pplx-decider-flash-v1.2 \
  export-webgpu-q4 --output artifacts/pplx-q4-reproduced
```

Copy `config.json`, `decision_config.json`, `joint_config.json`, `tokenizer.json`,
`LICENSE`, and `NOTICE` from the pinned source beside the packed files to obtain
the complete runtime file set. Preserve their original bytes.
