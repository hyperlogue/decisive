# Prepare the browser Q4 release

This uses the existing `export-webgpu-q4` command. It does not change model or
inference code. The package is the `q4-w4a8-e4-r4-g64` format consumed by the
browser runtime's `q4-e4` mode, including quantized embeddings and readout.

## Create the upload folder

```sh
cargo build --locked --release --no-default-features --features cli,pplx-flash
python3 scripts/pplx_flash/prepare_hf_q4.py prepare \
  --exporter target/release/decisive \
  --source models/pplx-decider-flash-v1.2 \
  --output artifacts/huggingface/pplx-decider-flash-v1.2-q4-e4
```

The command verifies every source checksum before exporting, checks the packed
layout and hashes, and independently checks samples against the original BF16
values. It copies only the runtime's configuration/tokenizer and license/notice,
and generates a model card, release manifest, and `SHA256SUMS`. It refuses to
overwrite an existing folder. Preparation never contacts Hugging Face.

Pass `--format q8-e8` to package the Q8 release with `export-webgpu-q8`
instead; its card template is `hf_q8_model_card.md`.

Review `README.md` and `release-manifest.json` in the generated directory. The
manifest distinguishes upstream provenance from the future Hub release revision.
The card states the validation scope; a full browser inference/quality run is a
separate check, not implied by file integrity.

```sh
python3 scripts/pplx_flash/prepare_hf_q4.py verify \
  --output artifacts/huggingface/pplx-decider-flash-v1.2-q4-e4
python3 -m unittest discover -s scripts/pplx_flash -p test_prepare_hf_q4.py
```

For an actual inference smoke test, serve the prepared package with the existing
range-capable server and use a browser adapter with `shader-f16`:

```sh
DECISIVE_WASM_FEATURES=decima,pplx-flash,wasm,webgpu,kernel-checks \
  bash scripts/build_decima_wasm.sh artifacts/web/q4-release/pkg
python3 scripts/pplx_flash/serve_phone.py \
  --package artifacts/web/q4-release/pkg \
  --model artifacts/huggingface/pplx-decider-flash-v1.2-q4-e4 \
  --packed artifacts/huggingface/pplx-decider-flash-v1.2-q4-e4
# Take the printed URL and append v/current/.
node scripts/pplx_flash/check_hf_q4_browser.mjs "$LOCAL_VERSION_URL"
```

Set `PLAYWRIGHT_MODULE` and `PRIVACY_BROWSER_EXECUTABLE` if Playwright/Chromium
are not available at their default locations. The test checks a known support
routing request through all 32 layers. It writes a JSON report to
`Q4_SMOKE_OUTPUT` (default `/tmp/pplx-q4-release-smoke.json`). A missing
`shader-f16` feature is a blocked inference check, not a passing model test.
The package's initial release manifest records `full_inference: false`.

## Publish after review

Set the intended repository ID and authenticate locally with `hf auth login`.
The upload syntax was checked with `huggingface_hub` 2.2.0. If `hf` is not
installed, use `uv tool run --from huggingface_hub==2.2.0 hf` in place of `hf`:

```sh
uv tool run --from huggingface_hub==2.2.0 hf auth login
```

Do not paste credentials into chat or put them in any release file. Create a
model repository with the intended visibility, then upload only the prepared
directory. For the public browser demo, the eventual release must allow anonymous
downloads; browser users must not need a Hugging Face token.

```sh
# Replace this placeholder with the chosen organization/repository.
MODEL_REPO_ID=ORG/pplx-decider-flash-v1.2-decisive-q4
uv tool run --from huggingface_hub==2.2.0 hf upload "$MODEL_REPO_ID" \
  artifacts/huggingface/pplx-decider-flash-v1.2-q4-e4 . \
  --repo-type model --commit-message "Add Decisive Q4-E4 browser weights"
```

Use the current [`hf upload` command](https://huggingface.co/docs/huggingface_hub/guides/upload)
and [model-card conventions](https://huggingface.co/docs/hub/model-cards).
Do not upload the parent `artifacts/`, build outputs, original BF16 shards,
browser storage, or local benchmark logs.

## Connect the webpage after publication

1. Record the **new repository's full commit SHA**, not the upstream source SHA
   inside `packed-q4-e4.json`.
2. Verify anonymous file access and HTTP range responses at that pinned revision.
3. Update the `pplx-flash-q4` catalog entry's revision and files from
   `release-manifest.json` → `download.files`. Its `download.total_bytes` is the
   actual size to display. Keep the existing published revision until the new
   release has been verified.
4. Keep the website adapter configured for `loadFlash` with WebGPU and
   `{mode: 'q4-e4', resident: false, profile: false}`. Keep the requested
   `predict(request, 1024)` limit. Check `shader-f16` as well as WebGPU for this
   model. The default `q4-e4` mode does not require packed integer dot products.
5. Test the pinned download, checksum verification, OPFS reload, one real
   inference, cancellation, and model deletion before enabling the catalog entry.

The publication command above is deliberately separate from preparation. No
repository is created, uploaded, or made public by the preparation scripts.
