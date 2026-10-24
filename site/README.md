# Privacy check

A static, WebGPU-only privacy demo using the upstream Decisive runtimes. Drafts and results stay in memory in the browser. There is no backend or analytics. Explicit model downloads come directly from Hugging Face; PNG sharing is generated locally.

## Models

`models.json` is the download catalog:

- **Decima Small** is the default. The pinned `amyrmahdy/decima-small` revision is `14d09cabc1b4e23106ae4413eef852a568ced77b`; its seven files total **506,679,429 bytes**. Each file has an exact size and SHA-256 digest. The shared Rust runtime receives the encoder as a browser `File` and reads its vocabulary in ranges instead of buffering the complete checkpoint in WASM memory.
- **pplx-decider-flash-v1.2-q4** uses `rickye26/pplx-decider-flash-v1.2-q4`, pinned to revision `a264abdd91f7f002d929bb038acffd51c453115d`. Its 42 runtime files total **2,387,682,594 bytes** (2.39 GB in the interface). This is the `q4-w4a8-e4-r4-g64` checkpoint, including packed embeddings and readout. The frontend passes OPFS `File` objects into `DecisiveModel`; Rust owns Q4 layer streaming, embedding caching, double buffering, kernel selection, and queue completion. Each sequence is limited to **1,024 tokens**; inputs are never silently truncated. Both the UI and Worker require a usable `shader-f16` GPU device before downloading or loading this model. Devices without it can use Decima Small. There is no fallback to the original BF16 weights or a CPU path.
- **pplx-decider-flash-v1.2-q8** uses `rickye26/pplx-decider-flash-v1.2-q8`, pinned to revision `10e640e6566ff93708ef61b83f1b3dd7c06182aa`, with the same 42-file layout in the `q8-w8a8-e8-r8-g64` format: **4,490,404,034 bytes** (4.49 GB). It stays closer to the BF16 model than Q4 with the same task accuracy, but each analysis streams twice the decoder bytes, so in Chrome on an M3 an analysis takes about 7.5 s versus 2–4 s for Q4. It has its own storage namespace and the same `shader-f16` requirement and 1,024-token limit.

The site builds the Decisive runtimes from this checkout. Transient adapter/device initialization failures are retried before reporting WebGPU unavailable.

## Build and preview

Requires Python 3, Node 24 with npm (font package and tests), Rust with the `wasm32-unknown-unknown` target, and the wasm-bindgen CLI version matching Cargo.lock (currently 0.2.129).

```sh
npm ci --prefix browser --ignore-scripts
npm ci --prefix site --ignore-scripts
npm test --prefix browser
node browser/node_modules/typescript/bin/tsc -p site/tsconfig.json
node --test site/tests/*.test.js
DECISIVE_WASM_FEATURES=decima,pplx-flash,wasm,webgpu,pplx-webgpu \
  bash scripts/build_decima_wasm.sh artifacts/web/privacy-build/pkg
python3 site/build.py --package artifacts/web/privacy-build/pkg
python3 -m http.server 8768 --directory site/dist
```

The Noto Sans SC font comes from the pinned `@fontsource/noto-sans-sc` package in `site/package-lock.json`; `build.py` copies its 400-weight subsets and license into `dist/assets/fonts/`. Browsers download only the subsets a page's characters need, and the service worker caches those on first use rather than precaching all of them. Use a fresh runtime output directory for each build. `python3 site/build.py` defaults to `artifacts/web/privacy-build/pkg`, matching the build command above; use `--package` for another output directory. It verifies the binary and recorded source hashes before copying the unchanged runtime. Model weights are not included in the static site. Production hosting needs HTTPS; localhost is supported for development.

The entry HTML loads scripts, workers, fonts, WASM, and the model catalog from one fingerprinted release directory. This prevents a previous service worker from mixing the new UI with old inference or storage code. Storage replies must match the selected model ID, pinned revision, total bytes, and file list before downloads can be enabled. Existing OPFS model files are preserved across app upgrades. `node site/tests/release-cache.mjs` verifies migration from an old cached Pplx worker and rejects forced mismatched replies.

The complete `dist/` output works under any URL directory. The service worker caches app assets under that directory; weights remain in OPFS and are not duplicated in Cache Storage. After setup, analysis works offline. Page loads always fetch the latest `index.html` from the network and fall back to the cached copy only when offline. A new service worker activates as soon as it installs; open tabs check for updates on focus and every 30 minutes, show a Reload prompt when `release.json` names a newer release, and reload by themselves while hidden if no draft or model operation would be lost. `node site/tests/release-cache.mjs` simulates a deploy to verify this.

## Interface and storage

- An intro explains the goal, the three steps, and what to expect, then opens Choose model. Each model appears as a radio card showing size, a Hugging Face link to the pinned revision, and a Download / Resume / Pause button; a downloaded model shows a Ready pill and a delete icon instead. A card that can't run here is greyed out and shows a Disabled badge in place of its button; the badge's tooltip gives the reason. Without `shader-f16` only the Q4 and Q8 cards are disabled. Without WebGPU every card is disabled and a red notice appears under the dialog title. Use model stays disabled until the selected model is fully downloaded, and Stop analysis appears while a model loads or runs. Choose model is the only model dialog; the old `#models` link opens it.
- The interface follows the system light or dark theme. Wide screens place the draft and analysis side by side; phones stack them. Entrances, result cards, score bars, dialogs, and the in-progress indicator animate briefly, and all motion is removed under reduced-motion preferences.
- English, Spanish, Simplified Chinese, and Arabic interfaces, including RTL and local PNG export. Scores and raw input/output are available in expandable details.
- The draft stays immediately below the header; analysis scrolls within the rest of the viewport. Result changes and disclosures animate, respecting reduced-motion preferences. Inference time and the runtime's reported input-token count appear above the results and clear when the draft changes; these numbers do not include model download or loading time.
- Downloads stream into OPFS with integrity checks, pause, resume, retry, and per-file state. The app requests persistent storage when a download starts and blocks downloads that would exceed the browser quota estimate.
- Each model has its own storage namespace. Deleting one model cannot erase another model's files. Existing full-size Pplx files from the earlier demo are listed in Choose model with a delete button; they are never loaded as Q4 weights or silently removed.
- Web Locks coordinate operations between tabs. File paths containing directories are encoded into safe OPFS filenames while retaining their original Hugging Face paths.
- Analysis runs automatically after a typing pause, with one active request and only the latest draft queued. Stale results are discarded; sharing is disabled until results match the draft. Switching models unloads the active engine and preserves the text.
- Exactly two messages form the model context: the fictional community invitation and the complete draft. Four questions cover eight visible categories: `noul` questions for names and financial details, and two `choice` questions whose options compete. One covers dates of birth, contact details, home address or exact location, and government IDs; the other covers access credentials and private keys or recovery phrases. Access credentials include passwords, PINs, one-time verification codes, API keys, access tokens, and session cookies. Pplx uses one merged `credential` option. Decima retains its separate `password`, `token`, and `otp` options; the visible credential score is their maximum, preserving the existing detection threshold without adding their probabilities. Both models retain the private-key, placeholder/example/public-value, and no-secret options. Results, details, and PNG exports show one access-credentials category; raw output retains the model's actual answers.
- Decima encodes the draft once per question. The configured predictor sets the model-specific `long: true` parameter: Decima scores the draft in sentence chunks of up to 32 tokens with the invitation as context, and each category takes its option's highest probability across chunks. Scoring the whole draft at once dilutes a single detail among the surrounding sentences, and chunking also lets drafts exceed Decima's 512-token context. Work grows with the number of chunks, about one per sentence.
- Icons appear at score 0.5 or above. Yellow/red indicate category sensitivity, not calibrated probabilities of harm. Model accuracy is not guaranteed. Decima's 512-token state-plus-question limit is enforced by the runtime; overlong input produces an error without silent truncation.
- PNG export freezes the message, model identity, and results together. Nothing is uploaded automatically.

OPFS belongs to a browser and origin, not a URL directory. The private review origin and final custom domain do not share downloads. Browser eviction or clearing site data can remove files.

## Verification

The Node tests cover the SDK cache lifecycle and ownership, batch cache sharing, cancellation, typed parameters, pinned model sizes, storage isolation, integrity checks, resumption, deletion, worker ordering, GPU feature checks/retries, stale-result suppression, and credential scoring for each model. The TypeScript Worker binds model-specific defaults in `lib/inference-config.ts`; the editor selects the secret question and reads its scores through `lib/privacy.js` using the selected model's engine. The request shape and other three questions are shared. Errors use Rust error codes instead of matching error-message text.

With Playwright available, serve the repository root and run:

```sh
node site/tests/browser.mjs
node site/tests/download-layout.mjs
node site/tests/webgpu-startup.mjs
node site/tests/pplx-browser.mjs
node browser/tests/wasm.mjs
node site/tests/real-model.mjs
node site/tests/decima-download.mjs
```

The browser suite uses small test downloads for actual OPFS and WASM hashing. It checks the published Q4 catalog and precision requirement, legacy cleanup, modal interactions, locales, offline reload, and sharing. Its UI-only inference fixture replays recorded model output; that adapter and fixtures are excluded from `dist/`. `pplx-browser.mjs` exercises Q4 downloading, loading through the configured SDK predictor, token-limit errors, OPFS reuse, cancellation, and scoped deletion with test doubles for GPU/inference. This verifies application plumbing, not model accuracy or GPU kernels.

The uploaded Q4 revision's runtime file sizes/hashes, anonymous access, HTTP byte ranges, and CORS have been verified. Run `node browser/tests/q4.mjs` on a GPU with `shader-f16` to compare the new Rust scheduler with the previous JavaScript scheduler and test cancellation recovery; serve the repository root and set `PPLX_Q4_MODEL_DIR` if the prepared checkpoint is elsewhere. Its smoke timings are not a performance benchmark. The new Rust scheduler still needs physical-GPU Q4 parity/performance validation; the unchanged kernel implementations alone do not establish scheduler performance. Actual Q4 inference still requires validation on a device with `shader-f16`; the available SwiftShader adapter lacks that feature. The release's 927 quantization/tensor checks are separate from this device-level smoke test. See `scripts/pplx_flash/check_hf_q4_browser.mjs` for the real inference check.

`browser/tests/wasm.mjs` checks the new Worker SDK against the existing Decima browser entry point, shared batch caches, per-item errors, parameter validation, and disposal. Set `PPLX_MODEL_DIR` to additionally check bounded BF16 checkpoint loading with shards larger than 4 GiB; that check does not run Pplx inference.

`browser/tests/errors.mjs` checks unsupported model/backend/format selections before any checkpoint reads, missing build features, and preservation of error codes/messages across the Worker. Set `PPLX_Q4_MODEL_DIR` on a WebGPU device without `shader-f16` to also test the missing-feature error using real Q4 metadata. The page maps unsupported selections, missing GPU features, and memory limits to specific messages in all interface languages; SDK errors retain technical details and supported alternatives.

The real-model smoke uses the pinned local Decima checkpoint and the actual application questions. The optional download test fetches the real ~507 MB checkpoint, runs the complete webpage analysis flow, and deletes its test model files. Environment options include `PLAYWRIGHT_MODULE`, `PRIVACY_BROWSER_EXECUTABLE`, `PRIVACY_TEST_URL`, `PRIVACY_REAL_TEST_URL`, and `PRIVACY_MODEL_DIR` (default `models/decima-small`). `PRIVACY_SOFTWARE_GPU=1` permits SwiftShader for correctness checks; its timings are not physical-GPU performance measurements. Phone and Safari behavior still require device testing.

The recorded Decima WebGPU smoke detected name/email in English, location in Chinese, and no disclosures in a neutral greeting. See `tests/fixtures/decima-webgpu-smoke.json`. The full Hugging Face download, SHA-256 verification, OPFS loading, webpage inference, and deletion test also passed. These are correctness checks, not a broad model-quality evaluation.

## GitHub Pages

The demo is published at <https://decisive.hyperlogue.tech/demo/private-info-detector/>. `.github/workflows/pages.yml` runs on pushes to `main` that touch the site or runtime sources, and on manual dispatch. Pull requests run the same build checks without publishing. Only the deployment job has Pages/OIDC write permissions. The wasm-bindgen CLI version is read from Cargo.lock so it stays matched to the Rust bindings. It compiles the strict TypeScript SDK and application Worker, runs SDK, frontend and Pages-layout tests, builds the runtime, bundles `site/dist`, and lets `site/pages.py` place it under `demo/private-info-detector/`. The domain root, `/demo/`, and unknown paths redirect to the demo. Before upload, `site/check_pages.py` verifies redirects, module/style references, service-worker assets, the WASM checksum, and absence of model weights or symlinks. The workflow replaces the complete Pages artifact, so other site content must be added to `site/pages.py`.

To preview the deployed layout locally:

```sh
python3 site/pages.py --out artifacts/pages
python3 site/check_pages.py artifacts/pages
python3 -m http.server 8770 --directory artifacts/pages
# open http://127.0.0.1:8770/demo/private-info-detector/
```

One-time setup: in repository Settings → Pages, choose GitHub Actions as the source and set the custom domain to `decisive.hyperlogue.tech`, with a DNS `CNAME` record pointing at the organization's `github.io` host. Before the domain is bound, the workflow passes the project-page base path to `site/pages.py` so the 404 redirect still works. Model weights stay on Hugging Face. See [GitHub's Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
