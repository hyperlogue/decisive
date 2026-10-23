# Privacy check

A static, WebGPU-only privacy demo using the upstream Decisive runtimes. Drafts and results stay in memory in the browser. There is no backend or analytics. Explicit model downloads come directly from Hugging Face; PNG sharing is generated locally.

## Models

`models.json` is the download catalog:

- **Decima Small** is the default. The pinned `amyrmahdy/decima-small` revision is `14d09cabc1b4e23106ae4413eef852a568ced77b`; its seven files total **506,679,429 bytes**. Each file has an exact size and SHA-256 digest. The frontend passes the encoder as a browser `File` to the optimized upstream `DecimaWebGpuModel.create`, which reads its vocabulary in ranges instead of buffering the complete checkpoint in WASM memory.
- **Pplx Flash · Q4** uses `rickye26/pplx-decider-flash-v1.2-q4`, pinned to revision `a264abdd91f7f002d929bb038acffd51c453115d`. Its 42 runtime files total **2,387,682,594 bytes** (2.39 GB in the interface). This is the `q4-w4a8-e4-r4-g64` checkpoint, including packed embeddings and readout. The frontend passes OPFS `File` objects into the unchanged upstream `loadFlash` runtime with `mode: 'q4-e4'` and streamed layer buffers. Each sequence is limited to **1,024 tokens**; inputs are never silently truncated. Both the UI and Worker require a usable `shader-f16` GPU device before downloading or loading this model. Devices without it can use Decima Small. There is no fallback to the original BF16 weights or a CPU path.

The site builds the Decisive runtimes from this checkout. Transient adapter/device initialization failures are retried before reporting WebGPU unavailable.

## Build and preview

Requires Python 3, Node 24 with npm (font package and tests), Rust with the `wasm32-unknown-unknown` target, and the wasm-bindgen CLI version matching Cargo.lock (currently 0.2.129).

```sh
npm ci --prefix site --ignore-scripts
node --test site/tests/*.test.js
DECISIVE_WASM_FEATURES=decima,pplx-flash,wasm,webgpu,kernel-checks \
  bash scripts/build_decima_wasm.sh artifacts/web/privacy-build/pkg
python3 site/build.py --package artifacts/web/privacy-build/pkg
python3 -m http.server 8768 --directory site/dist
```

The Noto Sans SC font comes from the pinned `@fontsource/noto-sans-sc` package in `site/package-lock.json`; `build.py` copies its 400-weight subsets and license into `dist/assets/fonts/`. Browsers download only the subsets a page's characters need, and the service worker caches those on first use rather than precaching all of them. Use a fresh runtime output directory for each build. `python3 site/build.py` defaults to `artifacts/web/privacy-build/pkg`, matching the build command above; use `--package` for another output directory. It verifies the binary and recorded source hashes before copying the unchanged runtime. Model weights are not included in the static site. Production hosting needs HTTPS; localhost is supported for development.

The entry HTML loads scripts, workers, fonts, WASM, and the model catalog from one fingerprinted release directory. This prevents a previous service worker from mixing the new UI with old inference or storage code. Storage replies must match the selected model ID, pinned revision, total bytes, and file list before downloads can be enabled. Existing OPFS model files are preserved across app upgrades. `node site/tests/release-cache.mjs` verifies migration from an old cached Pplx worker and rejects forced mismatched replies.

The complete `dist/` output works under any URL directory. The service worker caches app assets under that directory; weights remain in OPFS and are not duplicated in Cache Storage. After setup, analysis works offline.

## Interface and storage

- Setup shows WebGPU availability, a model picker, and the selected model's download size. Models & storage is a modal over the editor.
- The interface follows the system light or dark theme. Wide screens place the draft and analysis side by side; phones stack them. Entrances, result cards, score bars, dialogs, and the in-progress indicator animate briefly, and all motion is removed under reduced-motion preferences.
- English, Spanish, Simplified Chinese, and Arabic interfaces, including RTL and local PNG export. Scores and raw input/output are available in expandable details.
- The draft stays immediately below the header; analysis scrolls within the rest of the viewport. Result changes and disclosures animate, respecting reduced-motion preferences. Inference time and the runtime's reported input-token count appear above the results and clear when the draft changes; these numbers do not include model download or loading time.
- Downloads stream into OPFS with integrity checks, pause, resume, retry, and per-file state. Browser quota estimates are labeled approximate. Persistent storage requires a user action.
- Each model has its own storage namespace. Decima deletion cannot erase Q4 files. Existing full-size Pplx files from the earlier demo remain visible in a separate cleanup section; they are never loaded as Q4 weights or silently removed.
- Web Locks coordinate operations between tabs. File paths containing directories are encoded into safe OPFS filenames while retaining their original Hugging Face paths.
- Analysis runs automatically after a typing pause, with one active request and only the latest draft queued. Stale results are discarded; sharing is disabled until results match the draft. Switching models unloads the active engine and preserves the text.
- Exactly two messages form the model context: the fictional community invitation and the complete draft. Four questions cover ten categories: `noul` questions for names and for financial details, and two `choice` questions whose options compete. One covers dates of birth, contact details, home address or exact location, and government IDs; the other covers passwords, API keys and tokens, one-time codes, and private keys or recovery phrases. A choice category's score is its option's probability. Decima encodes the draft once per question, so grouping cuts the work per analysis from ten encodes to four. Requests set `long: true`: Decima scores the draft in sentence chunks of up to 32 tokens with the invitation as context, and each category takes its option's highest probability across chunks. Scoring the whole draft at once dilutes a single detail among the surrounding sentences, and chunking also lets drafts exceed Decima's 512-token context. Work grows with the number of chunks, about one per sentence.
- Icons appear at score 0.5 or above. Yellow/red indicate category sensitivity, not calibrated probabilities of harm. Model accuracy is not guaranteed. Decima's 512-token state-plus-question limit is enforced by the runtime; overlong input produces an error without silent truncation.
- PNG export freezes the message, model identity, and results together. Nothing is uploaded automatically.

OPFS belongs to a browser and origin, not a URL directory. The private review origin and final custom domain do not share downloads. Browser eviction or clearing site data can remove files.

## Verification

The Node tests cover pinned model sizes, unpublished-model guards, Decima's streamed encoder boundary, the Q4 adapter's configuration and 1,024-token limit, cancellation, per-model storage isolation, integrity checks, resumption, deletion, worker ordering, GPU feature checks/retries, and stale-result suppression.

With Playwright available, serve the repository root and run:

```sh
node site/tests/browser.mjs
node site/tests/webgpu-startup.mjs
node site/tests/pplx-browser.mjs
node site/tests/real-model.mjs
node site/tests/decima-download.mjs
```

The browser suite uses small test downloads for actual OPFS and WASM hashing. It checks the published Q4 catalog and precision requirement, legacy cleanup, modal interactions, locales, offline reload, and sharing. Its UI-only inference adapter replays recorded model output; that adapter and fixtures are excluded from `dist/`. `pplx-browser.mjs` exercises Q4 downloading, loading through the frontend adapter, token-limit errors, OPFS reuse, cancellation, and scoped deletion with test doubles for GPU/inference. This verifies application plumbing, not model accuracy or GPU kernels.

The uploaded Q4 revision's runtime file sizes/hashes, anonymous access, HTTP byte ranges, and CORS have been verified. Actual Q4 inference still requires validation on a device with `shader-f16`; the available SwiftShader adapter lacks that feature. The release's 927 quantization/tensor checks are separate from this device-level smoke test. See `scripts/pplx_flash/check_hf_q4_browser.mjs` for the real inference check.

The real-model smoke uses the pinned local Decima checkpoint and the actual application questions. The optional download test fetches the real ~507 MB checkpoint, runs the complete webpage analysis flow, and deletes its test model files. Environment options include `PLAYWRIGHT_MODULE`, `PRIVACY_BROWSER_EXECUTABLE`, `PRIVACY_TEST_URL`, `PRIVACY_REAL_TEST_URL`, and `PRIVACY_MODEL_DIR` (default `models/decima-small`). `PRIVACY_SOFTWARE_GPU=1` permits SwiftShader for correctness checks; its timings are not physical-GPU performance measurements. Phone and Safari behavior still require device testing.

The recorded Decima WebGPU smoke detected name/email in English, location in Chinese, and no disclosures in a neutral greeting. See `tests/fixtures/decima-webgpu-smoke.json`. The full Hugging Face download, SHA-256 verification, OPFS loading, webpage inference, and deletion test also passed. These are correctness checks, not a broad model-quality evaluation.

## GitHub Pages

The demo is published at <https://decisive.hyperlogue.tech/demo/private-info-detector/>. `.github/workflows/pages.yml` runs on pushes to `main` that touch the site or runtime sources, and on manual dispatch. Pull requests run the same build checks without publishing. Only the deployment job has Pages/OIDC write permissions. The wasm-bindgen CLI version is read from Cargo.lock so it stays matched to the Rust bindings. It runs the frontend and Pages-layout tests, builds the runtime, bundles `site/dist`, and lets `site/pages.py` place it under `demo/private-info-detector/`. The domain root, `/demo/`, and unknown paths redirect to the demo. Before upload, `site/check_pages.py` verifies redirects, module/style references, service-worker assets, the WASM checksum, and absence of model weights or symlinks. The workflow replaces the complete Pages artifact, so other site content must be added to `site/pages.py`.

To preview the deployed layout locally:

```sh
python3 site/pages.py --out artifacts/pages
python3 site/check_pages.py artifacts/pages
python3 -m http.server 8770 --directory artifacts/pages
# open http://127.0.0.1:8770/demo/private-info-detector/
```

One-time setup: in repository Settings → Pages, choose GitHub Actions as the source and set the custom domain to `decisive.hyperlogue.tech`, with a DNS `CNAME` record pointing at the organization's `github.io` host. Before the domain is bound, the workflow passes the project-page base path to `site/pages.py` so the 404 redirect still works. Model weights stay on Hugging Face. See [GitHub's Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
