#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash scripts/pplx_flash/bench_webgpu.sh benchmark-runs/name artifacts/web/bundle/pkg [--kernels-only]'
  echo 'Runs adapter and main-thread/worker kernel checks, then both full-model reference fixtures.'
  echo '--kernels-only skips the checkpoint and full-model checks.'
  exit 0
fi
out=${1:?usage: bash scripts/pplx_flash/bench_webgpu.sh benchmark-runs/name artifacts/web/bundle/pkg [--kernels-only]}
package=${2:?Wasm package directory required}
mode=${3:---full}
[[ $# -le 3 && ( "$mode" == --full || "$mode" == --kernels-only ) ]] || { echo 'expected --kernels-only or no third argument' >&2; exit 1; }
[[ "$out" == benchmark-runs/* && "$out" != *..* && ! -e "$out" ]] || { echo 'choose a new benchmark-runs/ directory' >&2; exit 1; }
[[ "$package" == artifacts/web/* && "$package" != *..* && -f "$package/build.json" ]] || { echo 'local Wasm package required' >&2; exit 1; }
runner=${DECISIVE_BROWSER_BENCH:-target/benchmarks/browser_bench}
browser=${CHROMIUM:-chromium}
args=(--browser "$browser")
if [[ ${DECISIVE_BROWSER_NO_SANDBOX:-0} == 1 ]]; then args+=(--no-sandbox); fi
if [[ ${DECISIVE_SOFTWARE_GPU:-0} == 1 ]]; then args+=(--software-gpu); fi
command -v "$runner" >/dev/null || { echo "Browser runner missing: $runner. Run scripts/build_bench.sh browser_bench --no-default-features --features browser-bench,pplx-flash" >&2; exit 1; }
command -v "$browser" >/dev/null || { echo 'Chromium missing; set CHROMIUM to the Chrome/Chromium executable' >&2; exit 1; }
python3 scripts/pplx_flash/verify_webgpu.py package "$package" "$mode"
mkdir -p "$out"
cp "$package/build.json" "$out/build.json"
"$runner" "${args[@]}" \
  --page /examples/browser/webgpu-adapter.html \
  --output "$out/adapter.json" --timeout-seconds 60
python3 scripts/pplx_flash/verify_webgpu.py adapter "$out/adapter.json"
for worker in 0 1; do
  "$runner" "${args[@]}" \
    --page "/examples/browser/pplx_flash_webgpu.html?package=/$package&worker=$worker" \
    --output "$out/kernels-worker-$worker.json" --timeout-seconds "${DECISIVE_KERNEL_TIMEOUT:-240}"
done
if [[ "$mode" == --full ]]; then
  for input in request multilingual; do
    "$runner" "${args[@]}" \
      --page "/examples/browser/pplx_flash.html?package=/$package&fixture=$input&backend=webgpu" \
      --output "$out/$input.json" --timeout-seconds "${DECISIVE_BROWSER_TIMEOUT:-3600}"
  done
fi
python3 scripts/pplx_flash/verify_webgpu.py results "$out" "$mode"
