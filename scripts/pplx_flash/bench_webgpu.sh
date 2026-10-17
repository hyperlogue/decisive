#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=${1:?usage: bash scripts/pplx_flash/bench_webgpu.sh benchmark-runs/name artifacts/web/bundle/pkg}
package=${2:?Wasm package directory required}
[[ "$out" == benchmark-runs/* && "$out" != *..* && ! -e "$out" ]] || { echo 'choose a new benchmark-runs/ directory' >&2; exit 1; }
[[ "$package" == artifacts/web/* && "$package" != *..* && -f "$package/build.json" ]] || { echo 'local Wasm package required' >&2; exit 1; }
args=()
if [[ ${DECISIVE_BROWSER_NO_SANDBOX:-0} == 1 ]]; then args+=(--no-sandbox); fi
if [[ ${DECISIVE_SOFTWARE_GPU:-0} == 1 ]]; then args+=(--software-gpu); fi
runner=${DECISIVE_BROWSER_BENCH:-target/benchmarks/browser_bench}
browser=${CHROMIUM:-chromium}
mkdir -p "$out"
cp "$package/build.json" "$out/build.json"
if python3 - "$package/build.json" <<'PY'
import json,sys
sys.exit('kernel-checks' not in json.load(open(sys.argv[1]))['features'].split(','))
PY
then
  for worker in 0 1; do
    "$runner" "${args[@]}" --browser "$browser" \
      --page "/examples/browser/pplx_flash_webgpu.html?package=/$package&worker=$worker" \
      --output "$out/kernels-worker-$worker.json" --timeout-seconds 240
  done
fi
for input in request multilingual; do
  "$runner" "${args[@]}" --browser "$browser" \
    --page "/examples/browser/pplx_flash.html?package=/$package&fixture=$input&backend=webgpu" \
    --output "$out/$input.json" --timeout-seconds "${DECISIVE_BROWSER_TIMEOUT:-3600}"
done
python3 - "$out" <<'PY'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1])
checks=[json.loads(p.read_text()) for p in root.glob('kernels-*.json')]
assert all(not r.get('error') and r.get('checks') for r in checks), 'kernel check failed'
runs=[json.loads((root/(name+'.json')).read_text()) for name in ['request','multilingual']]
for r in runs:
    assert not r.get('error') and r.get('response'), 'full-model check failed'
    assert r['backend']=='webgpu' and math.isfinite(r['reference_max_error']) and r['reference_max_error']<=1e-4
summary={'cases':2,'max_absolute_output_error':max(r['reference_max_error'] for r in runs),'bound':1e-4,'software_gpu':any(r.get('software_gpu_requested') for r in runs),'scope':'synthetic reference parity, not a task-quality benchmark'}
(root/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print('WebGPU checks passed; timings include layer streaming. Software adapters are correctness checks only.')
PY
