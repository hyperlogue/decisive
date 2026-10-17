#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=${1:?usage: bash scripts/pplx_flash/bench.sh benchmark-runs/name [wasm-package]}
[[ "$out" == benchmark-runs/* && "$out" != *..* && ! -e "$out" ]] || { echo 'choose a new directory under benchmark-runs/' >&2; exit 1; }
mkdir -p "$out"
cli=${DECISIVE_BIN:-target/release/decisive}
"$cli" capabilities --json > "$out/capabilities.json"
load_args=()
if [[ "${DECISIVE_PPLX_RESIDENT:-0}" == 1 ]]; then load_args+=(--resident); fi
for threads in 1 3; do
    for input in request multilingual; do
        "$cli" pplx-flash "${load_args[@]}" --threads "$threads" evaluate \
            --input "tests/fixtures/pplx_flash/$input.json" --rounds 3 \
            > "$out/native-$input-$threads.jsonl"
    done
done
if [[ -n "${2:-}" ]]; then
    package=$2
    [[ "$package" == artifacts/web/* && "$package" != *..* ]] || { echo 'package must be under artifacts/web/' >&2; exit 1; }
    browser_args=()
    if [[ "${DECISIVE_BROWSER_NO_SANDBOX:-0}" == 1 ]]; then browser_args+=(--no-sandbox); fi
    for input in request multilingual; do
        "${DECISIVE_BROWSER_BENCH:-target/benchmarks/browser_bench}" "${browser_args[@]}" --browser "${CHROMIUM:-chromium}" \
            --page "/examples/browser/pplx_flash.html?package=/$package&fixture=$input" \
            --output "$out/wasm-$input.json" --timeout-seconds 900
    done
fi
python3 - "$out" <<'PY'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1])
cases=json.loads(Path('tests/fixtures/pplx_flash/reference.json').read_text())['cases']
maximum=0.0
checked=0
def compare(actual,expected):
    global maximum
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(), 'response schema differs'
        for key in expected: compare(actual[key],expected[key])
    elif isinstance(expected,(int,float)):
        assert isinstance(actual,(int,float)) and math.isfinite(actual), 'non-finite or missing numeric output'
        delta=abs(actual-expected);maximum=max(maximum,delta)
        assert delta<=1e-4, f'output error {delta} exceeds 1e-4'
    else: assert actual==expected, f'decision/value changed: {actual!r} != {expected!r}'
for path in sorted(root.glob('native-*.jsonl'))+sorted(root.glob('wasm-*.json')):
    expected=cases[1 if 'multilingual' in path.name else 0]['response']
    runs=[json.loads(line) for line in path.read_text().splitlines()] if path.suffix=='.jsonl' else [json.loads(path.read_text())]
    for run in runs: compare(run['response'],expected);checked+=1
result={'checked_responses':checked,'max_absolute_output_error':maximum,'bound':1e-4,'scope':'synthetic reference parity, not a task-quality benchmark'}
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
