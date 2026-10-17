#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
destination=${1:-artifacts/web/pplx-flash-demo}
[[ "$destination" == artifacts/web/* && "$destination" != *..* ]] || { echo 'destination must be under artifacts/web' >&2; exit 1; }
[[ ! -e "$destination" ]] || { echo 'destination already exists; choose a new build name' >&2; exit 1; }
export DECISIVE_WASM_FEATURES=pplx-flash,wasm,webgpu
if [[ ${DECISIVE_PPLX_CHECKS:-0} == 1 ]]; then
  DECISIVE_WASM_FEATURES+=,kernel-checks
fi
if [[ ${DECISIVE_PPLX_FMA:-0} == 1 ]]; then
  DECISIVE_WASM_FEATURES+=,pplx-wasm-fma
fi
bash scripts/build_decima_wasm.sh "$destination/pkg"
cp examples/pplx-flash/*.html examples/pplx-flash/*.js examples/pplx-flash/model.json "$destination/"
python3 - "$destination" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
files=[*root.glob('*.js'),*root.glob('*.html'),root/'model.json']
(root/'browser-build.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2)+'\n')
PY
echo "Browser bundle: $destination"
echo "Serve it: python3 -m http.server 8080 --bind 127.0.0.1 --directory $destination"
