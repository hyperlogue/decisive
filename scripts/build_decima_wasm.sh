#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
destination=${1:-artifacts/web/pkg}
[[ "$destination" == artifacts/web/* && "$destination" != *..* ]] || { echo 'destination must be under artifacts/web' >&2; exit 1; }
[[ ! -e "$destination" ]] || { echo 'destination already exists; choose a new build name' >&2; exit 1; }
compiler=$(rustc --version)
bindings=$(wasm-bindgen --version)
features=${DECISIVE_WASM_FEATURES:-laya,decima,wasm,webgpu,fp16,int8,kernel-checks}
target_dir=${DECISIVE_WASM_TARGET_DIR:-target}
export RUSTFLAGS='-C target-feature=+simd128'
cargo build --locked --release --no-default-features --features "$features" \
    --target wasm32-unknown-unknown --target-dir "$target_dir" --lib
wasm-bindgen --target web --out-dir "$destination" \
    "$target_dir/wasm32-unknown-unknown/release/decisive.wasm"
python3 - "$destination" "$compiler" "$features" "$bindings" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
destination=Path(sys.argv[1])
paths=[Path('Cargo.toml'),Path('Cargo.lock'),Path('scripts/build_decima_wasm.sh'),
       *sorted(Path('src').rglob('*.rs')),*sorted(Path('src').rglob('*.wgsl'))]
data={'compiler':sys.argv[2],'bindings':sys.argv[4],'features':sys.argv[3],
      'rustflags':os.environ['RUSTFLAGS'],
      'wasm_sha256':hashlib.sha256((destination/'decisive_bg.wasm').read_bytes()).hexdigest(),
      'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(destination/'build.json').write_text(json.dumps(data,indent=2)+'\n')
PY
