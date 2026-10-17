#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
destination=${1:-artifacts/web/pkg-decima-threads}
[[ "$destination" == artifacts/web/* && "$destination" != *..* ]] || { echo 'destination must be under artifacts/web' >&2; exit 1; }
[[ ! -e "$destination" ]] || { echo 'destination already exists; choose a new build name' >&2; exit 1; }
toolchain=${DECISIVE_WASM_THREADS_TOOLCHAIN:-nightly-2026-08-07}
if [[ -n "${DECISIVE_NIGHTLY_CARGO:-}" ]]; then
    cargo_command=("$DECISIVE_NIGHTLY_CARGO")
    rustc_command=("${RUSTC:-rustc}")
else
    command -v rustup >/dev/null
    cargo_command=(rustup run "$toolchain" cargo)
    rustc_command=(rustup run "$toolchain" rustc)
fi
compiler=$("${rustc_command[@]}" --version)
[[ "$compiler" == *nightly* ]] || { echo 'shared-memory builds require nightly Rust with rust-src' >&2; exit 1; }
command -v wasm-bindgen >/dev/null
bindings=$(wasm-bindgen --version)
target_dir=${DECISIVE_WASM_THREADS_TARGET_DIR:-target/decima-wasm-threads}
features=${DECISIVE_WASM_THREADS_FEATURES:-laya,decima,wasm,webgpu,fp16,int8,kernel-checks,decima-wasm-threads}
export RUSTFLAGS='-C target-feature=+simd128,+atomics,+bulk-memory -C link-arg=--shared-memory -C link-arg=--max-memory=2147483648 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base'
"${cargo_command[@]}" build -Z build-std=panic_abort,std --locked --release \
    --no-default-features --features "$features" --target wasm32-unknown-unknown \
    --target-dir "$target_dir" --lib
wasm-bindgen --target web --out-dir "$destination" \
    "$target_dir/wasm32-unknown-unknown/release/decisive.wasm"
python3 - "$destination" "$compiler" "$features" "$bindings" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
destination=Path(sys.argv[1])
paths=[Path('Cargo.toml'),Path('Cargo.lock'),Path('scripts/build_decima_wasm_threads.sh'),
       *sorted(Path('src').rglob('*.rs')),*sorted(Path('src').rglob('*.wgsl'))]
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
data={'compiler':sys.argv[2],'bindings':sys.argv[4],'features':sys.argv[3],'rustflags':os.environ['RUSTFLAGS'],
      'wasm_sha256':hashlib.sha256((destination/'decisive_bg.wasm').read_bytes()).hexdigest(),
      'sources':manifest}
(destination/'build.json').write_text(json.dumps(data,indent=2)+'\n')
PY
