#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
name=${1:?usage: scripts/build_bench.sh BENCH_NAME [cargo build options]}
shift
manifest=$(mktemp)
trap 'rm -f "$manifest"' EXIT
cargo build --locked --release --bench "$name" --message-format=json-render-diagnostics "$@" > "$manifest"
python3 - "$name" "$manifest" <<'PY'
import json, os, shutil, sys
from pathlib import Path
name, manifest = sys.argv[1:]
artifacts = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
executable = next(Path(a['executable']) for a in artifacts
                  if a.get('reason') == 'compiler-artifact'
                  and a['target']['name'] == name and 'bench' in a['target']['kind']
                  and a.get('executable'))
destination = Path(os.environ.get('CARGO_TARGET_DIR', 'target')) / 'benchmarks' / name
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(executable, destination)
print(destination)
PY
