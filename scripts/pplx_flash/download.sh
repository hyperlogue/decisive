#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ ${1:-} == --help || $# -gt 1 ]]; then
  echo 'Usage: bash scripts/pplx_flash/download.sh [model-directory]'
  echo 'Downloads the pinned BF16 release (about 8.5 GiB), verifying SHA-256.'
  exit 0
fi
destination=${1:-models/pplx-decider-flash-v1.2}
if [[ -n ${DECISIVE_BIN:-} ]]; then
  exec "$DECISIVE_BIN" pplx-flash --model "$destination" fetch
fi
exec cargo run --locked --release --no-default-features --features cli,pplx-flash \
  -- pplx-flash --model "$destination" fetch
