#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"

cd "${ROOT}"
"${PYTHON_BIN}" grid_search.py \
    --device "${DEVICE}" \
    --output-dir results/full_grid \
    "$@" 2>&1 | tee results/full_grid.log
