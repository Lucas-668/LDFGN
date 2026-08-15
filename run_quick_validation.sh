#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"

cd "${ROOT}"
"${PYTHON_BIN}" quick_validate.py \
    --device "${DEVICE}" \
    "$@" 2>&1 | tee results/quick_validation.log
