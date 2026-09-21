#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export CUDA_VISIBLE_DEVICES="${M3_BENCH_GPU_ID:-0}"
exec "${ODEM_PYTHON_BIN:-python}" -m eval.m3bench.agent_m3bench "$@"
