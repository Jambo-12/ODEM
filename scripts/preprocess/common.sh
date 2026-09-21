#!/usr/bin/env bash
# Shared path and configuration helpers for both preprocessing entrypoints.

if [[ -z "${PROJECT_DIR:-}" ]]; then
    echo "[ERROR] preprocess/common.sh requires PROJECT_DIR to be set by the caller"
    exit 1
fi

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.env"
    set +a
fi

if [[ -n "${ODEM_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${ODEM_PYTHON_BIN}"
else
    PYTHON_BIN="$(command -v python || true)"
fi

PREPROCESS_CONFIG_PATH="${ODEM_PREPROCESS_CONFIG:-${PROJECT_DIR}/config/preprocess.yaml}"


preprocess_config_get() {
    local dataset="$1"
    local key="$2"
    "${PYTHON_BIN}" "${PROJECT_DIR}/preprocess_config.py" \
        --config "${PREPROCESS_CONFIG_PATH}" \
        --dataset "${dataset}" \
        --get "${key}"
}
