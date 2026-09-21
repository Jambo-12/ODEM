#!/usr/bin/env bash
# Preprocess Video-MME captions, visual embeddings, and text embeddings.
# Captioning and visual embedding reuse one GPU; text embeddings use an API.
#
# Usage:
#   bash scripts/preprocess/videomme.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/preprocess/common.sh"

DATASET_ID="videomme"
VIDEO_DIR="${VIDEOMME_VIDEO_DIR:-$(preprocess_config_get "${DATASET_ID}" video_dir)}"
PREPROCESS_DIR="${VIDEOMME_PREPROCESS_DIR:-$(preprocess_config_get "${DATASET_ID}" output_dir)}"
CAPTION_MODEL_PATH="${ODEM_CAPTION_MODEL_PATH:-$(preprocess_config_get "${DATASET_ID}" models.caption.path)}"
VICLIP_MODEL_PATH="${ODEM_VICLIP_MODEL_PATH:-$(preprocess_config_get "${DATASET_ID}" models.visual_embedding.checkpoint)}"
CAPTION_PROMPT="${VIDEOMME_CAPTION_PROMPT:-$(preprocess_config_get "${DATASET_ID}" caption_prompt)}"
CLIP_SECONDS="$(preprocess_config_get "${DATASET_ID}" clip_seconds)"
GPU_ID="${PREPROCESS_GPU_ID:-$(preprocess_config_get "${DATASET_ID}" gpu_id)}"
TEXT_MODEL="${TEXT_EMBEDDING_MODEL:-$(preprocess_config_get "${DATASET_ID}" models.text_embedding.name)}"
TEXT_EMBEDDING_BATCH_SIZE="${TEXT_EMBEDDING_BATCH_SIZE:-$(preprocess_config_get "${DATASET_ID}" models.text_embedding.batch_size)}"
FRAMES_PER_FEAT="${VISUAL_FRAMES_PER_FEAT:-$(preprocess_config_get "${DATASET_ID}" models.visual_embedding.frames_per_segment)}"


show_help() {
    cat <<'EOF'
Usage:
  bash scripts/preprocess/videomme.sh

Shared configuration:
  config/preprocess.yaml

Environment overrides:
  VIDEOMME_VIDEO_DIR         Raw Video-MME video directory
  VIDEOMME_PREPROCESS_DIR    Normalized preprocessing output directory
  ODEM_PREPROCESS_CONFIG     Shared preprocessing config
  ODEM_PYTHON_BIN            Python interpreter
  ODEM_CAPTION_MODEL_PATH    Repository-local Qwen2.5-VL model directory
  ODEM_VICLIP_MODEL_PATH     Repository-local ViCLIP checkpoint
  PREPROCESS_GPU_ID          GPU shared by captioning and visual embedding
  OPENAI_API_KEY             Text embedding service API key
  OPENAI_API_BASE            Text embedding service endpoint
EOF
}


check_required_path() {
    # Validate the interpreter, dataset, and config before expensive work begins.
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "[ERROR] Python interpreter is not executable: ${PYTHON_BIN}"
        exit 1
    fi
    if [[ ! -d "${VIDEO_DIR}" ]]; then
        echo "[ERROR] Video-MME video directory does not exist: ${VIDEO_DIR}"
        exit 1
    fi
    local required_file
    for required_file in "${PREPROCESS_CONFIG_PATH}" "${CAPTION_PROMPT}" "${VICLIP_MODEL_PATH}"; do
        if [[ ! -f "${required_file}" ]]; then
            echo "[ERROR] Required file does not exist: ${required_file}"
            exit 1
        fi
    done
    if [[ ! -d "${CAPTION_MODEL_PATH}" ]]; then
        echo "[ERROR] Caption model directory does not exist: ${CAPTION_MODEL_PATH}"
        exit 1
    fi
    if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU ID: ${GPU_ID}"
        exit 2
    fi
}


run_caption() {
    # Process videos sequentially and skip existing captions.json files.
    echo "[Stage 1/2] Generating Video-MME captions on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" "${PROJECT_DIR}/caption.py" \
        --dataset_name Video-MME \
        --video_dir "${VIDEO_DIR}" \
        --output_dir "${PREPROCESS_DIR}" \
        --clip_seconds "${CLIP_SECONDS}" \
        --model "${CAPTION_MODEL_PATH}" \
        --prompt_path "${CAPTION_PROMPT}"
}


run_visual_embedding() {
    # Generate ViCLIP vectors aligned with caption time ranges on the same GPU.
    echo "[Stage 2/2] Generating visual embeddings on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" "${PROJECT_DIR}/segment_feats_vis.py" \
        --video_dir "${VIDEO_DIR}" \
        --base_dir "${PREPROCESS_DIR}" \
        --model_path "${VICLIP_MODEL_PATH}" \
        --frames_per_feat "${FRAMES_PER_FEAT}" \
        --device cuda
}


run_text_embedding() {
    # Encode all four caption keys and rebuild stale or legacy 2D vectors.
    echo "[Stage 2/2] Generating four-key text embeddings"
    "${PYTHON_BIN}" "${PROJECT_DIR}/segment_feats_text.py" \
        --config "${PREPROCESS_CONFIG_PATH}" \
        --preprocess_dir "${PREPROCESS_DIR}" \
        --model "${TEXT_MODEL}" \
        --batch_size "${TEXT_EMBEDDING_BATCH_SIZE}"
}


wait_embedding_tasks() {
    # Wait for both parallel tasks and return a failure if either one fails.
    local visual_pid="$1"
    local text_pid="$2"
    local visual_status=0
    local text_status=0

    wait "${visual_pid}" || visual_status=$?
    wait "${text_pid}" || text_status=$?

    if [[ "${visual_status}" -ne 0 || "${text_status}" -ne 0 ]]; then
        echo "[ERROR] Embedding extraction failed: visual=${visual_status}, text=${text_status}"
        return 1
    fi
}


write_preprocess_manifest() {
    "${PYTHON_BIN}" "${PROJECT_DIR}/preprocess_metadata.py" \
        --config "${PREPROCESS_CONFIG_PATH}" \
        --dataset "${DATASET_ID}" \
        --video_dir "${VIDEO_DIR}" \
        --output_dir "${PREPROCESS_DIR}" \
        --caption_model_path "${CAPTION_MODEL_PATH}" \
        --caption_prompt "${CAPTION_PROMPT}" \
        --clip_seconds "${CLIP_SECONDS}" \
        --text_model "${TEXT_MODEL}" \
        --text_batch_size "${TEXT_EMBEDDING_BATCH_SIZE}" \
        --viclip_checkpoint "${VICLIP_MODEL_PATH}" \
        --frames_per_segment "${FRAMES_PER_FEAT}"
}


main() {
    # Run captioning first, then generate both retrieval embeddings in parallel.
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        show_help
        return 0
    fi
    check_required_path
    mkdir -p "${PREPROCESS_DIR}"

    echo "Video-MME video directory: ${VIDEO_DIR}"
    echo "Preprocessing output directory: ${PREPROCESS_DIR}"
    echo "Shared config: ${PREPROCESS_CONFIG_PATH}"
    echo "Caption model: ${CAPTION_MODEL_PATH}"
    echo "ViCLIP checkpoint: ${VICLIP_MODEL_PATH}"
    echo "Python interpreter: ${PYTHON_BIN}"
    echo "Preprocessing GPU: ${GPU_ID}"

    run_caption

    run_visual_embedding &
    visual_pid=$!
    run_text_embedding &
    text_pid=$!
    wait_embedding_tasks "${visual_pid}" "${text_pid}"
    write_preprocess_manifest

    echo "[DONE] Video-MME captions, visual embeddings, and text embeddings are ready"
}


main "$@"
