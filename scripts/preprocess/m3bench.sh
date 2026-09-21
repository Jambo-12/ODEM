#!/usr/bin/env bash
# Preprocess M3-Bench 30-second captions, visual embeddings, and four-key text embeddings.
# Captioning and visual embedding reuse one GPU; text embeddings use an API.
# This script scans only source videos and never reads official subtitles or QA annotations.
#
# Full single-GPU run:
#   bash scripts/preprocess/m3bench.sh
#
# Single-video validation:
#   bash scripts/preprocess/m3bench.sh --gpu_id 0 --video_name living_room_06

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/preprocess/common.sh"

DATASET_ID="m3bench"
CONFIG_VIDEO_DIR="$(preprocess_config_get "${DATASET_ID}" video_dir)"
if [[ -n "${M3_BENCH_VIDEO_DIR:-}" ]]; then
    VIDEO_DIR="${M3_BENCH_VIDEO_DIR}"
elif [[ -n "${M3_BENCH_ROOT:-}" ]]; then
    VIDEO_DIR="${M3_BENCH_ROOT}/videos"
else
    VIDEO_DIR="${CONFIG_VIDEO_DIR}"
fi
PREPROCESS_DIR="${M3_BENCH_PREPROCESS_DIR:-$(preprocess_config_get "${DATASET_ID}" output_dir)}"
EMBEDDING_CONFIG="${ODEM_PREPROCESS_CONFIG:-${PREPROCESS_CONFIG_PATH}}"
CAPTION_MODEL_PATH="${ODEM_CAPTION_MODEL_PATH:-$(preprocess_config_get "${DATASET_ID}" models.caption.path)}"
VICLIP_MODEL_PATH="${ODEM_VICLIP_MODEL_PATH:-$(preprocess_config_get "${DATASET_ID}" models.visual_embedding.checkpoint)}"
CAPTION_PROMPT="${M3_BENCH_CAPTION_PROMPT:-$(preprocess_config_get "${DATASET_ID}" caption_prompt)}"
EXPECTED_VIDEO_COUNT="${M3_BENCH_EXPECTED_VIDEO_COUNT:-$(preprocess_config_get "${DATASET_ID}" expected_video_count)}"
GPU_ID="${PREPROCESS_GPU_ID:-$(preprocess_config_get "${DATASET_ID}" gpu_id)}"
TEXT_MODEL="${TEXT_EMBEDDING_MODEL:-$(preprocess_config_get "${DATASET_ID}" models.text_embedding.name)}"
TEXT_BATCH_SIZE="${TEXT_EMBEDDING_BATCH_SIZE:-$(preprocess_config_get "${DATASET_ID}" models.text_embedding.batch_size)}"
FRAMES_PER_FEAT="${VISUAL_FRAMES_PER_FEAT:-$(preprocess_config_get "${DATASET_ID}" models.visual_embedding.frames_per_segment)}"
CLIP_SECONDS="$(preprocess_config_get "${DATASET_ID}" clip_seconds)"
VIDEO_NAME=""


show_help() {
    # Show script options, default paths, and common commands.
    cat <<'EOF'
Usage:
  bash scripts/preprocess/m3bench.sh [options]

Options:
  --gpu_id ID                Physical GPU shared by captioning and visual embedding; default: 0
  --video_name NAME          Process one video name without its extension
  --embedding_config PATH    Shared preprocessing config; API keys come from the environment
  --text_model NAME          Text embedding model name
  --text_batch_size N        Text embedding request batch size
  --frames_per_feat N        Frames sampled for each caption segment
  -h, --help                 Show this help

Full run:
  bash scripts/preprocess/m3bench.sh

Single-video validation:
  bash scripts/preprocess/m3bench.sh --gpu_id 0 --video_name living_room_06

Override defaults with M3_BENCH_ROOT, M3_BENCH_VIDEO_DIR,
M3_BENCH_PREPROCESS_DIR, M3_BENCH_EXPECTED_VIDEO_COUNT, PREPROCESS_GPU_ID,
ODEM_PREPROCESS_CONFIG, and ODEM_PYTHON_BIN.
This script never reads official M3-Bench subtitles or QA annotations.
EOF
}


parse_arguments() {
    # Parse M3-Bench options without modifying shared configuration files.
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpu_id)
                [[ $# -ge 2 ]] || { echo "[ERROR] --gpu_id requires a value"; exit 2; }
                GPU_ID="$2"
                shift 2
                ;;
            --video_name)
                [[ $# -ge 2 ]] || { echo "[ERROR] --video_name requires a value"; exit 2; }
                VIDEO_NAME="$2"
                shift 2
                ;;
            --embedding_config|--config)
                [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value"; exit 2; }
                EMBEDDING_CONFIG="$2"
                shift 2
                ;;
            --text_model)
                [[ $# -ge 2 ]] || { echo "[ERROR] --text_model requires a value"; exit 2; }
                TEXT_MODEL="$2"
                shift 2
                ;;
            --text_batch_size)
                [[ $# -ge 2 ]] || { echo "[ERROR] --text_batch_size requires a value"; exit 2; }
                TEXT_BATCH_SIZE="$2"
                shift 2
                ;;
            --frames_per_feat)
                [[ $# -ge 2 ]] || { echo "[ERROR] --frames_per_feat requires a value"; exit 2; }
                FRAMES_PER_FEAT="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "[ERROR] Unknown option: $1"
                show_help
                exit 2
                ;;
        esac
    done
}


check_positive_integer() {
    # Validate positive integer arguments such as counts, batch sizes, and frames.
    local value="$1"
    local name="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${name} must be a positive integer; got: ${value}"
        exit 2
    fi
}


video_exists() {
    # Single-video mode accepts all three extensions supported by captioning.
    local extension
    for extension in mp4 mkv webm; do
        if [[ -f "${VIDEO_DIR}/${VIDEO_NAME}.${extension}" ]]; then
            return 0
        fi
    done
    return 1
}


count_mp4_videos() {
    # The published M3-Bench layout stores MP4 files directly under videos/.
    local count=0
    while IFS= read -r -d '' _; do
        ((count += 1))
    done < <(find "${VIDEO_DIR}" -maxdepth 1 -type f -iname '*.mp4' -print0)
    echo "${count}"
}


check_runtime() {
    # Validate the interpreter, programs, dataset, config, and GPU before model loading.
    local required_file
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "[ERROR] Python interpreter is not executable: ${PYTHON_BIN}"
        exit 1
    fi
    if [[ ! -d "${VIDEO_DIR}" ]]; then
        echo "[ERROR] M3-Bench video directory does not exist: ${VIDEO_DIR}"
        exit 1
    fi
    for required_file in caption.py segment_feats_text.py segment_feats_vis.py; do
        if [[ ! -f "${PROJECT_DIR}/${required_file}" ]]; then
            echo "[ERROR] Missing shared program: ${PROJECT_DIR}/${required_file}"
            exit 1
        fi
    done
    if [[ ! -f "${EMBEDDING_CONFIG}" ]]; then
        echo "[ERROR] Shared preprocessing config does not exist: ${EMBEDDING_CONFIG}"
        exit 1
    fi
    if [[ ! -d "${CAPTION_MODEL_PATH}" ]]; then
        echo "[ERROR] Caption model directory does not exist: ${CAPTION_MODEL_PATH}"
        exit 1
    fi
    if [[ ! -f "${VICLIP_MODEL_PATH}" ]]; then
        echo "[ERROR] ViCLIP checkpoint does not exist: ${VICLIP_MODEL_PATH}"
        exit 1
    fi
    if [[ ! -f "${CAPTION_PROMPT}" ]]; then
        echo "[ERROR] Caption prompt does not exist: ${CAPTION_PROMPT}"
        exit 1
    fi
    if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU ID: ${GPU_ID}"
        exit 2
    fi
    check_positive_integer "${EXPECTED_VIDEO_COUNT}" "expected_video_count"
    check_positive_integer "${TEXT_BATCH_SIZE}" "text_batch_size"
    check_positive_integer "${FRAMES_PER_FEAT}" "frames_per_feat"

    if [[ -n "${VIDEO_NAME}" ]]; then
        if ! video_exists; then
            echo "[ERROR] Requested video was not found: ${VIDEO_NAME}"
            exit 1
        fi
    else
        local actual_video_count
        actual_video_count="$(count_mp4_videos)"
        if [[ "${actual_video_count}" -ne "${EXPECTED_VIDEO_COUNT}" ]]; then
            echo "[ERROR] Found ${actual_video_count} MP4 files; expected ${EXPECTED_VIDEO_COUNT}"
            exit 1
        fi
    fi
}


build_scope_arguments() {
    # Build scope arguments shared by full and single-video modes.
    SCOPE_ARGUMENTS=()
    COUNT_ARGUMENTS=()
    if [[ -n "${VIDEO_NAME}" ]]; then
        SCOPE_ARGUMENTS+=(--video_name "${VIDEO_NAME}")
    else
        COUNT_ARGUMENTS+=(--expected_video_count "${EXPECTED_VIDEO_COUNT}")
    fi
}


run_preflight() {
    # Inspect source videos and current outputs without loading the caption model.
    echo "[Preflight] Checking M3-Bench videos and caption task scope"
    "${PYTHON_BIN}" "${PROJECT_DIR}/caption.py" \
        --dataset_name M3-Bench \
        --video_dir "${VIDEO_DIR}" \
        --output_dir "${PREPROCESS_DIR}" \
        --clip_seconds "${CLIP_SECONDS}" \
        --model "${CAPTION_MODEL_PATH}" \
        --prompt_path "${CAPTION_PROMPT}" \
        "${COUNT_ARGUMENTS[@]}" \
        "${SCOPE_ARGUMENTS[@]}" \
        --check_only
}


run_caption() {
    # Process videos sequentially and skip complete captions.
    echo "[Stage 1/3] Generating 30-second captions on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" "${PROJECT_DIR}/caption.py" \
        --dataset_name M3-Bench \
        --video_dir "${VIDEO_DIR}" \
        --output_dir "${PREPROCESS_DIR}" \
        --clip_seconds "${CLIP_SECONDS}" \
        --model "${CAPTION_MODEL_PATH}" \
        --prompt_path "${CAPTION_PROMPT}" \
        "${COUNT_ARGUMENTS[@]}" \
        "${SCOPE_ARGUMENTS[@]}"
}


verify_caption_outputs() {
    # Stop on incomplete captions or failure markers to prevent vector misalignment.
    echo "[Validation 1/2] Checking caption completeness"
    "${PYTHON_BIN}" "${PROJECT_DIR}/caption.py" \
        --dataset_name M3-Bench \
        --video_dir "${VIDEO_DIR}" \
        --output_dir "${PREPROCESS_DIR}" \
        --clip_seconds "${CLIP_SECONDS}" \
        --model "${CAPTION_MODEL_PATH}" \
        --prompt_path "${CAPTION_PROMPT}" \
        "${COUNT_ARGUMENTS[@]}" \
        "${SCOPE_ARGUMENTS[@]}" \
        --check_only \
        --require_complete
}


run_visual_embedding() {
    # Read source videos by caption time range without using subtitles or QA annotations.
    echo "[Stage 2/3] Generating visual embeddings on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" "${PROJECT_DIR}/segment_feats_vis.py" \
        --video_dir "${VIDEO_DIR}" \
        --base_dir "${PREPROCESS_DIR}" \
        --model_path "${VICLIP_MODEL_PATH}" \
        "${COUNT_ARGUMENTS[@]}" \
        "${SCOPE_ARGUMENTS[@]}" \
        --frames_per_feat "${FRAMES_PER_FEAT}" \
        --device cuda
}


run_text_embedding() {
    # Encode environment, event, attention, and summary independently.
    echo "[Stage 2/3] Generating four-key text embeddings"
    "${PYTHON_BIN}" "${PROJECT_DIR}/segment_feats_text.py" \
        --config "${EMBEDDING_CONFIG}" \
        --preprocess_dir "${PREPROCESS_DIR}" \
        "${SCOPE_ARGUMENTS[@]}" \
        --model "${TEXT_MODEL}" \
        --batch_size "${TEXT_BATCH_SIZE}"
}


wait_embedding_tasks() {
    # Preserve both parallel exit codes and stop if either task fails.
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


verify_all_outputs() {
    # Validate all three outputs, segment counts, and both vector timelines.
    echo "[Stage 3/3] Validating captions and both embedding structures"
    "${PYTHON_BIN}" - "${VIDEO_DIR}" "${PREPROCESS_DIR}" \
        "${VIDEO_NAME}" "${EXPECTED_VIDEO_COUNT}" <<'PY'
import glob
import json
import os
import pickle
import sys


def parse_time(value):
    parts = [int(part) for part in str(value).split(":")]
    if len(parts) != 3:
        raise ValueError("Invalid timestamp: {}".format(value))
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


video_dir, preprocess_dir, video_name, expected_text = sys.argv[1:]
video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
if video_name:
    video_paths = [
        path for path in video_paths
        if os.path.splitext(os.path.basename(path))[0] == video_name
    ]
elif len(video_paths) != int(expected_text):
    raise SystemExit(
        "[ERROR] Final validation found {} videos; expected {}".format(
            len(video_paths), expected_text
        )
    )

failures = []
required_keys = (
    "start_time", "end_time", "environment", "event", "attention", "summary"
)
for video_path in video_paths:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(preprocess_dir, stem)
    caption_path = os.path.join(output_dir, "captions.json")
    text_path = os.path.join(output_dir, "segment_textual_embedding.pkl")
    visual_path = os.path.join(output_dir, "segment_visual_embedding.pkl")
    try:
        for required_path in (caption_path, text_path, visual_path):
            if not os.path.isfile(required_path):
                raise FileNotFoundError("Missing {}".format(required_path))
        with open(caption_path, "r", encoding="utf-8") as file_obj:
            captions = json.load(file_obj)
        with open(text_path, "rb") as file_obj:
            text_payload = pickle.load(file_obj)
        with open(visual_path, "rb") as file_obj:
            visual_payload = pickle.load(file_obj)

        if not captions:
            raise ValueError("captions.json is empty")
        if any(tuple(item.keys()) != required_keys for item in captions):
            raise ValueError("Caption fields or order do not match the four-key schema")
        starts = [parse_time(item["start_time"]) for item in captions]
        ends = [parse_time(item["end_time"]) for item in captions]
        segment_count = len(captions)
        for name, payload in (
            ("Text embeddings", text_payload),
            ("Visual embeddings", visual_payload),
        ):
            embeddings = payload.get("embeddings")
            if getattr(embeddings, "shape", (0,))[0] != segment_count:
                raise ValueError("{} segment count does not match captions".format(name))
            if list(payload.get("segment_ids", [])) != list(range(segment_count)):
                raise ValueError("{} segment_ids are not contiguous".format(name))
            if list(payload.get("start_times", [])) != starts:
                raise ValueError("{} start timeline does not match captions".format(name))
            if list(payload.get("end_times", [])) != ends:
                raise ValueError("{} end timeline does not match captions".format(name))
        if getattr(text_payload.get("embeddings"), "ndim", 0) != 3:
            raise ValueError("Text embeddings do not use the 3D multi-key schema")
        if text_payload.get("caption_keys") != [
            "environment", "event", "attention", "summary"
        ]:
            raise ValueError("Invalid text-embedding caption-key order")
    except Exception as error:
        failures.append("{}: {}".format(stem, error))

if failures:
    print("[ERROR] Final validation failed:")
    for failure in failures:
        print("  - {}".format(failure))
    raise SystemExit(1)
print("[DONE] Final validation passed for {} videos".format(len(video_paths)))
PY
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
        --text_batch_size "${TEXT_BATCH_SIZE}" \
        --viclip_checkpoint "${VICLIP_MODEL_PATH}" \
        --frames_per_segment "${FRAMES_PER_FEAT}"
}


main() {
    # Run preflight, single-GPU captioning, parallel embeddings, and final validation.
    parse_arguments "$@"
    check_runtime
    build_scope_arguments
    mkdir -p "${PREPROCESS_DIR}"

    echo "M3-Bench video directory: ${VIDEO_DIR}"
    echo "Preprocessing output directory: ${PREPROCESS_DIR}"
    echo "Shared config: ${EMBEDDING_CONFIG}"
    echo "Caption model: ${CAPTION_MODEL_PATH}"
    echo "ViCLIP checkpoint: ${VICLIP_MODEL_PATH}"
    echo "Preprocessing GPU: ${GPU_ID}"
    echo "Official subtitles and QA annotations: disabled"

    run_preflight
    run_caption
    verify_caption_outputs

    run_visual_embedding &
    visual_pid=$!
    run_text_embedding &
    text_pid=$!
    wait_embedding_tasks "${visual_pid}" "${text_pid}"

    verify_all_outputs
    write_preprocess_manifest
    echo "[DONE] M3-Bench captions, visual embeddings, and text embeddings are ready"
}


main "$@"
