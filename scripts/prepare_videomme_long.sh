#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${ODEM_PYTHON_BIN:-python}"
SOURCE_ROOT="${VIDEOMME_SOURCE_ROOT:-data/raw/videomme/source}"
ANNOTATION_ROOT="${SOURCE_ROOT}/annotations"
ARCHIVE_ROOT="${SOURCE_ROOT}/official"
ANNOTATION_PARQUET="${ANNOTATION_ROOT}/data/test-00000-of-00001.parquet"
ANNOTATION_JSON="data/raw/videomme/videoMME_long.json"
VIDEO_ROOT="data/raw/videomme/videos"


check_dependencies() {
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Python executable not found: ${PYTHON_BIN}" >&2
        exit 1
    fi
    if ! command -v hf >/dev/null 2>&1; then
        echo "Missing Hugging Face CLI. Install huggingface_hub first." >&2
        exit 1
    fi
    if ! "${PYTHON_BIN}" -c "import pyarrow.parquet" >/dev/null 2>&1; then
        echo "Missing PyArrow. Install pyarrow first." >&2
        exit 1
    fi
}


download_annotations() {
    mkdir -p "${ANNOTATION_ROOT}"
    hf download topyun/Video-MME-Long \
        data/test-00000-of-00001.parquet \
        --repo-type dataset \
        --local-dir "${ANNOTATION_ROOT}"
}


convert_annotations() {
    "${PYTHON_BIN}" - "${ANNOTATION_PARQUET}" "${ANNOTATION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

source = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = pq.read_table(source).to_pylist()
rows = [row for row in rows if str(row.get("duration", "")).lower() == "long"]
video_ids = {str(row["videoID"]) for row in rows}

if len(rows) != 900 or len(video_ids) != 300:
    raise RuntimeError(
        "Expected 900 questions and 300 videos, found "
        f"{len(rows)} questions and {len(video_ids)} videos"
    )

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as file:
    json.dump(rows, file, ensure_ascii=False, indent=2)

print(f"Wrote {len(rows)} questions for {len(video_ids)} videos to {output}")
PY
}


download_video_archives() {
    mkdir -p "${ARCHIVE_ROOT}"
    hf download lmms-lab/Video-MME \
        --repo-type dataset \
        --include "videos_chunked_*.zip" \
        --local-dir "${ARCHIVE_ROOT}"
}


extract_long_videos() {
    "${PYTHON_BIN}" - "${ANNOTATION_JSON}" "${ARCHIVE_ROOT}" "${VIDEO_ROOT}" <<'PY'
import json
import shutil
import sys
import zipfile
from pathlib import Path

annotation_path = Path(sys.argv[1])
archive_root = Path(sys.argv[2])
video_root = Path(sys.argv[3])

with annotation_path.open("r", encoding="utf-8") as file:
    annotations = json.load(file)

wanted = {str(row["videoID"]) for row in annotations}
archives = sorted(archive_root.glob("videos_chunked_*.zip"))
if not archives:
    raise FileNotFoundError(f"No videos_chunked_*.zip files found in {archive_root}")

video_root.mkdir(parents=True, exist_ok=True)
for archive_path in archives:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename).name
            path = Path(filename)
            video_id = path.stem
            if path.suffix.lower() != ".mp4" or video_id not in wanted:
                continue
            destination = video_root / f"{video_id}.mp4"
            if destination.exists():
                continue
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
PY
}


validate_prepared_data() {
    "${PYTHON_BIN}" - "${ANNOTATION_JSON}" "${VIDEO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

annotation_path = Path(sys.argv[1])
video_root = Path(sys.argv[2])

with annotation_path.open("r", encoding="utf-8") as file:
    annotations = json.load(file)

expected = {str(row["videoID"]) for row in annotations}
actual = {
    path.stem
    for path in video_root.iterdir()
    if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm"}
}

if len(annotations) != 900:
    raise RuntimeError(f"Expected 900 questions, found {len(annotations)}")
if len(expected) != 300:
    raise RuntimeError(f"Expected 300 video IDs, found {len(expected)}")
if actual != expected:
    raise RuntimeError(
        f"Video mismatch: missing={sorted(expected - actual)[:10]}, "
        f"unexpected={sorted(actual - expected)[:10]}"
    )

print("Video-MME Long validation passed: 300 videos and 900 questions")
PY
}


main() {
    check_dependencies
    download_annotations
    convert_annotations
    download_video_archives
    extract_long_videos
    validate_prepared_data
}


main
