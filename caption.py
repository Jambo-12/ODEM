'''
    Lightweight first-view captioning: split long videos into continuous 30-second segments.

    Single-GPU run:
     CUDA_VISIBLE_DEVICES=6 \
    python caption.py \
    --video_name "YcU9d-RPEog" \
    --output_dir data/processed/videomme/qwen2_5_vl_7b_30s

    Specify the dataset and output directory:
    python caption.py \
        --video_dir data/raw/videomme/videos \
        --output_dir /path/to/bionic_captions

    Optional annotation-manifest mode:
    python caption.py \
        --config /path/to/caption_config.yaml

    Save each video result to: <output_dir>/<video_name>/captions.json
'''
import os
import json
import torch
import argparse
import tempfile
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import cv2
from decord import VideoReader, cpu

import time
import shutil
import re
import math
from typing import Any, Dict, List, Optional, Tuple


# ========= Basic utilities =========

CAPTION_KEYS = ("environment", "event", "attention", "summary")
OUTPUT_KEYS = ("start_time", "end_time") + CAPTION_KEYS
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "Qwen2.5-VL-7B-Instruct")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "videomme",
    "qwen2_5_vl_7b_30s",
)
DEFAULT_VIDEO_DIR = os.path.join(PROJECT_DIR, "data", "raw", "videomme", "videos")
DEFAULT_PROMPT_PATH = os.path.join(
    PROJECT_DIR,
    "prompts",
    "prompt_caption.txt",
)
DEFAULT_MAX_WORDS_PER_KEY = 30
DEFAULT_MAX_NEW_TOKENS = 768
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm")
ANNOTATION_CONSISTENCY_FIELDS = (
    "video_path",
    "subtitle_path",
    "starting_timestamp_for_subtitles",
    "duration",
)

SegmentRange = Tuple[float, float]
CaptionContent = Dict[str, Optional[str]]
CaptionRecord = Dict[str, Optional[str]]


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to a fixed HH:MM:SS string.

    Container durations are often fractional, while timestamps are second-aligned.
    Rounding up keeps the final label at or after the true endpoint.
    """
    total_seconds = max(0, int(math.ceil(float(seconds) - 1e-6)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_video_duration(video_path: str) -> float:
    """
    Read the true video duration with Decord and return seconds.

    Compute duration from frame count and average FPS for continuous segmentation.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    del vr

    if video_fps <= 0 or total_frames <= 0:
        raise ValueError(f"Could not read a valid video duration: {video_path}")
    return total_frames / video_fps


def build_segment_ranges(video_duration: float, clip_seconds: int = 30) -> List[SegmentRange]:
    """
    Generate chronological intervals that cover the entire video.

    Every interval except the last has clip_seconds duration. The last interval ends
    at the true video endpoint so no tail content is omitted.
    """
    if video_duration <= 0:
        return []
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be greater than 0")

    segment_count = int(math.ceil(video_duration / clip_seconds))
    return [
        (segment_index * clip_seconds, min((segment_index + 1) * clip_seconds, video_duration))
        for segment_index in range(segment_count)
    ]


def extract_segment_decord(
    video_path: str,
    start: float,
    end: float,
    sample_fps: float,
    save_path: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Extract [start, end) with Decord and create a lightweight temporary video.

    Uniformly sample at sample_fps instead of loading every source frame. Samples remain
    chronological and cover the beginning, middle, and end of the segment.

    Args:
      - video_path: source video path;
      - start, end: segment boundaries in seconds;
      - sample_fps: sampling rate for the vision model;
      - save_path: temporary output path, created automatically when None.

    Returns:
      - (success, temporary video path).
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0")

    if save_path is None:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="seg_")
        os.close(tmp_fd)
    else:
        tmp_path = save_path

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = float(vr.get_avg_fps())
        total_frames = len(vr)

        start_fid = max(0, int(math.floor(start * video_fps)))
        end_fid = min(total_frames - 1, int(math.ceil(end * video_fps)) - 1)
        if start_fid > end_fid:
            return False, None

        # Keep at least three frames so a short tail remains a valid video input.
        sample_count = max(3, int(math.ceil((end - start) * sample_fps)))
        frame_ids = np.linspace(start_fid, end_fid, num=sample_count).round().astype(np.int64)
        frames = vr.get_batch(frame_ids.tolist()).asnumpy()

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"), # type: ignore[attr-defined]
            float(sample_fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create temporary video: {tmp_path}")

        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        del frames
        del vr
        import gc
        gc.collect()
        return True, tmp_path

    except Exception as e:
        print(f"[Decord error] {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False, None


def check_video_valid(video_path: str, min_frames: int = 3) -> bool:
    """
    Verify that the temporary video is readable and has enough frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frames >= min_frames


# ========= Qwen captioning =========

def init_qwen(model_name: str = DEFAULT_MODEL_PATH, device: str = "cuda") -> Tuple[Any, Any]:
    """
    Initialize the Qwen2.5-VL model and processor.

    CUDA mode uses bfloat16, Flash Attention, and automatic device mapping.
    CPU mode uses float32 for interface checks without a GPU.
    """
    print(f"Loading Qwen2.5-VL model from {model_name} ...")
    if not os.path.isdir(model_name):
        raise FileNotFoundError(f"Model directory does not exist: {model_name}")

    model_kwargs: Dict[str, Any] = {
        "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }
    if device == "cuda":
        model_kwargs.update({
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
        })
    else:
        # Place the model directly on the target device during loading.
        model_kwargs["device_map"] = device

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    print("✅ Qwen2.5-VL loaded.")
    return processor, model


def build_caption_prompt(
    max_words_per_key: int = DEFAULT_MAX_WORDS_PER_KEY,
    prompt_path: str = DEFAULT_PROMPT_PATH,
) -> str:
    """
    Read the coarse first-view caption prompt template from a file.

    The prompt records only the environment, central event, one attention focus, and a
    short summary. Each segment is observed independently. Datasets may provide a custom
    template without changing generation logic for other datasets.
    """
    if max_words_per_key <= 0:
        raise ValueError("max_words_per_key must be greater than 0")
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Caption prompt does not exist: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read().strip()

    return (
        template
        .replace("{max_words_per_key}", str(max_words_per_key))
        .replace("{max_total_words}", str(max_words_per_key * len(CAPTION_KEYS)))
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first complete JSON object from the model response.

    Normal output contains only JSON, but tolerate an occasional Markdown fence or prefix.
    """
    if not text:
        raise ValueError("Model response is empty")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    decoder = json.JSONDecoder()
    object_start = cleaned.find("{")
    if object_start < 0:
        raise ValueError("Model response contains no JSON object")

    parsed, _ = decoder.raw_decode(cleaned[object_start:])
    if not isinstance(parsed, dict):
        raise ValueError("Model returned JSON that is not an object")
    return parsed


def validate_caption_content(record: Dict[str, Any]) -> CaptionContent:
    """
    Validate and normalize the four generated caption fields.

    Accept only the specified fields and value types without inventing unobserved facts.
    event may be null; all other fields must be nonempty English strings. Word limits are
    prompt-controlled and are not enforced here.
    """
    if set(record.keys()) != set(CAPTION_KEYS):
        missing = sorted(set(CAPTION_KEYS) - set(record.keys()))
        extra = sorted(set(record.keys()) - set(CAPTION_KEYS))
        raise ValueError(f"Invalid fields; missing={missing}, extra={extra}")

    normalized: CaptionContent = {}
    for key in CAPTION_KEYS:
        value = record[key]
        if key == "event" and value is None:
            normalized[key] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field {key} must be a nonempty string")
        normalized[key] = " ".join(value.strip().split())

    return normalized


def _generate_qwen_text(
    model: Any,
    processor: Any,
    video_path: str,
    fps: float,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """
    Run one Qwen2.5-VL generation and return the decoded raw text.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": float(fps),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, raw_video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )
    # Normalize an optional helper result before expanding it with **.
    video_kwargs: Dict[str, Any] = raw_video_kwargs or {}
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def generate_caption_record(
    model: Any,
    processor: Any,
    video_path: str,
    fps: float,
    max_words_per_key: int = DEFAULT_MAX_WORDS_PER_KEY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_retries: int = 2,
    prompt_path: str = DEFAULT_PROMPT_PATH,
) -> CaptionContent:
    """
    Generate and validate a four-field first-view caption for one video segment.

    Normally call the model once. Retry unchanged visual evidence up to max_retries when
    JSON or fields are invalid. Raise after all attempts to avoid writing corrupt data.
    """
    prompt = build_caption_prompt(
        max_words_per_key=max_words_per_key,
        prompt_path=prompt_path,
    )
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            output_text = _generate_qwen_text(
                model=model,
                processor=processor,
                video_path=video_path,
                fps=fps,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            return validate_caption_content(extract_json_object(output_text))
        except (ValueError, TypeError) as e:
            last_error = e
            tqdm.write(f"[WARN] Output validation failed on attempt {attempt}/{max_retries}: {e}")

    raise RuntimeError(f"Caption remains invalid after {max_retries} attempts: {last_error}")


# ========= Result persistence and main processing =========

def save_json_atomic(data: Any, output_path: str) -> None:
    """
    Save JSON with atomic replacement to avoid partial files after interruption.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".caption_", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_partial_captions(
    partial_json: str,
    segment_ranges: List[SegmentRange],
) -> List[CaptionRecord]:
    """
    Read and validate a checkpoint, restoring only records matching current segments.

    Reject corrupt checkpoints or mismatched timelines to prevent mixing configurations.
    """
    if not os.path.exists(partial_json):
        return []

    with open(partial_json, "r", encoding="utf-8") as f:
        captions: Any = json.load(f)
    if not isinstance(captions, list) or len(captions) > len(segment_ranges):
        raise ValueError(f"Invalid checkpoint file: {partial_json}")

    for segment_index, record in enumerate(captions):
        start, end = segment_ranges[segment_index]
        if not isinstance(record, dict) or tuple(record.keys()) != OUTPUT_KEYS:
            raise ValueError(f"Invalid fields in checkpoint record {segment_index}")
        if record["start_time"] != format_timestamp(start) or record["end_time"] != format_timestamp(end):
            raise ValueError(f"Checkpoint record {segment_index} does not match the current timeline")
        validate_caption_content({key: record[key] for key in CAPTION_KEYS})
    return captions


def caption_video_segments(
    video_path: str,
    output_json: str,
    clip_seconds: int,
    fps: float,
    model: Any,
    processor: Any,
    max_words_per_key: int = DEFAULT_MAX_WORDS_PER_KEY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_retries: int = 2,
    prompt_path: str = DEFAULT_PROMPT_PATH,
) -> None:
    """
    Run complete first-view captioning for one video.

    Read duration, generate continuous 30-second intervals, create one four-field record
    per segment, and checkpoint after each segment. Atomically finalize captions.json.
    """
    video_duration = get_video_duration(video_path)
    segment_ranges = build_segment_ranges(video_duration, clip_seconds=clip_seconds)
    partial_json = output_json + ".partial.json"
    captions = load_partial_captions(partial_json, segment_ranges)
    tmp_dir = tempfile.mkdtemp(prefix="segcap_")

    print(f"\n▶ Processing {os.path.basename(video_path)}")
    print(f"▶ Duration: {video_duration:.2f}s, segments: {len(segment_ranges)}, resumed: {len(captions)}")
    print(f"▶ Temporary directory: {tmp_dir}")

    try:
        remaining_ranges = segment_ranges[len(captions):]
        progress = tqdm(
            enumerate(remaining_ranges, start=len(captions)),
            total=len(remaining_ranges),
            desc="Generating captions",
            unit="segment",
        )
        for segment_index, (start, end) in progress:
            tmp_path = os.path.join(tmp_dir, f"seg_{segment_index:05d}.mp4")
            ok, out_path = extract_segment_decord(
                video_path,
                start,
                end,
                sample_fps=fps,
                save_path=tmp_path,
            )
            if not ok or not out_path or not os.path.exists(out_path):
                raise RuntimeError(f"Failed to extract segment {segment_index}: {start:.2f}s-{end:.2f}s")
            if not check_video_valid(out_path, min_frames=3):
                raise RuntimeError(f"Segment {segment_index} is too short or corrupt")

            content = generate_caption_record(
                model=model,
                processor=processor,
                video_path=out_path,
                fps=fps,
                max_words_per_key=max_words_per_key,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
                prompt_path=prompt_path,
            )
            record = {
                "start_time": format_timestamp(start),
                "end_time": format_timestamp(end),
                **content,
            }
            captions.append(record)
            save_json_atomic(captions, partial_json)
            os.remove(out_path)
            tqdm.write(
                f"[Segment {segment_index}] {record['start_time']}-{record['end_time']} → {record['summary']}"
            )

        os.replace(partial_json, output_json)
        print(f"✅ Saved captions to {output_json}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"🧹 Cleaned temporary directory: {tmp_dir}")


# ========= Checkpoint task-lock utilities =========

def acquire_lock(lock_path: str) -> bool:
    """
    Atomically create a lock containing the current PID; return False if already locked.
    """
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def is_stale_lock(lock_path: str, stale_seconds: float = 300.0) -> bool:
    """
    Determine whether the lock is stale because its owner no longer exists.

    A dead PID is stale. Empty legacy locks expire after stale_seconds. Treat an existing
    but inaccessible process as active to avoid deleting a valid lock.
    """
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError:
        return False

    if content.isdigit():
        pid = int(content)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    try:
        lock_age = time.time() - os.path.getmtime(lock_path)
    except OSError:
        return False
    return lock_age > stale_seconds


def release_lock(lock_path: str) -> None:
    """
    Release the process lock for the current video.
    """
    if os.path.exists(lock_path):
        os.remove(lock_path)


# ========= Dataset task adaptation =========

def resolve_dataset_path(root_dir: str, relative_path: str, field_name: str) -> str:
    """
    Safely resolve an annotation-relative path inside the dataset directory.

    Annotation manifests may store relative video and subtitle paths. Reject absolute paths
    and traversal so malformed annotations cannot escape the dataset directory.
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"Annotation field {field_name} must be a nonempty string")
    if os.path.isabs(relative_path):
        raise ValueError(f"Annotation field {field_name} cannot use an absolute path: {relative_path}")

    root_path = os.path.realpath(root_dir)
    resolved_path = os.path.realpath(os.path.join(root_path, relative_path))
    try:
        inside_root = os.path.commonpath([root_path, resolved_path]) == root_path
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError(f"Annotation field {field_name} escapes the dataset directory: {relative_path}")
    return resolved_path


def load_annotation_video_tasks(
    video_root: str,
    annotation_files: List[str],
    subtitle_root: str = "",
    video_name: str = "",
) -> List[Tuple[str, str]]:
    """
    Build deduplicated video tasks from one or more annotation files.

    Use annotation video_id as the task name and video_path for reading. This preserves
    official IDs when filenames differ. Caption each video once and verify consistent metadata.
    """
    if not annotation_files:
        raise ValueError("annotation_files cannot be empty")

    signatures: Dict[str, Tuple[Any, ...]] = {}
    task_paths: Dict[str, str] = {}
    file_stems: Dict[str, str] = {}
    video_path_owners: Dict[str, str] = {}

    for annotation_file in annotation_files:
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(f"Annotation file does not exist: {annotation_file}")
        with open(annotation_file, "r", encoding="utf-8") as f:
            records: Any = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Annotation file root must be a list: {annotation_file}")

        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Record {record_index} in {annotation_file} is not an object"
                )

            raw_video_id = record.get("video_id")
            if not isinstance(raw_video_id, str) or not raw_video_id.strip():
                raise ValueError(
                    f"Record {record_index} in {annotation_file} has no valid video_id"
                )
            task_name = raw_video_id.strip()
            if os.path.basename(task_name) != task_name or task_name in (".", ".."):
                raise ValueError(f"video_id cannot contain a directory path: {task_name}")

            raw_video_path = record.get("video_path")
            video_path = resolve_dataset_path(video_root, raw_video_path, "video_path")
            file_stem, extension = os.path.splitext(os.path.basename(video_path))
            if extension.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                raise ValueError(f"Annotation references an unsupported video format: {raw_video_path}")
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Annotation video does not exist: {video_path}")

            if subtitle_root:
                subtitle_path = resolve_dataset_path(
                    subtitle_root,
                    record.get("subtitle_path"),
                    "subtitle_path",
                )
                if not os.path.isfile(subtitle_path):
                    raise FileNotFoundError(f"Annotation subtitle does not exist: {subtitle_path}")

            signature = tuple(record.get(key) for key in ANNOTATION_CONSISTENCY_FIELDS)
            if task_name in signatures and signatures[task_name] != signature:
                raise ValueError(f"Inconsistent metadata for one video_id: {task_name}")
            if video_path in video_path_owners and video_path_owners[video_path] != task_name:
                raise ValueError(
                    f"One video file is referenced by multiple video_ids: "
                    f"{video_path_owners[video_path]} and {task_name}"
                )

            signatures[task_name] = signature
            task_paths[task_name] = video_path
            file_stems[task_name] = file_stem
            video_path_owners[video_path] = task_name

    tasks = []
    for task_name in sorted(task_paths):
        if video_name and video_name not in (task_name, file_stems[task_name]):
            continue
        tasks.append((task_name, task_paths[task_name]))
    return tasks


def list_video_tasks(
    video_root: str,
    video_name: str = "",
    annotation_files: Optional[List[str]] = None,
    subtitle_root: str = "",
) -> List[Tuple[str, str]]:
    """
    Return videos sorted by task name for the selected input mode.

    With annotation_files, filter by the manifest and use video_id as the output directory.
    Otherwise scan every video under video_root to preserve Video-MME behavior.
    """
    if annotation_files:
        return load_annotation_video_tasks(
            video_root=video_root,
            annotation_files=annotation_files,
            subtitle_root=subtitle_root,
            video_name=video_name,
        )

    tasks: List[Tuple[str, str]] = []
    for filename in sorted(os.listdir(video_root)):
        stem, extension = os.path.splitext(filename)
        if extension.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        if video_name and stem != video_name:
            continue
        tasks.append((stem, os.path.join(video_root, filename)))
    return tasks


def summarize_task_outputs(
    video_tasks: List[Tuple[str, str]],
    output_root: str,
) -> Dict[str, int]:
    """
    Count completed, failed, partial, locked, and pending target tasks.

    Count markers independently to reveal stale failures beside completed captions.
    pending means the task directory has no known output or runtime marker.
    """
    summary = {
        "total": len(video_tasks),
        "completed": 0,
        "failed": 0,
        "partial": 0,
        "locked": 0,
        "pending": 0,
    }
    for task_name, _ in video_tasks:
        task_dir = os.path.join(output_root, task_name)
        markers = {
            "completed": os.path.join(task_dir, "captions.json"),
            "failed": os.path.join(task_dir, ".failed"),
            "partial": os.path.join(task_dir, "captions.json.partial.json"),
            "locked": os.path.join(task_dir, ".lock"),
        }
        existing_markers = []
        for status_name, marker_path in markers.items():
            if os.path.exists(marker_path):
                summary[status_name] += 1
                existing_markers.append(status_name)
        if not existing_markers:
            summary["pending"] += 1
    return summary


def print_task_summary(summary: Dict[str, int]) -> None:
    """Print caption task status in a stable reusable format."""
    print(
        "Task status: total={total}, completed={completed}, failed={failed}, "
        "partial={partial}, locked={locked}, pending={pending}".format(**summary)
    )


# ========= Batch entrypoint =========

def main(args: argparse.Namespace) -> int:
    """
    Build dataset tasks and run captioning in video order.

    Manifest mode supports explicit video_id mapping. Directory-scan mode supports Video-MME
    and M3-Bench. check_only validates paths, annotations, and outputs without loading the model.
    """
    print(f"Dataset          : {args.dataset_name or 'directory scan'}")
    print(f"Output directory : {args.output_dir}")
    print(f"Video directory  : {args.video_dir}")
    print(f"Device           : {args.device}")
    print(f"Segment length   : {args.clip_seconds} seconds")
    print(f"Sampling FPS     : {args.qwen_fps}")
    print(f"⭐ Caption Prompt: {args.prompt_path}")
    print(f"Words per field  : {args.max_words_per_key}")
    print(f"Maximum new tokens: {args.max_new_tokens}")
    if args.annotation_files:
        print(f"Annotation files : {', '.join(args.annotation_files)}")
    try:
        if not os.path.isdir(args.video_dir):
            raise FileNotFoundError(f"Video directory does not exist: {args.video_dir}")
        if not os.path.isdir(args.model):
            raise FileNotFoundError(f"Model directory does not exist: {args.model}")
        if not os.path.isfile(args.prompt_path):
            raise FileNotFoundError(f"Caption prompt does not exist: {args.prompt_path}")
        os.makedirs(args.output_dir, exist_ok=True)
        video_tasks = list_video_tasks(
            video_root=args.video_dir,
            video_name=args.video_name,
            annotation_files=args.annotation_files,
            subtitle_root=args.subtitle_dir,
        )
    except Exception as error:
        print(f"[ERROR] Caption preflight failed: {error}")
        return 1

    if not video_tasks:
        print(f"[ERROR] No matching video found: {args.video_name or args.video_dir}")
        return 1
    if args.expected_video_count > 0 and not args.video_name:
        if len(video_tasks) != args.expected_video_count:
            print(
                f"[ERROR] Manifest contains {len(video_tasks)} unique videos; "
                f"expected {args.expected_video_count}"
            )
            return 1

    task_source = "annotation manifest" if args.annotation_files else "video directory"
    print(f"Task source      : {task_source}")
    print(f"Target videos    : {len(video_tasks)}")
    summary = summarize_task_outputs(video_tasks, args.output_dir)
    print_task_summary(summary)

    if args.check_only:
        print("[DONE] Inputs, paths, and outputs checked without loading the vision model")
        if args.require_complete:
            complete = summary["completed"] == summary["total"]
            clean = summary["failed"] == 0 and summary["locked"] == 0
            if not complete or not clean:
                print("[ERROR] Caption outputs are incomplete or have failure/lock markers")
                return 1
        return 0

    processor, model = init_qwen(args.model, args.device)
    processing_failures = 0

    while True:
        any_task = False
        for video_name, video_path in video_tasks:
            video_dir = os.path.join(args.output_dir, video_name)
            os.makedirs(video_dir, exist_ok=True)
            output_json = os.path.join(video_dir, "captions.json")
            failed_path = os.path.join(video_dir, ".failed")
            lock_path = os.path.join(video_dir, ".lock")

            if os.path.exists(output_json) or os.path.exists(failed_path):
                continue
            if os.path.exists(lock_path) and is_stale_lock(lock_path):
                try:
                    os.remove(lock_path)
                    print(f"[WARN] Removed stale lock: {lock_path}")
                except OSError:
                    # Another process already removed it.
                    pass
            if not acquire_lock(lock_path):
                continue

            any_task = True
            print(
                f"\n[{time.strftime('%H:%M:%S')}] Locked {video_name} on device "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', args.device)}"
            )
            try:
                caption_video_segments(
                    video_path=video_path,
                    output_json=output_json,
                    clip_seconds=args.clip_seconds,
                    fps=args.qwen_fps,
                    model=model,
                    processor=processor,
                    max_words_per_key=args.max_words_per_key,
                    max_new_tokens=args.max_new_tokens,
                    max_retries=args.max_retries,
                    prompt_path=args.prompt_path,
                )
            except Exception as error:
                processing_failures += 1
                print(f"[ERROR] Failed to process {video_name}: {error}")
                with open(failed_path, "w", encoding="utf-8") as f:
                    f.write(f"{type(error).__name__}: {error}\n")
            finally:
                release_lock(lock_path)

        if not any_task:
            print(
                f"\n[{time.strftime('%H:%M:%S')}] No pending tasks remain on device "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', args.device)}"
            )
            break
        time.sleep(1)
    return 1 if processing_failures else 0


# ========= CLI arguments =========

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the caption CLI while preserving existing defaults."""
    parser = argparse.ArgumentParser(description="Generate continuous 30-second captions for long videos.")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional YAML config; CLI arguments take precedence",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Video-MME",
        help="Dataset label used only in logs",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help="Directory containing source .mp4/.mkv/.webm videos",
    )
    parser.add_argument(
        "--subtitle_dir",
        type=str,
        default="",
        help="Optional subtitle root; manifests validate paths but never use subtitle content",
    )
    parser.add_argument(
        "--annotation_json",
        dest="annotation_files",
        nargs="+",
        default=[],
        help="One or more JSON manifests; process only referenced videos",
    )
    parser.add_argument(
        "--output_dir", "--preprocess_dir",
        dest="output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Caption output root; --preprocess_dir is a compatibility alias",
    )
    parser.add_argument(
        "--clip_seconds", "--seconds_per_feat",
        dest="clip_seconds",
        type=int,
        default=30,
        help="Duration of each continuous segment in seconds",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Qwen model name or local path",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=DEFAULT_PROMPT_PATH,
        help="Caption prompt template path",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Execution device: cuda or cpu")
    parser.add_argument(
        "--qwen_fps",
        type=float,
        default=1.0,
        help="Uniform first-view sampling FPS",
    )
    parser.add_argument(
        "--max_words_per_key",
        type=int,
        default=DEFAULT_MAX_WORDS_PER_KEY,
        help="Target maximum English words per caption field",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens for the four-field JSON",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum generation attempts for an invalid segment response",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="",
        help="Process only this extension-free video name",
    )
    parser.add_argument(
        "--expected_video_count",
        type=int,
        default=0,
        help="Expected unique videos in manifest mode; 0 disables the check",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Check inputs, paths, and outputs without loading the model",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="With --check_only, fail on incomplete outputs or failure markers",
    )
    return parser


def load_config_defaults(config_path: str) -> Dict[str, Any]:
    """
    Read caption YAML with OmegaConf and convert it to argparse defaults.

    Accept only options declared by the caption CLI. Descriptive YAML fields are not passed
    to runtime functions, and API keys are never read here.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError("Reading --config requires omegaconf") from error

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Caption config does not exist: {config_path}")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Caption config root must be a mapping: {config_path}")

    config_to_argument = {
        "dataset_name": "dataset_name",
        "video_dir": "video_dir",
        "subtitle_dir": "subtitle_dir",
        "annotation_files": "annotation_files",
        "output_dir": "output_dir",
        "clip_seconds": "clip_seconds",
        "model": "model",
        "prompt_path": "prompt_path",
        "device": "device",
        "qwen_fps": "qwen_fps",
        "max_words_per_key": "max_words_per_key",
        "max_new_tokens": "max_new_tokens",
        "max_retries": "max_retries",
        "expected_video_count": "expected_video_count",
    }
    defaults = {}
    for config_key, argument_name in config_to_argument.items():
        if config_key in config:
            defaults[argument_name] = config[config_key]
    return defaults


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Read --config first so explicit CLI arguments override YAML defaults.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    known_args, _ = config_parser.parse_known_args(argv)

    parser = build_argument_parser()
    if known_args.config:
        parser.set_defaults(**load_config_defaults(known_args.config))
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments()
    raise SystemExit(main(args))
