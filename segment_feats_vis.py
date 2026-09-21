"""
Generate per-segment ViCLIP visual embeddings from captions.json time ranges.

Default directory-scan mode preserves Video-MME behavior:

    CUDA_VISIBLE_DEVICES=0 python segment_feats_vis.py \
        --video_dir /path/to/videos \
        --base_dir /path/to/caption/VideoMME

Optional manifest mode maps video_id values to source video files:

    CUDA_VISIBLE_DEVICES=0 python segment_feats_vis.py \
        --video_dir /path/to/videos \
        --base_dir /path/to/processed \
        --annotation_json /path/to/annotations.json
"""
import argparse
import gc
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "viclip-l-internvid-10m-flt"
DEFAULT_VICLIP_PRETRAINED = os.path.join(
    SCRIPT_DIR,
    "models",
    "ViCLIP",
    "ViClip-InternVid-10M-FLT.pth",
)
MODEL_CONFIGS = {
    MODEL_NAME: {
        "size": "l",
        "pretrained": DEFAULT_VICLIP_PRETRAINED,
    }
}
# Keep the legacy name for callers that import it directly.
model_cfgs = MODEL_CONFIGS
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv")
CAPTION_KEYS = ("environment", "event", "attention", "summary")
OUTPUT_KEYS = ("start_time", "end_time") + CAPTION_KEYS

CaptionRange = Tuple[int, float, float]
VideoTask = Tuple[str, str]


def initialize_viclip_runtime(device: str) -> Tuple[Any, Any, Any, Any]:
    """
    Initialize the execution device before importing and building ViCLIP.

    Building ViCLIP on CPU before the first lazy ``to(cuda)`` initialization can crash in
    some environments. CUDA mode initializes the driver first, then imports Decord and
    InternVid lazily. CPU mode performs no CUDA operation.
    """
    normalized_device = str(device).strip().lower()
    if normalized_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check device visibility and the driver")
        torch.cuda.init()
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        print(
            "CUDA initialized: {}".format(
                torch.cuda.get_device_name(device_index)
            )
        )

    import decord as decord_module
    from InternVid.viclip import frames2tensor, get_viclip, get_vid_feat

    decord_module.bridge.set_bridge("torch")
    return decord_module, frames2tensor, get_viclip, get_vid_feat


def parse_timestamp(timestamp: str) -> float:
    """
    Convert an HH:MM:SS caption timestamp to seconds.

    Text and visual pipelines share this rule to keep their timelines identical.
    """
    if not isinstance(timestamp, str):
        raise ValueError(
            "Timestamp must be a string; got {}".format(type(timestamp).__name__)
        )
    parts = timestamp.strip().split(":")
    if len(parts) != 3:
        raise ValueError("Invalid timestamp: {}".format(timestamp))
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("Invalid timestamp: {}".format(timestamp)) from error
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("Invalid timestamp: {}".format(timestamp))
    return float(hours * 3600 + minutes * 60 + seconds)


def load_caption_ranges(captions_path: str) -> List[CaptionRange]:
    """
    Read true segment time ranges from the current four-field captions.json.

    Generate segment_id from list order. Validate interval bounds and overlap without
    depending on caption text.
    """
    with open(captions_path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, list):
        raise ValueError("{} must contain a list".format(captions_path))

    ranges = []  # type: List[CaptionRange]
    previous_end = 0.0
    for segment_id, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("Caption {} is not a JSON object".format(segment_id))
        if tuple(item.keys()) != OUTPUT_KEYS:
            raise ValueError(
                "Caption {} has invalid fields or order; expected {}".format(
                    segment_id,
                    OUTPUT_KEYS,
                )
            )

        start_sec = parse_timestamp(item["start_time"])
        end_sec = parse_timestamp(item["end_time"])
        if end_sec <= start_sec:
            raise ValueError("Caption {} must end after it starts".format(segment_id))
        if segment_id > 0 and start_sec < previous_end:
            raise ValueError("Caption {} overlaps the previous segment".format(segment_id))
        ranges.append((segment_id, start_sec, end_sec))
        previous_end = end_sec
    return ranges


def collect_video_tasks(
    video_dir: str,
    recursive: bool = True,
    video_name: str = "",
) -> List[VideoTask]:
    """
    Collect directory-mode video tasks by filename.

    Preserve Video-MME scanning by using each extension-free filename as the task name.
    """
    tasks = []  # type: List[VideoTask]
    for video_path in collect_videos(video_dir, recursive=recursive):
        task_name = os.path.splitext(os.path.basename(video_path))[0]
        if video_name and task_name != video_name:
            continue
        tasks.append((task_name, video_path))
    return tasks


def collect_videos(
    video_dir: str,
    exts: Tuple[str, ...] = SUPPORTED_VIDEO_EXTENSIONS,
    recursive: bool = True,
) -> List[str]:
    """Preserve the legacy scan interface with configurable video extensions."""
    video_paths = []  # type: List[str]
    if recursive:
        for root, _, filenames in os.walk(video_dir):
            for filename in filenames:
                if filename.lower().endswith(exts):
                    video_paths.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, filename)
            if os.path.isfile(video_path) and filename.lower().endswith(exts):
                video_paths.append(video_path)
    video_paths.sort()
    return video_paths


def save_pickle_atomic(payload: Dict[str, Any], output_path: str) -> None:
    """Write a temporary file and atomically replace the final visual embeddings."""
    temporary_path = output_path + ".tmp"
    try:
        with open(temporary_path, "wb") as file_obj:
            pickle.dump(payload, file_obj)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def is_reusable_embedding_file(
    output_path: str,
    caption_ranges: List[CaptionRange],
    frames_per_feat: int,
) -> bool:
    """
    Check whether an existing visual PKL matches the current timeline and parameters.

    This strict reuse check serves manifest mode. Default Video-MME mode retains the
    existing file-exists skip behavior.
    """
    try:
        with open(output_path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        if not isinstance(payload, dict):
            return False
        embeddings = np.asarray(payload.get("embeddings"))
        expected_ids = [item[0] for item in caption_ranges]
        expected_starts = [item[1] for item in caption_ranges]
        expected_ends = [item[2] for item in caption_ranges]
        return bool(
            payload.get("model") == MODEL_NAME
            and payload.get("frames_per_segment") == frames_per_feat
            and payload.get("segment_ids") == expected_ids
            and np.allclose(payload.get("start_times"), expected_starts)
            and np.allclose(payload.get("end_times"), expected_ends)
            and embeddings.ndim == 2
            and embeddings.shape[0] == len(caption_ranges)
            and embeddings.shape[1] > 0
        )
    except (OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError):
        return False


class SegmentFeature:
    """
    Generate visual embeddings for each video's true captions.json time ranges.

    Each task carries an output video_id and source path, supporting both matching
    Video-MME names and manifest IDs that differ from filenames.
    """

    def __init__(
        self,
        video_path_list: Optional[List[str]] = None,
        base_dir: str = "",
        frames_per_feat: int = 8,
        device: str = "cuda",
        overwrite: bool = False,
        video_tasks: Optional[List[VideoTask]] = None,
        strict: bool = False,
        model_path: str = DEFAULT_VICLIP_PRETRAINED,
    ) -> None:
        if frames_per_feat <= 0:
            raise ValueError("frames_per_feat must be greater than 0")
        if video_tasks is None:
            video_tasks = []
            for video_path in video_path_list or []:
                task_name = os.path.splitext(os.path.basename(video_path))[0]
                video_tasks.append((task_name, video_path))
        self.video_tasks = video_tasks
        self.video_path_list = [video_path for _, video_path in video_tasks]
        self.base_dir = base_dir
        self.frames_per_feat = frames_per_feat
        self.device = device
        self.overwrite = overwrite
        self.strict = strict
        self.model_path = model_path

    def create_visual_embedding(self) -> Dict[str, int]:
        """
        Extract ViCLIP features for every captions.json interval in each target video.

        Uniformly sample a fixed number of frames per interval. Do not save a video if any
        segment fails, preventing text/visual index misalignment.
        """
        start_time = time.time()
        (
            decord_module,
            frames2tensor,
            get_viclip,
            get_vid_feat,
        ) = initialize_viclip_runtime(self.device)
        config = MODEL_CONFIGS[MODEL_NAME]
        model = get_viclip(config["size"], self.model_path)
        assert (
            isinstance(model, dict)
            and model["viclip"] is not None
            and model["tokenizer"] is not None
        )
        clip = model["viclip"].to(self.device).eval()
        print("ViCLIP model load time: {} seconds".format(round(time.time() - start_time, 3)))

        summary = {
            "total": len(self.video_tasks),
            "saved": 0,
            "skipped": 0,
            "failed": 0,
        }
        for task_index, (task_name, video_path) in enumerate(self.video_tasks, 1):
            base_name = os.path.basename(video_path)
            output_dir = os.path.join(self.base_dir, task_name)
            captions_path = os.path.join(output_dir, "captions.json")
            output_path = os.path.join(output_dir, "segment_visual_embedding.pkl")

            if not os.path.exists(captions_path):
                status = "ERROR" if self.strict else "SKIP"
                print(
                    "[{}/{}] [{}] {}: missing captions.json".format(
                        task_index,
                        len(self.video_tasks),
                        status,
                        task_name,
                    )
                )
                summary["failed" if self.strict else "skipped"] += 1
                continue
            if os.path.exists(output_path) and not self.overwrite and not self.strict:
                # Directory mode preserves Video-MME's file-exists skip behavior.
                print(
                    "[{}/{}] Existing output, skipping: {}".format(
                        task_index,
                        len(self.video_tasks),
                        task_name,
                    )
                )
                summary["skipped"] += 1
                continue
            try:
                caption_ranges = load_caption_ranges(captions_path)
                if not caption_ranges:
                    raise ValueError("captions.json is empty")
                if os.path.exists(output_path) and not self.overwrite:
                    if not self.strict or is_reusable_embedding_file(
                        output_path=output_path,
                        caption_ranges=caption_ranges,
                        frames_per_feat=self.frames_per_feat,
                    ):
                        print(
                            "[{}/{}] Existing output, skipping: {}".format(
                                task_index,
                                len(self.video_tasks),
                                task_name,
                            )
                        )
                        summary["skipped"] += 1
                        continue
                    print("[REBUILD] {}: visual embeddings are invalid or stale".format(task_name))
                video_reader = decord_module.VideoReader(video_path, num_threads=4)
                num_frames = len(video_reader)
                fps = float(video_reader.get_avg_fps())
                if fps <= 0 or num_frames <= 0:
                    raise ValueError(
                        "Invalid video frame metadata: fps={}, frames={}".format(fps, num_frames)
                    )
            except Exception as error:
                print(
                    "[{}/{}] [ERROR] Could not initialize {} ({}): {}".format(
                        task_index,
                        len(self.video_tasks),
                        task_name,
                        base_name,
                        error,
                    )
                )
                summary["failed" if self.strict else "skipped"] += 1
                continue

            print(
                "[{}/{}] {} | fps={:.2f}, frames={}, caption segments={}".format(
                    task_index,
                    len(self.video_tasks),
                    task_name,
                    fps,
                    num_frames,
                    len(caption_ranges),
                )
            )
            segment_features = []  # type: List[torch.Tensor]
            video_start_time = time.time()
            failure_reason = ""

            for segment_id, start_sec, end_sec in caption_ranges:
                start_frame = max(0, int(np.floor(start_sec * fps)))
                end_frame = min(num_frames - 1, int(np.ceil(end_sec * fps)) - 1)
                if start_frame > end_frame:
                    failure_reason = (
                        "Invalid time range for segment {}: {:.1f}-{:.1f}s".format(
                            segment_id,
                            start_sec,
                            end_sec,
                        )
                    )
                    break

                frame_indices = np.linspace(
                    start_frame,
                    end_frame,
                    num=self.frames_per_feat,
                    dtype=int,
                ).tolist()
                try:
                    batch = video_reader.get_batch(frame_indices)
                    frames = batch.numpy() if hasattr(batch, "numpy") else batch.asnumpy()
                    frames_tensor = frames2tensor(list(frames), device=self.device)
                    with torch.no_grad():
                        video_feature = get_vid_feat(frames_tensor, clip).cpu()
                    segment_features.append(video_feature)
                    del frames_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as error:
                    failure_reason = "Failed to read or encode segment {}: {}".format(
                        segment_id,
                        error,
                    )
                    break

            del video_reader
            gc.collect()
            if failure_reason or len(segment_features) != len(caption_ranges):
                print("[ERROR] {}: {}; result not saved".format(task_name, failure_reason))
                summary["failed"] += 1
                continue

            embeddings = torch.cat(segment_features, dim=0).numpy()
            payload = {
                "schema_version": "bionic_caption_v2",
                "model": MODEL_NAME,
                "frames_per_segment": self.frames_per_feat,
                "segment_ids": [item[0] for item in caption_ranges],
                "start_times": [item[1] for item in caption_ranges],
                "end_times": [item[2] for item in caption_ranges],
                "embeddings": embeddings,
            }
            save_pickle_atomic(payload, output_path)
            summary["saved"] += 1
            print(
                "Saved: {}, shape={}, elapsed {} seconds".format(
                    output_path,
                    embeddings.shape,
                    round(time.time() - video_start_time, 3),
                )
            )

        print(
            "Visual embedding summary: total={total}, saved={saved}, skipped={skipped}, "
            "failed={failed}".format(**summary)
        )
        print("Total visual embedding time: {} seconds".format(round(time.time() - start_time, 3)))
        return summary

    def run(self) -> Dict[str, int]:
        """Generate visual embeddings and return stage statistics."""
        return self.create_visual_embedding()


def parse_args() -> argparse.Namespace:
    """Parse arguments shared by directory and manifest modes."""
    parser = argparse.ArgumentParser(description="Generate visual embeddings by caption time range")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "data", "raw", "videomme", "videos"),
        help="Directory containing source videos; recursive by default",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(
            SCRIPT_DIR,
            "data",
            "processed",
            "videomme",
            "qwen2_5_vl_7b_30s",
        ),
        help="Preprocessing root containing each video's captions.json",
    )
    parser.add_argument(
        "--annotation_json",
        dest="annotation_files",
        nargs="+",
        default=[],
        help="Optional manifests; use video_id as output directory names",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="",
        help="Process one video_id or extension-free filename",
    )
    parser.add_argument(
        "--expected_video_count",
        type=int,
        default=0,
        help="Expected unique video count; 0 disables the check",
    )
    parser.add_argument("--no_recursive", action="store_true", help="Do not scan subdirectories")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device: cuda or cpu")
    parser.add_argument(
        "--frames_per_feat",
        type=int,
        default=8,
        help="Frames uniformly sampled from each caption interval",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_VICLIP_PRETRAINED,
        help="Repository-local ViCLIP checkpoint path",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing visual embeddings")
    return parser.parse_args()


def main() -> int:
    """Build target tasks, extract visual embeddings, and report failures."""
    args = parse_args()
    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError("Video directory does not exist: {}".format(args.video_dir))
    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError("Preprocessing directory does not exist: {}".format(args.base_dir))

    if args.annotation_files:
        # Load manifest mapping only in manifest mode.
        from preprocess_manifest import load_annotation_video_tasks

        video_tasks = load_annotation_video_tasks(
            video_root=args.video_dir,
            annotation_files=args.annotation_files,
            video_name=args.video_name,
        )
    else:
        video_tasks = collect_video_tasks(
            video_dir=args.video_dir,
            recursive=not args.no_recursive,
            video_name=args.video_name,
        )

    if not video_tasks:
        raise ValueError("No matching video tasks found")
    if args.expected_video_count > 0 and not args.video_name:
        if len(video_tasks) != args.expected_video_count:
            raise ValueError(
                "Found {} unique videos; expected {}".format(
                    len(video_tasks),
                    args.expected_video_count,
                )
            )

    print("Found {} video tasks: {}".format(len(video_tasks), args.video_dir))
    segment_feature = SegmentFeature(
        video_tasks=video_tasks,
        base_dir=args.base_dir,
        frames_per_feat=args.frames_per_feat,
        device=args.device,
        overwrite=args.overwrite,
        strict=bool(args.annotation_files),
        model_path=args.model_path,
    )
    summary = segment_feature.run()
    return 1 if args.annotation_files and summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
