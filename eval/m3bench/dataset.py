"""Read M3-Bench-robot annotations while isolating answer fields from the inference state."""

import json
import os
from typing import Any, Dict, List, Tuple


REQUIRED_PREPROCESS_FILES = (
    "captions.json",
    "segment_textual_embedding.pkl",
    "segment_visual_embedding.pkl",
)
EXPERIMENT_VERSION = "m3bench_streaming_episodic_v1"


def _timestamp_to_seconds(value: Any) -> float:
    """Convert an HH:MM:SS caption timestamp to seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        raise ValueError("Invalid caption timestamp: {}".format(value))
    hours, minutes, seconds = [float(part) for part in parts]
    return hours * 3600.0 + minutes * 60.0 + seconds


def caption_duration(preprocess_root: str, video_name: str) -> float:
    """Read the video end time from the complete caption timeline without launching ffprobe."""
    path = os.path.join(preprocess_root, video_name, "captions.json")
    with open(path, "r", encoding="utf-8") as file:
        captions = json.load(file)
    if not isinstance(captions, list) or not captions:
        raise ValueError("Caption is empty: {}".format(path))
    return max(_timestamp_to_seconds(item.get("end_time")) for item in captions)


def has_preprocess_files(preprocess_root: str, video_name: str) -> bool:
    """Check whether an M3-Bench video has all preprocessed files required by the graph."""
    video_dir = os.path.join(preprocess_root, video_name)
    return all(
        os.path.isfile(os.path.join(video_dir, filename))
        for filename in REQUIRED_PREPROCESS_FILES
    )


def load_robot_dataset(
    annotation_json: str,
    video_root: str,
    preprocess_root: str,
) -> List[Dict[str, Any]]:
    """
    Read the robot subset and place only safe question fields in ``questions``.

    ``reasoning`` is not copied into the result. ``answer`` remains gold data outside
    the runner and must be explicitly excluded again when constructing graph state.
    """
    with open(annotation_json, "r", encoding="utf-8") as file:
        annotations = json.load(file)
    if not isinstance(annotations, dict):
        raise ValueError("The top level of M3-Bench robot.json must be an object")

    videos = []  # type: List[Dict[str, Any]]
    for video_name, item in annotations.items():
        if not isinstance(item, dict):
            raise ValueError("The annotation for video {} is not an object".format(video_name))
        video_path = os.path.join(video_root, "{}.mp4".format(video_name))
        if not os.path.isfile(video_path):
            raise FileNotFoundError("M3-Bench video does not exist: {}".format(video_path))
        if not has_preprocess_files(preprocess_root, video_name):
            raise FileNotFoundError(
                "M3-Bench preprocessing is incomplete: {}".format(
                    os.path.join(preprocess_root, video_name)
                )
            )
        duration = caption_duration(preprocess_root, video_name)
        raw_questions = item.get("qa_list", [])
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError("Video {} does not contain qa_list".format(video_name))
        questions = []  # type: List[Dict[str, Any]]
        for question in raw_questions:
            if not isinstance(question, dict):
                continue
            before_clip = int(question.get("before_clip"))
            cutoff_time = min(duration, float(before_clip + 1) * 30.0)
            if cutoff_time <= 0:
                raise ValueError(
                    "Invalid before_clip for question {}".format(
                        question.get("question_id")
                    )
                )
            questions.append({
                "question_id": str(question.get("question_id") or ""),
                "question": str(question.get("question") or ""),
                "gold": str(question.get("answer") or ""),
                "type": [str(value) for value in question.get("type", [])],
                "timestamp": str(question.get("timestamp") or ""),
                "before_clip": before_clip,
                "cutoff_time_sec": cutoff_time,
            })
        videos.append({
            "video_id": video_name,
            "video_name": video_name,
            "video_path": video_path,
            "duration": duration,
            "questions": questions,
        })
    return videos


def select_shard(
    videos: List[Dict[str, Any]],
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    """Create stable round-robin shards in the original video order."""
    if num_shards <= 0 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("Invalid shard parameters {}/{}".format(shard_index, num_shards))
    return [
        video
        for index, video in enumerate(videos)
        if index % num_shards == shard_index
    ]


def existing_responses(
    output_path: str,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Build a resume index for existing complete question results."""
    if not os.path.isfile(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as file:
        videos = json.load(file)
    responses = {}  # type: Dict[Tuple[str, str], Dict[str, Any]]
    for video in videos:
        video_id = str(video.get("video_id") or "")
        for question in video.get("questions", []):
            if question.get("experiment_version") != EXPERIMENT_VERSION:
                continue
            detail = question.get("response_detail", {})
            if isinstance(detail, dict) and (
                detail.get("runtime_error") or detail.get("errors")
            ):
                continue
            prediction = str(question.get("prediction") or "").strip()
            if prediction:
                responses[(
                    video_id,
                    str(question.get("question_id") or ""),
                )] = dict(question)
    return responses


def save_json_atomic(value: Any, output_path: str) -> None:
    """Atomically save a prediction or evaluation file to avoid partial JSON after interruption."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, output_path)
