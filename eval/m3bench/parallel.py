"""Prepare, monitor, and merge sharded M3-Bench results without changing per-question inference."""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import performance_trace as performance_module
from performance_trace import PerformanceTracker

from eval.m3bench.dataset import EXPERIMENT_VERSION, save_json_atomic


QUESTION_PATTERN = re.compile(r"^\[QUESTION\] video_id=(\S+).* question=(\S+) ")
PERFORMANCE_NODE_ORDER = (
    "plan",
    "localization",
    "segment_asr",
    "memory",
    "perception",
    "attention",
    "reason",
    "reflect",
    "answer",
)


def _load_json(path: str, default: Any) -> Any:
    """Read JSON and return the supplied default when the file is missing or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return default


def _load_dataset(annotation_json: str) -> List[Dict[str, Any]]:
    """Build a result skeleton without reasoning in the original robot.json order."""
    annotations = _load_json(annotation_json, {})
    if not isinstance(annotations, dict):
        raise ValueError("The top level of M3-Bench robot.json must be an object")
    videos = []  # type: List[Dict[str, Any]]
    for video_id, video in annotations.items():
        questions = []  # type: List[Dict[str, Any]]
        for item in video.get("qa_list", []):
            if not isinstance(item, dict):
                continue
            questions.append({
                "question_id": str(item.get("question_id") or ""),
                "question": str(item.get("question") or ""),
                "gold": str(item.get("answer") or ""),
                "type": [str(value) for value in item.get("type", [])],
                "timestamp": str(item.get("timestamp") or ""),
                "before_clip": int(item.get("before_clip")),
            })
        videos.append({
            "video_id": str(video_id),
            "video_name": str(video_id),
            "questions": questions,
        })
    return videos


def _response_records(videos: Any) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Extract nonempty open-ended predictions for safe migration between single-process and sharded results."""
    records = {}  # type: Dict[Tuple[str, str], Dict[str, Any]]
    for video in videos if isinstance(videos, list) else []:
        if not isinstance(video, dict):
            continue
        video_id = str(video.get("video_id") or "")
        for question in video.get("questions", []):
            if not isinstance(question, dict):
                continue
            if question.get("experiment_version") != EXPERIMENT_VERSION:
                continue
            detail = question.get("response_detail", {})
            if isinstance(detail, dict) and (
                detail.get("runtime_error") or detail.get("errors")
            ):
                continue
            prediction = " ".join(str(question.get("prediction") or "").split())
            if prediction:
                records[(video_id, str(question.get("question_id") or ""))] = {
                    "prediction": prediction,
                    "response_detail": question.get("response_detail"),
                    "elapsed_sec": question.get("elapsed_sec"),
                    "experiment_version": EXPERIMENT_VERSION,
                }
    return records


def _performance_lines(path: str) -> Dict[Tuple[str, str], str]:
    """Read question-level performance records; final aggregate rows are recalculated after merging."""
    output = {}  # type: Dict[Tuple[str, str], str]
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                match = QUESTION_PATTERN.match(line)
                if match is not None:
                    output[(match.group(1), match.group(2))] = line.rstrip("\n")
    except FileNotFoundError:
        pass
    return output


def _write_performance(path: str, lines: Dict[Tuple[str, str], str]) -> None:
    """Write deduplicated question records and regenerate video and average statistics."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        for line in lines.values():
            file.write(line + "\n")
    os.replace(temporary_path, path)
    performance_module.NODE_ORDER = PERFORMANCE_NODE_ORDER
    PerformanceTracker(path).finalize()


def prepare(
    annotation_json: str,
    shard_root: str,
    combined_output: str,
    combined_performance: str,
    num_shards: int,
    max_videos: int,
    rerun: bool,
) -> None:
    """Create shard checkpoints, migrating only questions with both predictions and performance records."""
    expected_videos = _load_dataset(annotation_json)[:max_videos]
    combined_records = {} if rerun else _response_records(
        _load_json(combined_output, [])
    )
    combined_lines = {} if rerun else _performance_lines(combined_performance)
    for shard_index in range(num_shards):
        shard_dir = os.path.join(shard_root, "shard_{}".format(shard_index))
        prediction_path = os.path.join(shard_dir, "predictions.json")
        performance_path = os.path.join(shard_dir, "performance.log")
        shard_videos = [
            json.loads(json.dumps(video, ensure_ascii=False))
            for index, video in enumerate(expected_videos)
            if index % num_shards == shard_index
        ]
        old_records = {} if rerun else _response_records(
            _load_json(prediction_path, [])
        )
        old_lines = {} if rerun else _performance_lines(performance_path)
        available_records = dict(combined_records)
        available_records.update(old_records)
        available_lines = dict(combined_lines)
        available_lines.update(old_lines)
        retained_lines = {}  # type: Dict[Tuple[str, str], str]
        for video in shard_videos:
            video_id = str(video["video_id"])
            for question in video.get("questions", []):
                key = (video_id, str(question.get("question_id") or ""))
                record = available_records.get(key)
                if record is None or key not in available_lines:
                    continue
                question.update(record)
                retained_lines[key] = available_lines[key]
        save_json_atomic(shard_videos, prediction_path)
        _write_performance(performance_path, retained_lines)


def progress(
    shard_root: str,
    num_shards: int,
    expected_videos: int,
    expected_questions: int,
) -> None:
    """Show persisted video and open-ended question counts without calculating accuracy during generation."""
    completed_videos = 0
    completed_questions = 0
    for shard_index in range(num_shards):
        videos = _load_json(
            os.path.join(shard_root, "shard_{}".format(shard_index), "predictions.json"),
            [],
        )
        for video in videos if isinstance(videos, list) else []:
            questions = video.get("questions", []) if isinstance(video, dict) else []
            finished = sum(
                bool(str(question.get("prediction") or "").strip())
                for question in questions
                if isinstance(question, dict)
            )
            completed_questions += finished
            completed_videos += int(bool(questions) and finished == len(questions))
    print(
        "[Shard progress] videos {}/{}, questions {}/{}".format(
            completed_videos,
            expected_videos,
            completed_questions,
            expected_questions,
        ),
        flush=True,
    )


def _save_video_results(video: Dict[str, Any], result_root: str) -> None:
    """Write merged results in the per-video structure expected by the reference judge."""
    records = []  # type: List[Dict[str, Any]]
    for question in video.get("questions", []):
        records.append({
            "question_id": question.get("question_id"),
            "question": question.get("question"),
            "gold": question.get("gold"),
            "pred": question.get("prediction", ""),
            "type": question.get("type", []),
            "timestamp": question.get("timestamp", ""),
            "before_clip": question.get("before_clip"),
            "cutoff_time_sec": question.get("cutoff_time_sec"),
            "elapsed_sec": question.get("elapsed_sec"),
            "experiment_version": EXPERIMENT_VERSION,
        })
    save_json_atomic(
        {"video_id": video.get("video_id"), "results": records},
        os.path.join(result_root, "{}.json".format(video.get("video_id"))),
    )


def merge(
    annotation_json: str,
    shard_root: str,
    output_path: str,
    result_root: str,
    performance_path: str,
    num_shards: int,
    max_videos: int,
) -> None:
    """Strictly validate and merge all shards while preserving open-ended predictions and performance logs."""
    expected = _load_dataset(annotation_json)[:max_videos]
    expected_ids = [str(video["video_id"]) for video in expected]
    videos_by_id = {}  # type: Dict[str, Dict[str, Any]]
    performance_lines = {}  # type: Dict[Tuple[str, str], str]
    for shard_index in range(num_shards):
        shard_dir = os.path.join(shard_root, "shard_{}".format(shard_index))
        shard_videos = _load_json(os.path.join(shard_dir, "predictions.json"), [])
        for video in shard_videos if isinstance(shard_videos, list) else []:
            video_id = str(video.get("video_id") or "")
            if video_id in videos_by_id:
                raise ValueError("Duplicate video_id={} across shards".format(video_id))
            videos_by_id[video_id] = video
        performance_lines.update(
            _performance_lines(os.path.join(shard_dir, "performance.log"))
        )
    if set(videos_by_id) != set(expected_ids):
        raise ValueError(
            "Incomplete shard video set: expected={}, actual={}".format(
                len(expected_ids), len(videos_by_id)
            )
        )

    combined = []  # type: List[Dict[str, Any]]
    expected_keys = []  # type: List[Tuple[str, str]]
    for expected_video in expected:
        video_id = str(expected_video["video_id"])
        video = videos_by_id[video_id]
        questions = {
            str(question.get("question_id") or ""): question
            for question in video.get("questions", [])
        }
        ordered_questions = []  # type: List[Dict[str, Any]]
        for expected_question in expected_video.get("questions", []):
            question_id = str(expected_question.get("question_id") or "")
            question = questions.get(question_id)
            if question is None or not str(question.get("prediction") or "").strip():
                raise ValueError("Question {} is missing an open-ended prediction".format(question_id))
            ordered_questions.append(question)
            expected_keys.append((video_id, question_id))
        video["questions"] = ordered_questions
        combined.append(video)

    missing_performance = [key for key in expected_keys if key not in performance_lines]
    if missing_performance:
        raise ValueError("Missing performance records for {} questions".format(len(missing_performance)))
    save_json_atomic(combined, output_path)
    for video in combined:
        _save_video_results(video, result_root)
    _write_performance(
        performance_path,
        {key: performance_lines[key] for key in expected_keys},
    )
    print(
        "[Merge complete] videos {}, open-ended questions {}".format(
            len(combined), len(expected_keys)
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse shard preparation, progress monitoring, and result merging commands."""
    parser = argparse.ArgumentParser(description="M3-Bench parallel result manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--annotation_json", required=True)
    prepare_parser.add_argument("--shard_root", required=True)
    prepare_parser.add_argument("--combined_output", required=True)
    prepare_parser.add_argument("--combined_performance", required=True)
    prepare_parser.add_argument("--num_shards", type=int, required=True)
    prepare_parser.add_argument("--max_videos", type=int, required=True)
    prepare_parser.add_argument("--rerun", action="store_true")

    progress_parser = subparsers.add_parser("progress")
    progress_parser.add_argument("--shard_root", required=True)
    progress_parser.add_argument("--num_shards", type=int, required=True)
    progress_parser.add_argument("--expected_videos", type=int, required=True)
    progress_parser.add_argument("--expected_questions", type=int, required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--annotation_json", required=True)
    merge_parser.add_argument("--shard_root", required=True)
    merge_parser.add_argument("--output_path", required=True)
    merge_parser.add_argument("--result_root", required=True)
    merge_parser.add_argument("--performance_path", required=True)
    merge_parser.add_argument("--num_shards", type=int, required=True)
    merge_parser.add_argument("--max_videos", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """Run preparation, progress reporting, or final merging."""
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("num_shards must be greater than 0")
    if args.command == "prepare":
        prepare(
            args.annotation_json,
            args.shard_root,
            args.combined_output,
            args.combined_performance,
            args.num_shards,
            args.max_videos,
            args.rerun,
        )
    elif args.command == "progress":
        progress(
            args.shard_root,
            args.num_shards,
            args.expected_videos,
            args.expected_questions,
        )
    else:
        merge(
            args.annotation_json,
            args.shard_root,
            args.output_path,
            args.result_root,
            args.performance_path,
            args.num_shards,
            args.max_videos,
        )


if __name__ == "__main__":
    main()
