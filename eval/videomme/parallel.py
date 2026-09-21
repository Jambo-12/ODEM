"""Prepare, summarize, and merge sharded Video-MME results without changing per-question inference."""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import performance_trace as performance_module
from performance_trace import PerformanceTracker


VALID_OPTIONS = {"A", "B", "C", "D", "E"}
QUESTION_PATTERN = re.compile(r"^\[QUESTION\] video_id=(\S+).* question=(\S+) ")
PERFORMANCE_NODE_ORDER = (
    "plan",
    "localization",
    "segment_asr",
    "memory",
    "perception",
    "attention",
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


def _save_json_atomic(value: Any, path: str) -> None:
    """Atomically save JSON through a same-directory temporary file to prevent partial progress reads."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=4)
    os.replace(temporary_path, path)


def _load_dataset(input_json: str) -> List[Dict[str, Any]]:
    """Group flat Video-MME annotations by video in their original order."""
    raw_items = _load_json(input_json, [])
    if not isinstance(raw_items, list):
        raise ValueError("The top level of the Video-MME annotation file must be a list")
    videos = {}  # type: Dict[str, Dict[str, Any]]
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        video_id = str(item["video_id"])
        if video_id not in videos:
            videos[video_id] = {
                "video_id": video_id,
                "duration": item.get("duration"),
                "domain": item.get("domain"),
                "sub_category": item.get("sub_category"),
                "questions": [],
            }
        videos[video_id]["questions"].append({
            "question_id": item["question_id"],
            "task_type": item.get("task_type"),
            "question": item.get("question"),
            "options": item.get("options", []),
            "answer": item.get("answer"),
        })
    return list(videos.values())


def _response_records(videos: Any) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Extract existing valid answers for safe migration from a single process to multiple shards."""
    records = {}  # type: Dict[Tuple[str, str], Dict[str, Any]]
    for video in videos if isinstance(videos, list) else []:
        if not isinstance(video, dict):
            continue
        video_id = str(video.get("video_id", ""))
        for question in video.get("questions", []):
            if not isinstance(question, dict):
                continue
            response = str(question.get("response") or "").strip().upper()
            if response not in VALID_OPTIONS:
                continue
            records[(video_id, str(question.get("question_id", "")))] = {
                "response": response,
                "response_detail": question.get("response_detail"),
            }
    return records


def _performance_lines(path: str) -> Dict[Tuple[str, str], str]:
    """Read question-level performance records; aggregate rows are recalculated after merging."""
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
    """Write deduplicated question records and regenerate video and average statistics with the original tracker."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        for line in lines.values():
            file.write(line + "\n")
    os.replace(temporary_path, path)
    performance_module.NODE_ORDER = PERFORMANCE_NODE_ORDER
    PerformanceTracker(path).finalize()


def _accuracy(videos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy for current valid predictions, excluding unfinished questions from the denominator."""
    correct = 0
    evaluated = 0
    for video in videos:
        for question in video.get("questions", []):
            response = str(question.get("response") or "").strip().upper()
            answer = str(question.get("answer") or "").strip().upper()
            if response not in VALID_OPTIONS or answer not in VALID_OPTIONS:
                continue
            evaluated += 1
            correct += int(response == answer)
    value = float(correct) / float(evaluated) if evaluated else 0.0
    return {
        "dataset": "Video-MME",
        "correct": correct,
        "incorrect": evaluated - correct,
        "total_evaluated": evaluated,
        "accuracy": round(value, 6),
        "accuracy_percent": round(value * 100.0, 2),
    }


def prepare(
    input_json: str,
    shard_root: str,
    combined_output: str,
    combined_performance: str,
    num_shards: int,
    rerun: bool,
) -> None:
    """Create shard checkpoints, migrating only existing questions with both predictions and performance records."""
    expected_videos = _load_dataset(input_json)
    combined_records = {} if rerun else _response_records(
        _load_json(combined_output, [])
    )
    combined_lines = {} if rerun else _performance_lines(combined_performance)
    for shard_index in range(num_shards):
        shard_dir = os.path.join(shard_root, "shard_{}".format(shard_index))
        prediction_path = os.path.join(shard_dir, "predictions.json")
        accuracy_path = os.path.join(shard_dir, "accuracy.json")
        performance_path = os.path.join(shard_dir, "performance.log")
        shard_videos = [
            json.loads(json.dumps(video, ensure_ascii=False))
            for index, video in enumerate(expected_videos)
            if index % num_shards == shard_index
        ]
        shard_records = {} if rerun else _response_records(
            _load_json(prediction_path, [])
        )
        shard_lines = {} if rerun else _performance_lines(performance_path)
        available_lines = dict(combined_lines)
        available_lines.update(shard_lines)
        available_records = dict(combined_records)
        available_records.update(shard_records)
        retained_lines = {}  # type: Dict[Tuple[str, str], str]
        for video in shard_videos:
            video_id = str(video["video_id"])
            for question in video.get("questions", []):
                key = (video_id, str(question.get("question_id", "")))
                record = available_records.get(key)
                if record is None or key not in available_lines:
                    continue
                question["response"] = record["response"]
                if record.get("response_detail") is not None:
                    question["response_detail"] = record["response_detail"]
                retained_lines[key] = available_lines[key]
        _save_json_atomic(shard_videos, prediction_path)
        _save_json_atomic(_accuracy(shard_videos), accuracy_path)
        _write_performance(performance_path, retained_lines)


def progress(shard_root: str, num_shards: int, expected_videos: int, expected_questions: int) -> None:
    """Print aggregate video progress, question progress, and current accuracy across shards."""
    combined = []  # type: List[Dict[str, Any]]
    completed_videos = 0
    for shard_index in range(num_shards):
        videos = _load_json(
            os.path.join(shard_root, "shard_{}".format(shard_index), "predictions.json"),
            [],
        )
        if not isinstance(videos, list):
            continue
        combined.extend(videos)
        completed_videos += sum(
            bool(video.get("questions"))
            and all(
                str(question.get("response") or "").strip().upper() in VALID_OPTIONS
                for question in video.get("questions", [])
            )
            for video in videos
            if isinstance(video, dict)
        )
    summary = _accuracy(combined)
    print(
        "[Shard progress] videos {}/{}, questions {}/{}, accuracy {:.2f}% ({}/{})".format(
            completed_videos,
            expected_videos,
            summary["total_evaluated"],
            expected_questions,
            summary["accuracy_percent"],
            summary["correct"],
            summary["total_evaluated"],
        ),
        flush=True,
    )


def merge(
    input_json: str,
    shard_root: str,
    output_path: str,
    accuracy_path: str,
    performance_path: str,
    num_shards: int,
) -> None:
    """Strictly validate and merge all shards while preserving prediction, accuracy, and performance log formats."""
    expected = _load_dataset(input_json)
    expected_ids = [str(video["video_id"]) for video in expected]
    videos_by_id = {}  # type: Dict[str, Dict[str, Any]]
    performance_lines = {}  # type: Dict[Tuple[str, str], str]
    for shard_index in range(num_shards):
        shard_dir = os.path.join(shard_root, "shard_{}".format(shard_index))
        shard_videos = _load_json(os.path.join(shard_dir, "predictions.json"), [])
        for video in shard_videos if isinstance(shard_videos, list) else []:
            video_id = str(video.get("video_id", ""))
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
            str(question.get("question_id", "")): question
            for question in video.get("questions", [])
        }
        ordered_questions = []  # type: List[Dict[str, Any]]
        for expected_question in expected_video.get("questions", []):
            question_id = str(expected_question.get("question_id", ""))
            question = questions.get(question_id)
            if question is None:
                raise ValueError("Question {} is missing a shard result".format(question_id))
            response = str(question.get("response") or "").strip().upper()
            if response not in VALID_OPTIONS:
                raise ValueError("Question {} is missing a valid prediction".format(question_id))
            ordered_questions.append(question)
            expected_keys.append((video_id, question_id))
        video["questions"] = ordered_questions
        combined.append(video)

    missing_performance = [key for key in expected_keys if key not in performance_lines]
    if missing_performance:
        raise ValueError("Missing performance records for {} questions".format(len(missing_performance)))
    _save_json_atomic(combined, output_path)
    summary = _accuracy(combined)
    _save_json_atomic(summary, accuracy_path)
    ordered_lines = {key: performance_lines[key] for key in expected_keys}
    _write_performance(performance_path, ordered_lines)
    print(
        "[Merge complete] videos {}, questions {}, accuracy {:.2f}% ({}/{})".format(
            len(combined),
            summary["total_evaluated"],
            summary["accuracy_percent"],
            summary["correct"],
            summary["total_evaluated"],
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse sharded result management commands."""
    parser = argparse.ArgumentParser(description="Video-MME parallel result manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input_json", required=True)
    prepare_parser.add_argument("--shard_root", required=True)
    prepare_parser.add_argument("--combined_output", required=True)
    prepare_parser.add_argument("--combined_performance", required=True)
    prepare_parser.add_argument("--num_shards", type=int, required=True)
    prepare_parser.add_argument("--rerun", action="store_true")

    progress_parser = subparsers.add_parser("progress")
    progress_parser.add_argument("--shard_root", required=True)
    progress_parser.add_argument("--num_shards", type=int, required=True)
    progress_parser.add_argument("--expected_videos", type=int, required=True)
    progress_parser.add_argument("--expected_questions", type=int, required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--input_json", required=True)
    merge_parser.add_argument("--shard_root", required=True)
    merge_parser.add_argument("--output_path", required=True)
    merge_parser.add_argument("--accuracy_path", required=True)
    merge_parser.add_argument("--performance_path", required=True)
    merge_parser.add_argument("--num_shards", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """Run preparation, progress reporting, or final merging."""
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("num_shards must be greater than 0")
    if args.command == "prepare":
        prepare(
            args.input_json,
            args.shard_root,
            args.combined_output,
            args.combined_performance,
            args.num_shards,
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
            args.input_json,
            args.shard_root,
            args.output_path,
            args.accuracy_path,
            args.performance_path,
            args.num_shards,
        )


if __name__ == "__main__":
    main()
