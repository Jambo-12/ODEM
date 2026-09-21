"""Run M3-Bench questions and persist predictions and traces."""

import os
import time
from typing import Any, Dict, List, cast

from tqdm import tqdm

import performance_trace as performance_module
from eval import runtime
from eval.m3bench.dataset import EXPERIMENT_VERSION, existing_responses, save_json_atomic
from eval.m3bench.graph import DEFAULT_OPEN_ANSWER
from eval.m3bench.state import build_initial_state
from eval.m3bench.trace import write_trace
from graph_trace import collect_graph_events
from performance_trace import finish_question_tracking, start_question_tracking


performance_module.NODE_ORDER = (
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


def _video_result(video: Dict[str, Any]) -> Dict[str, Any]:
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
    return {"video_id": video.get("video_id"), "results": records}


def run_dataset(
    graph: Any,
    videos: List[Dict[str, Any]],
    output_path: str,
    result_root: str,
    log_root: str,
    rerun: bool,
    max_iterations: int,
    max_concurrency: int,
) -> None:
    """Run the unchanged M3-Bench question loop."""
    existing = {} if rerun else existing_responses(output_path)
    total_questions = sum(len(video.get("questions", [])) for video in videos)
    progress = tqdm(total=total_questions, desc="M3-Bench inference", unit="question")
    for video in videos:
        video_id = str(video["video_id"])
        log_path = os.path.join(log_root, "{}.log".format(video_id))
        result_path = os.path.join(result_root, "{}.json".format(video_id))
        log_initialized = os.path.isfile(log_path) and not rerun
        for question in video.get("questions", []):
            question_id = str(question["question_id"])
            previous = existing.get((video_id, question_id))
            if previous:
                question["prediction"] = previous["prediction"]
                for key in ("response_detail", "elapsed_sec"):
                    if key in previous:
                        question[key] = previous[key]
                question["experiment_version"] = EXPERIMENT_VERSION
                progress.update(1)
                continue

            initial_state = build_initial_state(video, question, max_iterations)
            events = []  # type: List[Dict[str, Any]]
            started_at = time.perf_counter()
            start_question_tracking(video_id, video_id, question_id)
            try:
                with runtime.suppress_normal_console_output():
                    events, final_state, runtime_error = collect_graph_events(
                        graph,
                        cast(Any, initial_state),
                        recursion_limit=max(40, max_iterations * 30),
                        max_concurrency=max_concurrency,
                    )
            finally:
                finish_question_tracking(events)
            elapsed_sec = round(time.perf_counter() - started_at, 3)
            prediction = " ".join(
                str(final_state.get("final_answer") or "").strip().split()
            ) or DEFAULT_OPEN_ANSWER
            final_state["final_answer"] = prediction
            question["prediction"] = prediction
            question["elapsed_sec"] = elapsed_sec
            question["experiment_version"] = EXPERIMENT_VERSION
            question["response_detail"] = {
                "answer_output": final_state.get("answer_output", {}),
                "trace_log": log_path,
                "runtime_error": runtime_error.splitlines()[0] if runtime_error else "",
                "errors": list(final_state.get("errors", [])),
            }
            write_trace(
                log_path,
                {
                    "video_id": video_id,
                    "video_name": video_id,
                    "question_id": question_id,
                    "question": question.get("question", ""),
                    "type": question.get("type", []),
                    "before_clip": question.get("before_clip"),
                    "cutoff_time_sec": question.get("cutoff_time_sec"),
                },
                events,
                final_state,
                runtime_error,
                append=log_initialized,
            )
            log_initialized = True
            save_json_atomic(videos, output_path)
            save_json_atomic(_video_result(video), result_path)
            if runtime_error:
                progress.write(
                    "[ERROR] video={}, question={}: {}".format(
                        video_id, question_id, runtime_error.splitlines()[0]
                    )
                )
            for error in final_state.get("errors", []):
                progress.write(
                    "[ERROR] video={}, question={}: {}".format(
                        video_id, question_id, error
                    )
                )
            progress.update(1)
        save_json_atomic(_video_result(video), result_path)
    save_json_atomic(videos, output_path)
    progress.close()
