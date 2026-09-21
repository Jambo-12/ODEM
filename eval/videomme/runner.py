"""Run Video-MME questions and persist predictions, traces, and accuracy."""

import os
import re
import sys
from typing import Any, Dict, List, Tuple, cast

from tqdm import tqdm

import performance_trace as performance_module
from eval import runtime
from eval.videomme.dataset import existing_records
from eval.videomme.state import build_initial_state
from eval.videomme.trace import write_trace
from graph_trace import collect_graph_events, normalize_option
from performance_trace import finish_question_tracking, start_question_tracking


performance_module.NODE_ORDER = (
    "plan",
    "localization",
    "segment_asr",
    "memory",
    "perception",
    "attention",
    "reflect",
    "answer",
)


def run_dataset(
    graph: Any,
    videos: List[Dict[str, Any]],
    video_names: Dict[str, str],
    video_root: str,
    preprocess_root: str,
    output_path: str,
    accuracy_path: str,
    log_root: str,
    rerun: bool,
    max_videos: int,
    max_iterations: int,
    max_concurrency: int,
    diagnostic_logging: bool,
) -> None:
    """Run the unchanged Video-MME question loop."""
    existing = {} if rerun else existing_records(output_path)

    selected = []  # type: List[Tuple[Dict[str, Any], str, str]]
    missing_videos = 0
    missing_preprocess = 0
    for video in videos:
        if len(selected) >= max_videos:
            break
        video_id = str(video["video_id"])
        video_name = video_names[video_id]
        video_path = runtime.resolve_video_path(video_root, video_name)
        if video_path is None:
            missing_videos += 1
            continue
        if not runtime.has_preprocess_files(preprocess_root, video_name):
            missing_preprocess += 1
            continue
        selected.append((video, video_name, video_path))

    if missing_videos or missing_preprocess:
        print(
            "Video-MME | skipped: missing videos {}, missing preprocessing {}".format(
                missing_videos, missing_preprocess
            )
        )

    total_questions = sum(
        len(video.get("questions", [])) for video, _, _ in selected
    )
    progress = tqdm(
        total=total_questions,
        desc="Video-MME",
        unit="question",
        dynamic_ncols=True,
        file=sys.stdout,
        postfix="accuracy -- | videos 0/{}".format(len(selected)),
        bar_format=(
            "{desc:<10} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        ),
    )

    for video_index, (video, video_name, video_path) in enumerate(selected, start=1):
        video_id = str(video["video_id"])
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_name)
        log_path = os.path.join(log_root, "{}.log".format(safe_name))
        log_initialized = diagnostic_logging and os.path.isfile(log_path) and not rerun
        for question in video.get("questions", []):
            question_id = str(question["question_id"])
            existing_record = existing.get((video_id, question_id))
            if existing_record is not None:
                question["response"] = existing_record["response"]
                if existing_record.get("response_detail") is not None:
                    question["response_detail"] = existing_record["response_detail"]
                accuracy = runtime.calculate_overall_accuracy(videos, "Video-MME")
                progress.set_postfix_str(
                    "accuracy {accuracy_percent:.2f}% ({correct}/{total_evaluated})"
                    " | videos {}/{}".format(video_index, len(selected), **accuracy),
                    refresh=False,
                )
                progress.update(1)
                continue

            initial_state = build_initial_state(
                video, video_name, video_path, question, max_iterations
            )
            events = []  # type: List[Dict[str, Any]]
            start_question_tracking(video_id, video_name, question_id)
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
            if runtime_error:
                progress.write(
                    "[ERROR] video_id={}, question_id={}: {}".format(
                        video_id, question_id, runtime_error.splitlines()[0]
                    )
                )
            for message in final_state.get("errors", []):
                progress.write(
                    "[ERROR] video_id={}, question_id={}: {}".format(
                        video_id, question_id, message
                    )
                )
            response = final_state.get("final_answer") or runtime.DEFAULT_FORCED_OPTION
            question["response"] = response
            final_state["final_answer"] = response
            ground_truth = normalize_option(question.get("answer"))
            evaluation = {
                "ground_truth": ground_truth or None,
                "answer_correct": (
                    normalize_option(response) == ground_truth if ground_truth else None
                ),
            }
            if diagnostic_logging:
                evaluation = write_trace(
                    log_path,
                    {
                        "video_id": video_id,
                        "video_name": video_name,
                        "question_id": question_id,
                        "task_type": question.get("task_type"),
                        "question": question.get("question"),
                        "options": question.get("options", []),
                        "ground_truth": question.get("answer"),
                    },
                    events,
                    final_state,
                    runtime_error,
                    append=log_initialized,
                )
                log_initialized = True
            detail = dict(final_state.get("answer_output", {}))
            detail.update({
                "ground_truth": evaluation.get("ground_truth"),
                "answer_correct": evaluation.get("answer_correct"),
            })
            if diagnostic_logging:
                detail["trace_log"] = log_path
            question["response_detail"] = detail

            accuracy = runtime.calculate_overall_accuracy(videos, "Video-MME")
            progress.set_postfix_str(
                "accuracy {accuracy_percent:.2f}% ({correct}/{total_evaluated})"
                " | videos {}/{}".format(video_index, len(selected), **accuracy),
                refresh=False,
            )
            progress.update(1)

        runtime.save_experiment_results(
            videos, output_path, accuracy_path, "Video-MME"
        )
    final_accuracy = runtime.save_experiment_results(
        videos, output_path, accuracy_path, "Video-MME"
    )
    progress.set_postfix_str(
        "accuracy {accuracy_percent:.2f}% ({correct}/{total_evaluated})"
        " | videos {}/{}".format(len(selected), len(selected), **final_accuracy),
        refresh=False,
    )
    progress.close()
