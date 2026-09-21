"""
Generate human-readable, node-by-node LangGraph inference traces.

This module has no standalone CLI. Benchmark agents call it to retain questions, graph flow,
candidate ranges, model responses, and final correctness without dumping the full shared state.
"""

import json
import os
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

NODE_TITLES = {
    "plan": "PLAN AGENT",
    "localization": "LOCALIZATION NODE",
    "perception": "PERCEPTION AGENT",
    "attention": "ATTENTION AGENT",
    "reason": "REASON AGENT",
    "reflect": "REFLECT AGENT",
    "answer": "ANSWER NODE",
}


def make_json_safe(value: Any) -> Any:
    """Recursively convert NumPy scalars and similar values for stable serialization."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def collect_graph_events(
    graph: Any,
    initial_state: Dict[str, Any],
    recursion_limit: int,
    max_concurrency: int = 1,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """Stream the graph and collect node updates and final state in execution order."""
    events = []  # type: List[Dict[str, Any]]
    node_counts = {}  # type: Dict[str, int]
    final_state = dict(initial_state)  # type: Dict[str, Any]
    runtime_error = ""
    try:
        updates = graph.stream(
            initial_state,
            {
                "recursion_limit": max(1, int(recursion_limit)),
                "max_concurrency": max(1, int(max_concurrency)),
            },
            stream_mode="updates",
        )
        for graph_update in updates:
            if not isinstance(graph_update, dict):
                continue
            for node_name, raw_output in graph_update.items():
                output = raw_output if isinstance(raw_output, dict) else {}
                node_counts[node_name] = node_counts.get(node_name, 0) + 1
                safe_output = make_json_safe(output)
                event = {
                    "sequence": len(events) + 1,
                    "node": str(node_name),
                    "invocation": node_counts[node_name],
                    "output": safe_output,
                }
                events.append(event)
                final_state.update(safe_output)
                if event_callback is not None:
                    event_callback(event)
    except Exception as error:
        runtime_error = "{}: {}\n{}".format(
            type(error).__name__,
            error,
            traceback.format_exc(),
        )
    return events, make_json_safe(final_state), runtime_error


def normalize_option(value: Any) -> str:
    """Normalize dataset or model answers to A through E when possible."""
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    match = re.search(r"(?:^|\b)([A-E])(?:\b|[.\):])", text)
    return match.group(1) if match else text


def _pretty(value: Any) -> str:
    """Format structured fields as stable indented experiment-log text."""
    return json.dumps(value, ensure_ascii=False, indent=2)


def _time_range_text(start_time: Any, end_time: Any) -> str:
    """Format candidate start and end seconds as compact log text."""
    try:
        return "{:.3f}s - {:.3f}s".format(float(start_time), float(end_time))
    except (TypeError, ValueError):
        return "unknown"


def _candidate_lines(output: Dict[str, Any]) -> List[str]:
    """Extract localization ranks, four-key scores, and fused multimodal scores."""
    lines = []  # type: List[str]
    candidates = output.get("candidate_segments", [])
    if not isinstance(candidates, list):
        return lines
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        lines.append(
            "  Rank {rank}: segment={segment}, retrieved_time={retrieved_time}, "
            "observation_time={observation_time}, "
            "best_key={best_key}, text={text}, visual={visual}, fused={fused}".format(
                rank=candidate.get("rank", "?"),
                segment=candidate.get("segment_id", "?"),
                retrieved_time=_time_range_text(
                    candidate.get("original_start_time"),
                    candidate.get("original_end_time"),
                ),
                observation_time=_time_range_text(
                    candidate.get("start_time"),
                    candidate.get("end_time"),
                ),
                best_key=candidate.get("best_caption_key", "n/a"),
                text=candidate.get("textual_score", "n/a"),
                visual=candidate.get("visual_score", "n/a"),
                fused=candidate.get("fused_score", "n/a"),
            )
        )
        key_scores = candidate.get("caption_key_scores")
        if isinstance(key_scores, dict):
            lines.append(
                "    Caption key scores: {}".format(
                    ", ".join(
                        "{}={}".format(key, key_scores.get(key, "n/a"))
                        for key in ("environment", "event", "attention", "summary")
                    )
                )
            )
    return lines


def _observation_without_raw(observation: Any) -> Dict[str, Any]:
    """Copy an observation while removing separately displayed raw model and ASR text."""
    if not isinstance(observation, dict):
        return {}
    return {
        key: value
        for key, value in observation.items()
        if key not in ("raw_model_output", "asr_transcript")
    }


def _speaker_transcript_lines(transcript: Any) -> List[str]:
    """Format a speaker transcript chronologically for manual inspection."""
    if not isinstance(transcript, dict):
        return ["Speaker ASR: <unavailable>"]
    utterances = transcript.get("utterances", [])
    if not isinstance(utterances, list):
        utterances = []
    lines = [
        "Speaker ASR: status={}, cached={}, utterances={}".format(
            transcript.get("status", "unknown"),
            transcript.get("cached", False),
            len(utterances),
        )
    ]
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        lines.append(
            "  [{}] {}: {}".format(
                _time_range_text(
                    utterance.get("start_time"),
                    utterance.get("end_time"),
                ),
                utterance.get("speaker", "SPEAKER_UNKNOWN"),
                utterance.get("text", ""),
            )
        )
    if transcript.get("error"):
        lines.append("  Error: {}".format(transcript.get("error")))
    return lines


def _format_event(event: Dict[str, Any]) -> List[str]:
    """Convert one node update into a log section focused on key decisions."""
    node_name = str(event.get("node", "unknown"))
    output = event.get("output", {})
    if not isinstance(output, dict):
        output = {}
    title = NODE_TITLES.get(node_name, node_name.upper())
    lines = [
        "",
        "=" * 88,
        "STEP {sequence:02d} | {title} | CALL {invocation}".format(
            sequence=int(event.get("sequence", 0)),
            title=title,
            invocation=int(event.get("invocation", 1)),
        ),
        "=" * 88,
    ]

    if node_name == "plan":
        lines.extend([
            "Question type: {}".format(output.get("question_type", "unknown")),
            "Retrieval plan:",
            str(output.get("retrieval_plan", "")),
            "Answer evidence criterion:",
            str(output.get("answer_evidence_criterion", "")),
            "Retrieval query:",
            str(output.get("retrieval_query", "")),
            "Raw LLM output:",
            str(output.get("plan_model_output", "") or "<unavailable>"),
        ])
    elif node_name == "localization":
        candidate = output.get("current_candidate", {})
        if not isinstance(candidate, dict):
            candidate = {}
        metadata = output.get("localization_metadata", {})
        lines.extend([
            "Selection status: {}".format(
                output.get("candidate_selection_status", "unknown")
            ),
            "Selected candidate: rank={}, segment={}, retrieved_time={}, observation_time={}".format(
                candidate.get("rank", "?"),
                candidate.get("segment_id", "?"),
                _time_range_text(
                    candidate.get("original_start_time"),
                    candidate.get("original_end_time"),
                ),
                _time_range_text(
                    candidate.get("start_time"),
                    candidate.get("end_time"),
                ),
            ),
            "Retrieval metadata:",
            _pretty(metadata if isinstance(metadata, dict) else {}),
            "Ranked candidates:",
        ])
        candidate_lines = _candidate_lines(output)
        lines.extend(candidate_lines or ["  <existing ranking reused or no candidate>"])
    elif node_name == "perception":
        observation = output.get("current_perception_observation", {})
        if not isinstance(observation, dict):
            observation = {}
        transcript = output.get(
            "current_asr_transcript",
            observation.get("asr_transcript", {}),
        )
        lines.extend([
            "Decision: question_type={}, relevance={}, next={}".format(
                output.get("question_type", "unknown"),
                output.get("candidate_relevance", "unknown"),
                output.get("perception_route", "unknown"),
            ),
            "Parsed VLM observation:",
            _pretty(_observation_without_raw(observation)),
            "Raw VLM output:",
            str(observation.get("raw_model_output", "") or "<unavailable>"),
        ])
        lines.extend(_speaker_transcript_lines(transcript))
    elif node_name == "attention":
        observation = output.get("current_attention_observation", {})
        if not isinstance(observation, dict):
            observation = {}
        lines.extend([
            "Parsed focused observation:",
            _pretty(_observation_without_raw(observation)),
            "Raw VLM output:",
            str(observation.get("raw_model_output", "") or "<unavailable>"),
        ])
    elif node_name == "reason":
        reason_result = output.get("reason_result", {})
        if not isinstance(reason_result, dict):
            reason_result = {}
        lines.extend([
            "Reason decision: candidate_answer={}, evidence_sufficient={}".format(
                reason_result.get("candidate_answer", ""),
                reason_result.get("evidence_sufficient", False),
            ),
            "Compact reasoning history:",
            _pretty(reason_result.get("reasoning_history", [])),
            "Raw LLM output:",
            str(output.get("reason_model_output", "") or "<unavailable>"),
        ])
    elif node_name == "reflect":
        if "reflect_result" in output:
            lines.extend([
                "Graph decision: route={}".format(
                    output.get("route", "unknown"),
                ),
                "Parsed routing result:",
                _pretty(output.get("reflect_result", {})),
                "Raw LLM output:",
                str(output.get("reflect_model_output", "") or "<unavailable>"),
            ])
        else:
            # Support dataset implementations that have not split Reason and Reflect.
            lines.extend([
                "Graph decision: route={}, verified_answer={}".format(
                    output.get("route", "unknown"),
                    output.get("verified_conclusion") or "<not verified>",
                ),
                "Answer criterion satisfied: {}".format(
                    output.get("answer_criterion_satisfied", False)
                ),
                "Parsed reflection:",
                _pretty(output.get("reflection_result", {})),
                "Raw LLM output:",
                str(output.get("reflect_model_output", "") or "<unavailable>"),
            ])
    elif node_name == "answer":
        lines.extend([
            "Final answer: {}".format(output.get("final_answer", "")),
            "Answer detail:",
            _pretty(output.get("answer_output", {})),
        ])
    else:
        lines.extend(["Node output:", _pretty(output)])

    node_errors = output.get("errors", [])
    if isinstance(node_errors, list) and node_errors:
        lines.extend(["Accumulated errors:", _pretty(node_errors)])
    return lines


def write_graph_trace(
    log_path: str,
    case_metadata: Dict[str, Any],
    events: List[Dict[str, Any]],
    final_state: Dict[str, Any],
    runtime_error: str = "",
    append: bool = False,
) -> Dict[str, Any]:
    """Write a readable single-question trace and return answer comparison results."""
    predicted = normalize_option(final_state.get("final_answer"))
    ground_truth = normalize_option(case_metadata.get("ground_truth"))
    answer_correct = predicted == ground_truth if ground_truth else None
    route_trace = " -> ".join(
        "{}#{}".format(
            str(event.get("node", "unknown")).upper(),
            event.get("invocation", 1),
        )
        for event in events
    ) or "<no node completed>"

    lines = [
        "# ODEM LANGGRAPH INFERENCE TRACE",
        "",
        "Video: {} ({})".format(
            case_metadata.get("video_name", ""),
            case_metadata.get("video_id", ""),
        ),
        "Question ID: {}".format(case_metadata.get("question_id", "")),
        "Task type: {}".format(case_metadata.get("task_type") or "unknown"),
        "Question: {}".format(case_metadata.get("question", "")),
        "Options:",
    ]
    lines.extend(
        "  {}".format(option)
        for option in case_metadata.get("options", [])
    )
    lines.extend([
        "",
        "Graph flow:",
        route_trace,
    ])
    for event in events:
        lines.extend(_format_event(event))

    verdict = (
        "CORRECT" if answer_correct is True
        else "INCORRECT" if answer_correct is False
        else "NOT_AVAILABLE"
    )
    lines.extend([
        "",
        "# FINAL EVALUATION",
        "Predicted answer: {}".format(predicted or "<empty>"),
        "Ground-truth answer: {}".format(ground_truth or "<unavailable>"),
        "Answer correctness: {}".format(verdict),
        "Execution status: {}".format("FAILED" if runtime_error else "COMPLETED"),
    ])
    if final_state.get("errors"):
        lines.extend([
            "Agent errors:",
            _pretty(final_state.get("errors", [])),
        ])
    if runtime_error:
        lines.extend([
            "Runtime error:",
            runtime_error.rstrip(),
        ])
    lines.extend(["", ""])

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    mode = "a" if append else "w"
    with open(log_path, mode, encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))
    return {
        "predicted_answer": predicted or None,
        "ground_truth": ground_truth or None,
        "answer_correct": answer_correct,
        "log_path": log_path,
    }
