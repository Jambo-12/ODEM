"""Produce concise Video-MME traces without affecting inference logic."""

from typing import Any, Dict, List

from graph_trace import write_graph_trace


def write_trace(
    log_path: str,
    case_metadata: Dict[str, Any],
    events: List[Dict[str, Any]],
    final_state: Dict[str, Any],
    runtime_error: str,
    append: bool,
) -> Dict[str, Any]:
    """Compress node events while preserving evaluation and resume interfaces."""
    log_events = []  # type: List[Dict[str, Any]]
    for event in events:
        copied = dict(event)
        output = dict(event.get("output", {}))
        node_name = event.get("node")
        if node_name == "plan":
            output["retrieval_plan"] = output.get("retrieval_target", "")
        elif node_name == "localization":
            candidates = []  # type: List[Dict[str, Any]]
            for query_result in output.get("query_results", []):
                if not isinstance(query_result, dict):
                    continue
                for hit in query_result.get("hits", []):
                    if not isinstance(hit, dict):
                        continue
                    candidates.append({
                        "rank": "{}:{}".format(
                            query_result.get("query_id", "query"), hit.get("rank")
                        ),
                        "segment_id": hit.get("segment_id"),
                        "original_start_time": hit.get("start_time"),
                        "original_end_time": hit.get("end_time"),
                        "start_time": hit.get("start_time"),
                        "end_time": hit.get("end_time"),
                        "best_caption_key": "summary",
                        "textual_score": hit.get("summary_score"),
                        "visual_score": hit.get("visual_score"),
                        "fused_score": hit.get("fused_score"),
                    })
            output = {
                "candidate_selection_status": output.get(
                    "localization_metadata", {}
                ).get("status", "unknown"),
                "candidate_segments": candidates,
                "localization_metadata": output.get("localization_metadata", {}),
                "errors": output.get("errors", []),
            }
        elif node_name == "segment_asr":
            output = {
                "asr_batch_status": output.get("asr_batch_status", {}),
                "errors": output.get("errors", []),
            }
        elif node_name == "memory":
            output = {
                "memory_assembly_status": output.get("memory_assembly_status", {}),
                "memory_context": output.get("memory_context", ""),
                "errors": output.get("errors", []),
            }
        elif node_name == "perception":
            evidence = output.get("latest_tool_evidence", [])
            first = evidence[0] if isinstance(evidence, list) and evidence else {}
            targets = []  # type: List[Dict[str, Any]]
            raw_outputs = []  # type: List[str]
            for item in evidence if isinstance(evidence, list) else []:
                if not isinstance(item, dict):
                    continue
                visual = item.get("visual", {})
                parsed_visual = {
                    key: value for key, value in visual.items()
                    if key != "raw_model_output"
                } if isinstance(visual, dict) else {}
                targets.append({
                    "segment_id": item.get("segment_id"),
                    "time_ranges": item.get("time_ranges"),
                    "visual": parsed_visual,
                    "asr": item.get("asr", {}),
                })
                if isinstance(visual, dict) and visual.get("raw_model_output"):
                    raw_outputs.append(
                        "[segment {}]\n{}".format(
                            item.get("segment_id"), visual["raw_model_output"]
                        )
                    )
            output = {
                "question_type": final_state.get("question_type", "unknown"),
                "candidate_relevance": "tool_selected",
                "perception_route": "reason",
                "current_perception_observation": {
                    "targets": targets,
                    "raw_model_output": "\n\n".join(raw_outputs),
                },
                "current_asr_transcript": first.get("asr", {}),
                "errors": output.get("errors", []),
            }
        elif node_name == "attention":
            evidence = output.get("latest_tool_evidence", [])
            first = evidence[0] if isinstance(evidence, list) and evidence else {}
            visual = dict(first.get("visual", {})) if isinstance(
                first.get("visual"), dict
            ) else {}
            visual["same_range_asr"] = first.get("asr", [])
            output = {
                "current_attention_observation": visual,
                "errors": output.get("errors", []),
            }
        elif node_name == "reason":
            reason_result = dict(output.get("reason_result", {}))
            reason_result["reasoning_history"] = output.get("iteration_history", [])
            output["reason_result"] = reason_result
        copied["output"] = output
        log_events.append(copied)
    return write_graph_trace(
        log_path=log_path,
        case_metadata=case_metadata,
        events=log_events,
        final_state=final_state,
        runtime_error=runtime_error,
        append=append,
    )
