"""Video-MME answer handling and LangGraph wiring."""

from typing import Any, Dict, Tuple

from langgraph.graph import END, START, StateGraph

from eval import runtime
from eval.videomme.state import VideoMMEState
from performance_trace import timed_node


VALID_ROUTES = ("replan", "perception", "attention")


def answer_node(state: VideoMMEState) -> Dict[str, Any]:
    """Save Reason's A-E best-effort conclusion without another model call."""
    result = state.get("reason_result", {})
    option = str(result.get("candidate_answer") or "").strip().upper()
    if option not in runtime.VALID_OPTION_LETTERS:
        option = runtime.DEFAULT_FORCED_OPTION
    history = state.get("iteration_history", [])
    basis = ""
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        basis = str(history[-1].get("reason_summary") or "").strip()
    if not basis:
        basis = str(result.get("reason_summary") or "").strip()
    return {
        "final_answer": option,
        "answer_output": {
            "option": option,
            "basis": basis or "Option {} is the current best-effort result.".format(option),
            "answer_status": (
                "verified" if result.get("evidence_sufficient") is True else "best_effort"
            ),
            "iteration_history": history[-3:] if isinstance(history, list) else [],
            "answer_evidence_criterion": state.get("answer_evidence_criterion", ""),
            "answer_criterion_satisfied": result.get("evidence_sufficient") is True,
        },
    }


def route_after_reason(state: VideoMMEState) -> str:
    result = state.get("reason_result", {})
    if result.get("evidence_sufficient") is True:
        return "answer"
    if int(state.get("iteration", 0)) >= int(state.get("max_iterations", 3)):
        return "answer"
    return "reflect"


def route_after_reflect(state: VideoMMEState) -> str:
    route = str(state.get("route", "replan"))
    return route if route in VALID_ROUTES else "replan"


def build_graph(modules: Tuple[Any, ...]) -> Any:
    (
        plan_agent,
        localization_node,
        segment_asr_node,
        memory_node,
        attention_agent,
        perception_agent,
        reason_agent,
        reflect_agent,
    ) = modules
    builder = StateGraph(VideoMMEState)
    builder.add_node("plan", timed_node("plan", plan_agent))
    builder.add_node("localization", timed_node("localization", localization_node))
    builder.add_node("segment_asr", timed_node("segment_asr", segment_asr_node))
    builder.add_node("memory", timed_node("memory", memory_node))
    builder.add_node("perception", timed_node("perception", perception_agent))
    builder.add_node("attention", timed_node("attention", attention_agent))
    builder.add_node("reason", timed_node("reason", reason_agent))
    builder.add_node("reflect", timed_node("reflect", reflect_agent))
    builder.add_node("answer", timed_node("answer", answer_node))
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "localization")
    builder.add_edge("localization", "segment_asr")
    builder.add_edge("segment_asr", "memory")
    builder.add_edge("memory", "reason")
    builder.add_edge("perception", "reason")
    builder.add_edge("attention", "reason")
    builder.add_conditional_edges(
        "reason", route_after_reason, {"answer": "answer", "reflect": "reflect"}
    )
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"replan": "plan", "perception": "perception", "attention": "attention"},
    )
    builder.add_edge("answer", END)
    return builder.compile(name="odem_videomme_streaming_episodic")
