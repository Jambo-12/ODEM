"""Structured shared state for the M3-Bench open-ended QA LangGraph."""

from typing import Any, Dict, List

from typing_extensions import TypedDict


class M3BenchState(TypedDict, total=False):
    """Store safe inputs, retrieval state, and answer-audit data for one question."""

    # Safe dataset inputs remain unchanged and exclude gold answers and official reasoning.
    video_id: str  # Video ID, filename, and preprocessing directory name.
    video_name: str  # Extension-free video name.
    video_path: str  # Absolute source-video path used by ASR and visual tools.
    video_duration_sec: float  # Total duration read from the caption timeline.
    question_id: str  # Current open-ended question ID.
    question: str  # Original question text.
    before_clip: int  # Last accessible 30-second clip, inclusive.
    cutoff_time_sec: float  # Upper bound for all memory, audio, and visual evidence.

    # Plan output, updated during initial planning or Reflect-requested replanning.
    question_type: str  # global or local; affects only retrieval-query organization.
    retrieval_target: str  # Core fact or relation to retrieve this pass.
    retrieval_query: str  # Main query encoded into Summary and ViCLIP spaces.
    sub_queries: List[str]  # One to four supplemental queries for distinct evidence.
    answer_evidence_criterion: str  # Evidence criterion required before accepting an answer.
    query_history: List[Dict[str, Any]]  # Compact main-query and sub-query history.
    plan_model_output: str  # Raw Plan response used only for diagnostics.
    replan_feedback: Dict[str, Any]  # Missing-evidence diagnosis passed from Reflect to Plan.

    # Summary + ViCLIP localization output restricted to the visible time range.
    query_results: List[Dict[str, Any]]  # Independent fused Top-5 and scores for each query.
    current_retrieved_segment_ids: List[int]  # Deduplicated segment IDs hit this pass.
    pending_asr_segment_ids: List[int]  # Newly retrieved segments awaiting transcription.
    localization_metadata: Dict[str, Any]  # Retrieval mode, weights, query count, and visibility stats.

    # Episodic memory keeps one merged record per segment across replanning passes.
    episodic_memory: List[Dict[str, Any]]  # Captions, ASR, matched queries, and retrieval scores.
    memory_context: str  # Serialized caption + ASR context C within the token budget.
    memory_context_segment_ids: List[int]  # Segment IDs currently represented in C.
    memory_token_count: int  # Approximate token count of C.
    memory_assembly_status: Dict[str, Any]  # Retention, truncation, and budget statistics.
    asr_batch_status: Dict[str, Any]  # ASR execution statistics for retrieved segments.

    # Structured Reflect requests to visual tools and accumulated evidence.
    tool_request: Dict[str, Any]  # Tool type, segment/range, spatial target, and purpose.
    tool_evidence: List[Dict[str, Any]]  # Accumulated audiovisual evidence from visual tools.
    latest_tool_evidence: List[Dict[str, Any]]  # Evidence added by the latest tool call.
    pending_evidence_source: str  # Primary evidence source for the next Reason pass.
    processed_segment_ids: List[int]  # Segments already inspected by Perception.
    attention_target_history: List[str]  # Attention ranges already inspected.

    # Reason output; iteration increases only when Reason executes.
    answer_state: Dict[str, Any]  # Natural-language candidate, evidence status, and missing information.
    reason_result: Dict[str, Any]  # Candidate answer, sufficiency, and compact reasoning summary.
    iteration_history: List[Dict[str, Any]]  # Reasoning summaries and Reflect actions for up to three passes.
    reason_model_output: str  # Raw Reason response used only for diagnostics.

    # Reflect output audits Reason and selects the next evidence action.
    reflect_result: Dict[str, Any]  # Route, diagnosis, and structured tool target.
    reflect_model_output: str  # Raw Reflect response used only for diagnostics.
    route: str  # replan, perception, or attention.

    # Graph control and final output.
    iteration: int  # Number of Reason executions, capped at 3.
    max_iterations: int  # Maximum Reason executions.
    working_memory: List[Dict[str, Any]]  # Short node logs, not the primary reasoning context.
    final_answer: str  # Final nonempty natural-language answer.
    answer_output: Dict[str, Any]  # Answer evidence, status, and pass summaries.
    errors: List[str]  # Accumulated recoverable node errors.


def build_initial_state(
    video: Dict[str, Any],
    question: Dict[str, Any],
    max_iterations: int,
) -> M3BenchState:
    """Create an initial state without official answers or reasoning."""
    return M3BenchState(
        video_id=str(video["video_id"]),
        video_name=str(video["video_name"]),
        video_path=str(video["video_path"]),
        video_duration_sec=float(video["duration"]),
        question_id=str(question["question_id"]),
        question=str(question["question"]),
        before_clip=int(question["before_clip"]),
        cutoff_time_sec=float(question["cutoff_time_sec"]),
        question_type="",
        retrieval_target="",
        retrieval_query="",
        sub_queries=[],
        answer_evidence_criterion="",
        query_history=[],
        plan_model_output="",
        replan_feedback={},
        query_results=[],
        current_retrieved_segment_ids=[],
        pending_asr_segment_ids=[],
        localization_metadata={},
        episodic_memory=[],
        memory_context="",
        memory_context_segment_ids=[],
        memory_token_count=0,
        memory_assembly_status={},
        asr_batch_status={},
        tool_request={},
        tool_evidence=[],
        latest_tool_evidence=[],
        pending_evidence_source="caption_asr_memory",
        processed_segment_ids=[],
        attention_target_history=[],
        answer_state={},
        reason_result={},
        iteration_history=[],
        reason_model_output="",
        reflect_result={},
        reflect_model_output="",
        route="replan",
        iteration=0,
        max_iterations=max(1, int(max_iterations)),
        working_memory=[],
        final_answer="",
        answer_output={},
        errors=[],
    )
