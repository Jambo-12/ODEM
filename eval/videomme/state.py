"""Structured shared state for the Video-MME episodic-memory LangGraph."""

from typing import Any, Dict, List

from typing_extensions import TypedDict


class VideoMMEState(TypedDict, total=False):
    """Store structured planning, retrieval, and evidence-audit state for one question."""

    # Dataset inputs remain unchanged throughout the question lifecycle.
    video_id: str  # Internal Video-MME video ID.
    video_name: str  # Extension-free filename and preprocessing directory name.
    video_path: str  # Absolute source-video path used by ASR and visual tools.
    question_id: str  # Current multiple-choice question ID.
    question: str  # Original question text.
    options: List[str]  # Candidate answers A through E.

    # Plan output, updated during initial planning or Reflect-requested replanning.
    question_type: str  # global or local; affects query design, not graph routing.
    retrieval_target: str  # Core evidence target for this pass.
    retrieval_query: str  # Main query encoded into Summary and ViCLIP spaces.
    sub_queries: List[str]  # One to four supplemental queries covering distinct evidence.
    answer_evidence_criterion: str  # Observable criterion required before Reason accepts an answer.
    query_history: List[Dict[str, Any]]  # Compact history of main and sub-queries.
    plan_model_output: str  # Raw Plan response used only for readable diagnostics.
    replan_feedback: Dict[str, Any]  # Missing-evidence diagnosis passed from Reflect to Plan.

    # Summary + ViCLIP localization output; cross-pass memory is not overwritten.
    query_results: List[Dict[str, Any]]  # Independent fused Top-5 and scores for each query.
    current_retrieved_segment_ids: List[int]  # Deduplicated segment IDs hit this pass.
    pending_asr_segment_ids: List[int]  # Newly retrieved segments awaiting transcription.
    localization_metadata: Dict[str, Any]  # Retrieval mode, fusion weights, query count, and status.

    # Episodic memory keeps one merged record per segment across replanning passes.
    episodic_memory: List[Dict[str, Any]]  # Captions, ASR, matched queries, and retrieval scores.
    memory_context: str  # Serialized caption + ASR context C within the token budget.
    memory_context_segment_ids: List[int]  # Segments whose full content remains in C.
    memory_token_count: int  # Approximate token count used only for budget control.
    memory_assembly_status: Dict[str, Any]  # Retention, truncation, and budget statistics.
    asr_batch_status: Dict[str, Any]  # Concurrent ASR success, cache, silence, and failure counts.

    # Structured requests from Reflect to optional visual tools.
    tool_request: Dict[str, Any]  # Route, target segment/range, spatial target, and purpose.
    tool_evidence: List[Dict[str, Any]]  # Accumulated direct evidence from visual tools.
    latest_tool_evidence: List[Dict[str, Any]]  # Evidence added by the latest tool call.
    pending_evidence_source: str  # Evidence source for the next Reason pass.
    processed_segment_ids: List[int]  # Segments already inspected by Perception.
    attention_target_history: List[str]  # Normalized Attention ranges already inspected.

    # Reason output; iteration increases only when Reason executes.
    answer_state: Dict[str, Any]  # Option support, tentative answer, and global sufficiency.
    reason_result: Dict[str, Any]  # Candidate answer, sufficiency, summary, and missing evidence.
    iteration_history: List[Dict[str, Any]]  # Compact reasoning and Reflect history for up to three passes.
    reason_model_output: str  # Raw Reason response used only for diagnostics.

    # Reflect output audits Reason and selects the next evidence action.
    reflect_result: Dict[str, Any]  # Route, diagnosis, and structured tool target.
    reflect_model_output: str  # Raw Reflect response used only for diagnostics.
    route: str  # replan, perception, attention, or answer.

    # Graph control and final output.
    iteration: int  # Number of Reason executions; other tools do not increment it.
    max_iterations: int  # Maximum Reason executions, defaulting to 3 for Video-MME.
    working_memory: List[Dict[str, Any]]  # Short node records, not the primary reasoning context.
    final_answer: str  # Final option A through E.
    answer_output: Dict[str, Any]  # Answer evidence, status, and pass summaries.
    errors: List[str]  # Accumulated recoverable node errors.


def build_initial_state(
    video: Dict[str, Any],
    video_name: str,
    video_path: str,
    question: Dict[str, Any],
    max_iterations: int,
) -> VideoMMEState:
    """Create the initial state shared by all graph nodes for one question."""
    return VideoMMEState(
        video_id=str(video["video_id"]),
        video_name=video_name,
        video_path=video_path,
        question_id=str(question["question_id"]),
        question=str(question["question"]),
        options=[str(option) for option in question.get("options", [])],
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
