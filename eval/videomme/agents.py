"""Implement the Plan, Reason, and Reflect agents in the Video-MME pipeline."""

import json
from typing import Any, Dict, List

from eval.common import (
    extract_json_object,
    format_options,
    invoke_llm,
    render_prompt,
)

from eval.videomme.state import VideoMMEState


VALID_OPTIONS = ("A", "B", "C", "D", "E")
VALID_ROUTES = ("replan", "perception", "attention")
GLOBAL_CUES = (
    "mainly about",
    "main topic",
    "central theme",
    "overall purpose",
    "overall content",
    "best summarizes",
    "best summary",
    "dominant activity",
)
DEFAULT_CRITERION = (
    "Observable Caption, speech, or direct visual evidence must distinguish the "
    "best answer from plausible alternatives."
)


def _append_error(state: VideoMMEState, message: str) -> List[str]:
    """Append an error to the current task-specific state."""
    return list(state.get("errors", [])) + [message]


def _append_memory(
    state: VideoMMEState, agent_name: str, summary: str
) -> List[Dict[str, Any]]:
    """Append a short node record for logging, not as full reasoning context."""
    return list(state.get("working_memory", [])) + [{
        "agent": agent_name,
        "summary": summary,
    }]


def _compact(value: Any, word_limit: int) -> str:
    """Compact free-form model text to prevent state growth across iterations."""
    return " ".join(str(value or "").strip().split()[:word_limit])


def _option(value: Any) -> str:
    """Accept only the Video-MME options A through E."""
    normalized = str(value or "").strip().upper()
    return normalized if normalized in VALID_OPTIONS else ""


class VideoMMEPlanAgent:
    """Analyze a question and generate one main query, 1-4 subqueries, and an evidence criterion."""

    def __init__(self, llm: Any, prompt_template: str) -> None:
        self.llm = llm
        self.prompt_template = prompt_template

    @staticmethod
    def _question_type(value: Any, question: str, previous: str) -> str:
        """Classify strictly on the first pass and preserve the question type when replanning."""
        if previous in ("global", "local"):
            return previous
        parsed = str(value or "").strip().lower()
        if parsed in ("global", "local"):
            return parsed
        lowered = str(question).lower()
        return "global" if any(cue in lowered for cue in GLOBAL_CUES) else "local"

    @staticmethod
    def _query(value: Any, fallback: str) -> str:
        """Normalize one neutral natural-language query."""
        normalized = _compact(value, 40)
        return normalized or _compact(fallback, 40) or "Relevant video evidence"

    @classmethod
    def _sub_queries(cls, value: Any, question: str, main_query: str) -> List[str]:
        """Accept strings or objects with a query field, deduplicate, and retain 1-4 items."""
        raw_items = value if isinstance(value, list) else []
        output = []  # type: List[str]
        seen = {main_query.lower()}
        for item in raw_items:
            raw_query = item.get("query") if isinstance(item, dict) else item
            query = cls._query(raw_query, "")
            if not query or query.lower() in seen:
                continue
            seen.add(query.lower())
            output.append(query)
            if len(output) >= 4:
                break
        if not output:
            output.append(
                "Locate evidence that distinguishes the candidate answers for: {}".format(
                    _compact(question, 28)
                )
            )
        return output

    def __call__(self, state: VideoMMEState) -> Dict[str, Any]:
        """Create the initial plan or rewrite a materially different retrieval query from Reflect gaps."""
        prompt = render_prompt(self.prompt_template, {
            "QUESTION": state.get("question", ""),
            "OPTIONS": format_options(state.get("options", [])),
            "PREVIOUS_RETRIEVAL_QUERY": state.get("retrieval_query", ""),
            "PREVIOUS_SUB_QUERIES": json.dumps(
                state.get("sub_queries", []), ensure_ascii=False
            ),
            "PREVIOUS_ANSWER_EVIDENCE_CRITERION": state.get(
                "answer_evidence_criterion", ""
            ),
            "MEMORY_CONTEXT": state.get("memory_context", ""),
            "ANSWER_STATE": json.dumps(state.get("answer_state", {}), ensure_ascii=False),
            "ITERATION_HISTORY": json.dumps(
                state.get("iteration_history", []), ensure_ascii=False
            ),
            "REFLECTION_FEEDBACK": json.dumps(
                state.get("replan_feedback", {}), ensure_ascii=False
            ),
        })
        raw_output = ""
        errors = list(state.get("errors", []))
        try:
            raw_output = invoke_llm(self.llm, prompt, component="plan_llm")
            parsed = extract_json_object(raw_output)
        except Exception as error:
            parsed = {}
            errors = _append_error(state, "Plan Agent failed: {}".format(error))

        question = state.get("question", "")
        question_type = self._question_type(
            parsed.get("question_type"), question, state.get("question_type", "")
        )
        retrieval_query = self._query(parsed.get("retrieval_query"), question)
        sub_queries = self._sub_queries(
            parsed.get("sub_queries"), question, retrieval_query
        )
        retrieval_target = _compact(parsed.get("retrieval_target"), 45)
        if not retrieval_target:
            retrieval_target = "Find episode evidence that distinguishes the answer options."
        criterion = _compact(parsed.get("answer_evidence_criterion"), 55)
        if not criterion:
            criterion = state.get("answer_evidence_criterion", "") or DEFAULT_CRITERION

        history = [
            dict(item) for item in state.get("query_history", []) if isinstance(item, dict)
        ]
        previous_queries = set()
        for item in history:
            previous_queries.add(str(item.get("retrieval_query", "")).strip().lower())
            previous_queries.update(
                str(query).strip().lower()
                for query in item.get("sub_queries", [])
                if str(query).strip()
            )
        if history and retrieval_query.lower() in previous_queries:
            missing_information = state.get("replan_feedback", {}).get(
                "missing_information", "alternative distinguishing evidence"
            )
            retrieval_query = self._query(
                "{} Focus on {}".format(retrieval_target, missing_information),
                question,
            )
            if retrieval_query.lower() in previous_queries:
                evidence_channel = "spoken" if len(history) == 1 else "visible"
                retrieval_query = self._query(
                    "Unobserved {} evidence for {}".format(
                        evidence_channel, missing_information
                    ),
                    question,
                )
        new_sub_queries = [
            query for query in sub_queries
            if query.lower() not in previous_queries
            and query.lower() != retrieval_query.lower()
        ]
        if not new_sub_queries:
            evidence_channel = "spoken" if len(history) == 1 else "visible"
            new_sub_queries = [self._query(
                "Alternative {} episode evidence for {}".format(
                    evidence_channel, retrieval_target
                ),
                question,
            )]
        sub_queries = new_sub_queries[:4]
        history.append({
            "plan_call": len(history) + 1,
            "retrieval_target": retrieval_target,
            "retrieval_query": retrieval_query,
            "sub_queries": sub_queries,
        })
        return {
            # Localization executes each structured Plan output with fused Summary + ViCLIP retrieval.
            "question_type": question_type,
            "retrieval_target": retrieval_target,
            "retrieval_query": retrieval_query,
            "sub_queries": sub_queries,
            "answer_evidence_criterion": criterion,
            "query_history": history,
            "plan_model_output": raw_output,
            "replan_feedback": {},
            "errors": errors,
            "working_memory": _append_memory(
                state,
                "plan",
                "Plan generated one main query and {} sub-queries.".format(
                    len(sub_queries)
                ),
            ),
        }


class VideoMMEReasonAgent:
    """Maintain answer state from all accumulated memory and use execution count as the iteration."""

    def __init__(self, llm: Any, prompt_template: str) -> None:
        self.llm = llm
        self.prompt_template = prompt_template

    @staticmethod
    def _compact_tool_evidence(value: Any) -> List[Dict[str, Any]]:
        """Remove long raw tool replies and retain only structured audiovisual results for Reason."""
        output = []  # type: List[Dict[str, Any]]
        for item in value if isinstance(value, list) else []:
            if not isinstance(item, dict):
                continue
            copied = {
                key: item.get(key)
                for key in ("source", "segment_id", "time_ranges", "objective", "spatial_target")
                if key in item
            }
            visual = item.get("visual", {})
            if isinstance(visual, dict):
                copied["visual"] = {
                    key: visual.get(key)
                    for key in (
                        "provisional_answer", "key_evidence", "unresolved_details",
                        "observation", "observed_time_range", "observed_time_ranges", "status", "error"
                    )
                    if key in visual
                }
            copied["asr"] = item.get("asr", {})
            output.append(copied)
        return output

    @staticmethod
    def _option_assessments(value: Any, previous: Any) -> Dict[str, Dict[str, Any]]:
        """Normalize the five options into a compact global support/refute/unknown state."""
        current = value if isinstance(value, dict) else {}
        old = previous if isinstance(previous, dict) else {}
        output = {}  # type: Dict[str, Dict[str, Any]]
        for letter in VALID_OPTIONS:
            item = current.get(letter, {})
            if not isinstance(item, dict):
                item = {}
            old_item = old.get(letter, {}) if isinstance(old.get(letter), dict) else {}
            status = str(item.get("status") or old_item.get("status") or "unknown").lower()
            if status not in ("support", "refute", "unknown"):
                status = "unknown"
            output[letter] = {
                "status": status,
                "basis": _compact(item.get("basis") or old_item.get("basis"), 24),
                "evidence_refs": [
                    str(ref) for ref in item.get(
                        "evidence_refs", old_item.get("evidence_refs", [])
                    )[:4]
                ] if isinstance(
                    item.get("evidence_refs", old_item.get("evidence_refs", [])), list
                ) else [],
            }
        return output

    def __call__(self, state: VideoMMEState) -> Dict[str, Any]:
        """Summarize accumulated evidence, update answer state, and add an iteration record of at most 80 words."""
        iteration = int(state.get("iteration", 0)) + 1
        max_iterations = max(1, int(state.get("max_iterations", 3)))
        previous_answer_state = state.get("answer_state", {})
        prompt = render_prompt(self.prompt_template, {
            "QUESTION": state.get("question", ""),
            "OPTIONS": format_options(state.get("options", [])),
            "QUESTION_TYPE": state.get("question_type", "local"),
            "RETRIEVAL_TARGET": state.get("retrieval_target", ""),
            "ANSWER_EVIDENCE_CRITERION": state.get("answer_evidence_criterion", ""),
            "MEMORY_CONTEXT": state.get("memory_context", ""),
            "TOOL_EVIDENCE": json.dumps(
                self._compact_tool_evidence(state.get("tool_evidence", [])),
                ensure_ascii=False,
            ),
            "PREVIOUS_ANSWER_STATE": json.dumps(previous_answer_state, ensure_ascii=False),
            "ITERATION_HISTORY": json.dumps(
                state.get("iteration_history", []), ensure_ascii=False
            ),
            "CURRENT_ITERATION": str(iteration),
            "MAX_ITERATIONS": str(max_iterations),
        })
        raw_output = ""
        errors = list(state.get("errors", []))
        try:
            raw_output = invoke_llm(self.llm, prompt, component="reason_llm")
            parsed = extract_json_object(raw_output)
        except Exception as error:
            parsed = {}
            errors = _append_error(state, "Reason Agent failed: {}".format(error))

        previous_option = _option(previous_answer_state.get("provisional_answer")) \
            if isinstance(previous_answer_state, dict) else ""
        candidate_answer = _option(parsed.get("candidate_answer")) or previous_option or "A"
        sufficient = parsed.get("evidence_sufficient") is True
        reason_summary = _compact(parsed.get("reason_summary"), 80)
        if not reason_summary:
            reason_summary = (
                "Current episodic and tool evidence favors option {}; decisive support is {}."
            ).format(candidate_answer, "sufficient" if sufficient else "still incomplete")
        missing_information = _compact(parsed.get("missing_information"), 40)
        assessments = self._option_assessments(
            parsed.get("option_assessments"),
            previous_answer_state.get("option_assessments", {})
            if isinstance(previous_answer_state, dict) else {},
        )
        answer_state = {
            "provisional_answer": candidate_answer,
            "evidence_sufficient": sufficient,
            "option_assessments": assessments,
            "missing_information": missing_information,
        }
        history = [
            dict(item) for item in state.get("iteration_history", []) if isinstance(item, dict)
        ]
        history.append({
            "iteration": iteration,
            "evidence_source": state.get("pending_evidence_source", "caption_asr_memory"),
            "reason_summary": reason_summary,
            "provisional_answer": candidate_answer,
            "evidence_sufficient": sufficient,
            "reflect_action": {},
        })
        history = history[-max_iterations:]
        result = {
            "candidate_answer": candidate_answer,
            "evidence_sufficient": sufficient,
            "reason_summary": reason_summary,
            "missing_information": missing_information,
            "forced_best_effort": iteration >= max_iterations and not sufficient,
        }
        return {
            # Increment iteration only here; after the third Reason pass, the graph forces a transition to Answer.
            "iteration": iteration,
            "answer_state": answer_state,
            "reason_result": result,
            "iteration_history": history,
            "reason_model_output": raw_output,
            "errors": errors,
            "working_memory": _append_memory(
                state,
                "reason",
                "Reason iteration {} selected option {}; evidence is {}.".format(
                    iteration,
                    candidate_answer,
                    "sufficient" if sufficient else "insufficient",
                ),
            ),
        }


class VideoMMEReflectAgent:
    """Audit Reason evidence gaps and select replan, Perception, or Attention."""

    def __init__(
        self,
        llm: Any,
        prompt_template: str,
        max_perception_segments: int = 2,
        max_attention_ranges: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_perception_segments = max(1, int(max_perception_segments))
        self.max_attention_ranges = max(1, int(max_attention_ranges))

    @staticmethod
    def _episode_catalog(state: VideoMMEState) -> List[Dict[str, Any]]:
        """Provide Reflect only the segment index needed to select a tool target, without copying full memory C."""
        output = []  # type: List[Dict[str, Any]]
        for episode in state.get("episodic_memory", []):
            if not isinstance(episode, dict):
                continue
            output.append({
                "segment_id": episode.get("segment_id"),
                "start_time": episode.get("start_time"),
                "end_time": episode.get("end_time"),
                "summary_score": episode.get("max_summary_score"),
                "visual_score": episode.get("max_visual_score"),
                "fused_score": episode.get(
                    "max_fused_score", episode.get("max_summary_score")
                ),
                "evidence_status": episode.get("evidence_status"),
                "memory_status": episode.get("memory_status"),
                "summary": (episode.get("caption") or {}).get("summary")
                if isinstance(episode.get("caption"), dict) else None,
            })
        return sorted(
            output,
            key=lambda item: -float(item.get("fused_score") or -1.0),
        )

    @staticmethod
    def _patch_history(state: VideoMMEState, action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write the tool and range selected by Reflect back to the latest Reason record."""
        history = [
            dict(item) for item in state.get("iteration_history", []) if isinstance(item, dict)
        ]
        if history:
            history[-1]["reflect_action"] = action
        return history

    def _fallback_perception(self, state: VideoMMEState) -> List[int]:
        processed = {int(value) for value in state.get("processed_segment_ids", [])}
        return [
            int(item["segment_id"])
            for item in self._episode_catalog(state)
            if item.get("segment_id") is not None and int(item["segment_id"]) not in processed
        ][:self.max_perception_segments]

    def __call__(self, state: VideoMMEState) -> Dict[str, Any]:
        """Return exactly one next action without answering again or changing Reason's answer judgment."""
        catalog = self._episode_catalog(state)
        prompt = render_prompt(self.prompt_template, {
            "QUESTION": state.get("question", ""),
            "OPTIONS": format_options(state.get("options", [])),
            "ANSWER_EVIDENCE_CRITERION": state.get("answer_evidence_criterion", ""),
            "REASON_RESULT": json.dumps(state.get("reason_result", {}), ensure_ascii=False),
            "ANSWER_STATE": json.dumps(state.get("answer_state", {}), ensure_ascii=False),
            "ITERATION_HISTORY": json.dumps(
                state.get("iteration_history", []), ensure_ascii=False
            ),
            "EPISODE_CATALOG": json.dumps(catalog, ensure_ascii=False),
            "PROCESSED_SEGMENT_IDS": json.dumps(
                state.get("processed_segment_ids", []), ensure_ascii=False
            ),
            "ATTENTION_TARGET_HISTORY": json.dumps(
                state.get("attention_target_history", []), ensure_ascii=False
            ),
            "CURRENT_ITERATION": str(state.get("iteration", 0)),
            "MAX_ITERATIONS": str(state.get("max_iterations", 3)),
        })
        raw_output = ""
        errors = list(state.get("errors", []))
        try:
            raw_output = invoke_llm(self.llm, prompt, component="reflect_llm")
            parsed = extract_json_object(raw_output)
        except Exception as error:
            parsed = {}
            errors = _append_error(state, "Reflect Agent failed: {}".format(error))

        requested_route = str(parsed.get("route", "")).strip().lower()
        objective = _compact(parsed.get("objective"), 45)
        if not objective:
            objective = _compact(
                state.get("reason_result", {}).get("missing_information"), 45
            ) or "Acquire evidence that distinguishes the leading answer options."
        existing_ids = set()
        for item in catalog:
            raw_segment_id = item.get("segment_id")
            if raw_segment_id is None:
                continue
            try:
                existing_ids.add(int(raw_segment_id))
            except (TypeError, ValueError):
                continue
        processed = {int(value) for value in state.get("processed_segment_ids", [])}
        requested_segments = parsed.get("segment_ids", [])
        if not isinstance(requested_segments, list):
            requested_segments = []
        perception_ids = []  # type: List[int]
        for item in requested_segments:
            try:
                segment_id = int(item)
            except (TypeError, ValueError):
                continue
            if (
                segment_id in existing_ids
                and segment_id not in processed
                and segment_id not in perception_ids
            ):
                perception_ids.append(segment_id)
            if len(perception_ids) >= self.max_perception_segments:
                break

        action = {}  # type: Dict[str, Any]
        route = requested_route if requested_route in VALID_ROUTES else ""
        if route == "perception" and perception_ids:
            action = {
                "route": "perception",
                "segment_ids": perception_ids,
                "time_ranges": [],
                "objective": objective,
            }
        elif route == "attention":
            try:
                raw_segment_id = parsed.get("segment_id")
                if raw_segment_id is None:
                    raise ValueError("missing segment_id")
                segment_id = int(raw_segment_id)
            except (TypeError, ValueError):
                segment_id = -1
            time_ranges = parsed.get("time_ranges", [])
            if not isinstance(time_ranges, list):
                time_ranges = []
            catalog_by_id = {
                int(item["segment_id"]): item for item in catalog
                if item.get("segment_id") is not None
            }
            history_keys = set(
                str(value) for value in state.get("attention_target_history", [])
            )
            valid_ranges = []  # type: List[List[float]]
            episode = catalog_by_id.get(segment_id)
            for item in time_ranges:
                if (
                    episode is None
                    or not isinstance(item, (list, tuple))
                    or len(item) != 2
                ):
                    continue
                try:
                    start_time = max(float(episode["start_time"]), float(item[0]))
                    end_time = min(float(episode["end_time"]), float(item[1]))
                except (KeyError, TypeError, ValueError):
                    continue
                if end_time <= start_time:
                    continue
                range_key = "{}:{:.3f}-{:.3f}".format(
                    segment_id, start_time, end_time
                )
                if range_key in history_keys:
                    continue
                valid_ranges.append([start_time, end_time])
                if len(valid_ranges) >= self.max_attention_ranges:
                    break
            if segment_id in existing_ids and valid_ranges:
                action = {
                    "route": "attention",
                    "segment_id": segment_id,
                    "segment_ids": [segment_id],
                    "time_ranges": valid_ranges,
                    "spatial_target": _compact(parsed.get("spatial_target"), 30),
                    "objective": objective,
                }
            else:
                route = ""
        elif route == "replan":
            action = {
                "route": "replan",
                "segment_ids": [],
                "time_ranges": [],
                "objective": objective,
            }
        else:
            route = ""

        if not route:
            fallback_ids = self._fallback_perception(state)
            if fallback_ids:
                route = "perception"
                action = {
                    "route": route,
                    "segment_ids": fallback_ids,
                    "time_ranges": [],
                    "objective": objective,
                }
            else:
                route = "replan"
                action = {
                    "route": route,
                    "segment_ids": [],
                    "time_ranges": [],
                    "objective": objective,
                }

        replan_feedback = {}
        if route == "replan":
            replan_feedback = {
                "reason": _compact(parsed.get("diagnosis"), 45)
                or "Current episodic memory does not satisfy the answer criterion.",
                "missing_information": objective,
                "previous_queries": state.get("query_history", []),
                "iteration_history": state.get("iteration_history", []),
            }
        history = self._patch_history(state, action)
        return {
            # tool_request is the sole Perception/Attention protocol; replanning returns control to Plan.
            "route": route,
            "tool_request": action,
            "replan_feedback": replan_feedback,
            "reflect_result": {
                "route": route,
                "diagnosis": _compact(parsed.get("diagnosis"), 45),
                "tool_request": action,
            },
            "iteration_history": history,
            "reflect_model_output": raw_output,
            "errors": errors,
            "working_memory": _append_memory(
                state,
                "reflect",
                "Reflect routed insufficient evidence to {}.".format(route),
            ),
        }
