"""Track node latency, model calls, and token usage independently for each question."""

import os
import re
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple


NODE_ORDER = (
    "plan",
    "global",
    "localization",
    "perception",
    "attention",
    "reflect",
    "answer",
)
CALL_KINDS = ("llm", "vqa", "asr")


@dataclass
class _ActiveQuestion:
    """Hold temporary performance data for the active question outside benchmark state."""

    video_id: str
    video_name: str
    question_id: str
    started_at: float
    node_durations: Dict[str, List[float]] = field(default_factory=dict)
    call_counts: Dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in CALL_KINDS}
    )
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_usage_by_component: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    unknown_token_usage_calls: int = 0


@dataclass
class _QuestionRecord:
    """Store a persistent question record used in video and dataset summaries."""

    video_id: str
    video_name: str
    question_id: str
    question_total_sec: float
    graph_route: List[str]
    node_calls: Dict[str, int]
    node_time_sec: Dict[str, float]
    llm_calls: int
    vqa_calls: int
    asr_calls: int
    reflect_cycles: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    unknown_token_usage_calls: int


class PerformanceTracker:
    """Maintain question-level performance records and generate a separate concise text log."""

    _QUESTION_PATTERN = re.compile(
        r"^\[QUESTION\] "
        r"video_id=(\S+) video=(\S+) question=(\S+) "
        r"total=([0-9.]+)s route=(\S+) nodes=(\S+) calls=(\S+) "
        r"tokens=in:(\d+),out:(\d+),total:(\d+) unknown_usage=(\d+)$"
    )

    def __init__(self, log_path: str) -> None:
        self.log_path = os.path.abspath(log_path)
        self._lock = threading.RLock()
        self._active = None  # type: Optional[_ActiveQuestion]
        self._records = {}  # type: Dict[Tuple[str, str], _QuestionRecord]
        self._load_existing_records()

    @property
    def has_active_question(self) -> bool:
        """Return whether a question is currently being timed."""
        with self._lock:
            return self._active is not None

    @staticmethod
    def _parse_named_numbers(value: str) -> Dict[str, float]:
        """Parse compact comma-separated name:value fields."""
        parsed = {}  # type: Dict[str, float]
        if value == "-":
            return parsed
        for item in value.split(","):
            if ":" not in item:
                continue
            name, raw_number = item.split(":", 1)
            try:
                parsed[name] = float(raw_number)
            except ValueError:
                continue
        return parsed

    @classmethod
    def _parse_node_metrics(
        cls,
        value: str,
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Parse node statistics in node:calls/seconds format."""
        calls = {}  # type: Dict[str, int]
        durations = {}  # type: Dict[str, float]
        if value == "-":
            return calls, durations
        for item in value.split(","):
            match = re.fullmatch(r"([^:]+):(\d+)/([0-9.]+)s", item)
            if match is None:
                continue
            node_name, raw_calls, raw_duration = match.groups()
            calls[node_name] = int(raw_calls)
            durations[node_name] = float(raw_duration)
        return calls, durations

    @classmethod
    def _parse_question_line(cls, line: str) -> Optional[_QuestionRecord]:
        """Restore question records from an existing log for deduplicated resume summaries."""
        match = cls._QUESTION_PATTERN.fullmatch(line.strip())
        if match is None:
            return None
        (
            video_id,
            video_name,
            question_id,
            raw_total_sec,
            raw_route,
            raw_nodes,
            raw_calls,
            raw_input_tokens,
            raw_output_tokens,
            raw_total_tokens,
            raw_unknown_usage,
        ) = match.groups()
        node_calls, node_time_sec = cls._parse_node_metrics(raw_nodes)
        call_counts = cls._parse_named_numbers(raw_calls)
        return _QuestionRecord(
            video_id=video_id,
            video_name=video_name,
            question_id=question_id,
            question_total_sec=float(raw_total_sec),
            graph_route=[] if raw_route == "-" else raw_route.split(">"),
            node_calls=node_calls,
            node_time_sec=node_time_sec,
            llm_calls=int(call_counts.get("llm", 0)),
            vqa_calls=int(call_counts.get("vqa", 0)),
            asr_calls=int(call_counts.get("asr", 0)),
            reflect_cycles=int(call_counts.get("cycles", 0)),
            input_tokens=int(raw_input_tokens),
            output_tokens=int(raw_output_tokens),
            total_tokens=int(raw_total_tokens),
            unknown_token_usage_calls=int(raw_unknown_usage),
        )

    def _load_existing_records(self) -> None:
        """Read only question rows from an existing log; always recalculate video and average rows."""
        if not os.path.isfile(self.log_path):
            return
        with open(self.log_path, "r", encoding="utf-8") as file:
            for line in file:
                record = self._parse_question_line(line)
                if record is None:
                    continue
                self._records[(record.video_id, record.question_id)] = record

    def start_question(
        self,
        video_id: str,
        video_name: str,
        question_id: str,
    ) -> None:
        """Create an independent statistics context for the current question before graph execution."""
        with self._lock:
            self._active = _ActiveQuestion(
                video_id=str(video_id),
                video_name=str(video_name),
                question_id=str(question_id),
                started_at=time.perf_counter(),
            )

    def record_node(self, node_name: str, duration_sec: float) -> None:
        """Append one node duration, retaining separate entries for repeated visits to the same node."""
        with self._lock:
            if self._active is None:
                return
            self._active.node_durations.setdefault(str(node_name), []).append(
                max(0.0, float(duration_sec))
            )

    def record_token_usage(
        self,
        component: str,
        call_kind: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        total_tokens: Optional[int],
    ) -> None:
        """Record a model call and its usage, explicitly marking incomplete fields as unknown."""
        with self._lock:
            if self._active is None:
                return
            normalized_kind = str(call_kind)
            if normalized_kind in self._active.call_counts:
                self._active.call_counts[normalized_kind] += 1

            known_input = _nonnegative_int(input_tokens)
            known_output = _nonnegative_int(output_tokens)
            known_total = _nonnegative_int(total_tokens)
            if known_total is None and known_input is not None and known_output is not None:
                known_total = known_input + known_output
            if known_input is None or known_output is None or known_total is None:
                self._active.unknown_token_usage_calls += 1

            component_usage = self._active.token_usage_by_component.setdefault(
                str(component),
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            if known_input is not None:
                self._active.input_tokens += known_input
                component_usage["input_tokens"] += known_input
            if known_output is not None:
                self._active.output_tokens += known_output
                component_usage["output_tokens"] += known_output
            if known_total is not None:
                self._active.total_tokens += known_total
                component_usage["total_tokens"] += known_total

    def finish_question(self, events: List[Dict[str, Any]]) -> None:
        """Stop timing the current question, replace any record with the same key, and refresh the log."""
        finished_at = time.perf_counter()
        with self._lock:
            active = self._active
            self._active = None
            if active is None:
                return
            node_calls = {
                node_name: len(durations)
                for node_name, durations in active.node_durations.items()
            }
            node_time_sec = {
                node_name: sum(durations)
                for node_name, durations in active.node_durations.items()
            }
            graph_route = [
                str(event.get("node", ""))
                for event in events
                if isinstance(event, dict) and event.get("node")
            ]
            record = _QuestionRecord(
                video_id=active.video_id,
                video_name=active.video_name,
                question_id=active.question_id,
                question_total_sec=max(0.0, finished_at - active.started_at),
                graph_route=graph_route,
                node_calls=node_calls,
                node_time_sec=node_time_sec,
                llm_calls=active.call_counts["llm"],
                vqa_calls=active.call_counts["vqa"],
                asr_calls=active.call_counts["asr"],
                # The new framework uses Reason for the evidence loop; retain Reflect compatibility for older graphs.
                reflect_cycles=node_calls.get(
                    "reason",
                    node_calls.get("reflect", 0),
                ),
                input_tokens=active.input_tokens,
                output_tokens=active.output_tokens,
                total_tokens=active.total_tokens,
                unknown_token_usage_calls=active.unknown_token_usage_calls,
            )
            self._records[(record.video_id, record.question_id)] = record
            self._write_log()

    @staticmethod
    def _safe_field(value: str) -> str:
        """Remove whitespace from log values to keep one question per line and simplify recovery."""
        return re.sub(r"\s+", "_", str(value).strip()) or "-"

    @classmethod
    def _format_question(cls, record: _QuestionRecord) -> str:
        """Format a question record as compact single-line text."""
        preferred_order = list(NODE_ORDER)
        if record.node_calls.get("reason", 0) > 0:
            preferred_order.insert(preferred_order.index("reflect"), "reason")
        node_names = [
            name for name in preferred_order if record.node_calls.get(name, 0) > 0
        ]
        node_names.extend(sorted(
            name
            for name in record.node_calls
            if name not in preferred_order and record.node_calls.get(name, 0) > 0
        ))
        nodes = ",".join(
            "{}:{}/{:.3f}s".format(
                name,
                record.node_calls[name],
                record.node_time_sec.get(name, 0.0),
            )
            for name in node_names
        ) or "-"
        route = ">".join(record.graph_route) or "-"
        return (
            "[QUESTION] video_id={video_id} video={video_name} "
            "question={question_id} total={total:.3f}s route={route} "
            "nodes={nodes} calls=llm:{llm},vqa:{vqa},asr:{asr},cycles:{cycles} "
            "tokens=in:{input_tokens},out:{output_tokens},total:{total_tokens} "
            "unknown_usage={unknown}"
        ).format(
            video_id=cls._safe_field(record.video_id),
            video_name=cls._safe_field(record.video_name),
            question_id=cls._safe_field(record.question_id),
            total=record.question_total_sec,
            route=route,
            nodes=nodes,
            llm=record.llm_calls,
            vqa=record.vqa_calls,
            asr=record.asr_calls,
            cycles=record.reflect_cycles,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            total_tokens=record.total_tokens,
            unknown=record.unknown_token_usage_calls,
        )

    @staticmethod
    def _average(values: List[float]) -> float:
        """Calculate the average of a nonempty numeric list."""
        return sum(values) / len(values) if values else 0.0

    def _video_lines(self, records: List[_QuestionRecord]) -> List[str]:
        """Regenerate per-video timing summaries from question records."""
        grouped = {}  # type: Dict[Tuple[str, str], List[_QuestionRecord]]
        for record in records:
            grouped.setdefault(
                (record.video_id, record.video_name),
                [],
            ).append(record)
        lines = []  # type: List[str]
        for (video_id, video_name), video_records in grouped.items():
            total_sec = sum(item.question_total_sec for item in video_records)
            lines.append(
                "[VIDEO] video_id={} video={} questions={} total={:.3f}s avg={:.3f}s".format(
                    self._safe_field(video_id),
                    self._safe_field(video_name),
                    len(video_records),
                    total_sec,
                    total_sec / len(video_records),
                )
            )
        return lines

    def _average_line(self, records: List[_QuestionRecord]) -> str:
        """Generate an overall average row consistent with ODEM online-stage accounting."""
        if not records:
            return "[AVERAGE] questions=0"
        complete_token_records = [
            record
            for record in records
            if record.unknown_token_usage_calls == 0
        ]
        avg_tokens = (
            "{:.2f}".format(self._average([
                float(record.total_tokens) for record in complete_token_records
            ]))
            if complete_token_records
            else "unknown"
        )
        preferred_order = list(NODE_ORDER)
        if any(record.node_calls.get("reason", 0) > 0 for record in records):
            preferred_order.insert(preferred_order.index("reflect"), "reason")
        node_calls = ",".join(
            "{}:{:.2f}".format(
                node_name,
                self._average([
                    float(record.node_calls.get(node_name, 0))
                    for record in records
                ]),
            )
            for node_name in preferred_order
        )
        node_time = ",".join(
            "{}:{:.3f}s".format(
                node_name,
                self._average([
                    record.node_time_sec.get(node_name, 0.0)
                    for record in records
                ]),
            )
            for node_name in preferred_order
        )
        return (
            "[AVERAGE] questions={questions} avg_question={avg_question:.3f}s "
            "avg_tokens={avg_tokens} token_questions={token_questions} "
            "avg_cycles={avg_cycles:.2f} llm_calls={llm_calls:.2f} "
            "vqa_calls={vqa_calls:.2f} asr_calls={asr_calls:.2f} "
            "node_calls={node_calls} node_time={node_time}"
        ).format(
            questions=len(records),
            avg_question=self._average([
                record.question_total_sec for record in records
            ]),
            avg_tokens=avg_tokens,
            token_questions=len(complete_token_records),
            avg_cycles=self._average([
                float(record.reflect_cycles) for record in records
            ]),
            llm_calls=self._average([
                float(record.llm_calls) for record in records
            ]),
            vqa_calls=self._average([
                float(record.vqa_calls) for record in records
            ]),
            asr_calls=self._average([
                float(record.asr_calls) for record in records
            ]),
            node_calls=node_calls,
            node_time=node_time,
        )

    def _write_log(self) -> None:
        """Atomically rewrite deduplicated question, video, and overall statistics."""
        output_dir = os.path.dirname(self.log_path)
        os.makedirs(output_dir, exist_ok=True)
        records = list(self._records.values())
        lines = [self._format_question(record) for record in records]
        if records:
            lines.append("")
            lines.extend(self._video_lines(records))
            lines.append("")
        lines.append(self._average_line(records))
        temporary_path = self.log_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
        os.replace(temporary_path, self.log_path)

    def finalize(self) -> None:
        """Refresh final video and dataset average statistics at the end of a run."""
        with self._lock:
            self._write_log()


def _nonnegative_int(value: Any) -> Optional[int]:
    """Normalize a usage value to a nonnegative integer, returning None when invalid."""
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized >= 0 else None


def _usage_mapping(usage: Any) -> Dict[str, Any]:
    """Support dictionaries, Pydantic objects, and OpenAI usage objects."""
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        try:
            dumped = usage.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except (TypeError, ValueError):
            pass
    result = {}  # type: Dict[str, Any]
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        if hasattr(usage, key):
            result[key] = getattr(usage, key)
    return result


_TRACKER = None  # type: Optional[PerformanceTracker]


def configure_performance_tracking(enabled: bool, log_path: str) -> None:
    """Enable or disable the global performance tracker without exposing it to node logic."""
    global _TRACKER
    try:
        _TRACKER = PerformanceTracker(log_path) if enabled else None
    except Exception as error:
        _TRACKER = None
        print("[WARN] Failed to initialize performance tracking: {}".format(error))


def start_question_tracking(
    video_id: str,
    video_name: str,
    question_id: str,
) -> None:
    """Safely start question tracking without allowing tracking errors to affect the graph."""
    if _TRACKER is None:
        return
    try:
        _TRACKER.start_question(video_id, video_name, question_id)
    except Exception as error:
        print("[WARN] Failed to start question performance tracking: {}".format(error))


def finish_question_tracking(events: List[Dict[str, Any]]) -> None:
    """Safely finish question tracking and refresh the log."""
    if _TRACKER is None:
        return
    try:
        _TRACKER.finish_question(events)
    except Exception as error:
        print("[WARN] Failed to save question performance statistics: {}".format(error))


def finalize_performance_tracking() -> None:
    """Safely refresh final average statistics."""
    if _TRACKER is None:
        return
    try:
        _TRACKER.finalize()
    except Exception as error:
        print("[WARN] Failed to summarize performance statistics: {}".format(error))


def is_question_tracking_active() -> bool:
    """Tell node wrappers whether timing is currently required."""
    return _TRACKER is not None and _TRACKER.has_active_question


def record_token_usage(
    component: str,
    call_kind: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
) -> None:
    """Safely record token usage for one model call."""
    if _TRACKER is None:
        return
    try:
        _TRACKER.record_token_usage(
            component,
            call_kind,
            input_tokens,
            output_tokens,
            total_tokens,
        )
    except Exception as error:
        print("[WARN] Failed to track token usage: {}".format(error))


def record_unknown_token_usage(component: str, call_kind: str) -> None:
    """Record a model call whose usage is unavailable."""
    record_token_usage(component, call_kind, None, None, None)


def record_usage_object(
    component: str,
    call_kind: str,
    usage: Any,
    embedding_call: bool = False,
) -> None:
    """Read input, output, and total token counts from common usage objects."""
    values = _usage_mapping(usage)
    input_tokens = values.get("input_tokens", values.get("prompt_tokens"))
    output_tokens = values.get("output_tokens", values.get("completion_tokens"))
    if embedding_call and output_tokens is None and input_tokens is not None:
        output_tokens = 0
    record_token_usage(
        component,
        call_kind,
        input_tokens,
        output_tokens,
        values.get("total_tokens"),
    )


def record_response_usage(
    component: str,
    call_kind: str,
    response: Any,
) -> None:
    """Find usage data in a LangChain message or OpenAI response."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            usage = metadata.get("token_usage") or metadata.get("usage")
    if usage is None:
        usage = getattr(response, "usage", None)
    if usage is None:
        record_unknown_token_usage(component, call_kind)
        return
    record_usage_object(component, call_kind, usage)


def timed_node(node_name: str, node: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap an existing LangGraph node to observe timing while passing inputs, outputs, and exceptions unchanged."""

    @wraps(node if hasattr(node, "__name__") else node.__call__)
    def wrapped(state: Any) -> Any:
        if not is_question_tracking_active():
            return node(state)
        started_at = time.perf_counter()
        try:
            return node(state)
        finally:
            if _TRACKER is not None:
                try:
                    _TRACKER.record_node(
                        node_name,
                        time.perf_counter() - started_at,
                    )
                except Exception as error:
                    print("[WARN] Failed to track node duration: {}".format(error))

    return wrapped
