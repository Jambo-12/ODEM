"""Evaluate open-ended predictions with the official M3-Agent GPT-4o semantic judge."""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI
from tqdm import tqdm

from eval.m3bench.agent_m3bench import load_config
from eval.m3bench.dataset import save_json_atomic
from eval.runtime import load_project_environment, require_environment


VERIFY_PROMPT = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
\t•\tDo not require exact wording or matching structure.
\t•\tSemantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
\t•\tOnly return "Yes" or "No", with no additional explanation or formatting.

Input fields:
\t•\tquestion: the question asked
\t•\tground_truth_answer: the correct answer
\t•\tagent_answer: the model's answer to be evaluated

Now evaluate the following input:

Input:
\t•\tquestion: {question}
\t•\tground_truth_answer: {ground_truth_answer}
\t•\tagent_answer: {agent_answer}

Output ('Yes' or 'No'):"""


class JudgeClient:
    """Call the officially specified model and retry transient API failures with official parameters."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int,
        retry_sleep_sec: float,
        request_timeout: float,
    ) -> None:
        self.model = model
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep_sec = max(0.0, float(retry_sleep_sec))
        self.request_timeout = max(1.0, float(request_timeout))
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Ask judge requests that are still retrying to stop promptly."""
        self._stop_event.set()

    def judge(self, question: str, gold: str, prediction: str) -> Tuple[bool, str]:
        """Mark empty predictions incorrect; use GPT to judge whether others entail the reference answer."""
        if not prediction.strip():
            return False, "(empty prediction)"
        prompt = VERIFY_PROMPT.format(
            question=question,
            ground_truth_answer=gold,
            agent_answer=prediction,
        )
        last_error = None  # type: Any
        for attempt in range(self.max_retries):
            if self._stop_event.is_set():
                raise RuntimeError("Judge evaluation was interrupted")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2048,
                    timeout=self.request_timeout,
                )
                raw = str(response.choices[0].message.content or "").strip()
                return "yes" in raw.lower()[:10], raw
            except Exception as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    if self._stop_event.wait(self.retry_sleep_sec):
                        raise RuntimeError("Judge evaluation was interrupted")
        raise RuntimeError(
            "Judge failed after {} attempts: {}".format(
                self.max_retries,
                last_error,
            )
        )


def _annotation_index(annotation_json: str) -> Dict[str, Dict[str, Any]]:
    """Build the evaluation index from official annotations while explicitly discarding reasoning."""
    with open(annotation_json, "r", encoding="utf-8") as file:
        annotations = json.load(file)
    indexed = {}  # type: Dict[str, Dict[str, Any]]
    for video_id, video in annotations.items():
        for question in video.get("qa_list", []):
            question_id = str(question["question_id"])
            indexed[question_id] = {
                "video_id": video_id,
                "question": str(question["question"]),
                "gold": str(question["answer"]),
                "type": [str(value) for value in question.get("type", [])],
            }
    return indexed


def _prediction_index(predictions_path: str) -> Dict[str, str]:
    """Read merged open-ended predictions without trusting duplicated reference answers within them."""
    with open(predictions_path, "r", encoding="utf-8") as file:
        videos = json.load(file)
    indexed = {}  # type: Dict[str, str]
    for video in videos:
        for question in video.get("questions", []):
            question_id = str(question.get("question_id") or "")
            if question_id in indexed:
                raise ValueError("Duplicate question_id={} in predictions".format(question_id))
            indexed[question_id] = str(question.get("prediction") or "").strip()
    return indexed


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the number of judged questions, correct answers, and accuracy."""
    total = len(records)
    correct = sum(record.get("correct") is True for record in records)
    return {
        "n_total": total,
        "n_correct": correct,
        "accuracy": round(float(correct) / float(total), 6) if total else 0.0,
        "accuracy_percent": round(100.0 * correct / total, 2) if total else 0.0,
    }


def _aggregate_by_type(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Count a question under each category when it has multiple labels."""
    buckets = {}  # type: Dict[str, List[Dict[str, Any]]]
    for record in records:
        for question_type in record.get("type", []) or ["_untyped_"]:
            buckets.setdefault(str(question_type), []).append(record)
    return {
        key: _aggregate(value)
        for key, value in sorted(buckets.items())
    }


def _aggregate_by_video(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate semantic judge results by video to identify per-video inference errors."""
    buckets = {}  # type: Dict[str, List[Dict[str, Any]]]
    for record in records:
        buckets.setdefault(str(record.get("video_id") or "unknown"), []).append(record)
    return {
        key: _aggregate(value)
        for key, value in sorted(buckets.items())
    }


def _load_completed_records(
    output_path: str,
    annotations: Dict[str, Dict[str, Any]],
    predictions: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Read successfully completed judge records and leave failed records for reevaluation."""
    if not os.path.isfile(output_path):
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as file:
            previous = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    completed = {}  # type: Dict[str, Dict[str, Any]]
    records = previous.get("per_question", []) if isinstance(previous, dict) else []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        question_id = str(record.get("question_id") or "")
        if question_id not in annotations or record.get("judge_error"):
            continue
        if str(record.get("prediction") or "").strip() != predictions[question_id]:
            continue
        if not isinstance(record.get("correct"), bool):
            continue
        completed[question_id] = dict(record)
    return completed


def _build_result(
    annotations: Dict[str, Dict[str, Any]],
    records_by_id: Dict[str, Dict[str, Any]],
    judge_model: str,
) -> Dict[str, Any]:
    """Assemble an interim evaluation result in official annotation order for immediate saving."""
    records = [
        records_by_id[question_id]
        for question_id in annotations
        if question_id in records_by_id
    ]
    judge_errors = sum(bool(record.get("judge_error")) for record in records)
    return {
        "dataset": "M3-Bench-robot",
        "judge_model": judge_model,
        "evaluation_complete": (
            len(records) == len(annotations) and judge_errors == 0
        ),
        "completed_questions": len(records),
        "expected_questions": len(annotations),
        "judge_errors": judge_errors,
        "overall": _aggregate(records),
        "by_type": _aggregate_by_type(records),
        "per_video": _aggregate_by_video(records),
        "per_question": records,
    }


def evaluate(
    predictions_path: str,
    annotation_json: str,
    output_path: str,
    config_path: str,
) -> Dict[str, Any]:
    """Run the official semantic judge concurrently and save per-question, overall, and category results."""
    config = load_config(config_path)
    annotations = _annotation_index(annotation_json)
    predictions = _prediction_index(predictions_path)
    missing = [question_id for question_id in annotations if question_id not in predictions]
    extra = [question_id for question_id in predictions if question_id not in annotations]
    if missing or extra:
        raise ValueError(
            "Incomplete prediction question set: missing={}, extra={}".format(
                len(missing),
                len(extra),
            )
        )
    judge = JudgeClient(
        api_key=require_environment("JUDGE_API_KEY"),
        base_url=(
            os.getenv("JUDGE_API_BASE") or "https://api.openai.com/v1"
        ).strip(),
        model=str(config.get("judge_model", "gpt-4o-2024-11-20")),
        max_retries=int(config.get("judge_max_retries", 20)),
        retry_sleep_sec=float(config.get("judge_retry_sleep_sec", 20)),
        request_timeout=float(config.get("judge_request_timeout", 30)),
    )
    records_by_id = _load_completed_records(
        output_path,
        annotations,
        predictions,
    )
    pending_items = [
        (question_id, item)
        for question_id, item in annotations.items()
        if question_id not in records_by_id
    ]

    def judge_one(question_id: str, item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        prediction = predictions[question_id]
        try:
            correct, raw = judge.judge(item["question"], item["gold"], prediction)
            error = ""
        except Exception as judge_error:
            correct = False
            raw = ""
            error = "{}: {}".format(type(judge_error).__name__, judge_error)
        return question_id, {
            "video_id": item["video_id"],
            "question_id": question_id,
            "question": item["question"],
            "gold": item["gold"],
            "prediction": prediction,
            "type": item["type"],
            "correct": correct,
            "judge_raw": raw,
            "judge_error": error,
        }

    workers = max(1, int(config.get("judge_workers", 4)))
    progress = tqdm(
        total=len(annotations),
        initial=len(records_by_id),
        desc="M3-Bench GPT Judge",
        unit="question",
        dynamic_ncols=True,
    )
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [
        executor.submit(judge_one, question_id, item)
        for question_id, item in pending_items
    ]
    try:
        for future in as_completed(futures):
            question_id, record = future.result()
            records_by_id[question_id] = record
            # Save atomically after every question so Ctrl+C or API errors do not lose completed results.
            save_json_atomic(
                _build_result(annotations, records_by_id, judge.model),
                output_path,
            )
            progress.update(1)
            current = _aggregate(list(records_by_id.values()))
            progress.set_postfix_str(
                "accuracy {:.2f}% ({}/{})".format(
                    current["accuracy_percent"],
                    current["n_correct"],
                    current["n_total"],
                ),
                refresh=False,
            )
            if record.get("judge_error"):
                raise RuntimeError(
                    "Judge request failed; current progress has been saved: {}".format(
                        record["judge_error"]
                    )
                )
    except BaseException:
        judge.stop()
        # Save futures completed before interruption but not yet consumed into the checkpoint as well.
        for future in futures:
            if not future.done() or future.cancelled():
                continue
            try:
                question_id, record = future.result()
            except BaseException:
                continue
            records_by_id[question_id] = record
        save_json_atomic(
            _build_result(annotations, records_by_id, judge.model),
            output_path,
        )
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        progress.close()

    result = _build_result(annotations, records_by_id, judge.model)
    save_json_atomic(result, output_path)
    return result


def main() -> None:
    """Parse evaluation inputs and print the final overall accuracy."""
    load_project_environment()
    parser = argparse.ArgumentParser(description="Official M3-Bench open-ended semantic evaluation")
    parser.add_argument(
        "--predictions",
        default=os.path.join(PROJECT_ROOT, "outputs", "m3bench", "predictions.json"),
    )
    parser.add_argument(
        "--annotations",
        default=os.path.join(
            PROJECT_ROOT,
            "data",
            "raw",
            "m3bench",
            "annotations",
            "robot.json",
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "outputs", "m3bench", "evaluation.json"),
    )
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "config.yaml"))
    args = parser.parse_args()
    result = evaluate(args.predictions, args.annotations, args.output, args.config)
    overall = result["overall"]
    print(
        "M3-Bench Judge: {:.2f}% ({}/{}), Judge errors {}".format(
            overall["accuracy_percent"],
            overall["n_correct"],
            overall["n_total"],
            result["judge_errors"],
        )
    )
    if result["judge_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
