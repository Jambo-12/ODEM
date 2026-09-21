"""Small shared runtime used by the public benchmark entrypoints."""

import argparse
import json
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from omegaconf import DictConfig

from graph_trace import normalize_option


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FORCED_OPTION = "A"
VALID_OPTION_LETTERS = ("A", "B", "C", "D", "E")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv")
PREPROCESS_FILES = (
    "captions.json",
    "segment_textual_embedding.pkl",
    "segment_visual_embedding.pkl",
)
ENV_FIELDS = {
    "openai_api_key": "OPENAI_API_KEY",
    "openai_api_base": "OPENAI_API_BASE",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "deepseek_api_base": "DEEPSEEK_API_BASE",
    "speaker_asr_api_key": "DASHSCOPE_API_KEY",
    "speaker_asr_api_base": "DASHSCOPE_API_BASE",
}


@contextmanager
def suppress_normal_console_output() -> Iterator[None]:
    """Hide model chatter while preserving exceptions and return values."""
    with open(os.devnull, "w", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            yield


def resolve_project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_project_environment() -> None:
    """Load simple KEY=VALUE entries from the repository-level .env file."""
    path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or not key or key[0].isdigit():
                continue
            if not key.replace("_", "").isalnum():
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


def parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def require_config_text(config: DictConfig, key: str) -> str:
    env_name = ENV_FIELDS.get(key)
    environment_value = os.getenv(env_name, "") if env_name else ""
    value = environment_value if key.endswith("_api_key") else (
        environment_value or config.get(key)
    )
    if value is None or not str(value).strip():
        source = env_name or key
        raise ValueError("missing required configuration: {}".format(source))
    return str(value).strip()


def require_environment(name: str) -> str:
    """Read a credential from the environment without any file fallback."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError("missing required environment variable: {}".format(name))
    return value


def create_reasoning_llm(config: DictConfig) -> ChatOpenAI:
    """Create the DeepSeek-V4-Flash model shared by all language agents."""
    provider = str(config.get("llm_provider", "deepseek")).strip().lower()
    model = str(config.get("deepseek_model", "deepseek-v4-flash")).strip()
    if provider != "deepseek" or model != "deepseek-v4-flash":
        raise ValueError("reasoning model must be deepseek-v4-flash")
    return ChatOpenAI(
        model=model,
        api_key=require_config_text(config, "deepseek_api_key"),
        base_url=require_config_text(config, "deepseek_api_base").rstrip("/"),
        temperature=float(config.get("llm_temperature", 0.0)),
        max_retries=int(config.get("llm_max_retries", 5)),
        timeout=float(config.get("llm_request_timeout", 120)),
    )


def resolve_video_path(video_root: str, video_name: str) -> Optional[str]:
    for extension in VIDEO_EXTENSIONS:
        path = os.path.join(video_root, video_name + extension)
        if os.path.isfile(path):
            return path
    return None


def has_preprocess_files(preprocess_root: str, video_name: str) -> bool:
    video_dir = os.path.join(preprocess_root, video_name)
    return all(os.path.isfile(os.path.join(video_dir, name)) for name in PREPROCESS_FILES)


def load_video_mme(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)

    videos = {}  # type: Dict[str, Dict[str, Any]]
    video_names = {}  # type: Dict[str, str]
    for record in records:
        video_id = str(record["video_id"])
        video_names[video_id] = str(record["videoID"])
        video = videos.setdefault(
            video_id,
            {
                "video_id": video_id,
                "duration": record.get("duration"),
                "domain": record.get("domain"),
                "sub_category": record.get("sub_category"),
                "questions": [],
            },
        )
        video["questions"].append({
            "question_id": record["question_id"],
            "task_type": record.get("task_type"),
            "question": record["question"],
            "options": record.get("options", []),
            "answer": record.get("answer"),
        })
    return list(videos.values()), video_names


def select_dataset_shard(
    videos: List[Dict[str, Any]], num_shards: int, shard_index: int
) -> List[Dict[str, Any]]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard {}/{}".format(shard_index, num_shards))
    return videos[shard_index::num_shards]


def save_json_atomic(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    os.replace(temporary_path, path)


def calculate_overall_accuracy(
    videos: List[Dict[str, Any]], dataset_name: str = "Video-MME"
) -> Dict[str, Any]:
    correct = 0
    total = 0
    for video in videos:
        for question in video.get("questions", []):
            response = question.get("response")
            answer = normalize_option(question.get("answer"))
            if response is None or not str(response).strip() or answer not in VALID_OPTION_LETTERS:
                continue
            total += 1
            correct += int(normalize_option(response) == answer)
    accuracy = float(correct) / total if total else 0.0
    return {
        "dataset": dataset_name,
        "correct": correct,
        "incorrect": total - correct,
        "total_evaluated": total,
        "accuracy": round(accuracy, 6),
        "accuracy_percent": round(accuracy * 100.0, 2),
    }


def save_experiment_results(
    videos: List[Dict[str, Any]],
    output_path: str,
    accuracy_path: str,
    dataset_name: str = "Video-MME",
) -> Dict[str, Any]:
    save_json_atomic(videos, output_path)
    accuracy = calculate_overall_accuracy(videos, dataset_name)
    save_json_atomic(accuracy, accuracy_path)
    return accuracy
