"""
Generate multi-key text embeddings aligned with four-field captions.json timelines.

Example:
    python segment_feats_text.py

Specify config and output directory:
    python segment_feats_text.py \
        --config config/preprocess.yaml \
        --preprocess_dir data/processed/videomme/qwen2_5_vl_7b_30s

Encode every caption field independently without field labels or concatenation. Output shape is
[segments, keys, dimensions] for per-key similarity. Read the API key from the environment and
the OpenAI-compatible base URL from shared YAML or the environment.
"""
import os
import time
import json
import pickle
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import OmegaConf

from encoder import encode_sentences

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "preprocess.yaml")
DEFAULT_PREPROCESS_DIR = os.path.join(
    SCRIPT_DIR,
    "data",
    "processed",
    "videomme",
    "qwen2_5_vl_7b_30s",
)
CAPTION_KEYS = ("environment", "event", "attention", "summary")
OUTPUT_KEYS = ("start_time", "end_time") + CAPTION_KEYS
NULLABLE_KEYS = ("event",)
DEFAULT_EMBEDDING_BATCH_SIZE = 128
CaptionKeyValues = Dict[str, Optional[str]]
CaptionItem = Tuple[int, float, float, CaptionKeyValues]


def load_embedding_api_config(config_path: str) -> Tuple[str, str]:
    """
    Read text-embedding API settings from the environment and shared YAML.

    API keys may only come from the environment; YAML declares names and endpoints.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = OmegaConf.load(config_path)
    service_config = OmegaConf.select(config, "services.text_embedding", default={})
    api_key_env = str(service_config.get("api_key_env", "OPENAI_API_KEY"))
    api_base_env = str(service_config.get("api_base_env", "OPENAI_API_BASE"))
    api_key = os.getenv(api_key_env)
    api_base = (
        os.getenv(api_base_env)
        or service_config.get("api_base")
        or config.get("openai_api_base")
    )
    if not api_key or not str(api_key).strip():
        raise ValueError(
            f"Missing text embedding API key; set {api_key_env}"
        )
    if not api_base or not str(api_base).strip():
        raise ValueError(
            f"Missing text embedding API base; set {api_base_env} or update {config_path}"
        )
    return str(api_key).strip(), str(api_base).strip().rstrip("/")


def configure_embedding_api(config_path: str) -> str:
    """
    Store the normalized embedding API settings in the current process environment.

    encoder.encode_sentences reads these variables for every request, so configure them first.
    Return the base URL only for non-secret runtime reporting.
    """
    api_key, api_base = load_embedding_api_config(config_path)
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    return api_base


def parse_timestamp(timestamp: str) -> float:
    """
    Convert an HH:MM:SS caption timestamp to seconds.

    Caption timestamps are second-aligned. Strictly validate three components to prevent
    invalid timelines from propagating to text and visual embeddings.
    """
    if not isinstance(timestamp, str):
        raise ValueError(f"Timestamp must be a string; got {type(timestamp).__name__}")
    parts = timestamp.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {timestamp}")
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"Invalid timestamp: {timestamp}") from error
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid timestamp: {timestamp}")
    return float(hours * 3600 + minutes * 60 + seconds)


def extract_caption_key_values(item: Dict[str, Any]) -> CaptionKeyValues:
    """
    Extract and normalize the four caption keys for one segment.

    Encode each nonempty field independently without labels or concatenation. event may be
    null; all other fields must contain valid text.
    """
    key_values = {}  # type: CaptionKeyValues
    for key in CAPTION_KEYS:
        if key not in item:
            raise ValueError(f"Caption is missing field {key}")
        value = item.get(key)
        if value is None:
            if key in NULLABLE_KEYS:
                key_values[key] = None
                continue
            raise ValueError(f"Field {key} cannot be null")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field {key} must be a nonempty string or null")
        key_values[key] = " ".join(value.strip().split())
    return key_values


def load_captions_list(captions_path: str) -> List[CaptionItem]:
    """
    Read captions.json in the current four-field list format.

    Generate segment_id deterministically from list order. Return tuples of
    (segment_id, start_sec, end_sec, key_values) after strict timeline validation.
    """
    with open(captions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{captions_path} must contain a list")

    captions: List[CaptionItem] = []
    previous_end = 0.0
    for segment_id, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Caption {segment_id} is not a JSON object")
        if tuple(item.keys()) != OUTPUT_KEYS:
            raise ValueError(
                f"Caption {segment_id} has invalid fields or order; expected {OUTPUT_KEYS}"
            )

        start_sec = parse_timestamp(item["start_time"])
        end_sec = parse_timestamp(item["end_time"])
        if end_sec <= start_sec:
            raise ValueError(f"Caption {segment_id} must end after it starts")
        if segment_id > 0 and start_sec < previous_end:
            raise ValueError(f"Caption {segment_id} overlaps the previous segment")

        key_values = extract_caption_key_values(item)
        captions.append((segment_id, start_sec, end_sec, key_values))
        previous_end = end_sec
    return captions


def to_numpy(arr: Any) -> np.ndarray:
    """Normalize encode_sentences output to a float32 NumPy array."""
    if hasattr(arr, "detach"):
        result = arr.detach().cpu().numpy()
        return np.asarray(result, dtype=np.float32)
    if hasattr(arr, "cpu") and hasattr(arr, "numpy"):
        result = arr.cpu().numpy()
        return np.asarray(result, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def encode_text_batches(
    sentences: List[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """
    Batch text-encoding requests to limit keys submitted for one long video.

    Preserve sentence order so the 3D segment/key array can be reconstructed.
    """
    if not sentences:
        raise ValueError("No caption keys to encode")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    batches = []  # type: List[np.ndarray]
    for start_index in range(0, len(sentences), batch_size):
        sentence_batch = sentences[start_index:start_index + batch_size]
        batch_embeddings = to_numpy(
            encode_sentences(
                sentence_list=sentence_batch,
                model_name=model_name,
            )
        )
        if batch_embeddings.ndim != 2:
            raise ValueError(
                f"Text embedding result must be a 2D array; got shape={batch_embeddings.shape}"
            )
        if batch_embeddings.shape[0] != len(sentence_batch):
            raise ValueError(
                f"Text embedding count mismatch: input={len(sentence_batch)}, "
                f"output={batch_embeddings.shape[0]}"
            )
        batches.append(batch_embeddings)
    return np.concatenate(batches, axis=0)


def build_multi_key_embeddings(
    captions: List[CaptionItem],
    model_name: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Independently encode valid keys and reconstruct a 3D array and 2D validity mask.

    embeddings has shape [segments, keys, dimensions]. Nullable keys use zero vectors with
    False in valid_mask; retrieval must apply the mask before maximum similarity.
    """
    sentences = []  # type: List[str]
    positions = []  # type: List[Tuple[int, int]]
    for segment_index, caption in enumerate(captions):
        key_values = caption[3]
        for key_index, key in enumerate(CAPTION_KEYS):
            value = key_values.get(key)
            if value is None:
                continue
            sentences.append(value)
            positions.append((segment_index, key_index))

    flat_embeddings = encode_text_batches(
        sentences=sentences,
        model_name=model_name,
        batch_size=batch_size,
    )
    embedding_dim = int(flat_embeddings.shape[1])
    embeddings = np.zeros(
        (len(captions), len(CAPTION_KEYS), embedding_dim),
        dtype=np.float32,
    )
    valid_mask = np.zeros(
        (len(captions), len(CAPTION_KEYS)),
        dtype=np.bool_,
    )
    for embedding_index, (segment_index, key_index) in enumerate(positions):
        embeddings[segment_index, key_index] = flat_embeddings[embedding_index]
        valid_mask[segment_index, key_index] = True
    return embeddings, valid_mask


def is_reusable_embedding_file(
    output_path: str,
    captions: List[CaptionItem],
    model_name: str,
) -> bool:
    """
    Check whether an existing PKL is a four-key array aligned with current captions.

    Reject legacy 2D vectors, changed field order, model mismatches, and timeline mismatches.
    """
    try:
        with open(output_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return False
        embeddings = np.asarray(payload.get("embeddings"))
        valid_mask = np.asarray(payload.get("valid_mask"))
        expected_ids = [item[0] for item in captions]
        expected_starts = [item[1] for item in captions]
        expected_ends = [item[2] for item in captions]
        return bool(
            str(payload.get("model")) == model_name
            and payload.get("caption_keys") == list(CAPTION_KEYS)
            and payload.get("segment_ids") == expected_ids
            and np.allclose(payload.get("start_times"), expected_starts)
            and np.allclose(payload.get("end_times"), expected_ends)
            and embeddings.ndim == 3
            and embeddings.shape[0] == len(captions)
            and embeddings.shape[1] == len(CAPTION_KEYS)
            and valid_mask.shape == embeddings.shape[:2]
        )
    except (OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError):
        return False


def save_pickle_atomic(payload: Dict[str, Any], output_path: str) -> None:
    """Write a temporary PKL and atomically replace the final file."""
    temporary_path = output_path + ".tmp"
    try:
        with open(temporary_path, "wb") as f:
            pickle.dump(payload, f)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

def process_preprocess_root(
    preprocess_root: str,
    model_name: str = "text-embedding-3-large",
    overwrite: bool = False,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    target_video_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Traverse every video directory under preprocess_root:
      - read captions.json;
      - generate segment_id from list order and parse true time bounds;
      - encode four caption keys independently without field labels;
      - save 3D vectors, valid-key masks, segment IDs, and timelines.

    Rebuild legacy 2D vectors. Reuse current vectors only when model, key order, timeline,
    and modification time match captions.json.
    """
    if not os.path.isdir(preprocess_root):
        raise FileNotFoundError(f"Preprocessing root does not exist: {preprocess_root}")
    if target_video_names is None:
        subdirs = [d for d in sorted(os.listdir(preprocess_root))
                   if os.path.isdir(os.path.join(preprocess_root, d))]
        strict_targets = False
    else:
        subdirs = sorted(set(target_video_names))
        strict_targets = True
    summary = {
        "total": len(subdirs),
        "saved": 0,
        "skipped": 0,
        "failed": 0,
    }

    for vid_name in subdirs:
        vid_dir = os.path.join(preprocess_root, vid_name)
        cap_json = os.path.join(vid_dir, "captions.json")
        out_pkl = os.path.join(vid_dir, "segment_textual_embedding.pkl")

        if not os.path.exists(cap_json):
            if strict_targets:
                print(f"[ERROR] {vid_name}: missing captions.json")
                summary["failed"] += 1
            else:
                print(f"[SKIP] {vid_name}: missing captions.json")
                summary["skipped"] += 1
            continue

        try:
            pairs = load_captions_list(cap_json)
            if len(pairs) == 0:
                print(f"[SKIP] {vid_name}: captions.json is empty")
                summary["skipped"] += 1
                continue

            if os.path.exists(out_pkl) and not overwrite:
                output_is_current = os.path.getmtime(out_pkl) >= os.path.getmtime(cap_json)
                output_is_current = output_is_current and is_reusable_embedding_file(
                    output_path=out_pkl,
                    captions=pairs,
                    model_name=model_name,
                )
                if output_is_current:
                    print(f"[SKIP] {vid_name}: valid four-key text embeddings already exist")
                    summary["skipped"] += 1
                    continue
                print(f"[REBUILD] {vid_name}: text embeddings are legacy or stale")

            t0 = time.time()
            embeddings, valid_mask = build_multi_key_embeddings(
                captions=pairs,
                model_name=model_name,
                batch_size=batch_size,
            )

            obj = {
                "model": model_name,
                "caption_keys": list(CAPTION_KEYS),
                "segment_ids": [item[0] for item in pairs],
                "start_times": [item[1] for item in pairs],
                "end_times": [item[2] for item in pairs],
                "embeddings": embeddings,
                "valid_mask": valid_mask,
            }
            save_pickle_atomic(obj, out_pkl)
            summary["saved"] += 1
            print(
                f"[SAVED] {vid_name}: {out_pkl} | "
                f"shape={embeddings.shape} | elapsed {time.time()-t0:.3f}s"
            )

        except Exception as e:
            print(f"[ERROR] {vid_name}: {e}")
            summary["failed"] += 1
    print(
        "Text embedding summary: directories={total}, saved={saved}, skipped={skipped}, "
        "failed={failed}".format(**summary)
    )
    return summary


def main() -> int:
    """
    Parse arguments, load API settings, and generate text embeddings in batches.
    """
    parser = argparse.ArgumentParser(description="Generate independent multi-key text embeddings")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Shared preprocessing YAML; API keys come from its declared environment variable")
    parser.add_argument("--preprocess_dir", type=str,
                        default=DEFAULT_PREPROCESS_DIR,
                        help="Preprocessing root directory")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="",
        help="Source video root referenced by manifests",
    )
    parser.add_argument(
        "--annotation_json",
        dest="annotation_files",
        nargs="+",
        default=[],
        help="Optional manifests; process only referenced video IDs",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="",
        help="Process one video_id or extension-free filename",
    )
    parser.add_argument(
        "--expected_video_count",
        type=int,
        default=0,
        help="Expected video count; 0 disables the check",
    )
    parser.add_argument("--model", type=str, default="text-embedding-3-large",
                        help="Text embedding model passed to encode_sentences")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing text embeddings")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Caption keys submitted per embedding request",
    )
    args = parser.parse_args()

    api_base = configure_embedding_api(args.config)
    print(f"Config file: {os.path.abspath(args.config)}")
    print(f"Text embedding API endpoint: {api_base}")
    print("Text embedding API key: loaded from the environment")
    target_video_names = None  # type: Optional[List[str]]
    if args.annotation_files:
        if not args.video_dir:
            parser.error("--annotation_json requires --video_dir")
        # Load dataset mapping only in manifest mode.
        from preprocess_manifest import load_annotation_video_tasks

        tasks = load_annotation_video_tasks(
            video_root=args.video_dir,
            annotation_files=args.annotation_files,
            video_name=args.video_name,
        )
        target_video_names = [task_name for task_name, _ in tasks]
    elif args.video_name:
        target_video_names = [args.video_name]

    if not target_video_names and (args.annotation_files or args.video_name):
        parser.error("No matching preprocessing videos found")
    if args.expected_video_count > 0 and not args.video_name:
        actual_count = len(target_video_names or [])
        if actual_count != args.expected_video_count:
            parser.error(
                "Manifest contains {} unique videos; expected {}".format(
                    actual_count,
                    args.expected_video_count,
                )
            )

    summary = process_preprocess_root(
        preprocess_root=args.preprocess_dir,
        model_name=args.model,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        target_video_names=target_video_names,
    )
    return 1 if summary["failed"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
