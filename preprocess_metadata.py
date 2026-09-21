"""Write a non-secret manifest for completed preprocessing outputs."""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict

from preprocess_config import (
    DEFAULT_CONFIG_PATH,
    get_config_value,
    load_preprocess_config,
)


REQUIRED_VIDEO_OUTPUTS = (
    "captions.json",
    "segment_textual_embedding.pkl",
    "segment_visual_embedding.pkl",
)


def count_completed_videos(output_dir: str) -> int:
    if not os.path.isdir(output_dir):
        return 0
    completed = 0
    for name in os.listdir(output_dir):
        video_dir = os.path.join(output_dir, name)
        if not os.path.isdir(video_dir):
            continue
        if all(os.path.isfile(os.path.join(video_dir, item)) for item in REQUIRED_VIDEO_OUTPUTS):
            completed += 1
    return completed


def build_manifest(
    config: Dict[str, Any],
    dataset: str,
    video_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": dataset,
        "preprocess_id": os.path.basename(os.path.normpath(output_dir)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "video_dir": os.path.abspath(video_dir),
            "expected_video_count": get_config_value(
                config, "expected_video_count", dataset=dataset
            ),
        },
        "output": {
            "root": os.path.abspath(output_dir),
            "completed_video_count": count_completed_videos(output_dir),
            "per_video_files": list(REQUIRED_VIDEO_OUTPUTS),
        },
        "caption": {
            "model": get_config_value(config, "models.caption.name"),
            "model_path": get_config_value(config, "models.caption.path"),
            "clip_seconds": get_config_value(config, "clip_seconds", dataset=dataset),
            "prompt_path": get_config_value(config, "caption_prompt", dataset=dataset),
        },
        "text_embedding": {
            "model": get_config_value(config, "models.text_embedding.name"),
            "batch_size": get_config_value(config, "models.text_embedding.batch_size"),
            "caption_keys": ["environment", "event", "attention", "summary"],
        },
        "visual_embedding": {
            "model": get_config_value(config, "models.visual_embedding.name"),
            "checkpoint": get_config_value(
                config, "models.visual_embedding.checkpoint"
            ),
            "frames_per_segment": get_config_value(
                config, "models.visual_embedding.frames_per_segment"
            ),
        },
    }


def save_json_atomic(payload: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".manifest_",
        suffix=".json",
        dir=os.path.dirname(output_path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)
            file_obj.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the preprocessing manifest.json")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", required=True, choices=("videomme", "m3bench"))
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--caption_model_path", required=True)
    parser.add_argument("--caption_prompt", required=True)
    parser.add_argument("--clip_seconds", required=True, type=int)
    parser.add_argument("--text_model", required=True)
    parser.add_argument("--text_batch_size", required=True, type=int)
    parser.add_argument("--viclip_checkpoint", required=True)
    parser.add_argument("--frames_per_segment", required=True, type=int)
    args = parser.parse_args()

    config = load_preprocess_config(args.config)
    manifest = build_manifest(
        config=config,
        dataset=args.dataset,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
    )
    # Record the values actually passed by the shell entrypoint so CLI and
    # environment overrides never make the manifest disagree with the run.
    manifest["caption"].update(
        {
            "model_path": os.path.abspath(args.caption_model_path),
            "clip_seconds": args.clip_seconds,
            "prompt_path": os.path.abspath(args.caption_prompt),
        }
    )
    manifest["text_embedding"].update(
        {
            "model": args.text_model,
            "batch_size": args.text_batch_size,
        }
    )
    manifest["visual_embedding"].update(
        {
            "checkpoint": os.path.abspath(args.viclip_checkpoint),
            "frames_per_segment": args.frames_per_segment,
        }
    )
    output_path = os.path.join(args.output_dir, "manifest.json")
    save_json_atomic(manifest, output_path)
    print("Preprocessing manifest: {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
