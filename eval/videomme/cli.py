"""Command-line interface for Video-MME evaluation."""

import argparse
import os

from eval import runtime


PROJECT_ROOT = runtime.PROJECT_ROOT
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "videomme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-MME Streaming Episodic LangGraph")
    parser.add_argument("--config", default=os.path.join(EVAL_DIR, "config.yaml"))
    parser.add_argument(
        "--input_json",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "videomme", "videoMME_long.json"),
    )
    parser.add_argument(
        "--video_root",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "videomme", "videos"),
    )
    parser.add_argument(
        "--preprocess_root",
        default=os.path.join(
            PROJECT_ROOT, "data", "processed", "videomme", "qwen2_5_vl_7b_30s"
        ),
    )
    parser.add_argument(
        "--output_path", default=os.path.join(DEFAULT_OUTPUT_ROOT, "predictions.json")
    )
    parser.add_argument(
        "--accuracy_path", default=os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy.json")
    )
    parser.add_argument("--log_root", default=os.path.join(DEFAULT_OUTPUT_ROOT, "logs"))
    parser.add_argument(
        "--performance_log_path",
        default=os.path.join(DEFAULT_OUTPUT_ROOT, "performance.log"),
    )
    parser.add_argument("--max_videos", type=int, default=300)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total evaluation shards; 1 preserves single-process behavior",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based shard index handled by this process",
    )
    parser.add_argument(
        "--perception_enable_asr", type=runtime.parse_boolean, default=None
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--draw_graph", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_videos <= 0:
        raise ValueError("max_videos must be greater than 0")
    if args.num_shards <= 0:
        raise ValueError("num_shards must be greater than 0")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            "shard_index must be in [0, {}); got {}".format(
                args.num_shards, args.shard_index
            )
        )
