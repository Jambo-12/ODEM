"""Command-line interface for M3-Bench evaluation."""

import argparse
import os

from eval import runtime


PROJECT_ROOT = runtime.PROJECT_ROOT
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "m3bench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M3-Bench open-ended episodic LangGraph")
    parser.add_argument("--config", default=os.path.join(EVAL_DIR, "config.yaml"))
    parser.add_argument(
        "--annotation_json",
        default=os.path.join(
            PROJECT_ROOT, "data", "raw", "m3bench", "annotations", "robot.json"
        ),
    )
    parser.add_argument(
        "--video_root",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "m3bench", "videos"),
    )
    parser.add_argument(
        "--preprocess_root",
        default=os.path.join(
            PROJECT_ROOT, "data", "processed", "m3bench", "qwen2_5_vl_7b_30s"
        ),
    )
    parser.add_argument(
        "--output_path", default=os.path.join(DEFAULT_OUTPUT_ROOT, "predictions.json")
    )
    parser.add_argument(
        "--result_root", default=os.path.join(DEFAULT_OUTPUT_ROOT, "results")
    )
    parser.add_argument("--log_root", default=os.path.join(DEFAULT_OUTPUT_ROOT, "logs"))
    parser.add_argument(
        "--performance_log_path",
        default=os.path.join(DEFAULT_OUTPUT_ROOT, "performance.log"),
    )
    parser.add_argument("--max_videos", type=int, default=100)
    parser.add_argument("--video_name", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--draw_graph", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_videos <= 0:
        raise ValueError("max_videos must be greater than 0")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard {}/{}".format(args.shard_index, args.num_shards))
