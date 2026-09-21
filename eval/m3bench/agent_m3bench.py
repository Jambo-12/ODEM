"""Run the M3-Bench evaluation."""

import os

from performance_trace import configure_performance_tracking, finalize_performance_tracking

from eval import runtime
from eval.config import load_config
from eval.m3bench.cli import EVAL_DIR, parse_args, validate_args
from eval.m3bench.dataset import load_robot_dataset, select_shard
from eval.m3bench.factory import build_agents
from eval.m3bench.graph import build_graph
from eval.m3bench.runner import run_dataset


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = load_config(args.config)

    videos = load_robot_dataset(
        args.annotation_json,
        args.video_root,
        args.preprocess_root,
    )
    if args.video_name:
        videos = [video for video in videos if video["video_name"] == args.video_name]
        if not videos:
            raise ValueError("M3-Bench video not found: {}".format(args.video_name))
    videos = select_shard(videos[:args.max_videos], args.num_shards, args.shard_index)
    if not videos:
        print(
            "M3-Bench shard {}/{} has no video tasks".format(
                args.shard_index, args.num_shards
            )
        )
        return

    with runtime.suppress_normal_console_output():
        modules = build_agents(
            config,
            args.preprocess_root,
            preload_vlm=not args.draw_graph,
        )
    graph = build_graph(modules)
    if args.draw_graph:
        graph_path = os.path.join(EVAL_DIR, "langgraph_m3bench.png")
        graph.get_graph().draw_mermaid_png(output_file_path=graph_path)
        print("M3-Bench LangGraph saved to: {}".format(graph_path))
        return

    configure_performance_tracking(
        bool(config.get("performance_tracking_enabled", True)),
        args.performance_log_path,
    )
    try:
        run_dataset(
            graph=graph,
            videos=videos,
            output_path=args.output_path,
            result_root=args.result_root,
            log_root=args.log_root,
            rerun=bool(args.rerun or config.get("rerun", False)),
            max_iterations=int(config.get("max_iterations", 3)),
            max_concurrency=int(config.get("graph_max_concurrency", 1)),
        )
    finally:
        finalize_performance_tracking()


if __name__ == "__main__":
    main()
