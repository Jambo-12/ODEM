"""Run the Video-MME evaluation."""

import os

from performance_trace import configure_performance_tracking, finalize_performance_tracking

from eval import runtime
from eval.config import load_config
from eval.videomme.cli import EVAL_DIR, parse_args, validate_args
from eval.videomme.dataset import load_dataset, select_shard
from eval.videomme.factory import build_agents
from eval.videomme.graph import answer_node, build_graph
from eval.videomme.runner import run_dataset


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = load_config(args.config)
    if args.perception_enable_asr is not None:
        config.perception_enable_asr = args.perception_enable_asr

    with runtime.suppress_normal_console_output():
        modules = build_agents(
            config,
            args.preprocess_root,
            preload_vlm=not args.draw_graph,
        )
    graph = build_graph(modules)
    if args.draw_graph:
        graph_path = os.path.join(EVAL_DIR, "langgraph_videomme.png")
        graph.get_graph().draw_mermaid_png(output_file_path=graph_path)
        print("Video-MME LangGraph saved to: {}".format(graph_path))
        return

    videos, video_names = load_dataset(args.input_json)
    videos = select_shard(videos, args.num_shards, args.shard_index)
    if not videos:
        raise ValueError(
            "Current shard has no videos: shard_index={}, num_shards={}".format(
                args.shard_index, args.num_shards
            )
        )

    configure_performance_tracking(
        bool(config.get("performance_tracking_enabled", True)),
        args.performance_log_path,
    )
    try:
        run_dataset(
            graph=graph,
            videos=videos,
            video_names=video_names,
            video_root=args.video_root,
            preprocess_root=args.preprocess_root,
            output_path=args.output_path,
            accuracy_path=args.accuracy_path,
            log_root=args.log_root,
            rerun=bool(args.rerun or config.get("rerun", False)),
            max_videos=min(args.max_videos, len(videos)),
            max_iterations=int(config.get("max_iterations", 3)),
            max_concurrency=int(config.get("graph_max_concurrency", 1)),
            diagnostic_logging=bool(config.get("diagnostic_logging_enabled", False)),
        )
    finally:
        finalize_performance_tracking()


if __name__ == "__main__":
    main()
