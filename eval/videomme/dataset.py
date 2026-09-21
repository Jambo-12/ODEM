"""Video-MME dataset loading and resume records."""

import json
import os
from typing import Any, Dict, List, Tuple

from eval import runtime


def load_dataset(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    return runtime.load_video_mme(path)


def select_shard(
    videos: List[Dict[str, Any]], num_shards: int, shard_index: int
) -> List[Dict[str, Any]]:
    return runtime.select_dataset_shard(videos, num_shards, shard_index)


def existing_records(path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        videos = json.load(file)

    records = {}  # type: Dict[Tuple[str, str], Dict[str, Any]]
    if not isinstance(videos, list):
        return records
    for video in videos:
        if not isinstance(video, dict):
            continue
        video_id = str(video.get("video_id", ""))
        for question in video.get("questions", []):
            if not isinstance(question, dict):
                continue
            response = question.get("response")
            if response is None or not str(response).strip():
                continue
            records[(video_id, str(question.get("question_id", "")))] = {
                "response": response,
                "response_detail": question.get("response_detail"),
            }
    return records
