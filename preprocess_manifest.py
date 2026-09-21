"""
Read dataset manifests and map output video IDs to source video paths.

This module provides optional manifest input adaptation. The public Video-MME and
M3-Bench entrypoints currently use directory-scan mode. Example:

    from preprocess_manifest import load_annotation_video_tasks

    tasks = load_annotation_video_tasks(
        video_root="/path/to/videos",
        annotation_files=["/path/to/annotations.json"],
    )
"""
import json
import os
from typing import Dict, List, Tuple


SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm")
VideoTask = Tuple[str, str]


def resolve_dataset_path(root_dir: str, relative_path: str, field_name: str) -> str:
    """
    Safely resolve an annotation-relative path under the dataset root.

    Reject absolute paths and traversal outside the dataset directory.
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("Annotation field {} must be a nonempty string".format(field_name))
    if os.path.isabs(relative_path):
        raise ValueError(
            "Annotation field {} cannot use an absolute path: {}".format(field_name, relative_path)
        )

    root_path = os.path.realpath(root_dir)
    resolved_path = os.path.realpath(os.path.join(root_path, relative_path))
    try:
        inside_root = os.path.commonpath([root_path, resolved_path]) == root_path
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError(
            "Annotation field {} escapes the dataset directory: {}".format(field_name, relative_path)
        )
    return resolved_path


def validate_video_id(video_id: object, annotation_file: str, record_index: int) -> str:
    """Validate that video_id is safe as a preprocessing subdirectory name."""
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError(
            "Annotation file {} record {} has no valid video_id".format(
                annotation_file,
                record_index,
            )
        )
    task_name = video_id.strip()
    if os.path.basename(task_name) != task_name or task_name in (".", ".."):
        raise ValueError("video_id cannot contain a directory path: {}".format(task_name))
    return task_name


def load_annotation_video_tasks(
    video_root: str,
    annotation_files: List[str],
    video_name: str = "",
) -> List[VideoTask]:
    """
    Return deduplicated, sorted ``(video_id, video_path)`` tasks from manifests.

    ``video_name`` may be an official video_id or an extension-free filename. Keep one task
    when multiple questions share a video_id so preprocessing runs only once.
    """
    if not annotation_files:
        raise ValueError("annotation_files cannot be empty")
    if not os.path.isdir(video_root):
        raise FileNotFoundError("Video directory does not exist: {}".format(video_root))

    task_paths = {}  # type: Dict[str, str]
    file_stems = {}  # type: Dict[str, str]
    path_owners = {}  # type: Dict[str, str]

    for annotation_file in annotation_files:
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError("Annotation file does not exist: {}".format(annotation_file))
        with open(annotation_file, "r", encoding="utf-8") as file_obj:
            records = json.load(file_obj)
        if not isinstance(records, list):
            raise ValueError("Annotation file root must be a list: {}".format(annotation_file))

        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    "Annotation file {} record {} is not an object".format(
                        annotation_file,
                        record_index,
                    )
                )
            task_name = validate_video_id(
                record.get("video_id"),
                annotation_file,
                record_index,
            )
            video_path = resolve_dataset_path(
                video_root,
                record.get("video_path"),
                "video_path",
            )
            file_stem, extension = os.path.splitext(os.path.basename(video_path))
            if extension.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                raise ValueError("Annotation references an unsupported video format: {}".format(video_path))
            if not os.path.isfile(video_path):
                raise FileNotFoundError("Annotation video does not exist: {}".format(video_path))
            if task_name in task_paths and task_paths[task_name] != video_path:
                raise ValueError("One video_id points to multiple videos: {}".format(task_name))
            if video_path in path_owners and path_owners[video_path] != task_name:
                raise ValueError(
                    "One video file is referenced by multiple video_ids: {} and {}".format(
                        path_owners[video_path],
                        task_name,
                    )
                )
            task_paths[task_name] = video_path
            file_stems[task_name] = file_stem
            path_owners[video_path] = task_name

    tasks = []  # type: List[VideoTask]
    for task_name in sorted(task_paths):
        if video_name and video_name not in (task_name, file_stems[task_name]):
            continue
        tasks.append((task_name, task_paths[task_name]))
    return tasks
