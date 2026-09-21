"""Artifact loading and validation shared by benchmark retrieval nodes."""

import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np


TEXT_EMBEDDING_FILE = "segment_textual_embedding.pkl"
VISUAL_EMBEDDING_FILE = "segment_visual_embedding.pkl"
CAPTION_FILE = "captions.json"
RETRIEVAL_KEYS = ("environment", "event", "attention", "summary")


class RetrievalArtifacts:
    """Read the existing caption and embedding schemas without changing them."""

    @staticmethod
    def _load_pickle(path: str) -> Any:
        if not os.path.isfile(path):
            raise FileNotFoundError("Preprocessing file does not exist: {}".format(path))
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _validate_segment_ids(segment_ids: List[int], row_count: int) -> None:
        if len(segment_ids) != row_count:
            raise ValueError("segment_ids count does not match embedding rows")
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("segment_ids contains duplicates")

    @classmethod
    def _unpack_textual_embeddings(
        cls,
        payload: Any,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
        if not isinstance(payload, dict):
            legacy_embeddings = np.asarray(payload)
            if legacy_embeddings.ndim == 2:
                raise ValueError(
                    "Legacy 2D text vectors detected; regenerate [N, 4, D] four-key vectors"
                )
            raise ValueError("Text embedding PKL must be a metadata dictionary")
        embeddings = np.asarray(payload.get("embeddings"), dtype=np.float32)
        if embeddings.ndim == 2:
            raise ValueError(
                "Legacy 2D text vectors detected; regenerate [N, 4, D] four-key vectors"
            )
        if (
            embeddings.ndim != 3
            or embeddings.shape[1] != len(RETRIEVAL_KEYS)
            or embeddings.shape[2] <= 0
        ):
            raise ValueError(
                "Text embeddings must have shape [N, 4, D]; got {}".format(
                    embeddings.shape
                )
            )
        if payload.get("caption_keys") != list(RETRIEVAL_KEYS):
            raise ValueError(
                "Text embedding caption_keys must be ordered as {}".format(RETRIEVAL_KEYS)
            )

        valid_mask = np.asarray(payload.get("valid_mask"), dtype=np.bool_)
        if valid_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "valid_mask must have shape [N, 4]; got {}".format(valid_mask.shape)
            )
        if np.any(~np.any(valid_mask, axis=1)):
            raise ValueError("A text segment has no valid caption keys")

        segment_ids = [int(value) for value in payload.get("segment_ids", [])]
        cls._validate_segment_ids(segment_ids, embeddings.shape[0])
        return embeddings, valid_mask, segment_ids, payload

    @classmethod
    def _unpack_visual_embeddings(
        cls,
        payload: Any,
    ) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
        if isinstance(payload, dict):
            embeddings = np.asarray(payload.get("embeddings"), dtype=np.float32)
            segment_ids = [int(value) for value in payload.get("segment_ids", [])]
            metadata = payload
        else:
            embeddings = np.asarray(payload, dtype=np.float32)
            segment_ids = list(range(len(embeddings)))
            metadata = {}
        if embeddings.ndim != 2 or embeddings.shape[1] <= 0:
            raise ValueError(
                "Visual embeddings must be a 2D array; got shape={}".format(
                    embeddings.shape
                )
            )
        cls._validate_segment_ids(segment_ids, embeddings.shape[0])
        return embeddings, segment_ids, metadata

    @staticmethod
    def _cosine_scores(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        query_vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if query_vector.shape[0] != embeddings.shape[1]:
            raise ValueError(
                "Query dimension {} does not match segment dimension {}".format(
                    query_vector.shape[0],
                    embeddings.shape[1],
                )
            )
        query_norm = max(float(np.linalg.norm(query_vector)), 1e-12)
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        denominator = np.maximum(embedding_norms * query_norm, 1e-12)
        return np.matmul(embeddings, query_vector) / denominator

    @staticmethod
    def _load_captions(path: str) -> Dict[int, Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as file:
            captions = json.load(file)
        if not isinstance(captions, list):
            raise ValueError("captions.json must contain a list")
        return {index: item for index, item in enumerate(captions)}

    @staticmethod
    def _time_map(metadata: Dict[str, Any], key: str) -> Dict[int, float]:
        segment_ids = metadata.get("segment_ids", [])
        values = metadata.get(key, [])
        if len(segment_ids) != len(values):
            return {}
        return {
            int(segment_id): float(value)
            for segment_id, value in zip(segment_ids, values)
        }

    @classmethod
    def _aligned_time_maps(
        cls,
        textual_meta: Dict[str, Any],
        visual_meta: Dict[str, Any],
        shared_ids: List[int],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        textual_starts = cls._time_map(textual_meta, "start_times")
        textual_ends = cls._time_map(textual_meta, "end_times")
        visual_starts = cls._time_map(visual_meta, "start_times")
        visual_ends = cls._time_map(visual_meta, "end_times")
        for segment_id in shared_ids:
            if (
                segment_id not in textual_starts
                or segment_id not in textual_ends
                or segment_id not in visual_starts
                or segment_id not in visual_ends
            ):
                raise ValueError(
                    "Segment {} is missing text or visual timeline metadata".format(segment_id)
                )
            if (
                not np.isclose(textual_starts[segment_id], visual_starts[segment_id])
                or not np.isclose(textual_ends[segment_id], visual_ends[segment_id])
            ):
                raise ValueError(
                    "Text and visual timelines differ for segment {}".format(segment_id)
                )
        return textual_starts, textual_ends
