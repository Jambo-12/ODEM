"""Implement Summary-ViCLIP fused retrieval, ASR, memory assembly, and visual tools for Video-MME."""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from openai import OpenAI

from eval.common import format_options, render_prompt
from eval.retrieval import (
    CAPTION_FILE,
    TEXT_EMBEDDING_FILE,
    VISUAL_EMBEDDING_FILE,
    RetrievalArtifacts,
)
from performance_trace import record_unknown_token_usage, record_usage_object

from eval.videomme.state import VideoMMEState


SUMMARY_KEY = "summary"
MEMORY_KEYS = ("environment", "event", "attention", "summary")


def _append_memory(
    state: VideoMMEState, node_name: str, summary: str
) -> List[Dict[str, Any]]:
    """Append a short node-level log record."""
    return list(state.get("working_memory", [])) + [{
        "agent": node_name,
        "summary": summary,
    }]


def _unique_ints(values: Iterable[Any]) -> List[int]:
    """Deduplicate in input order and ignore values that cannot be converted to integers."""
    output = []  # type: List[int]
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number not in output:
            output.append(number)
    return output


def _episode_map(memory: Any) -> Dict[int, Dict[str, Any]]:
    """Copy episodes from state and build a segment_id index."""
    output = {}  # type: Dict[int, Dict[str, Any]]
    if not isinstance(memory, list):
        return output
    for item in memory:
        if not isinstance(item, dict) or item.get("segment_id") is None:
            continue
        try:
            output[int(item["segment_id"])] = dict(item)
        except (TypeError, ValueError):
            continue
    return output


def _slice_cached_asr(
    transcript: Any,
    start_time: float,
    end_time: float,
    segment_id: Any,
) -> Optional[Dict[str, Any]]:
    """Crop utterances from existing ASR that fully covers the target range to avoid duplicate audio API calls."""
    if not isinstance(transcript, dict):
        return None
    if transcript.get("status") not in ("success", "no_audio", "no_speech"):
        return None
    try:
        raw_cached_start = transcript.get("start_time")
        raw_cached_end = transcript.get("end_time")
        if raw_cached_start is None or raw_cached_end is None:
            return None
        cached_start = float(raw_cached_start)
        cached_end = float(raw_cached_end)
    except (TypeError, ValueError):
        return None
    if cached_start > start_time + 1e-3 or cached_end < end_time - 1e-3:
        return None
    utterances = []  # type: List[Dict[str, Any]]
    for item in transcript.get("utterances", []):
        if not isinstance(item, dict):
            continue
        try:
            raw_utterance_start = item.get("start_time")
            raw_utterance_end = item.get("end_time")
            if raw_utterance_start is None or raw_utterance_end is None:
                continue
            utterance_start = float(raw_utterance_start)
            utterance_end = float(raw_utterance_end)
        except (TypeError, ValueError):
            continue
        if utterance_end < start_time or utterance_start > end_time:
            continue
        utterances.append(dict(item))
    return {
        "segment_id": segment_id,
        "start_time": start_time,
        "end_time": end_time,
        "status": transcript.get("status"),
        "utterances": utterances,
        "cached": True,
        "cache_source": "episodic_memory",
    }


class VideoMMELocalizationNode:
    """Fuse Summary and ViCLIP similarity for each query and accumulate deduplicated episodes."""

    def __init__(
        self,
        preprocess_root: str,
        text_api_key: str,
        text_api_base: str,
        text_model: str,
        viclip_size: str,
        viclip_pretrained: str,
        device: str,
        textual_weight: float,
        visual_weight: float,
        top_k: int = 5,
    ) -> None:
        # Initialize text and visual query encoders on first use, then reuse them across questions.
        self.preprocess_root = str(preprocess_root)
        self.text_api_key = str(text_api_key)
        self.text_api_base = str(text_api_base).rstrip("/")
        self.text_model = str(text_model)
        self.viclip_size = str(viclip_size)
        self.viclip_pretrained = str(viclip_pretrained)
        self.device = str(device)
        self.textual_weight = float(textual_weight)
        self.visual_weight = float(visual_weight)
        self.top_k = max(1, int(top_k))
        self._text_client = None  # type: Optional[OpenAI]
        self._viclip = None  # type: Optional[Any]
        self._viclip_tokenizer = None  # type: Optional[Any]

    @staticmethod
    def _load_pickle(path: str) -> Any:
        return RetrievalArtifacts._load_pickle(path)

    @staticmethod
    def _unpack_textual_embeddings(
        payload: Any,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
        return RetrievalArtifacts._unpack_textual_embeddings(payload)

    @staticmethod
    def _unpack_visual_embeddings(
        payload: Any,
    ) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
        return RetrievalArtifacts._unpack_visual_embeddings(payload)

    @staticmethod
    def _time_map(metadata: Dict[str, Any], key: str) -> Dict[int, float]:
        return RetrievalArtifacts._time_map(metadata, key)

    @staticmethod
    def _load_captions(path: str) -> Dict[int, Dict[str, Any]]:
        return RetrievalArtifacts._load_captions(path)

    @staticmethod
    def _cosine_scores(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        return RetrievalArtifacts._cosine_scores(query, embeddings)

    @staticmethod
    def _aligned_time_maps(
        textual_meta: Dict[str, Any],
        visual_meta: Dict[str, Any],
        shared_ids: List[int],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        return RetrievalArtifacts._aligned_time_maps(
            textual_meta, visual_meta, shared_ids
        )

    def _encode_text(self, query: str, model_name: str) -> np.ndarray:
        """Encode queries with the same model used for preprocessed summaries and record embedding usage."""
        if self._text_client is None:
            self._text_client = OpenAI(
                api_key=self.text_api_key,
                base_url=self.text_api_base,
            )
        client = self._text_client
        if client is None:
            raise RuntimeError("Failed to initialize the text embedding client")
        try:
            response = client.embeddings.create(
                input=[query],
                model=model_name,
            )
        except Exception:
            record_unknown_token_usage("localization_embedding", "embedding")
            raise
        usage = getattr(response, "usage", None)
        if usage is None:
            record_unknown_token_usage("localization_embedding", "embedding")
        else:
            record_usage_object(
                "localization_embedding", "embedding", usage, embedding_call=True
            )
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    def _encode_visual(self, query: str) -> np.ndarray:
        """Use the ViCLIP text encoder to create query vectors in the preprocessed video-vector space."""
        if self._viclip is None or self._viclip_tokenizer is None:
            if not os.path.isfile(self.viclip_pretrained):
                raise FileNotFoundError(
                    "ViCLIP weights do not exist: {}".format(self.viclip_pretrained)
                )
            from InternVid.viclip import get_viclip

            model_data = get_viclip(self.viclip_size, self.viclip_pretrained)
            self._viclip = model_data["viclip"].to(self.device).eval()
            self._viclip_tokenizer = model_data["tokenizer"]
        viclip = self._viclip
        tokenizer = self._viclip_tokenizer
        if viclip is None or tokenizer is None:
            raise RuntimeError("Failed to initialize the ViCLIP query encoder")
        with torch.no_grad():
            embedding = viclip.get_text_features(query, tokenizer, {})
        return embedding.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _queries(state: VideoMMEState) -> List[Tuple[str, str]]:
        """Collect the main query and up to four subqueries, deduplicating by text."""
        plan_call = max(1, len(state.get("query_history", [])))
        prefix = "plan_{}".format(plan_call)
        query_items = [(
            "{}.retrieval_query".format(prefix),
            state.get("retrieval_query", ""),
        )]
        query_items.extend(
            ("{}.sub_query_{}".format(prefix, index), value)
            for index, value in enumerate(state.get("sub_queries", [])[:4], start=1)
        )
        normalized = []  # type: List[Tuple[str, str]]
        seen = set()
        for query_id, raw_query in query_items:
            query = " ".join(str(raw_query or "").strip().split())
            key = query.lower()
            if not query or key in seen:
                continue
            seen.add(key)
            normalized.append((query_id, query))
        if not normalized:
            fallback = " ".join(str(state.get("question", "")).strip().split())
            normalized.append((
                "{}.retrieval_query".format(prefix),
                fallback or "Relevant video evidence",
            ))
        return normalized

    @staticmethod
    def _caption_fields(caption: Dict[str, Any]) -> Dict[str, Any]:
        """Retain only the four caption-memory fields defined by the current experiment."""
        return {key: caption.get(key) for key in MEMORY_KEYS}

    @staticmethod
    def _merge_match(
        episode: Dict[str, Any],
        query_id: str,
        query: str,
        summary_score: float,
        visual_score: Optional[float],
        fused_score: float,
        rank: int,
    ) -> None:
        """Merge text, visual, and fused scores when different queries retrieve the same segment."""
        matches = [
            str(item) for item in episode.get("matched_queries", [])
            if str(item).strip()
        ]
        if query_id not in matches:
            matches.append(query_id)
        episode["matched_queries"] = matches
        query_texts = dict(episode.get("matched_query_texts", {}))
        query_texts[query_id] = query
        episode["matched_query_texts"] = query_texts
        summary_scores = dict(episode.get("summary_scores", {}))
        summary_scores[query_id] = round(float(summary_score), 6)
        episode["summary_scores"] = summary_scores
        visual_scores = dict(episode.get("visual_scores", {}))
        visual_scores[query_id] = (
            round(float(visual_score), 6) if visual_score is not None else None
        )
        episode["visual_scores"] = visual_scores
        fused_scores = dict(episode.get("fused_scores", {}))
        fused_scores[query_id] = round(float(fused_score), 6)
        episode["fused_scores"] = fused_scores
        ranks = dict(episode.get("query_ranks", {}))
        ranks[query_id] = int(rank)
        episode["query_ranks"] = ranks
        episode["max_summary_score"] = max(
            [float(value) for value in summary_scores.values()] or [float("-inf")]
        )
        valid_visual_scores = [
            float(value) for value in visual_scores.values() if value is not None
        ]
        episode["max_visual_score"] = (
            max(valid_visual_scores) if valid_visual_scores else None
        )
        episode["max_fused_score"] = max(
            [float(value) for value in fused_scores.values()] or [float("-inf")]
        )

    def _validate_text_model(self, metadata: Dict[str, Any]) -> str:
        """Confirm that the online query encoder exactly matches the model recorded in the text PKL."""
        model_name = str(metadata.get("model") or "").strip()
        if not model_name:
            raise ValueError("Text-vector PKL is missing the model field")
        if model_name != self.text_model:
            raise ValueError(
                "Query encoder model {} does not match text-vector model {}".format(
                    self.text_model, model_name
                )
            )
        return model_name

    def _retrieve_summary_only(
        self,
        state: VideoMMEState,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Use Summary-only ranking when the visual path fails so question inference can continue."""
        video_dir = os.path.join(self.preprocess_root, state.get("video_name", ""))
        payload = self._load_pickle(os.path.join(video_dir, TEXT_EMBEDDING_FILE))
        embeddings, valid_mask, segment_ids, metadata = self._unpack_textual_embeddings(
            payload
        )
        caption_keys = list(metadata.get("caption_keys", []))
        if SUMMARY_KEY not in caption_keys:
            raise ValueError("Text-vector PKL is missing the summary key")
        summary_index = caption_keys.index(SUMMARY_KEY)
        summary_embeddings = embeddings[:, summary_index, :]
        summary_valid = valid_mask[:, summary_index]
        if not np.any(summary_valid):
            raise ValueError("Text-vector PKL contains no valid Summary embeddings")
        model_name = self._validate_text_model(metadata)
        start_times = self._time_map(metadata, "start_times")
        end_times = self._time_map(metadata, "end_times")
        captions = self._load_captions(os.path.join(video_dir, CAPTION_FILE))

        query_results = []  # type: List[Dict[str, Any]]
        for query_id, query in self._queries(state):
            summary_scores = self._cosine_scores(
                self._encode_text(query, model_name), summary_embeddings
            )
            valid_rows = np.flatnonzero(summary_valid)
            ranked_rows = valid_rows[
                np.argsort(-summary_scores[valid_rows], kind="stable")
            ]
            hits = []  # type: List[Dict[str, Any]]
            for rank, raw_row in enumerate(ranked_rows[:self.top_k], start=1):
                row = int(raw_row)
                segment_id = int(segment_ids[row])
                if segment_id not in captions:
                    raise ValueError("Caption is missing segment_id={}".format(segment_id))
                if segment_id not in start_times or segment_id not in end_times:
                    raise ValueError("Text vectors are missing the segment timeline: {}".format(segment_id))
                summary_score = round(float(summary_scores[row]), 6)
                hits.append({
                    "rank": rank,
                    "segment_id": segment_id,
                    "start_time": float(start_times[segment_id]),
                    "end_time": float(end_times[segment_id]),
                    "summary_score": summary_score,
                    "visual_score": None,
                    "fused_score": summary_score,
                    "summary": captions[segment_id].get("summary"),
                })
            query_results.append({
                "query_id": query_id,
                "query": query,
                "hits": hits,
            })
        return query_results, {
            "status": "summary_only_fallback",
            "retrieval_mode": "summary_only_fallback",
            "embedding_model": model_name,
            "summary_key_index": summary_index,
            "top_k_per_query": self.top_k,
            "query_count": len(query_results),
            "textual_weight": 1.0,
            "visual_weight": 0.0,
        }

    def _retrieve(
        self,
        state: VideoMMEState,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Calculate Summary, ViCLIP, and weighted fused scores for each query and return the top five."""
        video_dir = os.path.join(self.preprocess_root, state.get("video_name", ""))
        textual_payload = self._load_pickle(
            os.path.join(video_dir, TEXT_EMBEDDING_FILE)
        )
        visual_payload = self._load_pickle(
            os.path.join(video_dir, VISUAL_EMBEDDING_FILE)
        )
        (
            textual_embeddings,
            textual_valid_mask,
            textual_ids,
            textual_meta,
        ) = self._unpack_textual_embeddings(textual_payload)
        visual_embeddings, visual_ids, visual_meta = self._unpack_visual_embeddings(
            visual_payload
        )
        caption_keys = list(textual_meta.get("caption_keys", []))
        if SUMMARY_KEY not in caption_keys:
            raise ValueError("Text-vector PKL is missing the summary key")
        summary_index = caption_keys.index(SUMMARY_KEY)
        textual_rows = {
            segment_id: index for index, segment_id in enumerate(textual_ids)
        }
        visual_rows = {
            segment_id: index for index, segment_id in enumerate(visual_ids)
        }
        shared_ids = [
            segment_id
            for segment_id in sorted(set(textual_ids).intersection(visual_ids))
            if textual_valid_mask[textual_rows[segment_id], summary_index]
        ]
        if not shared_ids:
            raise ValueError("Text summaries and visual vectors have no common valid segment_id")

        summary_embeddings = textual_embeddings[
            [textual_rows[segment_id] for segment_id in shared_ids], summary_index, :
        ]
        visual_matrix = visual_embeddings[
            [visual_rows[segment_id] for segment_id in shared_ids]
        ]
        model_name = self._validate_text_model(textual_meta)
        start_times, end_times = self._aligned_time_maps(
            textual_meta, visual_meta, shared_ids
        )
        captions = self._load_captions(os.path.join(video_dir, CAPTION_FILE))

        query_results = []  # type: List[Dict[str, Any]]
        for query_id, query in self._queries(state):
            summary_scores = self._cosine_scores(
                self._encode_text(query, model_name), summary_embeddings
            )
            visual_scores = self._cosine_scores(
                self._encode_visual(query), visual_matrix
            )
            fused_scores = (
                self.textual_weight * summary_scores
                + self.visual_weight * visual_scores
            )
            ranked_rows = np.argsort(-fused_scores, kind="stable")[:self.top_k]
            hits = []  # type: List[Dict[str, Any]]
            for rank, raw_row in enumerate(ranked_rows, start=1):
                row = int(raw_row)
                segment_id = shared_ids[row]
                if segment_id not in captions:
                    raise ValueError("Caption is missing segment_id={}".format(segment_id))
                hits.append({
                    "rank": rank,
                    "segment_id": segment_id,
                    "start_time": float(start_times[segment_id]),
                    "end_time": float(end_times[segment_id]),
                    "summary_score": round(float(summary_scores[row]), 6),
                    "visual_score": round(float(visual_scores[row]), 6),
                    "fused_score": round(float(fused_scores[row]), 6),
                    "summary": captions[segment_id].get("summary"),
                })
            query_results.append({
                "query_id": query_id,
                "query": query,
                "hits": hits,
            })
        return query_results, {
            "status": "ready",
            "retrieval_mode": "summary_viclip_weighted_fusion",
            "text_embedding_model": model_name,
            "visual_embedding_model": visual_meta.get("model", "ViCLIP"),
            "summary_key_index": summary_index,
            "textual_weight": self.textual_weight,
            "visual_weight": self.visual_weight,
            "top_k_per_query": self.top_k,
            "query_count": len(query_results),
        }

    def __call__(self, state: VideoMMEState) -> Dict[str, Any]:
        """Complete multi-query retrieval, segment deduplication, and episode merging across replans."""
        errors = list(state.get("errors", []))
        try:
            query_results, metadata = self._retrieve(state)
        except Exception as fusion_error:
            try:
                query_results, metadata = self._retrieve_summary_only(state)
                metadata["fusion_error"] = str(fusion_error)
                errors.append(
                    "Localization visual fusion failed; Summary-only fallback was used: {}".format(
                        fusion_error
                    )
                )
            except Exception as fallback_error:
                query_results = []
                metadata = {
                    "status": "failed",
                    "retrieval_mode": "summary_viclip_weighted_fusion",
                    "fusion_error": str(fusion_error),
                    "fallback_error": str(fallback_error),
                }
                errors.append(
                    "Localization Node failed after visual fusion and Summary "
                    "fallback: fusion={}; fallback={}".format(
                        fusion_error, fallback_error
                    )
                )

        episodes = _episode_map(state.get("episodic_memory", []))
        previous_ids = set(episodes)
        current_ids = []  # type: List[int]
        captions = {}  # type: Dict[int, Dict[str, Any]]
        if query_results:
            caption_path = os.path.join(
                self.preprocess_root, state.get("video_name", ""), CAPTION_FILE
            )
            captions = self._load_captions(caption_path)

        for query_result in query_results:
            query_id = str(query_result.get("query_id", ""))
            query = str(query_result.get("query", ""))
            for hit in query_result.get("hits", []):
                segment_id = int(hit["segment_id"])
                if segment_id not in current_ids:
                    current_ids.append(segment_id)
                episode = episodes.get(segment_id)
                if episode is None:
                    caption = captions[segment_id]
                    episode = {
                        "segment_id": segment_id,
                        "start_time": float(hit["start_time"]),
                        "end_time": float(hit["end_time"]),
                        "caption": self._caption_fields(caption),
                        "asr": {},
                        "matched_queries": [],
                        "matched_query_texts": {},
                        "summary_scores": {},
                        "visual_scores": {},
                        "fused_scores": {},
                        "query_ranks": {},
                        "max_summary_score": float(hit["summary_score"]),
                        "max_visual_score": hit.get("visual_score"),
                        "max_fused_score": float(hit["fused_score"]),
                        "evidence_status": "unreviewed",
                        "source": "preprocessed_caption",
                    }
                    episodes[segment_id] = episode
                self._merge_match(
                    episode,
                    query_id,
                    query,
                    float(hit["summary_score"]),
                    (
                        float(hit["visual_score"])
                        if hit.get("visual_score") is not None else None
                    ),
                    float(hit["fused_score"]),
                    int(hit["rank"]),
                )

        new_ids = [segment_id for segment_id in current_ids if segment_id not in previous_ids]
        metadata = dict(metadata)  # type: Dict[str, Any]
        metadata.update({
            "unique_segment_count": len(current_ids),
            "new_segment_count": len(new_ids),
            "current_segment_ids": current_ids,
        })
        return {
            # Top-five ranking for each query, allowing retrieval results to be reviewed in logs.
            "query_results": query_results,
            # Query hits for the current pass and the subset that actually requires new ASR.
            "current_retrieved_segment_ids": current_ids,
            "pending_asr_segment_ids": new_ids,
            # episodic_memory accumulates across replans; repeated segments update only sources and scores.
            "episodic_memory": list(episodes.values()),
            "localization_metadata": metadata,
            "errors": errors,
            "working_memory": _append_memory(
                state,
                "localization",
                "Summary-ViCLIP retrieval returned {} unique episodes; {} require ASR.".format(
                    len(current_ids), len(new_ids)
                ),
            ),
        }


class SegmentASRNode:
    """Run controlled concurrent ASR on newly added deduplicated caption segments in this pass."""

    def __init__(self, speaker_asr: Any, max_concurrency: int = 4) -> None:
        if speaker_asr is None:
            raise ValueError("Video-MME segment ASR requires QwenSpeakerASR configuration")
        self.speaker_asr = speaker_asr
        self.max_concurrency = max(1, int(max_concurrency))

    def __call__(self, state: VideoMMEState) -> Dict[str, Any]:
        """Transcribe only new segments; record failures in episode state without blocking memory construction."""
        episodes = _episode_map(state.get("episodic_memory", []))
        pending = [
            segment_id for segment_id in _unique_ints(
                state.get("pending_asr_segment_ids", [])
            )
            if segment_id in episodes and not episodes[segment_id].get("asr")
        ]
        errors = list(state.get("errors", []))
        results = {}  # type: Dict[int, Dict[str, Any]]

        def run_one(segment_id: int) -> Tuple[int, Dict[str, Any]]:
            episode = episodes[segment_id]
            candidate = {
                "segment_id": segment_id,
                "start_time": episode["start_time"],
                "end_time": episode["end_time"],
            }
            try:
                return segment_id, self.speaker_asr(
                    state.get("video_path", ""), candidate
                )
            except Exception as error:
                return segment_id, {
                    "segment_id": segment_id,
                    "start_time": episode["start_time"],
                    "end_time": episode["end_time"],
                    "status": "failed",
                    "utterances": [],
                    "cached": False,
                    "error": str(error),
                }

        if pending:
            workers = min(self.max_concurrency, len(pending))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(run_one, segment_id): segment_id for segment_id in pending}
                for future in as_completed(futures):
                    segment_id, transcript = future.result()
                    results[segment_id] = transcript

        counts = {"success": 0, "no_audio": 0, "no_speech": 0, "failed": 0, "cached": 0}
        for segment_id in pending:
            transcript = results[segment_id]
            episodes[segment_id]["asr"] = transcript
            status = str(transcript.get("status", "failed"))
            counts[status if status in counts else "failed"] += 1
            if transcript.get("cached") is True:
                counts["cached"] += 1
            if status == "failed":
                errors.append(
                    "Speaker ASR failed on segment {}: {}".format(
                        segment_id, transcript.get("error", "unknown error")
                    )
                )

        return {
            # Write ASR directly to its episode so Memory Assembly later serializes captions and dialogue together.
            "episodic_memory": list(episodes.values()),
            "pending_asr_segment_ids": [],
            "asr_batch_status": {
                "requested": len(pending),
                "max_concurrency": self.max_concurrency,
                **counts,
            },
            "errors": errors,
            "working_memory": _append_memory(
                state,
                "segment_asr",
                "ASR processed {} new episodes with concurrency {}.".format(
                    len(pending), self.max_concurrency
                ),
            ),
        }


class EpisodicMemoryNode:
    """Assemble accumulated captions and ASR into context C under a shared token budget."""

    def __init__(self, max_tokens: int = 8000) -> None:
        self.max_tokens = max(256, int(max_tokens))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Control context with a conservative character estimate without requiring another tokenizer model."""
        return max(1, int(math.ceil(len(text) / 3.5)))

    @staticmethod
    def _clean_utterances(asr: Any) -> List[Dict[str, Any]]:
        """Sort ASR utterances by time and remove exact duplicates."""
        if not isinstance(asr, dict) or not isinstance(asr.get("utterances"), list):
            return []
        output = []  # type: List[Dict[str, Any]]
        seen = set()
        for item in sorted(
            (item for item in asr["utterances"] if isinstance(item, dict)),
            key=lambda item: float(item.get("start_time", 0.0)),
        ):
            text = " ".join(str(item.get("text", "")).split())
            key = (round(float(item.get("start_time", 0.0)), 2), text.lower())
            if not text or key in seen:
                continue
            seen.add(key)
            output.append({
                "speaker": item.get("speaker", "SPEAKER_UNKNOWN"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "text": text,
            })
        return output

    @classmethod
    def _serialize_episode(cls, episode: Dict[str, Any]) -> str:
        """Format an episode as a timestamped evidence block that Reason can cite."""
        caption = episode.get("caption", {})
        if not isinstance(caption, dict):
            caption = {}
        lines = [
            "[EPISODE segment_id={segment_id} time={start:.3f}-{end:.3f}s "
            "source=preprocessed_caption summary_score={summary:.6f} "
            "visual_score={visual} fused_score={fused:.6f}]".format(
                segment_id=episode.get("segment_id"),
                start=float(episode.get("start_time", 0.0)),
                end=float(episode.get("end_time", 0.0)),
                summary=float(episode.get("max_summary_score", 0.0)),
                visual=(
                    "{:.6f}".format(float(episode["max_visual_score"]))
                    if episode.get("max_visual_score") is not None else "unavailable"
                ),
                fused=float(
                    episode.get(
                        "max_fused_score", episode.get("max_summary_score", 0.0)
                    )
                ),
            )
        ]
        for key in MEMORY_KEYS:
            value = caption.get(key)
            if value is not None and str(value).strip():
                lines.append("{}: {}".format(key, " ".join(str(value).split())))
        asr = episode.get("asr", {})
        status = asr.get("status", "unavailable") if isinstance(asr, dict) else "unavailable"
        lines.append("asr_status: {}".format(status))
        for utterance in cls._clean_utterances(asr):
            lines.append(
                "asr [{:.3f}-{:.3f}s] {}: {}".format(
                    float(utterance.get("start_time", 0.0)),
                    float(utterance.get("end_time", 0.0)),
                    utterance.get("speaker", "SPEAKER_UNKNOWN"),
                    utterance.get("text", ""),
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _packing_order(
        episodes: Dict[int, Dict[str, Any]], query_results: Any
    ) -> List[int]:
        """First cover high-ranked results for every query, then fill by global fused relevance."""
        order = []  # type: List[int]
        results = query_results if isinstance(query_results, list) else []
        maximum_rank = max(
            [len(item.get("hits", [])) for item in results if isinstance(item, dict)] or [0]
        )
        for rank_index in range(maximum_rank):
            for result in results:
                hits = result.get("hits", []) if isinstance(result, dict) else []
                if rank_index >= len(hits):
                    continue
                try:
                    segment_id = int(hits[rank_index]["segment_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if segment_id in episodes and segment_id not in order:
                    order.append(segment_id)
        remaining = sorted(
            (segment_id for segment_id in episodes if segment_id not in order),
            key=lambda segment_id: (
                -float(
                    episodes[segment_id].get(
                        "max_fused_score",
                        episodes[segment_id].get("max_summary_score", -1.0),
                    )
                ),
                float(episodes[segment_id].get("start_time", 0.0)),
            ),
        )
        return order + remaining

    def __call__(self, state: VideoMMEState) -> Dict[str, Any]:
        """Generate C by query coverage and relevance, retaining compact state markers for trimmed segments."""
        episodes = _episode_map(state.get("episodic_memory", []))
        order = self._packing_order(episodes, state.get("query_results", []))
        blocks = []  # type: List[str]
        included = []  # type: List[int]
        omitted = []  # type: List[int]
        used_tokens = 0
        for segment_id in order:
            block = self._serialize_episode(episodes[segment_id])
            block_tokens = self._estimate_tokens(block)
            if used_tokens + block_tokens <= self.max_tokens:
                blocks.append(block)
                included.append(segment_id)
                used_tokens += block_tokens
                episodes[segment_id]["memory_status"] = "in_context"
            else:
                omitted.append(segment_id)
                episodes[segment_id]["memory_status"] = "trimmed_from_context"

        if omitted:
            marker = "[TRIMMED_EPISODES segment_ids={} reason=token_budget]".format(
                ",".join(str(value) for value in omitted)
            )
            marker_tokens = self._estimate_tokens(marker)
            if used_tokens + marker_tokens <= self.max_tokens:
                blocks.append(marker)
                used_tokens += marker_tokens

        context = "\n\n".join(blocks)
        return {
            # C is the bounded context consumed by Reason; complete structured episodes remain in state.
            "episodic_memory": list(episodes.values()),
            "memory_context": context,
            "memory_context_segment_ids": included,
            "memory_token_count": used_tokens,
            "memory_assembly_status": {
                "max_tokens": self.max_tokens,
                "estimated_tokens": used_tokens,
                "included_segment_ids": included,
                "trimmed_segment_ids": omitted,
            },
            "pending_evidence_source": "caption_asr_memory",
            "latest_tool_evidence": [],
            "working_memory": _append_memory(
                state,
                "memory",
                "Memory C retained {} episodes and trimmed {}.".format(
                    len(included), len(omitted)
                ),
            ),
        }


class VideoMMEPerceptionTool:
    """Obtain visual observations and matching-range ASR concurrently for complete segments selected by Reflect."""

    def __init__(self, prompt_template: str, vlm: Any, speaker_asr: Any, max_segments: int = 2) -> None:
        self.prompt_template = prompt_template
        self.vlm = vlm
        self.speaker_asr = speaker_asr
        self.max_segments = max(1, int(max_segments))

    @staticmethod
    def _safe_asr(speaker_asr: Any, video_path: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return speaker_asr(video_path, candidate)
        except Exception as error:
            return {
                "segment_id": candidate.get("segment_id"),
                "start_time": candidate.get("start_time"),
                "end_time": candidate.get("end_time"),
                "status": "failed",
                "utterances": [],
                "cached": False,
                "error": str(error),
            }

    def __call__(self, state: VideoMMEState) -> Dict[str, Any]:
        """Observe unprocessed target segments, starting VLM and ASR concurrently for each target."""
        request = state.get("tool_request", {})
        episodes = _episode_map(state.get("episodic_memory", []))
        processed = _unique_ints(state.get("processed_segment_ids", []))
        requested_ids = _unique_ints(request.get("segment_ids", []))[:self.max_segments]
        target_ids = [value for value in requested_ids if value in episodes and value not in processed]
        errors = list(state.get("errors", []))
        new_evidence = []  # type: List[Dict[str, Any]]

        for segment_id in target_ids:
            episode = episodes[segment_id]
            candidate = {
                "segment_id": segment_id,
                "start_time": episode["start_time"],
                "end_time": episode["end_time"],
            }
            prompt = render_prompt(self.prompt_template, {
                "QUESTION": state.get("question", ""),
                "OPTIONS": format_options(state.get("options", [])),
                "MEMORY_CONTEXT": state.get("memory_context", ""),
                "TOOL_REQUEST": json.dumps(request, ensure_ascii=False),
                "TARGET_SEGMENT": json.dumps(candidate, ensure_ascii=False),
            })
            cached_transcript = _slice_cached_asr(
                episode.get("asr"),
                float(candidate["start_time"]),
                float(candidate["end_time"]),
                segment_id,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                visual_future = executor.submit(
                    self.vlm, state.get("video_path", ""), candidate, prompt
                )
                asr_future = None
                if cached_transcript is None:
                    asr_future = executor.submit(
                        self._safe_asr,
                        self.speaker_asr,
                        state.get("video_path", ""),
                        candidate,
                    )
                try:
                    visual = visual_future.result()
                except Exception as error:
                    visual = {"status": "failed", "observation": "", "error": str(error)}
                    errors.append("Perception VLM failed on segment {}: {}".format(segment_id, error))
                transcript = (
                    cached_transcript
                    if cached_transcript is not None
                    else asr_future.result() if asr_future is not None else {}
                )
            if transcript.get("status") == "failed":
                errors.append(
                    "Speaker ASR failed on Perception segment {}: {}".format(
                        segment_id, transcript.get("error", "unknown error")
                    )
                )
            evidence = {
                "source": "perception_asr",
                "segment_id": segment_id,
                "time_ranges": [[candidate["start_time"], candidate["end_time"]]],
                "objective": request.get("objective", ""),
                "visual": visual,
                "asr": transcript,
            }
            new_evidence.append(evidence)
            processed.append(segment_id)
            episodes[segment_id]["evidence_status"] = "perception_observed"

        all_evidence = [
            dict(item) for item in state.get("tool_evidence", []) if isinstance(item, dict)
        ] + new_evidence
        return {
            # Combine visual and matching-range ASR into unified evidence accumulated across Reason passes.
            "tool_evidence": all_evidence,
            "latest_tool_evidence": new_evidence,
            "episodic_memory": list(episodes.values()),
            "processed_segment_ids": processed,
            "pending_evidence_source": "perception_asr",
            "errors": errors,
            "working_memory": _append_memory(
                state,
                "perception",
                "Perception inspected {} new segments with parallel ASR.".format(len(new_evidence)),
            ),
        }


class VideoMMEAttentionTool:
    """Obtain high-resolution visual observations and matching ASR concurrently for exact ranges selected by Reflect."""

    def __init__(
        self,
        prompt_template: str,
        vlm: Any,
        speaker_asr: Any,
        max_ranges: int = 3,
        max_total_seconds: float = 15.0,
        max_concurrency: int = 4,
    ) -> None:
        self.prompt_template = prompt_template
        self.vlm = vlm
        self.speaker_asr = speaker_asr
        self.max_ranges = max(1, int(max_ranges))
        self.max_total_seconds = max(0.5, float(max_total_seconds))
        self.max_concurrency = max(2, int(max_concurrency))

    @staticmethod
    def _range_key(segment_id: int, start_time: float, end_time: float) -> str:
        return "{}:{:.3f}-{:.3f}".format(segment_id, start_time, end_time)

    def _normalize_ranges(
        self, request: Dict[str, Any], episode: Dict[str, Any], history: set
    ) -> Tuple[List[List[float]], List[str]]:
        """Constrain Attention ranges to their containing segments and filter exact duplicates."""
        segment_id = int(episode["segment_id"])
        segment_start = float(episode["start_time"])
        segment_end = float(episode["end_time"])
        ranges = []  # type: List[List[float]]
        keys = []  # type: List[str]
        total = 0.0
        for item in request.get("time_ranges", []):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                start_time = max(segment_start, float(item[0]))
                end_time = min(segment_end, float(item[1]))
            except (TypeError, ValueError):
                continue
            duration = end_time - start_time
            if duration <= 0 or total >= self.max_total_seconds:
                continue
            end_time = min(end_time, start_time + self.max_total_seconds - total)
            key = self._range_key(segment_id, start_time, end_time)
            if key in history:
                continue
            ranges.append([start_time, end_time])
            keys.append(key)
            total += end_time - start_time
            if len(ranges) >= self.max_ranges:
                break
        return ranges, keys

    def __call__(self, state: VideoMMEState) -> Dict[str, Any]:
        """Observe multiple exact ranges once and pass merged visual and per-range ASR evidence to Reason."""
        request = dict(state.get("tool_request", {}))
        episodes = _episode_map(state.get("episodic_memory", []))
        history = set(str(value) for value in state.get("attention_target_history", []))
        errors = list(state.get("errors", []))
        try:
            raw_segment_id = request.get("segment_id")
            if raw_segment_id is None:
                raise ValueError("missing segment_id")
            segment_id = int(raw_segment_id)
        except (TypeError, ValueError):
            segment_id = -1
        episode = episodes.get(segment_id)
        ranges, new_keys = self._normalize_ranges(request, episode, history) if episode else ([], [])
        new_evidence = []  # type: List[Dict[str, Any]]

        if episode is not None and ranges:
            attention_input = {
                "segment_id": segment_id,
                "time_ranges": ranges,
                "spatial_target": request.get("spatial_target", ""),
                "detail_to_verify": request.get("objective", ""),
            }
            prompt = render_prompt(self.prompt_template, {
                "QUESTION": state.get("question", ""),
                "OPTIONS": format_options(state.get("options", [])),
                "MEMORY_CONTEXT": state.get("memory_context", ""),
                "TOOL_REQUEST": json.dumps(attention_input, ensure_ascii=False),
            })
            cached_transcripts = []  # type: List[Optional[Dict[str, Any]]]
            for index, (start_time, end_time) in enumerate(ranges):
                cached_transcripts.append(_slice_cached_asr(
                    episode.get("asr"),
                    start_time,
                    end_time,
                    "{}:attention:{}".format(segment_id, index),
                ))
            missing_count = sum(item is None for item in cached_transcripts)
            workers = min(self.max_concurrency, 1 + missing_count)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                visual_future = executor.submit(
                    self.vlm, state.get("video_path", ""), attention_input, prompt
                )
                asr_futures = {}  # type: Dict[int, Any]
                for index, (start_time, end_time) in enumerate(ranges):
                    if cached_transcripts[index] is not None:
                        continue
                    candidate = {
                        "segment_id": "{}:attention:{}".format(segment_id, index),
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                    asr_futures[index] = executor.submit(
                        VideoMMEPerceptionTool._safe_asr,
                        self.speaker_asr,
                        state.get("video_path", ""),
                        candidate,
                    )
                try:
                    visual = visual_future.result()
                except Exception as error:
                    visual = {"status": "failed", "observation": "", "error": str(error)}
                    errors.append("Attention VLM failed on segment {}: {}".format(segment_id, error))
                transcripts = []  # type: List[Dict[str, Any]]
                for index in range(len(ranges)):
                    cached = cached_transcripts[index]
                    transcripts.append(
                        cached if cached is not None else asr_futures[index].result()
                    )
            for transcript in transcripts:
                if transcript.get("status") == "failed":
                    errors.append(
                        "Speaker ASR failed on Attention segment {}: {}".format(
                            segment_id, transcript.get("error", "unknown error")
                        )
                    )
            new_evidence.append({
                "source": "attention_asr",
                "segment_id": segment_id,
                "time_ranges": ranges,
                "objective": request.get("objective", ""),
                "spatial_target": request.get("spatial_target", ""),
                "visual": visual,
                "asr": transcripts,
            })
            episodes[segment_id]["evidence_status"] = "attention_observed"
            history.update(new_keys)
        else:
            errors.append("Attention received no new valid target range.")

        all_evidence = [
            dict(item) for item in state.get("tool_evidence", []) if isinstance(item, dict)
        ] + new_evidence
        return {
            # Permanently register Attention range keys to prevent identical observation requests from Reflect.
            "tool_evidence": all_evidence,
            "latest_tool_evidence": new_evidence,
            "episodic_memory": list(episodes.values()),
            "attention_target_history": sorted(history),
            "pending_evidence_source": "attention_asr",
            "errors": errors,
            "working_memory": _append_memory(
                state,
                "attention",
                "Attention inspected {} new target group with parallel ASR.".format(
                    len(new_evidence)
                ),
            ),
        }
