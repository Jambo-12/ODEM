"""Transcribe the current candidate at Perception start and cache speaker logs in this process."""

import base64
import json
import os
import re
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from performance_trace import record_unknown_token_usage, record_usage_object


DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_REQUEST_TIMEOUT = 180.0
DEFAULT_API_RETRIES = 5
DEFAULT_VALIDATION_RETRIES = 2


class SpeakerASRValidationError(ValueError):
    """Indicate that the API returned text but the transcription JSON violates the shared-state protocol."""


class QwenSpeakerASR:
    """
    Use the Qwen Omni API for speaker diarization and transcription of a Localization candidate.

    This object is called only by Perception. Successful results are cached in the current
    Python process for reuse when the same segment is observed again or enters Attention,
    without creating new offline preprocessing files.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        prompt_template: str,
        audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        api_retries: int = DEFAULT_API_RETRIES,
        validation_retries: int = DEFAULT_VALIDATION_RETRIES,
    ) -> None:
        self.model = str(model).strip()
        self.prompt_template = prompt_template
        if not self.model:
            raise ValueError("ASR model name must not be empty")
        if "{{CLIP_DURATION}}" not in self.prompt_template:
            raise ValueError("ASR prompt is missing the {{CLIP_DURATION}} placeholder")
        self.audio_sample_rate = max(8000, int(audio_sample_rate))
        self.validation_retries = max(0, int(validation_retries))
        self.client = OpenAI(
            api_key=str(api_key).strip(),
            base_url=str(api_base).strip(),
            timeout=max(1.0, float(request_timeout)),
            max_retries=max(0, int(api_retries)),
        )
        self._cache = {}  # type: Dict[Tuple[str, float, float], Dict[str, Any]]
        self._audio_stream_cache = {}  # type: Dict[str, bool]
        self._lock = threading.RLock()

    @staticmethod
    def _candidate_bounds(candidate: Dict[str, Any]) -> Tuple[float, float]:
        """Read absolute candidate boundaries and reject empty segments."""
        raw_start = candidate.get("start_time")
        raw_end = candidate.get("end_time")
        if raw_start is None or raw_end is None:
            raise ValueError("ASR candidate is missing valid time boundaries")
        try:
            start_time = float(raw_start)
            end_time = float(raw_end)
        except (TypeError, ValueError) as error:
            raise ValueError("ASR candidate is missing valid time boundaries") from error
        if end_time <= start_time:
            raise ValueError("ASR candidate time range is empty")
        return start_time, end_time

    @staticmethod
    def _encode_audio(audio_path: str) -> str:
        """Encode a temporary WAV file as Base64 data accepted by the Qwen Omni API."""
        with open(audio_path, "rb") as audio_file:
            encoded = base64.b64encode(audio_file.read()).decode("utf-8")
        return "data:;base64,{}".format(encoded)

    @staticmethod
    def _content_text(content: Any) -> str:
        """Support streaming text deltas represented as strings or lists."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []  # type: List[str]
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                elif hasattr(item, "text"):
                    parts.append(str(item.text or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        """Extract the first valid JSON object from a model reply that may contain code fences."""
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                return value
        return {}

    @staticmethod
    def _parse_relative_time(value: Any) -> float:
        """Convert a number or MM:SS/HH:MM:SS text to seconds relative to the segment."""
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            raise ValueError("ASR timestamp is empty")
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            return float(text)
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("Unable to parse ASR timestamp: {}".format(text))
        values = [float(part) for part in parts]
        if len(values) == 2:
            return values[0] * 60.0 + values[1]
        return values[0] * 3600.0 + values[1] * 60.0 + values[2]

    @classmethod
    def _normalize_result(
        cls,
        raw_output: str,
        segment_start: float,
        segment_end: float,
    ) -> List[Dict[str, Any]]:
        """
        Validate the response and convert model-produced segment-relative times to
        absolute times in the source video.

        An empty list is allowed when the model explicitly reports no speech. If it
        reports speech but supplies no valid transcript, retry to avoid treating a
        formatting error as silence.
        """
        parsed = cls._extract_json_object(raw_output)
        if not parsed:
            raise SpeakerASRValidationError("ASR response does not contain a valid JSON object")
        raw_utterances = parsed.get("utterances")
        if not isinstance(raw_utterances, list):
            raise SpeakerASRValidationError("ASR response is missing an utterances list")

        clip_duration = segment_end - segment_start
        utterances = []  # type: List[Dict[str, Any]]
        for item in raw_utterances:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            try:
                relative_start = cls._parse_relative_time(item.get("start_time"))
                relative_end = cls._parse_relative_time(item.get("end_time"))
            except (TypeError, ValueError):
                continue
            relative_start = min(clip_duration, max(0.0, relative_start))
            relative_end = min(clip_duration, max(0.0, relative_end))
            if relative_end <= relative_start:
                continue
            utterances.append({
                "speaker": str(
                    item.get("speaker") or "SPEAKER_UNKNOWN"
                ).strip(),
                "start_time": round(segment_start + relative_start, 3),
                "end_time": round(segment_start + relative_end, 3),
                "text": text,
            })

        if parsed.get("has_speech") is True and not utterances:
            raise SpeakerASRValidationError(
                "ASR reports speech but contains no valid transcription entries"
            )
        utterances.sort(
            key=lambda item: (item["start_time"], item["end_time"])
        )
        return utterances

    def _has_audio_stream(self, video_path: str) -> bool:
        """Cache audio-track detection to avoid repeated ffprobe calls for questions on the same video."""
        resolved_path = os.path.realpath(video_path)
        with self._lock:
            cached = self._audio_stream_cache.get(resolved_path)
        if cached is not None:
            return cached
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            resolved_path,
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if result.returncode != 0:
            raise RuntimeError("ffprobe failed to read the video: {}".format(result.stderr.strip()))
        has_audio = bool(result.stdout.strip())
        with self._lock:
            self._audio_stream_cache[resolved_path] = has_audio
        return has_audio

    def _extract_audio(
        self,
        video_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
    ) -> None:
        """Extract only the current candidate as mono WAV without reading or transcribing outside audio."""
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "{:.3f}".format(start_time),
            "-i",
            video_path,
            "-t",
            "{:.3f}".format(end_time - start_time),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.audio_sample_rate),
            "-c:a",
            "pcm_s16le",
            "-y",
            output_path,
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise RuntimeError("ffmpeg audio extraction failed: {}".format(result.stderr.strip()))

    def _invoke(self, audio_path: str, clip_duration: float) -> str:
        """Send one WAV in text-only output mode and combine Qwen's streaming response."""
        prompt = self.prompt_template.replace(
            "{{CLIP_DURATION}}",
            "{:.3f}".format(clip_duration),
        )
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": self._encode_audio(audio_path),
                                "format": "wav",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                modalities=["text"],
                stream=True,
                stream_options={"include_usage": True},
            )
            parts = []  # type: List[str]
            usage = None
            for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                parts.append(
                    self._content_text(getattr(delta, "content", None))
                )
        except Exception:
            record_unknown_token_usage("speaker_asr", "asr")
            raise

        if usage is None:
            record_unknown_token_usage("speaker_asr", "asr")
        else:
            record_usage_object("speaker_asr", "asr", usage)
        return "".join(parts).strip()

    def _transcribe_audio(
        self,
        audio_path: str,
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """Call and validate one audio segment, retrying the model only for malformed responses."""
        last_error = None  # type: Optional[SpeakerASRValidationError]
        for _ in range(self.validation_retries + 1):
            raw_output = self._invoke(audio_path, end_time - start_time)
            try:
                return self._normalize_result(raw_output, start_time, end_time)
            except SpeakerASRValidationError as error:
                last_error = error
        if last_error is None:
            raise RuntimeError("ASR produced no result")
        raise last_error

    def transcribe(
        self,
        video_path: str,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transcribe the current candidate and return a stable shared-state structure.

        All returned times are absolute seconds in the source video. Missing audio and
        silence are normal states; network, parsing, and ffmpeg errors are left for
        Perception to catch so the existing visual-only path can continue.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError("ASR source video does not exist: {}".format(video_path))
        start_time, end_time = self._candidate_bounds(candidate)
        cache_key = (
            os.path.realpath(video_path),
            round(start_time, 3),
            round(end_time, 3),
        )
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            result = dict(cached)
            result["segment_id"] = candidate.get("segment_id")
            result["utterances"] = [
                dict(item) for item in cached.get("utterances", [])
            ]
            result["cached"] = True
            return result

        if not self._has_audio_stream(video_path):
            result = {
                "segment_id": candidate.get("segment_id"),
                "start_time": start_time,
                "end_time": end_time,
                "status": "no_audio",
                "utterances": [],
                "cached": False,
            }
        else:
            with tempfile.TemporaryDirectory(prefix="odem_speaker_asr_") as temp_dir:
                audio_path = os.path.join(temp_dir, "candidate.wav")
                self._extract_audio(
                    video_path,
                    audio_path,
                    start_time,
                    end_time,
                )
                utterances = self._transcribe_audio(
                    audio_path,
                    start_time,
                    end_time,
                )
            result = {
                "segment_id": candidate.get("segment_id"),
                "start_time": start_time,
                "end_time": end_time,
                "status": "success" if utterances else "no_speech",
                "utterances": utterances,
                "cached": False,
            }

        with self._lock:
            cached_result = dict(result)
            cached_result["utterances"] = [
                dict(item) for item in result.get("utterances", [])
            ]
            self._cache[cache_key] = cached_result
        return result
