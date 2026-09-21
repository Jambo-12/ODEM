"""
Use a local Qwen2.5-VL model to provide visual reasoning for Perception and Attention.

The model is preloaded when the main program starts and remains in GPU memory throughout
batch inference. Both visual nodes share the same instance. Video windows strictly use
the real second-level time range from the current benchmark state, and source-video
absolute timestamps are burned into sampled frames so the model can reliably align
observations with the video timeline.

Single-GPU usage example:
    VIDEOMME_GPU_ID=0 bash scripts/run_videomme.sh \
        --max_videos 10 \
        --rerun
"""

import json
import math
import os
import re
import tempfile
import threading
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from performance_trace import record_token_usage, record_unknown_token_usage


DEFAULT_MAX_PIXELS = 360 * 420
MIN_CLIP_DURATION_SEC = 0.1
SPARSE_VIDEO_FPS = 1.0


class QwenVLM:
    """Wrap Qwen2.5-VL model loading, timestamped video cropping, and visual inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        perception_num_frames: int = 32,
        perception_fps: Optional[float] = None,
        attention_fps: float = 2.0,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        attention_max_pixels: Optional[int] = None,
        perception_max_new_tokens: int = 768,
        attention_max_new_tokens: int = 768,
        annotate_timestamps: bool = True,
        use_flash_attention: bool = True,
        serialize_inference: bool = True,
    ) -> None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError("Qwen2.5-VL model directory does not exist: {}".format(model_path))
        if perception_num_frames < 2 or perception_num_frames % 2 != 0:
            raise ValueError("Perception uniform sample count must be an even number greater than or equal to 2")
        if perception_fps is not None and perception_fps <= 0:
            raise ValueError("Perception FPS must be greater than 0")
        if attention_fps <= 0:
            raise ValueError("Attention FPS must be greater than 0")
        if max_pixels <= 0:
            raise ValueError("max_pixels must be greater than 0")
        if attention_max_pixels is not None and attention_max_pixels <= 0:
            raise ValueError("attention_max_pixels must be greater than 0")

        self.model_path = model_path
        self.device = device
        self.perception_num_frames = int(perception_num_frames)
        # None retains fixed-frame compatibility mode; Video-MME uses duration-based FPS mode.
        self.perception_fps = (
            float(perception_fps) if perception_fps is not None else None
        )
        self.attention_fps = float(attention_fps)
        # Perception uses a lower resolution to control GPU-memory and token costs for longer candidates.
        self.max_pixels = int(max_pixels)
        # Attention observes only short focused ranges and can use a separate higher per-frame resolution.
        self.attention_max_pixels = int(
            attention_max_pixels
            if attention_max_pixels is not None
            else max_pixels
        )
        self.perception_max_new_tokens = max(1, int(perception_max_new_tokens))
        self.attention_max_new_tokens = max(1, int(attention_max_new_tokens))
        self.annotate_timestamps = bool(annotate_timestamps)
        self.use_flash_attention = bool(use_flash_attention)
        self.serialize_inference = bool(serialize_inference)

        self._model = None  # type: Optional[Any]
        self._processor = None  # type: Optional[Any]
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load_model(self) -> Tuple[Any, Any]:
        """Load the model and processor, or return the shared GPU-resident instances if already loaded."""
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor

            print("Preloading the Qwen2.5-VL perception model")
            model_kwargs = {
                "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            }  # type: Dict[str, Any]
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
                if self.use_flash_attention:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                model_kwargs["device_map"] = self.device

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            model.eval()
            processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = model
            self._processor = processor
            print("Qwen2.5-VL perception model loaded")
        return self._model, self._processor

    def preload(self) -> None:
        """Load the model before graph execution so later visual calls reuse the resident instance."""
        self._load_model()

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as an absolute timestamp suitable for burning into video frames."""
        milliseconds = max(0, int(round(float(seconds) * 1000)))
        hours, remainder = divmod(milliseconds, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        whole_seconds, milliseconds = divmod(remainder, 1000)
        return "{:02d}:{:02d}:{:02d}.{:03d}".format(
            hours,
            minutes,
            whole_seconds,
            milliseconds,
        )

    @classmethod
    def _annotate_frame_timestamp(
        cls,
        frame: Any,
        absolute_seconds: float,
    ) -> Any:
        """Draw a high-contrast absolute timestamp at the top left that remains legible after scaling."""
        height, width = frame.shape[:2]
        label = "[ABS {}]".format(cls._format_timestamp(absolute_seconds))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.55, min(1.4, width / 1200.0))
        thickness = max(1, int(round(font_scale * 2)))
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )
        padding = max(5, int(round(font_scale * 8)))
        left = padding
        top = padding
        right = min(width - 1, left + text_width + padding * 2)
        bottom = min(height - 1, top + text_height + baseline + padding * 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), -1)
        text_origin = (left + padding, top + padding + text_height)
        cv2.putText(
            frame,
            label,
            text_origin,
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def _validate_time_range(
        start_time: Any,
        end_time: Any,
    ) -> Tuple[float, float]:
        """Validate a second-level time range to prevent empty clips or ranges before the video start."""
        try:
            start_second = max(0.0, float(start_time))
            end_second = float(end_time)
        except (TypeError, ValueError) as error:
            raise ValueError("Perception target is missing valid start_time/end_time values") from error
        if end_second - start_second < MIN_CLIP_DURATION_SEC:
            raise ValueError(
                "Perception time range is too short or invalid: {:.3f}-{:.3f}".format(
                    start_second,
                    end_second,
                )
            )
        return start_second, end_second

    @classmethod
    def _create_video_clip(
        cls,
        video_path: str,
        start_time: float,
        end_time: float,
        annotate_timestamps: bool,
    ) -> str:
        """Crop one real time range using the multi-range video concatenation implementation."""
        return cls._create_video_clip_from_ranges(
            video_path,
            [[start_time, end_time]],
            annotate_timestamps,
        )

    @classmethod
    def _create_video_clip_from_ranges(
        cls,
        video_path: str,
        time_ranges: List[List[float]],
        annotate_timestamps: bool,
    ) -> str:
        """
        Concatenate multiple absolute source-video time ranges into one temporary video.

        Each frame retains its absolute source-video timestamp, allowing the model to
        recognize time jumps even between noncontiguous ranges. All ranges produce only
        one temporary video so Attention makes exactly one VLM call.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError("Source video does not exist: {}".format(video_path))

        validated_ranges = []  # type: List[Tuple[float, float]]
        for time_range in time_ranges:
            if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
                continue
            validated_ranges.append(cls._validate_time_range(
                time_range[0],
                time_range[1],
            ))
        if not validated_ranges:
            raise ValueError("No valid time range is available for Attention")

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Unable to open source video: {}".format(video_path))

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Invalid video FPS, frame count, or resolution: {}".format(video_path))

        file_descriptor, clip_path = tempfile.mkstemp(
            prefix="odem_perception_",
            suffix=".mp4",
        )
        os.close(file_descriptor)
        writer = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            if os.path.exists(clip_path):
                os.remove(clip_path)
            raise RuntimeError("Unable to create a temporary video clip")

        written_frames = 0
        try:
            for start_time, end_time in validated_ranges:
                start_frame = max(0, int(start_time * fps))
                end_frame = min(
                    frame_count,
                    max(start_frame + 1, int(end_time * fps + 0.999)),
                )
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for frame_index in range(start_frame, end_frame):
                    success, frame = capture.read()
                    if not success:
                        break
                    if annotate_timestamps:
                        frame = cls._annotate_frame_timestamp(
                            frame,
                            frame_index / fps,
                        )
                    writer.write(frame)
                    written_frames += 1
        finally:
            writer.release()
            capture.release()

        if written_frames == 0:
            if os.path.exists(clip_path):
                os.remove(clip_path)
            raise RuntimeError("No video frames were read from the specified time range")
        return clip_path

    @classmethod
    def _create_sparse_video(
        cls,
        video_path: str,
        frame_timestamps: List[Dict[str, Any]],
        annotate_timestamps: bool,
    ) -> str:
        """
        Read discrete frames at absolute times and write a low-frame-rate video for
        Perception inference.

        Each frame in the temporary video corresponds to exactly one selected moment;
        adjacent irrelevant video content is neither decoded nor passed to the model.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError("Source video does not exist: {}".format(video_path))

        timestamps = []  # type: List[float]
        for item in frame_timestamps:
            if not isinstance(item, dict):
                continue
            raw_timestamp = item.get("timestamp")
            if raw_timestamp is None:
                continue
            try:
                timestamps.append(max(0.0, float(raw_timestamp)))
            except (TypeError, ValueError):
                continue
        if not timestamps:
            raise ValueError("Perception did not receive valid sampled-frame timestamps")

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Unable to open source video: {}".format(video_path))

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if source_fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Invalid video FPS, frame count, or resolution: {}".format(video_path))

        file_descriptor, sparse_path = tempfile.mkstemp(
            prefix="odem_perception_sparse_",
            suffix=".mp4",
        )
        os.close(file_descriptor)
        writer = cv2.VideoWriter(
            sparse_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            SPARSE_VIDEO_FPS,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            if os.path.exists(sparse_path):
                os.remove(sparse_path)
            raise RuntimeError("Unable to create the sparse Perception video")

        written_frames = 0
        try:
            for timestamp in timestamps:
                frame_index = min(
                    frame_count - 1,
                    max(0, int(round(timestamp * source_fps))),
                )
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    continue
                if annotate_timestamps:
                    frame = cls._annotate_frame_timestamp(
                        frame,
                        frame_index / source_fps,
                    )
                writer.write(frame)
                written_frames += 1
        finally:
            writer.release()
            capture.release()

        if written_frames != len(timestamps):
            if os.path.exists(sparse_path):
                os.remove(sparse_path)
            raise RuntimeError(
                "Perception planned to sample {} frames but read {} frames".format(
                    len(timestamps),
                    written_frames,
                )
            )
        return sparse_path

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        """Extract a JSON object from model output, retaining raw observation text on failure."""
        cleaned = str(text).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        object_start = cleaned.find("{")
        if object_start >= 0:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(cleaned[object_start:])
                if isinstance(parsed, dict):
                    return parsed
            except (TypeError, ValueError):
                pass
        return {
            "observation": cleaned,
            "parse_status": "unstructured",
        }

    def _generate(
        self,
        clip_path: str,
        prompt: str,
        fps: Optional[float],
        max_new_tokens: int,
        component: str,
        nframes: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a video with the specified sampling method and per-frame pixel limit while tracking tokens."""
        if (fps is None) == (nframes is None):
            raise ValueError("Video sampling must specify exactly one of fps or nframes")
        try:
            model, processor = self._load_model()
            video_content = {
                "type": "video",
                "video": clip_path,
                "max_pixels": int(
                    self.max_pixels if max_pixels is None else max_pixels
                ),
            }  # type: Dict[str, Any]
            if nframes is not None:
                # qwen-vl-utils uniformly selects the specified number of frames across the full video timeline.
                video_content["nframes"] = int(nframes)
            else:
                if fps is None:
                    raise ValueError("fps must not be empty when sampling by frame rate")
                video_content["fps"] = float(fps)
            content = [video_content]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            # Serialize preprocessing, tensor transfer, and generation to prevent concurrent targets using GPU memory.
            lock_context = (
                self._inference_lock
                if self.serialize_inference
                else nullcontext()
            )
            with lock_context:
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs, raw_video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                )
                video_kwargs = raw_video_kwargs or {}  # type: Dict[str, Any]
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs
                )
                model_device = next(model.parameters()).device
                inputs = inputs.to(model_device)
                attention_mask = getattr(inputs, "attention_mask", None)
                input_tokens = int(
                    attention_mask.sum().item()
                    if attention_mask is not None
                    else inputs.input_ids.numel()
                )

                if model_device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(model_device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=False
                    )
                if model_device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(model_device)
                generated_ids_trimmed = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(
                        inputs.input_ids,
                        generated_ids,
                    )
                ]
                output_tokens = sum(
                    int(output_ids.numel())
                    for output_ids in generated_ids_trimmed
                )
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
        except Exception:
            record_unknown_token_usage(component, "vqa")
            raise

        record_token_usage(
            component,
            "vqa",
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
        )
        parsed_output = self._extract_json_object(output_text)
        # Preserve the visual model's unnormalized raw reply for diagnosing inference errors in readable graph logs.
        parsed_output["raw_model_output"] = output_text
        return parsed_output

    def _observe_window(
        self,
        video_path: str,
        start_time: Any,
        end_time: Any,
        prompt: str,
        num_frames: int,
        max_new_tokens: int,
        component: str,
    ) -> Dict[str, Any]:
        """Crop a video window, uniformly sample a fixed number of frames, and delete the temporary clip after inference."""
        start_second, end_second = self._validate_time_range(start_time, end_time)
        duration = end_second - start_second
        # Use the center of each equal-width interval to avoid endpoint bias while covering the full candidate.
        frame_timestamps = [
            {
                "timestamp": start_second
                + (frame_index + 0.5) * duration / num_frames,
            }
            for frame_index in range(num_frames)
        ]
        clip_path = self._create_sparse_video(
            video_path,
            frame_timestamps,
            self.annotate_timestamps,
        )
        try:
            result = self._generate(
                clip_path=clip_path,
                prompt=prompt,
                fps=None,
                max_new_tokens=max_new_tokens,
                component=component,
                nframes=num_frames,
            )
        finally:
            if os.path.exists(clip_path):
                os.remove(clip_path)
        result.setdefault("observed_time_range", [start_second, end_second])
        return result

    def _observe_ranges(
        self,
        video_path: str,
        time_ranges: Any,
        prompt: str,
        fps: float,
        max_new_tokens: int,
        component: str,
    ) -> Dict[str, Any]:
        """Concatenate multiple absolute time ranges and complete all detailed observations in one generation."""
        if not isinstance(time_ranges, list):
            raise ValueError("Attention time_ranges must be a list")
        validated_ranges = []  # type: List[List[float]]
        for time_range in time_ranges:
            if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
                continue
            start_second, end_second = self._validate_time_range(
                time_range[0],
                time_range[1],
            )
            validated_ranges.append([start_second, end_second])
        if not validated_ranges:
            raise ValueError("Attention did not receive a valid observation time range")

        clip_path = self._create_video_clip_from_ranges(
            video_path,
            validated_ranges,
            self.annotate_timestamps,
        )
        try:
            result = self._generate(
                clip_path=clip_path,
                prompt=prompt,
                fps=fps,
                max_new_tokens=max_new_tokens,
                component=component,
                max_pixels=self.attention_max_pixels,
            )
        finally:
            if os.path.exists(clip_path):
                os.remove(clip_path)
        result.setdefault("observed_time_ranges", validated_ranges)
        return result

    def observe_perception(
        self,
        video_path: str,
        candidate: Dict[str, Any],
        prompt: str,
    ) -> Dict[str, Any]:
        """Cover the complete Localization candidate at the configured frame rate and return minimal evidence."""
        start_time, end_time = self._validate_time_range(
            candidate.get("start_time"),
            candidate.get("end_time"),
        )
        num_frames = self.perception_num_frames
        if self.perception_fps is not None:
            num_frames = max(
                2,
                int(math.ceil((end_time - start_time) * self.perception_fps)),
            )
            # Qwen processes video in pairs of frames, so round an odd count up by one.
            if num_frames % 2 != 0:
                num_frames += 1
        result = self._observe_window(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            prompt=prompt,
            num_frames=num_frames,
            max_new_tokens=self.perception_max_new_tokens,
            component="perception_vlm",
        )
        result.setdefault("question_type", "local")
        result.setdefault("provisional_answer", None)
        result.setdefault("key_evidence", [])
        result.setdefault("attention_metadata", None)
        result.setdefault("route", "attention")
        result.setdefault("observation", "")
        result.setdefault("sampling_frame_count", num_frames)
        if self.perception_fps is not None:
            result.setdefault("sampling_fps", self.perception_fps)
        return result

    def observe_attention(
        self,
        video_path: str,
        attention_input: Dict[str, Any],
        prompt: str,
    ) -> Dict[str, Any]:
        """Perform one denser detailed VLM observation over one or more focused ranges."""
        result = self._observe_ranges(
            video_path=video_path,
            time_ranges=attention_input.get("time_ranges", []),
            prompt=prompt,
            fps=self.attention_fps,
            max_new_tokens=self.attention_max_new_tokens,
            component="attention_vlm",
        )
        result.setdefault("provisional_answer", None)
        result.setdefault("key_evidence", [])
        result.setdefault("unresolved_details", [])
        result.setdefault("observation", "")
        return result
