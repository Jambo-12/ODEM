"""Construct M3-Bench agents and tools from configuration."""

import os
from typing import Any, Tuple

from omegaconf import DictConfig

from eval import runtime
from eval.common import load_prompt
from eval.m3bench.agents import (
    M3BenchPlanAgent,
    M3BenchReasonAgent,
    M3BenchReflectAgent,
)
from eval.m3bench.nodes import (
    EpisodicMemoryNode,
    M3BenchAttentionTool,
    M3BenchLocalizationNode,
    M3BenchPerceptionTool,
    SegmentASRNode,
)
from speaker_asr import QwenSpeakerASR
from vlm import QwenVLM


PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def build_agents(
    config: DictConfig,
    preprocess_root: str,
    preload_vlm: bool = True,
) -> Tuple[Any, ...]:
    """Create the unchanged M3-Bench agents, retrieval, ASR, and visual tools."""
    required = lambda key: runtime.require_config_text(config, key)
    reasoning_llm = runtime.create_reasoning_llm(config)
    plan_agent = M3BenchPlanAgent(
        reasoning_llm,
        load_prompt(os.path.join(PROMPT_DIR, "plan.txt")),
    )
    localization_node = M3BenchLocalizationNode(
        preprocess_root=preprocess_root,
        text_api_key=required("openai_api_key"),
        text_api_base=required("openai_api_base"),
        text_model=str(config.get("localization_text_model", "text-embedding-3-large")),
        viclip_size=str(config.get("localization_viclip_size", "l")),
        viclip_pretrained=required("localization_viclip_pretrained"),
        device=str(config.get("localization_device", "cuda")),
        textual_weight=float(config.get("localization_textual_weight", 11.0)),
        visual_weight=float(config.get("localization_visual_weight", 18.0)),
        top_k=int(config.get("localization_top_k", 5)),
    )
    shared_vlm = QwenVLM(
        model_path=required("perception_model_path"),
        device=str(config.get("perception_device", "cuda")),
        perception_num_frames=int(config.get("perception_num_frames", 32)),
        perception_fps=float(config.get("perception_fps", 0.5)),
        attention_fps=float(config.get("attention_fps", 2.0)),
        max_pixels=int(config.get("perception_max_pixels", 360 * 420)),
        attention_max_pixels=int(config.get("attention_max_pixels", 420 * 560)),
        perception_max_new_tokens=int(config.get("perception_max_new_tokens", 768)),
        attention_max_new_tokens=int(config.get("attention_max_new_tokens", 768)),
        annotate_timestamps=bool(config.get("perception_annotate_timestamps", True)),
        use_flash_attention=bool(config.get("perception_use_flash_attention", True)),
        serialize_inference=bool(config.get("perception_serialize_inference", True)),
    )
    if preload_vlm:
        shared_vlm.preload()

    if not bool(config.get("perception_enable_asr", True)):
        raise ValueError("M3-Bench requires perception_enable_asr=true")
    speaker_asr = QwenSpeakerASR(
        api_key=required("speaker_asr_api_key"),
        api_base=required("speaker_asr_api_base"),
        model=required("speaker_asr_model"),
        prompt_template=load_prompt(
            runtime.resolve_project_path(
                str(config.get("speaker_asr_prompt_path", "prompts/speaker_asr.txt"))
            )
        ),
        audio_sample_rate=int(config.get("speaker_asr_audio_sample_rate", 16000)),
        request_timeout=float(config.get("speaker_asr_request_timeout", 180)),
        api_retries=int(config.get("speaker_asr_api_retries", 5)),
        validation_retries=int(config.get("speaker_asr_validation_retries", 2)),
    )
    segment_asr_node = SegmentASRNode(
        speaker_asr=speaker_asr.transcribe,
        max_concurrency=int(config.get("asr_max_concurrency", 4)),
    )
    memory_node = EpisodicMemoryNode(
        max_tokens=int(config.get("episodic_memory_max_tokens", 8000)),
    )
    perception_tool = M3BenchPerceptionTool(
        prompt_template=load_prompt(os.path.join(PROMPT_DIR, "perception.txt")),
        vlm=shared_vlm.observe_perception,
        speaker_asr=speaker_asr.transcribe,
        max_segments=int(config.get("perception_tool_max_segments", 2)),
    )
    attention_tool = M3BenchAttentionTool(
        prompt_template=load_prompt(os.path.join(PROMPT_DIR, "attention.txt")),
        vlm=shared_vlm.observe_attention,
        speaker_asr=speaker_asr.transcribe,
        max_ranges=int(config.get("attention_max_ranges", 3)),
        max_total_seconds=float(config.get("attention_max_total_seconds", 15.0)),
        max_concurrency=int(config.get("asr_max_concurrency", 4)),
    )
    reason_agent = M3BenchReasonAgent(
        reasoning_llm,
        load_prompt(os.path.join(PROMPT_DIR, "reason.txt")),
    )
    reflect_agent = M3BenchReflectAgent(
        reasoning_llm,
        load_prompt(os.path.join(PROMPT_DIR, "reflect.txt")),
        max_perception_segments=int(config.get("perception_tool_max_segments", 2)),
        max_attention_ranges=int(config.get("attention_max_ranges", 3)),
    )
    return (
        plan_agent,
        localization_node,
        segment_asr_node,
        memory_node,
        attention_tool,
        perception_tool,
        reason_agent,
        reflect_agent,
    )
