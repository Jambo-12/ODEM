"""Prompt and model-response helpers shared by both public benchmarks."""

import json
import os
import re
from typing import Any, Dict, List

from performance_trace import record_response_usage, record_unknown_token_usage


def load_prompt(prompt_path: str) -> str:
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError("Prompt file does not exist: {}".format(prompt_path))
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def render_prompt(template: str, values: Dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []  # type: List[str]
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def invoke_llm(llm: Any, prompt: str, component: str = "llm") -> str:
    try:
        response = llm.invoke(prompt)
    except Exception:
        record_unknown_token_usage(component, "llm")
        raise
    record_response_usage(component, "llm", response)
    content = response.content if hasattr(response, "content") else response
    return normalize_message_content(content)


def extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if match is None:
        return {}
    try:
        value = json.loads(match.group(0))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def format_options(options: List[str]) -> str:
    if not options:
        return "(No candidate answers provided.)"
    return "\n".join(str(option) for option in options)
