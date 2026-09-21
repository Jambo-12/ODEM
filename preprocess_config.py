"""Load the shared ODEM preprocessing configuration.

This module only resolves configuration and repository-relative paths. It does
not alter any captioning or embedding behavior. The small CLI is used by the
existing shell entrypoints so all three datasets read the same YAML file.
"""

import argparse
import os
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "preprocess.yaml")
PATH_KEYS = {
    "models.caption.path",
    "models.visual_embedding.checkpoint",
    "video_dir",
    "output_dir",
    "caption_prompt",
}


def _to_plain_dict(value: Any, source: str) -> Dict[str, Any]:
    data = OmegaConf.to_container(value, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("Configuration node must be a mapping: {}".format(source))
    return data


def load_preprocess_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the complete preprocessing config without exposing secrets."""
    absolute_path = os.path.abspath(config_path)
    if not os.path.isfile(absolute_path):
        raise FileNotFoundError("Preprocessing config does not exist: {}".format(absolute_path))
    return _to_plain_dict(OmegaConf.load(absolute_path), absolute_path)


def resolve_project_path(value: str) -> str:
    """Resolve a configured path relative to the repository root."""
    expanded = os.path.expanduser(str(value).strip())
    if not expanded:
        raise ValueError("Configured path cannot be empty")
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(PROJECT_DIR, expanded))


def get_config_value(
    config: Dict[str, Any],
    key: str,
    dataset: Optional[str] = None,
) -> Any:
    """Read a dotted global key or a key from one dataset section."""
    if dataset:
        datasets = config.get("datasets")
        if not isinstance(datasets, dict) or dataset not in datasets:
            raise KeyError("Unknown preprocessing dataset: {}".format(dataset))
        dataset_config = datasets[dataset]
        if not isinstance(dataset_config, dict):
            raise ValueError("Dataset configuration must be a mapping: {}".format(dataset))
        if key in dataset_config:
            value = dataset_config[key]
            if key in PATH_KEYS:
                return resolve_project_path(str(value))
            return value

    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError("Missing preprocessing config field: {}".format(key))
        current = current[part]
    if key in PATH_KEYS:
        return resolve_project_path(str(current))
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description="Read the shared ODEM preprocessing config")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", choices=("videomme", "m3bench"))
    parser.add_argument("--get", required=True, dest="key")
    args = parser.parse_args()

    config = load_preprocess_config(args.config)
    value = get_config_value(config, args.key, dataset=args.dataset)
    if isinstance(value, bool):
        print("true" if value else "false")
    elif isinstance(value, (str, int, float)):
        print(value)
    else:
        raise ValueError("The CLI can output only scalar config values: {}".format(args.key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
