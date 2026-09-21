"""Shared configuration loading for benchmark entrypoints."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str) -> DictConfig:
    """Load a benchmark config and merge its optional base config."""
    config_path = Path(path).resolve()
    overlay = OmegaConf.load(config_path)
    base_reference = str(overlay.get("base_config") or "").strip()
    if not base_reference:
        return overlay

    del overlay["base_config"]
    base_path = (config_path.parent / base_reference).resolve()
    return OmegaConf.merge(OmegaConf.load(base_path), overlay)
