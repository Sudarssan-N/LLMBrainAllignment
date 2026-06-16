"""Configuration loading.

A thin layer over a YAML file that resolves the active device and exposes per-model
settings. Kept deliberately simple (dict-backed) so research code can read/override freely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    family: str = "unknown"
    type: str = "base"          # base | instruct
    layers: list[int] | None = None


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path = DEFAULT_CONFIG

    # ---- convenience accessors -------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    def resolve_device(self) -> str:
        dev = self.raw.get("device", "auto")
        if dev != "auto":
            return dev
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def derived_dir(self, *parts: str) -> Path:
        base = REPO_ROOT / self.raw["paths"]["data_derived"]
        p = base.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def raw_dir(self) -> Path:
        p = REPO_ROOT / self.raw["paths"]["data_raw"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def hf_cache(self) -> str:
        p = REPO_ROOT / self.raw["paths"]["hf_cache"]
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def model_spec(self, key: str | None = None) -> ModelSpec:
        key = key or self.raw.get("default_model")
        models = self.raw.get("models", {})
        if key not in models:
            raise KeyError(
                f"Model '{key}' not in config. Known: {sorted(models)}"
            )
        m = models[key]
        return ModelSpec(
            key=key,
            hf_id=m["hf_id"],
            family=m.get("family", "unknown"),
            type=m.get("type", "base"),
            layers=m.get("layers"),
        )


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=path)
