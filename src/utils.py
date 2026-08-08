"""
Configuration loading + path resolution for the LLM-RAG project.

The project root is derived from this file's location (two levels up),
so the package is portable across machines as long as the folder
layout is preserved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def pick_device(preferred: str | None = None) -> str:
    """Return a safe torch device string (CUDA → MPS → CPU).

    Importing torch is deferred so this module can be used without it
    (e.g. for path/config inspection or running the data loader alone).
    """
    if preferred:
        return preferred
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Project root = the directory that contains src/, config/, data/, etc.
# __file__ is .../LLM-RAG/src/utils.py  →  parent.parent is .../LLM-RAG/
PROJECT_ROOT: Path = Path(os.path.abspath(os.path.dirname(__file__))).parent

# Default config location, relative to project root.
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Keys whose values are paths and must be resolved to absolute form.
# Each entry is (section, key).
_PATH_KEYS: list[tuple[str, str]] = [
    ("data", "registry_csv"),
    ("data", "registry_parquet"),
    ("data", "chunk_metadata"),
    ("data", "faiss_index"),
    ("paths", "visualizations"),
    ("paths", "qdrant"),
]


def get_project_root() -> Path:
    """Return the absolute path to the LLM-RAG project root."""
    return PROJECT_ROOT


def _resolve(rel_or_abs: str) -> str:
    """Resolve a possibly-relative path against PROJECT_ROOT.

    Absolute paths are returned unchanged so users can override
    individual entries in config.yaml with an absolute path if needed.
    """
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml and resolve all path entries to absolute paths.

    Args:
        config_path: optional override; defaults to <root>/config/config.yaml.

    Returns:
        Nested dict with path entries rewritten to absolute paths.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for section, key in _PATH_KEYS:
        if section in cfg and key in cfg[section]:
            cfg[section][key] = _resolve(cfg[section][key])

    return cfg
