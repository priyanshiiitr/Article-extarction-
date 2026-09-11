"""Configuration loading.

Design rule for this project: **no module ever hardcodes a path or a model
name**. Everything comes from ``config/config.yaml``, which can be overridden
per-environment by environment variables. That is what makes the same code
runnable on a laptop, in CI, and on a server without editing source.

Precedence (lowest to highest):
    1. ``config/config.yaml``
    2. variables from a local ``.env`` file
    3. real environment variables

Override pattern: ``NEWSKG__<SECTION>__<KEY>`` -- a double underscore means
"go one level deeper". ``NEWSKG__NER__MODEL_NAME=foo`` sets ``ner.model_name``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# The project root is two levels up from this file (src/config.py -> src -> root).
# Anchoring on __file__ rather than os.getcwd() means `python scripts/foo.py`
# and `pytest` from any directory resolve the exact same paths.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

ENV_PREFIX = "NEWSKG"
ENV_NESTED_DELIMITER = "__"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    sample_dir: str = "data/sample"
    db_path: str = "data/knowledge_graph.db"


class IngestionConfig(BaseModel):
    source_file: str = "data/sample/articles.jsonl"
    min_body_chars: int = 120


class PreprocessingConfig(BaseModel):
    unicode_form: str = "NFKC"
    strip_html: bool = True
    sentence_segmenter: str = "spacy"
    spacy_model: str = "en_core_web_sm"


class NerConfig(BaseModel):
    # default_factory, not a bare list literal: a mutable default would be
    # shared across every NerConfig instance (see the notes in schemas.py).
    extractors: list[str] = Field(default_factory=lambda: ["spacy", "transformer", "gazetteer"])
    transformer_model: str = "dslim/bert-base-NER"
    device: int = -1
    batch_size: int = 8
    min_score: float = 0.0


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "rich"


class Config(BaseModel):
    """Typed view over config.yaml.

    Using pydantic rather than a raw dict buys us three things:
    type coercion (``"8"`` -> ``8`` for values coming from env vars, which are
    always strings), validation at load time instead of a crash deep in the
    pipeline, and editor autocomplete on ``cfg.preprocessing.spacy_model``.
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    ner: NerConfig = Field(default_factory=NerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def path(self, relative: str) -> Path:
        """Resolve a config path string against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


def _set_nested(target: dict[str, Any], keys: list[str], value: Any) -> None:
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def _env_overrides() -> dict[str, Any]:
    """Collect NEWSKG__SECTION__KEY environment variables into a nested dict."""
    overrides: dict[str, Any] = {}
    prefix = ENV_PREFIX + ENV_NESTED_DELIMITER
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(prefix):
            continue
        path = raw_key[len(prefix):].lower().split(ENV_NESTED_DELIMITER)
        _set_nested(overrides, path, raw_value)
    return overrides


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Config:
    """Load and cache the project configuration.

    Cached with ``lru_cache`` because config is read in many modules and is
    immutable for the lifetime of a process; re-reading the YAML on every call
    would be pure waste. Tests that need a different config call
    ``load_config.cache_clear()``.
    """
    # Load .env if present. Secrets live there and it is gitignored -- they are
    # never read from config.yaml, which IS committed.
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:  # pragma: no cover - dotenv is optional
        pass

    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    return Config(**_deep_merge(raw, _env_overrides()))
