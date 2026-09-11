"""Shared, cached loaders for heavyweight NLP models.

WHY THIS MODULE EXISTS
----------------------
Loading a model is expensive and loading it repeatedly is a classic pipeline
performance bug. ``spacy.load("en_core_web_sm")`` costs roughly a second and
tens of MB of RAM. Called once per document over 10,000 documents that is
nearly three hours of pure loading, plus repeated allocation churn.

So: load once per process, cache, reuse. ``functools.lru_cache`` keyed on the
model name and the excluded components gives us a process-level singleton
without a global variable or an explicit registry.

A NOTE ON PROCESS MODELS
------------------------
This cache is per-process. When we parallelise in Phase 11 with a process pool,
each worker holds its own copy -- which is correct (models are not safely
shareable across processes) but means memory scales with worker count. That is
precisely why you size a worker pool by RAM / model-size, not by CPU count.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

from src.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from spacy.language import Language

logger = get_logger(__name__)


class SpacyUnavailableError(RuntimeError):
    """Raised when spaCy or a requested model is not installed."""


@lru_cache(maxsize=4)
def load_spacy(model_name: str = "en_core_web_sm", exclude: tuple[str, ...] = ()) -> "Language":
    """Load and cache a spaCy pipeline.

    ``exclude`` drops components at load time so they are never even
    constructed. This matters: for sentence segmentation we need the tokenizer
    and the dependency parser, but not the lemmatizer or the NER head. Dropping
    what you do not need is the cheapest speedup available in a spaCy pipeline.

    Note ``exclude`` is a tuple, not a list, because lru_cache keys must be
    hashable -- a small detail that bites people who try to cache on a list.
    """
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover
        raise SpacyUnavailableError(
            "spaCy is not installed. Install it with: pip install -e '.[nlp]'"
        ) from exc

    try:
        nlp = spacy.load(model_name, exclude=list(exclude))
    except OSError as exc:
        raise SpacyUnavailableError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc

    logger.info("Loaded spaCy model '%s' (components: %s)", model_name, nlp.pipe_names)
    return nlp


def spacy_available(model_name: str = "en_core_web_sm") -> bool:
    """Check availability without raising, for graceful degradation."""
    try:
        load_spacy(model_name)
        return True
    except SpacyUnavailableError:
        return False


def describe_pipeline(model_name: str = "en_core_web_sm", exclude: Sequence[str] = ()) -> str:
    nlp = load_spacy(model_name, tuple(exclude))
    return f"{model_name}: {', '.join(nlp.pipe_names)}"
