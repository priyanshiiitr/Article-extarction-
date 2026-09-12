"""Context embeddings for entity resolution.

WHY EMBEDDINGS ARE NEEDED HERE
------------------------------
Every other feature we compute is symbolic: does the name match, is the role the
same, is the country the same. Those fail when a document simply does not state
the attribute -- which is most of the time. News rarely repeats "Narendra Modi,
Prime Minister of India" in every article.

What IS always present is the surrounding prose, and prose about a diplomat
reads nothing like prose about a cricket administrator:

    "arrived in Kazan for the BRICS Summit, held bilateral talks..."
    "lost an appeal in a London court over a dispute with the BCCI..."

An embedding turns each of those into a vector positioned by MEANING, so we can
measure that difference numerically. This is the feature that catches
Narendra/Lalit Modi even when roles and countries are missing.

WHY THIS PARTICULAR MODEL
-------------------------
``intfloat/multilingual-e5-small``: 118M parameters, already cached locally, and
multilingual -- which matters for a corpus of wire copy that carries
transliterated names and occasional non-English quotes. We call it through
plain ``transformers`` rather than adding ``sentence-transformers`` as a
dependency, because the pooling step is six lines and a new dependency in an
environment with a pinned torch is a real risk.

TWO DETAILS THAT ARE EASY TO GET WRONG
--------------------------------------
1. E5 models REQUIRE a prefix -- "query: " or "passage: ". They were trained
   with it, and omitting it measurably degrades the embeddings. We use
   "query: " for both sides since our comparison is symmetric.
2. MEAN POOLING MUST BE MASKED. A naive mean over the sequence axis averages in
   the PAD positions, so a short text gets its meaning diluted by padding. The
   mask-weighted mean below is the correct form.

WHAT EMBEDDINGS DO NOT SOLVE
----------------------------
They measure TOPICAL similarity, not identity. Two different Indian ministers
attending the same summit produce nearly identical context vectors. So a high
embedding score is weak evidence FOR a match, while a low score is strong
evidence AGAINST one. That asymmetry is why embeddings are one weighted feature
among several rather than the whole scorer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from src.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class EmbeddingUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _load(model_name: str):
    """Load and cache the tokenizer and model (once per process)."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingUnavailableError("transformers/torch not installed") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()  # disable dropout; we are doing inference only
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingUnavailableError(f"could not load {model_name}: {exc}") from exc

    logger.info("Loaded embedding model '%s'", model_name)
    return tokenizer, model, torch


class ContextEmbedder:
    """Encode entity context strings into normalised vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._tokenizer, self._model, self._torch = _load(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalised embeddings.

        Normalising to unit length means the dot product IS the cosine
        similarity, so downstream comparison is a single matrix multiply
        instead of a division per pair.
        """
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        torch = self._torch
        vectors: list[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = [f"query: {t.strip() or 'unknown'}" for t in texts[start : start + self.batch_size]]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            # no_grad: we never backpropagate here, and keeping the autograd
            # graph would waste memory proportional to batch size.
            with torch.no_grad():
                output = self._model(**encoded)

            hidden = output.last_hidden_state                 # (b, seq, dim)
            mask = encoded["attention_mask"].unsqueeze(-1)    # (b, seq, 1)
            # MASKED mean: sum only real tokens, divide by their count. A plain
            # .mean(1) would average PAD positions into the result.
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(vectors)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, rescaled from [-1, 1] into [0, 1].

    The rescale matters because these scores are combined in a WEIGHTED SUM
    with other features that all live in [0, 1]. A raw cosine of -0.4 would
    subtract from the total and make the weights uninterpretable.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    raw = float(np.dot(a, b))
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def embeddings_available(model_name: str = DEFAULT_MODEL) -> bool:
    try:
        _load(model_name)
        return True
    except EmbeddingUnavailableError:
        return False
