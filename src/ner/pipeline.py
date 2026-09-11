"""Phase 3 orchestration: run the configured extractors and merge their output.

Which extractors run is a CONFIG decision, not a code decision. That matters
practically: the transformer is ~17x slower than spaCy, so a first pass over a
large archive might run spaCy + gazetteer only, and a second, higher-quality
pass adds the transformer. Same code, different config.
"""

from __future__ import annotations

from typing import Sequence

from src.config import Config, load_config
from src.logging_utils import get_logger
from src.ner.base import EntityExtractor, verify_mentions
from src.ner.gazetteer import GazetteerExtractor
from src.ner.merge import MergeStats, merge_documents
from src.ner.spacy_ner import SpacyEntityExtractor
from src.schemas import Document

logger = get_logger(__name__)


def build_extractors(cfg: Config | None = None) -> list[EntityExtractor]:
    """Construct the extractors named in config, degrading gracefully.

    If the transformer is requested but unavailable (not installed, or no
    network on first run), we log a WARNING and continue with the others. A
    degraded result the operator is TOLD about is better than a dead pipeline;
    silently degrading with no log line would be indefensible.
    """
    cfg = cfg or load_config()
    extractors: list[EntityExtractor] = []

    for name in cfg.ner.extractors:
        if name == "spacy":
            extractors.append(SpacyEntityExtractor(model_name=cfg.preprocessing.spacy_model))
        elif name == "gazetteer":
            extractors.append(GazetteerExtractor())
        elif name == "transformer":
            try:
                from src.ner.transformer_ner import TransformerEntityExtractor

                extractors.append(
                    TransformerEntityExtractor(
                        model_name=cfg.ner.transformer_model,
                        device=cfg.ner.device,
                        batch_size=cfg.ner.batch_size,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - any load failure degrades
                logger.warning(
                    "Transformer NER unavailable (%s). Continuing without it.", exc
                )
        else:
            raise ValueError(f"Unknown extractor {name!r} in config.ner.extractors")

    if not extractors:
        raise ValueError("No NER extractors configured; nothing would be extracted.")

    logger.info("NER extractors active: %s", [e.name for e in extractors])
    return extractors


def run_ner(
    documents: Sequence[Document],
    cfg: Config | None = None,
    extractors: Sequence[EntityExtractor] | None = None,
) -> MergeStats:
    """Annotate documents in place with merged mentions."""
    cfg = cfg or load_config()
    extractors = extractors or build_extractors(cfg)

    # Each extractor processes the WHOLE batch before the next one starts, so
    # every model is loaded once and batching is used to the full extent.
    per_extractor = [extractor.extract_batch(documents) for extractor in extractors]

    stats = merge_documents(documents, per_extractor)

    # The denormalisation tripwire: Mention.text must still equal
    # document.text[start:end] for every mention we just produced.
    problems = 0
    for document in documents:
        for problem in verify_mentions(document):
            logger.error("%s: %s", document.article_id, problem)
            problems += 1
    if problems:
        raise RuntimeError(f"{problems} mention offset mismatches after NER -- aborting")

    # Filter by the configured confidence floor AFTER merging, so a low-scoring
    # mention still gets the chance to be beaten (or confirmed) by a better one.
    if cfg.ner.min_score > 0:
        for document in documents:
            before = len(document.mentions)
            document.mentions = [
                m for m in document.mentions if m.score >= cfg.ner.min_score
            ]
            if before != len(document.mentions):
                logger.debug(
                    "%s: dropped %d mentions below min_score=%.2f",
                    document.article_id,
                    before - len(document.mentions),
                    cfg.ner.min_score,
                )

    return stats
