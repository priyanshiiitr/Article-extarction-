"""Phase 5 orchestration: run the extractors, deduplicate, validate.

DEDUPLICATION IS NOT HOUSEKEEPING
---------------------------------
The same fact is genuinely stated more than once. Our documents prepend the
title, so "Prime Minister Modi reviews the AI mission" produces
(Modi, holds_position, Prime Minister) from BOTH the headline and the body.

Two different things could be meant by "duplicate":

  * SAME MENTIONS  -- identical subject span, predicate and object span. That
    is one fact extracted twice by two extractors; collapse it.
  * SAME MEANING   -- different spans, same (subject text, predicate, object
    text). That is the SAME fact stated twice in the article, which is weak
    evidence that it is true.

We collapse both, but we COUNT the second kind into ``evidence_count`` rather
than discarding it, because repetition is corroboration. A fact asserted in
three sentences deserves more confidence than one asserted in passing -- and
Phase 6 will use exactly this when weighing conflicting role claims.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from src.config import Config, load_config
from src.logging_utils import get_logger
from src.relation_extraction.base import RelationExtractor, verify_relations
from src.relation_extraction.dependency import DependencyRelationExtractor
from src.relation_extraction.patterns import PatternRelationExtractor
from src.schemas import Document, Relation

logger = get_logger(__name__)


@dataclass
class RelationStats:
    documents: int = 0
    raw: int = 0
    after_dedup: int = 0
    below_threshold: int = 0
    by_predicate: Counter = field(default_factory=Counter)
    by_extractor: Counter = field(default_factory=Counter)
    via_coref: int = 0

    def summary(self) -> str:
        return (
            f"docs={self.documents} raw={self.raw} kept={self.after_dedup} "
            f"dropped_low_conf={self.below_threshold} via_coref={self.via_coref}"
        )


def build_extractors(cfg: Config | None = None) -> list[RelationExtractor]:
    cfg = cfg or load_config()
    extractors: list[RelationExtractor] = []
    for name in cfg.relations.extractors:
        if name == "pattern":
            extractors.append(PatternRelationExtractor())
        elif name == "dependency":
            extractors.append(DependencyRelationExtractor(cfg.preprocessing.spacy_model))
        else:
            raise ValueError(f"Unknown relation extractor {name!r}")
    if not extractors:
        raise ValueError("No relation extractors configured.")
    logger.info("Relation extractors active: %s", [e.name for e in extractors])
    return extractors


def _semantic_key(relation: Relation) -> tuple[str, str, str]:
    """Identity of a FACT, independent of which words expressed it."""
    return (
        relation.subject_text.strip().lower(),
        relation.predicate,
        relation.object_text.strip().lower(),
    )


def deduplicate(relations: Sequence[Relation]) -> list[Relation]:
    """Collapse repeated facts, keeping the best-supported instance.

    "Best" means highest confidence; ties break toward the instance whose
    subject was NOT coreference-resolved, because a literal name is stronger
    evidence than a resolved pronoun.
    """
    best: dict[tuple[str, str, str], Relation] = {}
    counts: Counter = Counter()

    for relation in relations:
        key = _semantic_key(relation)
        counts[key] += 1
        current = best.get(key)
        if current is None:
            best[key] = relation
            continue
        candidate_rank = (relation.confidence, not relation.subject_via_coref)
        current_rank = (current.confidence, not current.subject_via_coref)
        if candidate_rank > current_rank:
            best[key] = relation

    merged: list[Relation] = []
    for key, relation in best.items():
        merged.append(relation.model_copy(update={"evidence_count": counts[key]}))
    return sorted(merged, key=lambda r: (r.sentence_index, -r.confidence))


def run_relation_extraction(
    documents: Sequence[Document],
    cfg: Config | None = None,
    extractors: Sequence[RelationExtractor] | None = None,
) -> RelationStats:
    """Annotate documents in place with extracted relations."""
    cfg = cfg or load_config()
    extractors = extractors or build_extractors(cfg)

    per_extractor = [extractor.extract_batch(documents) for extractor in extractors]
    stats = RelationStats(documents=len(documents))

    for index, document in enumerate(documents):
        candidates: list[Relation] = []
        for results in per_extractor:
            candidates.extend(results[index])
        stats.raw += len(candidates)

        merged = deduplicate(candidates)

        kept: list[Relation] = []
        for relation in merged:
            if relation.confidence < cfg.relations.min_confidence:
                stats.below_threshold += 1
                continue
            kept.append(relation)
            stats.by_predicate[relation.predicate] += 1
            stats.by_extractor[relation.extractor] += 1
            if relation.subject_via_coref or relation.object_via_coref:
                stats.via_coref += 1

        document.relations = kept
        stats.after_dedup += len(kept)

    problems = 0
    for document in documents:
        for problem in verify_relations(document):
            logger.error("%s: %s", document.article_id, problem)
            problems += 1
    if problems:
        raise RuntimeError(f"{problems} relation integrity problems -- aborting")

    logger.info("Relation extraction complete: %s", stats.summary())
    return stats
