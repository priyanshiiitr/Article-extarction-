"""Run the whole pipeline in memory, with progress reporting.

WHY THIS MODULE EXISTS
----------------------
Until now every stage was driven by a script that read a JSONL file and wrote
another one. That is the right shape for a batch job -- each stage is
restartable and its output inspectable -- but it is the wrong shape for a UI,
which needs to hold everything in memory and report progress as it goes.

So this is an ORCHESTRATOR over the same stage functions, not a reimplementation
of them. Nothing here knows how NER or coreference works; it only knows the
order and how to report where it is. If the two ever diverge in behaviour, that
is a bug in this file, not a second implementation to maintain.

PROFILES: full vs lite
----------------------
The models the full pipeline uses need roughly 4.5 GB of RAM:

    LingMess coreference  590M params  ~2.4 GB
    BERT NER              110M params  ~1.5 GB
    e5-small embeddings   118M params  ~0.5 GB

Streamlit Community Cloud gives you 1 GB. So there is a second profile that
swaps every heavy component for the lightweight alternative we already built
and documented:

    full  transformer NER + neural coref + embeddings   ~4.5 GB, ~14 s/doc
    lite  spaCy+gazetteer NER + rule coref, no embeds   ~0.6 GB, <1 s/doc

This costs no new code -- every stage already takes a config-selectable backend
with a graceful fallback. That design decision, made back in Phase 3, is what
makes a constrained deployment a config change rather than a rewrite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from src.config import Config, load_config
from src.coreference.pipeline import build_resolver as build_coref_resolver, run_coreference
from src.entity_resolution.profiles import build_all_profiles
from src.entity_resolution.resolver import CanonicalEntity, resolve_entities
from src.logging_utils import get_logger
from src.ner.pipeline import build_extractors as build_ner_extractors, run_ner
from src.preprocessing.processor import preprocess_articles, verify_offsets
from src.relation_extraction.pipeline import run_relation_extraction
from src.schemas import Article, Document

logger = get_logger(__name__)

ProgressCallback = Callable[[str, float, str], None]
"""(stage_name, fraction_complete, detail) -> None"""


@dataclass
class PipelineResult:
    documents: list[Document] = field(default_factory=list)
    entities: list[CanonicalEntity] = field(default_factory=list)
    review_queue: list = field(default_factory=list)
    # Kept so a UI can show WHICH two local entities a review pair refers to
    # without rebuilding every profile on each render.
    profiles: list = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Which components ACTUALLY loaded, as opposed to which were requested.
    #
    # Every stage degrades gracefully when a model cannot be loaded -- the
    # transformer NER and the neural coreference resolver both fall back to
    # lighter components and log a WARNING. That is correct for a batch job,
    # where an operator reads the logs, but it is invisible in a web UI: the
    # run just finishes suspiciously fast and nobody knows why.
    #
    # Recording the realised backends turns "it was fast" into an answerable
    # question, and `degraded` says plainly whether you got what you asked for.
    backends: dict[str, str] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())


def apply_profile(cfg: Config, profile: str) -> Config:
    """Return a config adjusted for the requested resource profile.

    Returns a COPY rather than mutating: ``load_config`` is lru_cached, so
    mutating the shared object would silently change behaviour for every other
    caller in the process -- exactly the kind of action-at-a-distance bug that
    caching invites.
    """
    adjusted = cfg.model_copy(deep=True)
    if profile == "lite":
        adjusted.ner.extractors = ["spacy", "gazetteer"]
        adjusted.coreference.backend = "rules"
        adjusted.entity_resolution.use_embeddings = False
    return adjusted


# Rough share of total runtime per stage, used only to drive a progress bar.
# Coreference dominates in full mode (measured at ~94% of wall clock), which is
# why a naive "one fifth per stage" bar would sit at 40% for two minutes and
# then jump to done.
_STAGE_WEIGHTS_FULL = {
    "preprocess": 0.02,
    "ner": 0.06,
    "coreference": 0.84,
    "relations": 0.03,
    "entity_resolution": 0.05,
}
_STAGE_WEIGHTS_LITE = {
    "preprocess": 0.25,
    "ner": 0.35,
    "coreference": 0.10,
    "relations": 0.20,
    "entity_resolution": 0.10,
}


def run_pipeline(
    articles: Sequence[Article],
    cfg: Config | None = None,
    profile: str = "full",
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    """Run preprocessing -> NER -> coref -> relations -> entity resolution."""
    cfg = apply_profile(cfg or load_config(), profile)
    result = PipelineResult()
    weights = _STAGE_WEIGHTS_FULL if profile == "full" else _STAGE_WEIGHTS_LITE
    completed = 0.0

    def report(stage: str, detail: str) -> None:
        if progress:
            progress(stage, min(completed, 1.0), detail)

    if not articles:
        result.warnings.append("no articles to process")
        return result

    # --- Phase 2 ---------------------------------------------------------
    report("preprocess", f"cleaning and segmenting {len(articles)} articles")
    started = time.perf_counter()
    documents = preprocess_articles(list(articles), cfg=cfg)
    for document in documents:
        for problem in verify_offsets(document):
            result.warnings.append(f"{document.article_id}: {problem}")
    result.timings["preprocess"] = time.perf_counter() - started
    completed += weights["preprocess"]
    result.counts["sentences"] = sum(len(d.sentences) for d in documents)

    # --- Phase 3 ---------------------------------------------------------
    report("ner", f"extracting entities ({', '.join(cfg.ner.extractors)})")
    started = time.perf_counter()
    # Built here rather than inside run_ner so we can see which extractors
    # actually instantiated. build_extractors() drops any that fail to load.
    extractors = build_ner_extractors(cfg)
    active_ner = [e.name for e in extractors]
    result.backends["ner"] = ", ".join(active_ner)
    for requested in cfg.ner.extractors:
        if requested not in active_ner:
            result.degraded.append(f"NER extractor '{requested}' failed to load")
    run_ner(documents, cfg=cfg, extractors=extractors)
    result.timings["ner"] = time.perf_counter() - started
    completed += weights["ner"]
    result.counts["mentions"] = sum(len(d.mentions) for d in documents)

    # --- Phase 4 ---------------------------------------------------------
    report("coreference", f"resolving coreference ({cfg.coreference.backend})")
    started = time.perf_counter()
    # Same reasoning as NER: build_resolver() silently returns the rule-based
    # resolver when the neural model cannot load, and that fallback is the most
    # likely explanation for a "full" run finishing suspiciously fast --
    # coreference is ~94% of full-profile runtime.
    resolver = build_coref_resolver(cfg)
    result.backends["coreference"] = resolver.name
    if cfg.coreference.backend == "fastcoref" and resolver.name != "fastcoref":
        result.degraded.append(
            "neural coreference unavailable - fell back to rules "
            "(no nominal coreference, so 'The Indian Prime Minister' will not link)"
        )
    run_coreference(documents, cfg=cfg, resolver=resolver)
    result.timings["coreference"] = time.perf_counter() - started
    completed += weights["coreference"]
    result.counts["coref_clusters"] = sum(len(d.coref_clusters) for d in documents)

    # --- Phase 5 ---------------------------------------------------------
    report("relations", "extracting relations")
    started = time.perf_counter()
    run_relation_extraction(documents, cfg=cfg)
    result.timings["relations"] = time.perf_counter() - started
    completed += weights["relations"]
    result.counts["relations"] = sum(len(d.relations) for d in documents)

    # --- Phase 6 ---------------------------------------------------------
    report("entity_resolution", "resolving entities across articles")
    started = time.perf_counter()
    profiles = build_all_profiles(documents)
    entities, review_queue, er_stats = resolve_entities(
        profiles, cfg=cfg, embed=cfg.entity_resolution.use_embeddings
    )
    result.timings["entity_resolution"] = time.perf_counter() - started
    result.backends["entity_resolution"] = (
        "embeddings + attributes" if cfg.entity_resolution.use_embeddings
        else "attributes only (no embeddings)"
    )
    completed = 1.0
    report("done", f"{len(entities)} canonical entities")

    result.documents = documents
    result.profiles = profiles
    result.entities = entities
    result.review_queue = review_queue
    result.counts.update(
        {
            "articles": len(documents),
            "profiles": er_stats.profiles,
            "entities": len(entities),
            "review_pairs": len(review_queue),
            "vetoed": er_stats.vetoed,
        }
    )

    logger.info(
        "Pipeline complete in %.1fs: %s", result.total_seconds, result.counts
    )
    return result


def persist(result: PipelineResult, cfg: Config | None = None) -> dict[str, int]:
    """Write a pipeline result into the SQLite knowledge graph."""
    from src.storage.store import (
        build_entity_relations,
        connect,
        database_stats,
        load_documents,
        load_entities,
        load_review_queue,
    )

    cfg = cfg or load_config()
    with connect(cfg.path(cfg.paths.db_path)) as connection:
        load_documents(connection, result.documents)
        load_entities(connection, result.entities)
        build_entity_relations(connection)
        load_review_queue(connection, result.review_queue)
        return database_stats(connection)
