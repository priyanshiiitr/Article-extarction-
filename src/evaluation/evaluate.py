"""Score the pipeline against the manually labelled gold set.

MATCHING POLICY -- and why it is reported explicitly
----------------------------------------------------
Whether a prediction "matches" gold is a CHOICE, and different choices give
very different numbers. Papers that omit this are not comparable. We report
both, because the gap between them is itself informative:

  STRICT   same character span AND same label. This is the standard for NER
           benchmarks (CoNLL uses it), and it punishes boundary errors as
           harshly as complete misses -- predicting "the 16th BRICS Summit"
           when gold says "16th BRICS Summit" scores zero.

  RELAXED  overlapping spans AND same label. Credits a system that found the
           entity but disagreed about the determiner.

A large strict/relaxed gap means your boundaries are wrong, not your detection
-- a completely different problem to fix, and one you would never see from a
single F1 number.

Relations and entities are scored on NORMALISED TEXT rather than spans, because
at that level the fact is what matters, not which characters expressed it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.entity_resolution.normalize import normalize_name
from src.evaluation.metrics import (
    ClusteringReport,
    PRF,
    b_cubed,
    evaluate_clustering,
    muc,
    prf_from_sets,
)
from src.logging_utils import get_logger
from src.schemas import Document

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    ner_strict: PRF = field(default_factory=PRF)
    ner_relaxed: PRF = field(default_factory=PRF)
    ner_by_label: dict[str, PRF] = field(default_factory=dict)
    ner_detection_only: PRF = field(default_factory=PRF)

    coref_muc: PRF = field(default_factory=PRF)
    coref_b3: tuple[float, float, float] = (0.0, 0.0, 0.0)

    relations: PRF = field(default_factory=PRF)
    relations_by_predicate: dict[str, PRF] = field(default_factory=dict)

    entity_resolution: ClusteringReport | None = None

    articles_evaluated: int = 0
    notes: list[str] = field(default_factory=list)


def load_gold(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_span(
    document: Document, sentence_index: int, surface: str
) -> tuple[int, int] | None:
    """Turn a (sentence, surface) gold reference into character offsets.

    Resolving at load time rather than storing offsets in the gold file means
    the gold set survives preprocessing changes. If the surface is not found we
    return None and record it, rather than silently scoring against a span that
    does not exist.
    """
    sentence = next((s for s in document.sentences if s.index == sentence_index), None)
    if sentence is None:
        return None
    window = document.text[sentence.start : sentence.end]
    offset = window.find(surface)
    if offset < 0:
        return None
    return sentence.start + offset, sentence.start + offset + len(surface)


def _gold_ner(document: Document, article_gold: dict) -> tuple[set, set, list[str]]:
    """Return (strict_keys, relaxed_spans, unresolved)."""
    strict: set[tuple[int, int, str]] = set()
    spans: set[tuple[int, int, str]] = set()
    unresolved: list[str] = []
    for item in article_gold.get("mentions", []):
        resolved = _resolve_span(document, item["sent"], item["text"])
        if resolved is None:
            unresolved.append(f"{document.article_id} S{item['sent']} {item['text']!r}")
            continue
        key = (resolved[0], resolved[1], item["label"])
        strict.add(key)
        spans.add(key)
    return strict, spans, unresolved


def _relaxed_match(
    predicted: set[tuple[int, int, str]], gold: set[tuple[int, int, str]]
) -> PRF:
    """Overlap-based matching: same label and any character overlap.

    Greedy one-to-one: each gold span may be claimed by at most one prediction,
    so a system cannot inflate recall by emitting five overlapping guesses.
    """
    remaining = set(gold)
    true_positives = 0
    for p_start, p_end, p_label in sorted(predicted):
        hit = None
        for g_start, g_end, g_label in remaining:
            if p_label == g_label and p_start < g_end and g_start < p_end:
                hit = (g_start, g_end, g_label)
                break
        if hit:
            remaining.discard(hit)
            true_positives += 1
    return PRF(
        true_positives=true_positives,
        false_positives=len(predicted) - true_positives,
        false_negatives=len(gold) - true_positives,
    )


def _normalize_argument(text: str) -> str:
    return " ".join(text.lower().replace(".", "").split())


def _build_alias_map(gold: dict[str, Any]) -> dict[str, str]:
    """Map every gold alias to its gold entity key.

    This is what lets relation evaluation work at ENTITY level: "Modi",
    "PM Modi" and "Narendra Modi" all resolve to E_NARENDRA, so a predicted
    relation phrased with one surface matches a gold relation phrased with
    another.

    WHY THIS MATTERS: scoring relations on raw strings conflates two different
    failures. Our first run scored `met` at F1 0.000 because gold said
    "Narendra Modi" and the system said "Modi" -- the RELATION was perfectly
    correct and only the surface form differed. Relation quality and entity
    quality are entangled, and the evaluation has to decide which one it is
    measuring. This measures the relation.
    """
    alias_map: dict[str, str] = {}
    for entity_key, aliases in gold.get("entities", {}).items():
        if entity_key.startswith("_"):
            continue
        for alias in aliases:
            alias_map[_normalize_argument(alias)] = entity_key
    return alias_map


def evaluate(
    documents: Sequence[Document],
    gold: dict[str, Any],
    entities: Sequence[Any] | None = None,
) -> EvaluationReport:
    """Score NER, coreference, relations and (optionally) entity resolution."""
    report = EvaluationReport()
    by_id = {d.article_id: d for d in documents}
    alias_map = _build_alias_map(gold)

    def canonical(text: str) -> str:
        """Resolve an argument to its gold entity key, or its normalised text."""
        normalized = _normalize_argument(text)
        return alias_map.get(normalized, normalized)

    gold_entity_clusters: dict[str, list[str]] = {}
    predicted_entity_clusters: dict[str, list[str]] = {}

    for article_id, article_gold in gold["articles"].items():
        document = by_id.get(article_id)
        if document is None:
            report.notes.append(f"gold article {article_id} not found in predictions")
            continue
        report.articles_evaluated += 1

        # ---- NER --------------------------------------------------------
        gold_strict, gold_spans, unresolved = _gold_ner(document, article_gold)
        report.notes.extend(f"unresolved gold span: {u}" for u in unresolved)

        predicted_strict = {(m.start, m.end, m.label) for m in document.mentions}

        report.ner_strict.true_positives += len(predicted_strict & gold_strict)
        report.ner_strict.false_positives += len(predicted_strict - gold_strict)
        report.ner_strict.false_negatives += len(gold_strict - predicted_strict)

        relaxed = _relaxed_match(predicted_strict, gold_spans)
        report.ner_relaxed.true_positives += relaxed.true_positives
        report.ner_relaxed.false_positives += relaxed.false_positives
        report.ner_relaxed.false_negatives += relaxed.false_negatives

        # Detection without typing: did we find the span at all, ignoring the
        # label? The gap between this and strict is your TYPING error rate,
        # which is a different fix from a detection error.
        detection_predicted = {(s, e) for s, e, _ in predicted_strict}
        detection_gold = {(s, e) for s, e, _ in gold_strict}
        detection = prf_from_sets(detection_predicted, detection_gold)
        report.ner_detection_only.true_positives += detection.true_positives
        report.ner_detection_only.false_positives += detection.false_positives
        report.ner_detection_only.false_negatives += detection.false_negatives

        for label in {label for _, _, label in gold_strict} | {
            label for _, _, label in predicted_strict
        }:
            bucket = report.ner_by_label.setdefault(label, PRF())
            p = {k for k in predicted_strict if k[2] == label}
            g = {k for k in gold_strict if k[2] == label}
            bucket.true_positives += len(p & g)
            bucket.false_positives += len(p - g)
            bucket.false_negatives += len(g - p)

        # ---- Coreference -------------------------------------------------
        gold_clusters: list[list[tuple[int, int]]] = []
        for cluster in article_gold.get("coref_clusters", []):
            members = []
            for item in cluster:
                resolved = _resolve_span(document, item["sent"], item["text"])
                if resolved:
                    members.append(resolved)
            if len(members) > 1:
                gold_clusters.append(members)

        # GOLD-MENTION EVALUATION.
        #
        # Coreference must be scored on the mentions the annotation actually
        # covers. Our gold set annotates the MAIN clusters, not every referring
        # expression in the article -- so scoring "system mentions" counts every
        # extra predicted cluster ("the ruling", "the day") as a false positive
        # and produced MUC P=0.25 for a system that was mostly right.
        #
        # The two standard settings, and why the choice must be reported:
        #   SYSTEM MENTIONS -- the model proposes its own mentions. Measures the
        #       whole pipeline, but requires EXHAUSTIVE annotation to be fair.
        #   GOLD MENTIONS   -- mentions are given; only the CLUSTERING is
        #       scored. Isolates coreference from mention detection, which is
        #       what we want here and what most coref papers report.
        gold_spans_in_clusters = {span for cluster in gold_clusters for span in cluster}
        predicted_clusters = []
        for cluster in document.coref_clusters:
            members = [
                (m.start, m.end)
                for m in cluster.mentions
                if (m.start, m.end) in gold_spans_in_clusters
            ]
            if len(members) > 1:
                predicted_clusters.append(members)

        if gold_clusters:
            article_muc = muc(predicted_clusters, gold_clusters)
            report.coref_muc.true_positives += article_muc.true_positives
            report.coref_muc.false_positives += article_muc.false_positives
            report.coref_muc.false_negatives += article_muc.false_negatives

        # ---- Relations ---------------------------------------------------
        gold_relations = {
            (canonical(r["subject"]), r["predicate"], canonical(r["object"]))
            for r in article_gold.get("relations", [])
        }
        predicted_relations = {
            (canonical(r.subject_text), r.predicate, canonical(r.object_text))
            for r in document.relations
        }
        article_relations = prf_from_sets(predicted_relations, gold_relations)
        report.relations.true_positives += article_relations.true_positives
        report.relations.false_positives += article_relations.false_positives
        report.relations.false_negatives += article_relations.false_negatives

        for predicate in {r[1] for r in gold_relations} | {r[1] for r in predicted_relations}:
            bucket = report.relations_by_predicate.setdefault(predicate, PRF())
            p = {r for r in predicted_relations if r[1] == predicate}
            g = {r for r in gold_relations if r[1] == predicate}
            bucket.true_positives += len(p & g)
            bucket.false_positives += len(p - g)
            bucket.false_negatives += len(g - p)

        # ---- Entity resolution gold, gathered across articles -------------
        for link in article_gold.get("entity_links", []):
            resolved = _resolve_span(document, 0, link["text"])
            key = f"{article_id}|{_normalize_argument(link['text'])}"
            gold_entity_clusters.setdefault(link["entity"], []).append(key)

    # ---- B-cubed over the union of all articles --------------------------
    all_gold_clusters: list[list[str]] = []
    all_predicted_clusters: list[list[str]] = []
    for article_id, article_gold in gold["articles"].items():
        document = by_id.get(article_id)
        if document is None:
            continue
        for cluster in article_gold.get("coref_clusters", []):
            members = [
                f"{article_id}:{item['sent']}:{item['text']}"
                for item in cluster
                if _resolve_span(document, item["sent"], item["text"])
            ]
            if len(members) > 1:
                all_gold_clusters.append(members)
        for cluster in document.coref_clusters:
            members = []
            for mention in cluster.mentions:
                sentence = document.sentence_containing(mention.start)
                if sentence:
                    members.append(f"{article_id}:{sentence.index}:{mention.text}")
            if len(members) > 1:
                all_predicted_clusters.append(members)
    report.coref_b3 = b_cubed(all_predicted_clusters, all_gold_clusters)

    # ---- Entity resolution ------------------------------------------------
    if entities is not None and gold_entity_clusters:
        for entity in entities:
            members = []
            for mention_id in entity.mention_ids:
                article_id = mention_id.split(":", 1)[0]
                if article_id not in gold["articles"]:
                    continue
                document = by_id.get(article_id)
                if document is None:
                    continue
                mention = next(
                    (m for m in document.mentions if m.mention_id == mention_id), None
                )
                if mention is None:
                    continue
                members.append(f"{article_id}|{_normalize_argument(mention.text)}")
            if members:
                predicted_entity_clusters[entity.entity_id] = members

        report.entity_resolution = evaluate_clustering(
            [sorted(set(v)) for v in predicted_entity_clusters.values()],
            [sorted(set(v)) for v in gold_entity_clusters.values()],
        )

    return report
