"""Rule-based coreference. A DOCUMENTED BASELINE, not a production system.

WHY HAVE THIS AT ALL
--------------------
Three honest reasons:
  1. INSURANCE. The neural model is a 1.7 GB download and needs ~13 s per
     document on CPU. The pipeline must still run, degraded but functional,
     when it is unavailable or too slow.
  2. A MEASURABLE FLOOR. "The transformer is better" is a claim. Phase 10 can
     only quantify how much better if there is something to compare against.
  3. TEACHING. Writing the rules makes it concrete WHY the neural model wins:
     you can see exactly which linguistic knowledge the rules cannot encode.

WHAT IT IMPLEMENTS
------------------
  R1  Exact match of named mentions      "Modi" ... "Modi"
  R2  Surname containment (PERSON only)  "Modi" is a token of "Narendra Modi"
  R3  Recency-based pronoun resolution   "He" -> nearest preceding PERSON

WHAT IT FUNDAMENTALLY CANNOT DO -- and this is the point
--------------------------------------------------------
  * GENDER. It has no idea that "Narendra" is male and "Nirmala" is female.
    With both in one document, "She" attaches to whichever person is nearer.
    We mitigate only with a CONSISTENCY constraint (one cluster cannot absorb
    both "he" and "she"), which is not the same as knowing gender.
  * SYNTAX. "Modi met Putin. He said..." -- "He" is far more likely to be Modi,
    the SUBJECT, than Putin, the object. Recency alone picks Putin. Wrong.
  * WORLD KNOWLEDGE. "the lender" -> "the New Development Bank" requires
    knowing a bank is a lender. No rule list contains that.
  * SEMANTIC PLAUSIBILITY. "It was signed on Wednesday" -- only meaning tells
    you "It" is a document rather than a person.

Every one of those is something a transformer learns implicitly during
pretraining, which is the whole argument for the neural approach.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.coreference.base import build_cluster
from src.logging_utils import get_logger
from src.schemas import CorefCluster, Document, Mention

logger = get_logger(__name__)

# Pronoun groups, and what kind of antecedent each can take.
MASCULINE = {"he", "him", "his", "himself"}
FEMININE = {"she", "her", "hers", "herself"}
NEUTER = {"it", "its", "itself"}
PLURAL = {"they", "them", "their", "theirs", "themselves"}

PERSON_PRONOUNS = MASCULINE | FEMININE
NON_PERSON_LABELS = {"ORG", "COUNTRY", "LOCATION", "EVENT"}

_ALL_PRONOUNS = sorted(MASCULINE | FEMININE | NEUTER | PLURAL, key=len, reverse=True)
_PRONOUN_RE = re.compile(r"\b(" + "|".join(_ALL_PRONOUNS) + r")\b", re.IGNORECASE)

_HONORIFIC_RE = re.compile(r"^(mr|mrs|ms|dr|prof|sir|shri|smt)\.?\s+", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


class _UnionFind:
    """Disjoint-set structure, used to turn pairwise links into clusters.

    Coreference is an EQUIVALENCE relation: if A links to B and B links to C,
    then A, B and C form one cluster even though A and C were never compared.
    Union-Find computes exactly that transitive closure, in near-constant time
    per operation thanks to path compression.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression: point every node on the way directly at the root,
        # so later lookups are O(1) instead of walking the chain again.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for item in self._parent:
            result.setdefault(self.find(item), []).append(item)
        return result


def normalize_name(text: str) -> str:
    """Lowercase and strip honorifics, so "Mr. Modi" and "Modi" compare equal."""
    cleaned = _HONORIFIC_RE.sub("", text.strip())
    return _PUNCT_RE.sub("", cleaned).strip().lower()


class RuleBasedCorefResolver:
    """Heuristic within-document coreference. Baseline quality by design."""

    name = "rules"

    def __init__(self, max_sentence_distance: int = 2) -> None:
        # How far back a pronoun may look. Two sentences covers the large
        # majority of pronoun-antecedent distances in news prose. Widening it
        # raises recall but raises wrong links faster.
        self.max_sentence_distance = max_sentence_distance

    def _pronoun_spans(self, document: Document) -> list[tuple[int, int, str]]:
        """Find pronouns that are not already inside an entity mention."""
        occupied = [(m.start, m.end) for m in document.mentions]
        spans: list[tuple[int, int, str]] = []
        for match in _PRONOUN_RE.finditer(document.text):
            start, end = match.start(), match.end()
            if any(s <= start and end <= e for s, e in occupied):
                continue
            spans.append((start, end, match.group(0).lower()))
        return spans

    def _sentence_index(self, document: Document, position: int) -> int:
        sentence = document.sentence_containing(position)
        return sentence.index if sentence else -1

    def resolve(self, document: Document) -> list[CorefCluster]:
        entity_mentions: list[Mention] = [
            m
            for m in document.mentions
            if m.label in {"PERSON", "ORG", "COUNTRY", "LOCATION", "EVENT"}
        ]
        if not entity_mentions:
            return []

        union = _UnionFind()
        span_of_key: dict[str, tuple[int, int]] = {}

        def key(start: int, end: int) -> str:
            k = f"{start}-{end}"
            span_of_key[k] = (start, end)
            union.find(k)
            return k

        # R1 + R2 -- link named mentions of the same type.
        for i, first in enumerate(entity_mentions):
            key(first.start, first.end)
            norm_first = normalize_name(first.text)
            if not norm_first:
                continue
            for second in entity_mentions[i + 1 :]:
                if first.label != second.label:
                    continue
                norm_second = normalize_name(second.text)
                if not norm_second:
                    continue

                exact = norm_first == norm_second
                # Surname containment, PERSON only. "Modi" is a proper subset of
                # the tokens of "Narendra Modi". Restricted to PERSON because
                # for ORG it produces nonsense -- "Bank" would match every bank
                # in the article.
                tokens_first, tokens_second = set(norm_first.split()), set(norm_second.split())
                contained = first.label == "PERSON" and (
                    tokens_first < tokens_second or tokens_second < tokens_first
                )
                if exact or contained:
                    union.union(key(first.start, first.end), key(second.start, second.end))

        # R3 -- pronouns attach to the nearest preceding compatible entity.
        # cluster_gender enforces CONSISTENCY (one cluster cannot hold both "he"
        # and "she"). That is not gender knowledge: it cannot tell you that
        # "Nirmala" is female, only that a single cluster should not be both.
        cluster_gender: dict[str, str] = {}

        for start, end, pronoun in self._pronoun_spans(document):
            pronoun_sentence = self._sentence_index(document, start)
            wants_person = pronoun in PERSON_PRONOUNS

            best: Mention | None = None
            for mention in entity_mentions:
                if mention.end > start:  # the antecedent must PRECEDE the pronoun
                    continue
                distance = pronoun_sentence - self._sentence_index(document, mention.start)
                if distance < 0 or distance > self.max_sentence_distance:
                    continue
                if wants_person and mention.label != "PERSON":
                    continue
                if not wants_person and mention.label not in NON_PERSON_LABELS:
                    continue
                if best is None or mention.start > best.start:
                    best = mention  # nearest preceding wins -- the whole heuristic

            if best is None:
                continue

            root = union.find(key(best.start, best.end))
            gender = "M" if pronoun in MASCULINE else "F" if pronoun in FEMININE else ""
            if gender and cluster_gender.get(root, gender) != gender:
                continue  # would mix "he" and "she" in one cluster
            if gender:
                cluster_gender[root] = gender

            union.union(root, key(start, end))

        clusters: list[CorefCluster] = []
        for index, (_, members) in enumerate(sorted(union.groups().items())):
            spans = sorted(span_of_key[m] for m in members)
            cluster = build_cluster(document, spans, index, method=self.name, score=0.5)
            if cluster is not None:
                clusters.append(cluster)

        # Renumber so IDs stay contiguous after singleton clusters were dropped.
        for index, cluster in enumerate(clusters):
            cluster.cluster_id = f"{document.article_id}:coref{index:03d}"
        return clusters

    def resolve_batch(self, documents: Sequence[Document]) -> list[list[CorefCluster]]:
        return [self.resolve(document) for document in documents]
