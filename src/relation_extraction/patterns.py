"""Pattern-based extraction for relations expressed by JUXTAPOSITION.

WHY PATTERNS ARE THE RIGHT TOOL HERE (and not elsewhere)
--------------------------------------------------------
Some relations in news prose are not expressed by a verb at all. They are
expressed by putting two things next to each other:

    "Prime Minister Narendra Modi"          role, then person -> holds_position
    "Indian Prime Minister Narendra Modi"   + demonym         -> represents
    "Mukesh Ambani, chairman of Reliance"   appositive        -> works_for

There is no verb for a dependency parse to hang off. The relation lives in the
ADJACENCY. So patterns are not a lazy choice for these -- they are the
structurally correct one. Conversely they would be a poor choice for "met" or
"discussed", where a verb exists and its arguments can be arbitrarily far away;
those go to the dependency extractor.

The general principle worth stating in an interview: match the extraction
method to how the relation is GRAMMATICALLY REALISED, not to a blanket
preference for rules or models.

HOW ADJACENCY IS DECIDED
------------------------
We do not compare raw string positions with a magic character budget. We check
that the text BETWEEN two mentions contains nothing but whitespace, or a short
connector we explicitly allow (a comma, "of", "the"). That makes the rule
readable and keeps it from silently matching across half a sentence.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.logging_utils import get_logger
from src.relation_extraction.base import (
    ResolvedArgument,
    build_relation,
    types_are_valid,
)
from src.schemas import Document, Mention, Relation

logger = get_logger(__name__)

# Patterns are deterministic string matches, so confidence reflects how often
# the PATTERN implies the relation, not model uncertainty. Direct juxtaposition
# ("Prime Minister Modi") is near-certain; an appositive across a comma is a
# little less so, because commas have many other uses.
ADJACENT_CONFIDENCE = 0.90
APPOSITIVE_CONFIDENCE = 0.78
OF_CONFIDENCE = 0.85

# Text permitted between two mentions for them still to count as adjacent.
_JOINERS = {"", ",", "the", ", the", "of", "of the", ", who is", "-", "&"}

_DEMONYM_PREFIX = "GAZ_DEMONYM:"


def _between(document: Document, left: Mention, right: Mention) -> str:
    return document.text[left.end : right.start].strip().lower()


def _adjacent(document: Document, left: Mention, right: Mention) -> bool:
    """True when only whitespace or an allowed connector separates the two."""
    return _between(document, left, right) in _JOINERS


def _argument(mention: Mention, text: str | None = None, label: str | None = None) -> ResolvedArgument:
    return ResolvedArgument(
        mention=mention,
        text=text if text is not None else mention.text,
        label=label if label is not None else mention.label,
        via_coref=False,
    )


class PatternRelationExtractor:
    """Extracts holds_position, represents and works_for from adjacency."""

    name = "pattern"

    def extract(self, document: Document) -> list[Relation]:
        relations: list[Relation] = []

        # Group mentions by sentence. A relation expressed by juxtaposition
        # never spans a sentence boundary, so this both prunes the search and
        # prevents nonsense matches across a full stop.
        by_sentence: dict[int, list[Mention]] = {}
        for mention in document.mentions:
            by_sentence.setdefault(mention.sentence_index, []).append(mention)

        for mentions in by_sentence.values():
            mentions.sort(key=lambda m: m.start)
            relations.extend(self._from_sentence(document, mentions))

        return relations

    def _from_sentence(self, document: Document, mentions: list[Mention]) -> list[Relation]:
        relations: list[Relation] = []

        for index, mention in enumerate(mentions):
            nxt = mentions[index + 1] if index + 1 < len(mentions) else None
            prv = mentions[index - 1] if index > 0 else None

            # ---- PATTERN 1: <ROLE> <PERSON>  ->  holds_position --------------
            # "Prime Minister Narendra Modi", "Finance Minister Sitharaman"
            if mention.label == "ROLE" and nxt is not None and nxt.label == "PERSON":
                if _adjacent(document, mention, nxt):
                    relations.append(
                        self._make(document, _argument(nxt), "holds_position",
                                   _argument(mention), ADJACENT_CONFIDENCE, "role+person")
                    )

                    # ---- PATTERN 2: <DEMONYM> <ROLE> <PERSON> -> represents --
                    # "Indian Prime Minister Narendra Modi" tells us WHO Modi
                    # represents. The demonym mention is labelled MISC, but the
                    # RELATION's object is the country it denotes.
                    #
                    # Deliberate normalisation: object_mention_id still points
                    # at the "Indian" span (that is the evidence), while
                    # object_text/label carry the canonical "India"/COUNTRY.
                    # The mention records WHERE the fact came from; the relation
                    # records WHAT it means. Those are different questions.
                    if prv is not None and prv.raw_label.startswith(_DEMONYM_PREFIX):
                        if _adjacent(document, prv, mention):
                            country = prv.raw_label[len(_DEMONYM_PREFIX):]
                            relations.append(
                                self._make(
                                    document,
                                    _argument(nxt),
                                    "represents",
                                    _argument(prv, text=country, label="COUNTRY"),
                                    ADJACENT_CONFIDENCE,
                                    "demonym+role+person",
                                )
                            )

            # ---- PATTERN 3: <PERSON>, <ROLE>  ->  holds_position -------------
            # Appositive: "Narendra Modi, the Prime Minister, said..."
            if mention.label == "PERSON" and nxt is not None and nxt.label == "ROLE":
                if _adjacent(document, mention, nxt):
                    relations.append(
                        self._make(document, _argument(mention), "holds_position",
                                   _argument(nxt), APPOSITIVE_CONFIDENCE, "person+role")
                    )

            # ---- PATTERN 4: <ROLE> of <ORG>  ->  works_for -------------------
            # "chairman of Reliance" -- attach to the nearest PERSON in the
            # sentence that the role already belongs to.
            if mention.label == "ROLE" and nxt is not None and nxt.label == "ORG":
                if _between(document, mention, nxt) in {"of", "of the", "at", "at the"}:
                    holder = self._role_holder(document, mentions, mention)
                    if holder is not None:
                        relations.append(
                            self._make(document, _argument(holder), "works_for",
                                       _argument(nxt), OF_CONFIDENCE, "role+of+org")
                        )

            # ---- PATTERN 5: coreference-mediated role attachment -------------
            # "The Indian Prime Minister is scheduled to hold talks..." has a
            # ROLE and a demonym but NO name, so patterns 1-2 find nothing.
            # Coreference knows this phrase refers to Narendra Modi, so we can
            # still emit holds_position and represents.
            #
            # This is the second place Phase 4 pays for itself, and it is the
            # reason nominal coreference mattered: without it these sentences
            # contribute no facts at all.
            if mention.label == "ROLE" and mention.coref_cluster_id:
                holder = self._coref_person(document, mention.coref_cluster_id)
                if holder is not None:
                    relations.append(
                        self._make(
                            document, _argument(holder), "holds_position",
                            _argument(mention), APPOSITIVE_CONFIDENCE, "coref+role",
                            via_coref_subject=True,
                        )
                    )
                    if prv is not None and prv.raw_label.startswith(_DEMONYM_PREFIX):
                        if _adjacent(document, prv, mention):
                            country = prv.raw_label[len(_DEMONYM_PREFIX):]
                            relations.append(
                                self._make(
                                    document, _argument(holder), "represents",
                                    _argument(prv, text=country, label="COUNTRY"),
                                    APPOSITIVE_CONFIDENCE, "coref+demonym+role",
                                    via_coref_subject=True,
                                )
                            )

        return [r for r in relations if r is not None]

    def _coref_person(self, document: Document, cluster_id: str) -> Mention | None:
        """Find the PERSON mention that a coreference cluster refers to."""
        for cluster in document.coref_clusters:
            if cluster.cluster_id != cluster_id:
                continue
            representative = cluster.representative
            if representative is None or not representative.ner_mention_id:
                return None
            for candidate in document.mentions:
                if (
                    candidate.mention_id == representative.ner_mention_id
                    and candidate.label == "PERSON"
                ):
                    return candidate
        return None

    def _role_holder(
        self, document: Document, mentions: list[Mention], role: Mention
    ) -> Mention | None:
        """Find the PERSON a role belongs to: immediately after, else before.

        "chairman of Reliance" on its own has no holder. In "Mukesh Ambani,
        chairman of Reliance" the holder precedes; in "Prime Minister Narendra
        Modi" it follows. We check the following mention first because the
        title-first construction is far more common in news leads.
        """
        ordered = sorted(mentions, key=lambda m: m.start)
        position = ordered.index(role)

        after = ordered[position + 1] if position + 1 < len(ordered) else None
        if after is not None and after.label == "PERSON" and _adjacent(document, role, after):
            return after

        for candidate in reversed(ordered[:position]):
            if candidate.label == "PERSON":
                return candidate
        return None

    def _make(
        self,
        document: Document,
        subject: ResolvedArgument,
        predicate: str,
        obj: ResolvedArgument,
        confidence: float,
        trigger: str,
        via_coref_subject: bool = False,
    ) -> Relation | None:
        if not types_are_valid(predicate, subject.label, obj.label):
            return None
        if via_coref_subject:
            subject.via_coref = True
        return build_relation(
            document=document,
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=confidence,
            extractor=self.name,
            trigger=trigger,
        )

    def extract_batch(self, documents: Sequence[Document]) -> list[list[Relation]]:
        return [self.extract(document) for document in documents]
