"""Merge mentions from several extractors into one consistent annotation layer.

THE PROBLEM
-----------
Three extractors run over the same text and they disagree. Real examples from
our corpus:

    "Sitharaman"            spaCy=ORG        transformer=PERSON
    "BRICS"                 spaCy=LOCATION   transformer=ORG
    "Kazan Declaration"     spaCy=PERSON     transformer=MISC   gazetteer=EVENT
    "the 16th BRICS Summit" spaCy=ORG        gazetteer=EVENT("16th BRICS Summit")
    "India"                 spaCy=LOCATION   gazetteer=COUNTRY

Someone has to decide. Doing it badly poisons every later stage, because a
mention typed ORG will never be considered as a candidate for a PERSON entity.

THE APPROACH: A TRUST MATRIX
----------------------------
Not "the highest score wins" -- the scores are not comparable across
extractors. spaCy's 0.75 is a fixed constant we invented, the transformer's
0.99 is a softmax output, and the gazetteer's 0.90 means "a string matched".
Comparing them directly would be meaningless.

Instead we encode what we actually know about each extractor's competence PER
LABEL, from the evidence we gathered:

  * The transformer is markedly better on PERSON and ORG (it got Sitharaman
    and BRICS right where spaCy did not).
  * spaCy is fine on LOCATION and DATE and is the only model with EVENT.
  * The gazetteer is near-certain on COUNTRY and ROLE, because those are
    closed-set lookups rather than predictions.
  * MISC is a dustbin label from every source and should almost always lose.

This table IS the merge policy, written down in one place instead of scattered
through if-statements. That matters because it is a set of assumptions, and
assumptions must be visible to be challenged -- which is exactly what Phase 10
will do by measuring them against labelled data.

THIS IS A HEURISTIC, NOT A LEARNED MODEL
----------------------------------------
In a mature system you would learn these weights: take a labelled dev set,
treat each extractor's prediction as a feature, and train a small classifier
(or a CRF) to pick the winner -- this is "stacking"/ensembling. We hand-set
them because we do not yet have labelled data. Phase 10 builds that data, and
at that point these numbers become measurable instead of asserted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from src.logging_utils import get_logger
from src.schemas import Document, Mention

logger = get_logger(__name__)

# How much we trust each extractor for each label, in [0, 1].
# Read a row as: "when THIS extractor claims THIS label, how much do we believe it?"
TRUST: dict[str, dict[str, float]] = {
    "transformer": {
        "PERSON": 0.95,   # clearly the best: found "Modi" in the headline
        "ORG": 0.88,      # got BRICS right where spaCy said LOCATION
        "LOCATION": 0.85,
        "MISC": 0.30,     # CoNLL MISC is a dustbin: nationalities+events+products
    },
    "spacy": {
        "PERSON": 0.70,   # called "Sitharaman" an ORG and "Kazan Declaration" a PERSON
        "ORG": 0.60,
        "LOCATION": 0.80,
        "EVENT": 0.55,    # inconsistent, but the only MODEL with an EVENT type
        "DATE": 0.90,     # dates are highly patterned; spaCy is reliable here
        "MISC": 0.25,
    },
    "gazetteer": {
        "COUNTRY": 0.98,  # closed set; if it matched, it is a country
        "ROLE": 0.95,     # no model provides this at all
        "EVENT": 0.90,    # pattern-based and precise on "<Caps> Summit"
        # Raised from 0.70 to 0.82 after observing a concrete failure: spaCy
        # labelled "AI" as GPE/LOCATION, and at 0.70 that beat our TOPIC match,
        # so "AI" entered the graph as a place. A curated-list TOPIC hit has
        # LOW recall but HIGH precision -- when it fires it is almost always
        # right -- so it should outrank a model's LOCATION guess (0.80).
        # Recorded here as evidence of why Phase 10 must LEARN these weights on
        # labelled data instead of me asserting them.
        "TOPIC": 0.82,
        "MISC": 0.40,     # demonyms; beats model MISC because it resolves a country
    },
}

DEFAULT_TRUST = 0.50

# Tie-breaker only. A more specific type is preferred when trust is equal --
# MISC means "we could not tell", so it should never beat a real answer.
LABEL_SPECIFICITY: dict[str, int] = {
    "COUNTRY": 5, "ROLE": 5, "EVENT": 4, "TOPIC": 4,
    "PERSON": 3, "ORG": 3,
    "LOCATION": 2, "DATE": 2,
    "MISC": 0,
}


@dataclass
class MergeStats:
    """What the merge actually did -- so it can be inspected, not guessed at."""

    input_mentions: int = 0
    output_mentions: int = 0
    conflicts: int = 0
    winners_by_extractor: Counter = field(default_factory=Counter)
    dropped_by_extractor: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        return (
            f"in={self.input_mentions} out={self.output_mentions} "
            f"conflicts={self.conflicts} winners={dict(self.winners_by_extractor)}"
        )


def _priority(mention: Mention) -> tuple[float, int, int, float]:
    """Rank a mention within a conflict group. Higher tuple wins.

    Python compares tuples element by element, so this expresses a strict
    preference order in one expression:

      1. trust        -- competence of this extractor for this label
      2. specificity  -- a real type beats MISC
      3. span length  -- "Narendra Modi" beats "Modi"; longer spans carry more
                         information and are usually the correct boundary
      4. score        -- final tiebreak within one extractor
    """
    trust = TRUST.get(mention.extractor, {}).get(mention.label, DEFAULT_TRUST)
    specificity = LABEL_SPECIFICITY.get(mention.label, 1)
    return (trust, specificity, mention.length, mention.score)


def _group_overlapping(mentions: list[Mention]) -> list[list[Mention]]:
    """Partition mentions into groups of mutually overlapping spans.

    A sweep over start-sorted mentions: extend the current group while the next
    mention starts before the group's furthest end, otherwise close the group.
    O(n log n) for the sort, O(n) for the sweep -- rather than the O(n^2) of
    comparing every mention to every other one.
    """
    if not mentions:
        return []

    ordered = sorted(mentions, key=lambda m: (m.start, -m.end))
    groups: list[list[Mention]] = [[ordered[0]]]
    group_end = ordered[0].end

    for mention in ordered[1:]:
        if mention.start < group_end:      # overlaps the open group
            groups[-1].append(mention)
            group_end = max(group_end, mention.end)
        else:
            groups.append([mention])
            group_end = mention.end
    return groups


def merge_mentions(
    mention_lists: Sequence[list[Mention]],
    stats: MergeStats | None = None,
) -> list[Mention]:
    """Combine several extractors' mentions for ONE document.

    Returns a list of non-overlapping mentions sorted by position.
    """
    stats = stats if stats is not None else MergeStats()

    all_mentions = [m for mentions in mention_lists for m in mentions]
    stats.input_mentions += len(all_mentions)
    if not all_mentions:
        return []

    merged: list[Mention] = []
    for group in _group_overlapping(all_mentions):
        if len(group) == 1:
            winner = group[0]
        else:
            stats.conflicts += 1
            winner = max(group, key=_priority)
            for loser in group:
                if loser is not winner:
                    stats.dropped_by_extractor[loser.extractor] += 1
        stats.winners_by_extractor[winner.extractor] += 1
        merged.append(winner)

    stats.output_mentions += len(merged)
    return sorted(merged, key=lambda m: m.start)


def merge_documents(
    documents: Sequence[Document],
    mention_lists_per_extractor: Sequence[Sequence[list[Mention]]],
) -> MergeStats:
    """Merge in place for a batch of documents.

    ``mention_lists_per_extractor`` is indexed [extractor][document]; we
    transpose it to [document][extractor] so each document's candidates are
    resolved together.
    """
    stats = MergeStats()
    for doc_index, document in enumerate(documents):
        per_extractor = [
            extractor_results[doc_index] for extractor_results in mention_lists_per_extractor
        ]
        document.mentions = merge_mentions(per_extractor, stats)
    logger.info("Merged NER output: %s", stats.summary())
    return stats
