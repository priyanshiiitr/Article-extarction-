"""The coreference-resolver interface, plus alignment back to NER mentions.

WHAT THIS STAGE MUST AND MUST NOT DO
------------------------------------
MUST:  produce clusters of character spans, and link each cluster to the Phase 3
       entity mentions that fall inside it.
MUST NOT: modify Document.text. Ever.

The temptation is to "resolve" the text by substituting names for pronouns.
That is wrong for five separate reasons, and they are worth being able to
recite:

  1. IT DESTROYS OFFSETS.   "He" (2 chars) -> "Narendra Modi" (13 chars) shifts
     every later character by 11. Every mention, sentence and future annotation
     in the document then points at the wrong words. This alone disqualifies it.
  2. IT IS LOSSY.           You can no longer recover what the journalist wrote,
     so provenance -- showing a user the original sentence -- becomes impossible.
  3. IT IS UNGRAMMATICAL.   "Modi said he would go" becomes "Modi said Narendra
     Modi would go", which then confuses the parser in Phase 5.
  4. IT OVER-APPLIES.       Not every "he" in a document is the same person. One
     article in our corpus contains TWO prime ministers.
  5. IT DISCARDS UNCERTAINTY. The model produced a probability; a rewritten
     string is a hard commitment with the confidence thrown away.

So we store SPANS and IDS. Downstream stages ask "what does this mention
resolve to?" and receive an ID plus a representative span -- never a rewritten
string.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from src.logging_utils import get_logger
from src.schemas import (
    CorefCluster,
    CorefMention,
    Document,
    choose_representative,
    classify_mention_form,
)

logger = get_logger(__name__)


@runtime_checkable
class CoreferenceResolver(Protocol):
    """Anything that can group coreferring spans within one document."""

    name: str

    def resolve(self, document: Document) -> list[CorefCluster]:
        ...

    def resolve_batch(self, documents: Sequence[Document]) -> list[list[CorefCluster]]:
        ...


def make_cluster_id(article_id: str, index: int) -> str:
    """Deterministic cluster ID, stable for the same document and ordering."""
    return f"{article_id}:coref{index:03d}"


def build_cluster(
    document: Document,
    spans: Sequence[tuple[int, int]],
    index: int,
    method: str,
    score: float = 1.0,
) -> CorefCluster | None:
    """Assemble a validated CorefCluster from raw character spans.

    Single construction point, for the same reason ``build_mention`` is: it
    guarantees every backend produces clusters with identical invariants --
    in-bounds spans, deduplicated, sorted by position, form classified, and a
    representative chosen by one rule.

    Returns None for a degenerate cluster (fewer than two distinct mentions).
    A "cluster" of one is not coreference -- it is just a mention, and keeping
    singletons would bloat storage with no information. Note this IS a choice:
    some evaluation metrics count singletons, so Phase 10 must know we drop them.
    """
    seen: set[tuple[int, int]] = set()
    mentions: list[CorefMention] = []

    for start, end in spans:
        if start < 0 or end > len(document.text) or start >= end:
            logger.warning(
                "%s: dropping out-of-bounds coref span (%d,%d)",
                document.article_id,
                start,
                end,
            )
            continue
        if (start, end) in seen:
            continue
        seen.add((start, end))

        surface = document.text[start:end]
        mentions.append(
            CorefMention(
                start=start,
                end=end,
                text=surface,
                form=classify_mention_form(surface),
            )
        )

    if len(mentions) < 2:
        return None

    mentions.sort(key=lambda m: m.start)
    return CorefCluster(
        cluster_id=make_cluster_id(document.article_id, index),
        article_id=document.article_id,
        mentions=mentions,
        representative_index=choose_representative(mentions),
        score=score,
        method=method,
    )


# When one coref span contains SEVERAL entity mentions, which one is "the"
# entity? Largest-overlap is wrong, and we found two real failures with it:
#
#   "Former Indian Premier League chairman Lalit Modi"
#        -> picked "Indian Premier League" (ORG, 21 chars)
#           over    "Lalit Modi"            (PERSON, 10 chars)
#   "Prime Minister Narendra Modi"
#        -> picked "Prime Minister" (ROLE, 14) over "Narendra Modi" (PERSON, 13)
#
# In both cases the longer span is a MODIFIER and the shorter one is the thing
# being referred to. Two signals fix it:
#   1. label priority -- a PERSON is a referent; a ROLE is a description of one;
#   2. head position  -- English noun phrases with pre-modifiers are head-FINAL,
#      so the mention ending at the span's end is usually the head.
LINK_LABEL_PRIORITY: dict[str, int] = {
    "PERSON": 0,
    "ORG": 1,
    "EVENT": 2,
    "TOPIC": 3,
    "LOCATION": 4,
    "COUNTRY": 5,
    "ROLE": 6,
    "DATE": 7,
    "MISC": 8,
}


def align_clusters_to_mentions(document: Document) -> int:
    """Connect coreference clusters to Phase 3 entity mentions, both ways.

    Coreference generates its OWN spans, which will not exactly match NER's.
    Typical mismatches from our corpus:

        NER:   "Narendra Modi"                  (56, 69)
        coref: "Prime Minister Narendra Modi"   (41, 69)   <- includes the title

    So exact span equality is too strict. We match a coref mention to an NER
    mention when one CONTAINS the other, preferring the largest overlap. That
    is deliberately permissive in one direction only: we never link two spans
    that merely touch.

    KNOWN LIMITATION: head detection here is positional, not syntactic. For a
    POST-modified phrase such as "the Board of Control for Cricket in India"
    the true head is "Board", but the span ends with "India", so the country
    wins. Fixing that properly needs the dependency parse (spaCy already gives
    us one) to find the real head token. We deliberately do NOT paper over it
    with a coverage threshold tuned on eleven documents -- that would be
    overfitting to the sample corpus rather than solving the problem.

    Returns the number of NER mentions successfully linked.
    """
    linked = 0
    for cluster in document.coref_clusters:
        for coref_mention in cluster.mentions:
            best_key: tuple[int, int, int] | None = None
            best_mention = None

            for ner_mention in document.mentions:
                # Require containment in one direction or the other, so that a
                # one-character brush does not create a spurious link.
                contains = (
                    coref_mention.start <= ner_mention.start
                    and ner_mention.end <= coref_mention.end
                ) or (
                    ner_mention.start <= coref_mention.start
                    and coref_mention.end <= ner_mention.end
                )
                if not contains:
                    continue

                overlap = min(coref_mention.end, ner_mention.end) - max(
                    coref_mention.start, ner_mention.start
                )
                if overlap <= 0:
                    continue

                # Lower tuple sorts better: referential label first, then the
                # head-final position, then the larger overlap.
                key = (
                    LINK_LABEL_PRIORITY.get(ner_mention.label, 9),
                    0 if ner_mention.end == coref_mention.end else 1,
                    -overlap,
                )
                if best_key is None or key < best_key:
                    best_key, best_mention = key, ner_mention

            if best_mention is not None:
                coref_mention.ner_mention_id = best_mention.mention_id
                best_mention.coref_cluster_id = cluster.cluster_id
                linked += 1
    return linked


def verify_clusters(document: Document) -> list[str]:
    """Check cluster spans still match the document text (the usual tripwire)."""
    problems: list[str] = []
    for cluster in document.coref_clusters:
        if len(cluster.mentions) < 2:
            problems.append(f"{cluster.cluster_id}: singleton cluster stored")
        for mention in cluster.mentions:
            if mention.end > len(document.text) or mention.start < 0:
                problems.append(f"{cluster.cluster_id}: span out of bounds")
                continue
            actual = document.text[mention.start : mention.end]
            if actual != mention.text:
                problems.append(
                    f"{cluster.cluster_id}: stored {mention.text!r} != actual {actual!r}"
                )
    return problems
