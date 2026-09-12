"""Relation-extractor interface, the shared Relation builder, and argument
resolution -- the point where Phase 4's coreference output is finally consumed.

ARGUMENT RESOLUTION IS THE INTERESTING PART
-------------------------------------------
A parser hands us token spans. We need MENTION IDS. Three things have to happen:

  1. Map a token span back to an entity mention by character overlap.
  2. If the span is a pronoun ("He"), find its coreference cluster and use the
     cluster's representative mention instead.
  3. Record whether step 2 fired, because a coref-resolved argument carries the
     coreference model's error on top of the parser's, and should therefore be
     trusted less.

Step 2 is the entire payoff of Phase 4. Without it, "He described the
partnership as special" yields the useless triple (He, described, partnership).
With it, the subject is Narendra Modi.

Note what we still do NOT do: rewrite the text. We look up an ID and read a
representative from a cluster. Document.text is untouched and every offset in
the system stays valid.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from src.logging_utils import get_logger
from src.schemas import Document, Mention, Relation, make_relation_id

logger = get_logger(__name__)


@runtime_checkable
class RelationExtractor(Protocol):
    """Anything that can extract relations from an annotated document."""

    name: str

    def extract(self, document: Document) -> list[Relation]:
        ...

    def extract_batch(self, documents: Sequence[Document]) -> list[list[Relation]]:
        ...


class ResolvedArgument:
    """A relation argument after mention lookup and coreference resolution."""

    __slots__ = ("mention", "text", "label", "via_coref")

    def __init__(self, mention: Mention, text: str, label: str, via_coref: bool) -> None:
        self.mention = mention
        self.text = text
        self.label = label
        self.via_coref = via_coref


def mention_at(document: Document, start: int, end: int) -> Mention | None:
    """Find the entity mention overlapping a character span.

    Prefers the mention with the greatest overlap. Parser token spans and NER
    mention spans rarely align exactly -- the parser gives us the single token
    "Modi" while NER found "Narendra Modi" -- so exact equality would find
    almost nothing.
    """
    best: Mention | None = None
    best_overlap = 0
    for mention in document.mentions:
        overlap = min(end, mention.end) - max(start, mention.start)
        if overlap > best_overlap:
            best, best_overlap = mention, overlap
    return best


def resolve_argument(
    document: Document,
    start: int,
    end: int,
    allow_coref: bool = True,
) -> ResolvedArgument | None:
    """Turn a character span into a usable relation argument.

    Returns None when the span is not an entity and cannot be resolved to one,
    which is the correct outcome for "reporters" or "the airport" -- spans that
    are grammatically arguments but not entities we track.
    """
    # Consult coreference FIRST, for pronominal AND nominal mentions.
    #
    # Restricting this to pronouns was a real bug: "The two leaders discussed
    # energy cooperation" produced nothing, because "The two leaders" is a
    # NOMINAL, not a pronoun, and is not an entity mention either -- so the
    # whole sentence contributed no facts. Nominal anaphora is common in news
    # prose ("the lender", "the Finance Minister", "the two leaders") and
    # skipping it throws away a large share of the available relations.
    #
    # NAMED coref mentions are deliberately NOT redirected: "Narendra Modi" IS
    # the entity, so replacing it with a cluster representative would be a
    # pointless indirection that could only introduce error.
    if allow_coref:
        covering = _coref_mention_covering(document, start, end)
        if covering is not None and covering[1].form in {"PRONOMINAL", "NOMINAL"}:
            resolved = _representative_argument(document, covering[0])
            if resolved is not None:
                return resolved

    mention = mention_at(document, start, end)
    if mention is None:
        return None

    surface = document.text[start:end]
    if not _is_pronoun(surface):
        return ResolvedArgument(mention, mention.text, mention.label, via_coref=False)

    # A pronoun that coreference could not resolve is not a usable argument:
    # "He" names nothing on its own.
    cluster = _cluster_covering(document, start, end)
    if cluster is None:
        return None

    representative = cluster.representative
    if representative is None:
        return None

    # Prefer the entity mention the representative is linked to, so the
    # argument is a real Mention with a real label rather than a bare string.
    if representative.ner_mention_id:
        for candidate in document.mentions:
            if candidate.mention_id == representative.ner_mention_id:
                return ResolvedArgument(
                    candidate, candidate.text, candidate.label, via_coref=True
                )

    return ResolvedArgument(mention, representative.text, mention.label, via_coref=True)


def _is_pronoun(surface: str) -> bool:
    from src.schemas import PRONOUNS

    return surface.strip().lower() in PRONOUNS


def _cluster_covering(document: Document, start: int, end: int):
    """Find the coreference cluster containing this exact span."""
    found = _coref_mention_covering(document, start, end)
    return found[0] if found else None


def _coref_mention_covering(document: Document, start: int, end: int):
    """Return (cluster, coref_mention) for the SMALLEST span covering [start,end).

    Smallest matters: a token can sit inside both "The Indian Prime Minister"
    and some larger enclosing span. The tightest one is the mention that
    actually refers, so it is the one whose form we should trust.
    """
    best = None
    best_width = None
    for cluster in document.coref_clusters:
        for mention in cluster.mentions:
            if mention.start <= start and end <= mention.end:
                width = mention.end - mention.start
                if best_width is None or width < best_width:
                    best, best_width = (cluster, mention), width
    return best


def _representative_arguments(document: Document, cluster) -> list[ResolvedArgument]:
    """Build arguments from a cluster's representative mention.

    Returns a LIST because of coordinate (plural) anaphora. The coreference
    model links "The two leaders" to the span "Modi and Putin", and collapsing
    that to a single argument silently drops half of every fact in the
    sentence: "The two leaders discussed energy" is true of BOTH of them.

    So when the representative span covers several entity mentions of the same
    type, we return all of them and the caller emits one relation per member.
    That is the correct reading of a plural antecedent.
    """
    representative = cluster.representative
    if representative is None:
        return []

    # Entity mentions falling inside the representative's span.
    inside = [
        m
        for m in document.mentions
        if representative.start <= m.start and m.end <= representative.end
    ]
    people = [m for m in inside if m.label == "PERSON"]
    if len(people) > 1:
        return [ResolvedArgument(m, m.text, m.label, via_coref=True) for m in people]

    if not representative.ner_mention_id:
        return []
    for candidate in document.mentions:
        if candidate.mention_id == representative.ner_mention_id:
            return [ResolvedArgument(candidate, candidate.text, candidate.label, via_coref=True)]
    return []


def _representative_argument(document: Document, cluster) -> ResolvedArgument | None:
    """Single-argument convenience wrapper, for callers that want one answer."""
    arguments = _representative_arguments(document, cluster)
    return arguments[0] if arguments else None


def resolve_arguments(
    document: Document,
    start: int,
    end: int,
    allow_coref: bool = True,
) -> list[ResolvedArgument]:
    """Like ``resolve_argument`` but returns EVERY referent of the span.

    Used by the dependency extractor so a plural antecedent produces one
    relation per member rather than an arbitrary single one.
    """
    if allow_coref:
        covering = _coref_mention_covering(document, start, end)
        if covering is not None and covering[1].form in {"PRONOMINAL", "NOMINAL"}:
            resolved = _representative_arguments(document, covering[0])
            if resolved:
                return resolved

    single = resolve_argument(document, start, end, allow_coref=allow_coref)
    return [single] if single is not None else []


def build_relation(
    document: Document,
    subject: ResolvedArgument,
    predicate: str,
    obj: ResolvedArgument,
    confidence: float,
    extractor: str,
    trigger: str = "",
    evidence_start: int = -1,
    evidence_end: int = -1,
) -> Relation | None:
    """Construct a validated Relation. The single entry point for extractors.

    Returns None for degenerate relations -- specifically, an entity related to
    ITSELF. That arises constantly from coreference ("Modi said he would go"
    resolves both arguments to Modi) and a self-loop is never a useful fact.
    """
    if subject.mention.mention_id == obj.mention.mention_id:
        return None

    sentence = document.sentence_containing(subject.mention.start)
    if evidence_start < 0 and sentence is not None:
        evidence_start, evidence_end = sentence.start, sentence.end

    return Relation(
        relation_id=make_relation_id(
            document.article_id, subject.mention.mention_id, predicate, obj.mention.mention_id
        ),
        article_id=document.article_id,
        subject_mention_id=subject.mention.mention_id,
        subject_text=subject.text,
        subject_label=subject.label,
        predicate=predicate,
        object_mention_id=obj.mention.mention_id,
        object_text=obj.text,
        object_label=obj.label,
        sentence_index=sentence.index if sentence else -1,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
        confidence=max(0.0, min(1.0, confidence)),
        extractor=extractor,
        trigger=trigger,
        subject_via_coref=subject.via_coref,
        object_via_coref=obj.via_coref,
    )


# Type constraints per relation. A relation whose arguments have the wrong
# types is almost always an extraction error, so this table is a cheap, highly
# effective precision filter -- it rejects "(Kazan) -[discussed]-> (AI)" before
# it reaches the graph.
#
# This is the relation-extraction equivalent of the NER trust matrix: an
# explicit, inspectable policy rather than conditions scattered through code.
ARGUMENT_TYPES: dict[str, tuple[set[str], set[str]]] = {
    "holds_position": ({"PERSON"}, {"ROLE"}),
    "represents": ({"PERSON"}, {"COUNTRY"}),
    "works_for": ({"PERSON"}, {"ORG"}),
    "attended": ({"PERSON"}, {"EVENT"}),
    "met": ({"PERSON"}, {"PERSON", "ORG"}),
    "discussed": ({"PERSON", "ORG"}, {"TOPIC", "EVENT"}),
    "said": ({"PERSON", "ORG"}, {"TOPIC", "EVENT"}),
    "located_in": ({"EVENT", "ORG"}, {"LOCATION", "COUNTRY"}),
    "signed": ({"PERSON", "ORG"}, {"EVENT"}),
}


def types_are_valid(predicate: str, subject_label: str, object_label: str) -> bool:
    allowed = ARGUMENT_TYPES.get(predicate)
    if allowed is None:
        return True
    subject_ok, object_ok = allowed
    return subject_label in subject_ok and object_label in object_ok


def verify_relations(document: Document) -> list[str]:
    """Check that every relation still points at mentions that exist."""
    problems: list[str] = []
    mention_ids = {m.mention_id for m in document.mentions}
    for relation in document.relations:
        if relation.subject_mention_id not in mention_ids:
            problems.append(f"{relation.relation_id}: subject mention missing")
        if relation.object_mention_id not in mention_ids:
            problems.append(f"{relation.relation_id}: object mention missing")
        if relation.evidence_start >= 0 and relation.evidence_end > len(document.text):
            problems.append(f"{relation.relation_id}: evidence span out of bounds")
    return problems
