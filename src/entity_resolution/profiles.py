"""Build the records that entity resolution actually compares.

THE UNIT OF RESOLUTION IS NOT A MENTION
---------------------------------------
The obvious design is to resolve individual mentions against each other. That
is wrong, for two reasons:

  1. WASTEFUL. One article mentions Modi six times. Resolving all six
     separately does six times the work to reach the same answer, and Phase 4
     already told us they are the same person.
  2. EVIDENCE-POOR. A bare mention "Modi" carries almost nothing. But the
     DOCUMENT knows he is a Prime Minister, represents India, attended the
     BRICS Summit, and appeared on 22 October 2024. That is the evidence that
     separates him from Lalit Modi -- and it exists only at document level.

So the unit is a LOCAL ENTITY: everything one document says about one entity.
Coreference clusters give us the grouping for free, and relations give us the
attributes. This is where Phases 4 and 5 are consumed.

    mentions  --(coreference)-->  local entity  --(relations)-->  profile
                                        |
                                        +--(entity resolution)--> canonical entity

Collapsing mentions into local entities first also improves PRECISION, not just
speed: "Modi" alone is ambiguous, but "Modi, Prime Minister, representing
India, at the BRICS Summit" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from src.entity_resolution.normalize import normalize_name
from src.logging_utils import get_logger
from src.schemas import Document, Mention

logger = get_logger(__name__)

# Types we resolve. LOCATION and COUNTRY are deliberately excluded: they are
# already normalised by the gazetteer to canonical names, so "India" is always
# "India" and there is nothing to resolve. Resolving them anyway would add cost
# and risk with no benefit.
RESOLVABLE_TYPES = {"PERSON", "ORG"}

# Upstream NER false positives that must not become entities. Our corpus
# produced "Rs" (rupees) as a PERSON from "Rs. 4,500 crore", and a currency
# abbreviation turning into a person in the knowledge graph is exactly the kind
# of quiet nonsense that erodes trust in the whole system.
#
# Two guards, both deliberately conservative:
#   * a stoplist of known non-names;
#   * a minimum length, since a one- or two-character "name" carries no
#     identifying information and cannot be resolved responsibly anyway.
# This filters at the ER boundary rather than deleting the mention, so the
# evidence stays in the document for inspection and for Phase 10 to score.
NON_ENTITY_TOKENS = {
    "rs", "usd", "eur", "inr", "gbp", "crore", "lakh", "bn", "mn",
    "ceo", "cfo", "gdp", "ai",
}
MIN_NAME_CHARS = 3


def is_resolvable(normalized: str) -> bool:
    """Whether a normalised name is worth resolving at all."""
    if len(normalized) < MIN_NAME_CHARS:
        return False
    return normalized not in NON_ENTITY_TOKENS


@dataclass
class EntityProfile:
    """Everything one document knows about one entity.

    This is the record that blocking indexes and scoring compares.
    """

    profile_id: str
    article_id: str
    source: str
    published_at: datetime
    entity_type: str

    # The best surface form seen in this document -- used for display and as
    # the string blocking and fuzzy matching operate on.
    canonical_surface: str
    normalized: str
    # Every surface form for this entity in this document. These become the
    # entity's ALIASES, and are what makes alias matching possible later.
    surfaces: set[str] = field(default_factory=set)

    # Attributes harvested from Phase 5 relations. These are the features that
    # separate two people who share a surname.
    roles: set[str] = field(default_factory=set)
    countries: set[str] = field(default_factory=set)
    orgs: set[str] = field(default_factory=set)
    topics: set[str] = field(default_factory=set)
    events: set[str] = field(default_factory=set)

    # Sentences this entity appears in, for embedding-based context similarity.
    context: str = ""

    mention_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"{self.canonical_surface} ({self.entity_type})"]
        if self.roles:
            bits.append(f"roles={sorted(self.roles)}")
        if self.countries:
            bits.append(f"countries={sorted(self.countries)}")
        if self.orgs:
            bits.append(f"orgs={sorted(self.orgs)}")
        return " ".join(bits)


def _best_surface(mentions: Sequence[Mention]) -> str:
    """Pick the most informative surface form among a local entity's mentions.

    Longest wins: "Narendra Modi" carries strictly more identifying information
    than "Modi", and a full name is dramatically easier to resolve correctly
    than a bare surname. Ties break on earliest position for determinism --
    without that, profile IDs could differ between runs on identical input.
    """
    return max(mentions, key=lambda m: (len(m.text), -m.start)).text


def build_profiles(document: Document) -> list[EntityProfile]:
    """Build one profile per local entity in a document."""
    resolvable = [m for m in document.mentions if m.label in RESOLVABLE_TYPES]
    if not resolvable:
        return []

    # Group by coreference cluster. A mention with no cluster is its own group,
    # keyed on its mention_id so it cannot collide with a cluster id.
    groups: dict[str, list[Mention]] = {}
    for mention in resolvable:
        key = mention.coref_cluster_id or f"solo:{mention.mention_id}"
        groups.setdefault(key, []).append(mention)

    # Index relations by subject mention so attribute lookup is O(1) per
    # mention instead of a scan of every relation per mention.
    by_subject: dict[str, list] = {}
    for relation in document.relations:
        by_subject.setdefault(relation.subject_mention_id, []).append(relation)

    profiles: list[EntityProfile] = []
    for index, (_, mentions) in enumerate(sorted(groups.items())):
        mentions.sort(key=lambda m: m.start)
        canonical = _best_surface(mentions)
        entity_type = mentions[0].label
        normalized = normalize_name(canonical, entity_type)

        if not is_resolvable(normalized):
            logger.debug(
                "%s: skipping unresolvable name %r", document.article_id, canonical
            )
            continue

        profile = EntityProfile(
            profile_id=f"{document.article_id}:p{index:03d}",
            article_id=document.article_id,
            source=document.source,
            published_at=document.published_at,
            entity_type=entity_type,
            canonical_surface=canonical,
            normalized=normalized,
            surfaces={m.text for m in mentions},
            mention_ids=[m.mention_id for m in mentions],
        )

        sentence_indices: set[int] = set()
        for mention in mentions:
            if mention.sentence_index >= 0:
                sentence_indices.add(mention.sentence_index)

            for relation in by_subject.get(mention.mention_id, []):
                if relation.predicate == "holds_position":
                    profile.roles.add(relation.object_text.lower())
                elif relation.predicate == "represents":
                    profile.countries.add(relation.object_text)
                elif relation.predicate == "works_for":
                    profile.orgs.add(relation.object_text)
                elif relation.predicate in {"discussed", "said"}:
                    profile.topics.add(relation.object_text.lower())
                elif relation.predicate in {"attended", "signed"}:
                    profile.events.add(relation.object_text)

        # Context = the sentences this entity actually appears in, not the whole
        # article. Using the whole article would make every entity in a document
        # look identical to every other, destroying the signal.
        parts = [
            document.text[s.start : s.end]
            for s in document.sentences
            if s.index in sentence_indices
        ]
        profile.context = " ".join(parts)[:1000]

        profiles.append(profile)

    return profiles


def attach_bare_surnames(profiles: list[EntityProfile]) -> list[EntityProfile]:
    """Attach bare-surname profiles to the full name in the SAME document.

    WHY THIS EXISTS -- it fixes a real false merge we observed.
    The Lalit Modi article contains both "Lalit Modi" and a bare "Modi" that
    coreference failed to link. Left as separate profiles, that bare "Modi"
    became a BRIDGE: the direct pair (Narendra Modi, Lalit Modi) was correctly
    vetoed, but

        Modi[Lalit's article]  <->  Modi[Narendra's article]   score 0.714

    matched on identical strings, and transitive closure then fused Narendra
    and Lalit Modi into one entity. A veto on the direct pair is worthless if an
    ambiguous third record can route around it.

    THE PRINCIPLE: resolve ambiguity with LOCAL evidence before global evidence.
    A bare surname inside a document that contains exactly ONE full name sharing
    that surname almost certainly refers to that person -- news writers
    introduce someone in full and then use the surname. This is the strongest
    signal available, and it costs one pass over each document.

    The "exactly one" condition is what makes it safe. A document mentioning
    both Narendra Modi and Lalit Modi is genuinely ambiguous, so we attach
    nothing and let scoring (and the review queue) handle it.
    """
    by_article: dict[str, list[EntityProfile]] = {}
    for profile in profiles:
        by_article.setdefault(profile.article_id, []).append(profile)

    absorbed: set[str] = set()

    for article_profiles in by_article.values():
        people = [p for p in article_profiles if p.entity_type == "PERSON"]
        full_names = [p for p in people if len(p.normalized.split()) > 1]
        bare_names = [p for p in people if len(p.normalized.split()) == 1]

        for bare in bare_names:
            candidates = [
                full for full in full_names if full.normalized.split()[-1] == bare.normalized
            ]
            if len(candidates) != 1:
                # Zero candidates: nothing to attach to.
                # Two or more: genuinely ambiguous inside this document, so
                # attaching would be a guess. Leave it to scoring.
                continue

            target = candidates[0]
            target.surfaces.update(bare.surfaces)
            target.roles.update(bare.roles)
            target.countries.update(bare.countries)
            target.orgs.update(bare.orgs)
            target.topics.update(bare.topics)
            target.events.update(bare.events)
            target.mention_ids.extend(bare.mention_ids)
            if bare.context and bare.context not in target.context:
                target.context = f"{target.context} {bare.context}".strip()[:1000]
            absorbed.add(bare.profile_id)
            logger.debug(
                "%s: attached bare %r to %r",
                bare.article_id,
                bare.canonical_surface,
                target.canonical_surface,
            )

    if absorbed:
        logger.info("Attached %d bare-surname profiles within their documents", len(absorbed))
    return [p for p in profiles if p.profile_id not in absorbed]


def build_all_profiles(documents: Sequence[Document]) -> list[EntityProfile]:
    profiles: list[EntityProfile] = []
    for document in documents:
        profiles.extend(build_profiles(document))
    profiles = attach_bare_surnames(profiles)
    logger.info(
        "Built %d entity profiles from %d documents", len(profiles), len(documents)
    )
    return profiles
