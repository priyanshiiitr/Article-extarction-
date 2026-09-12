"""Pairwise features for entity resolution.

Each feature answers one question about a candidate pair and returns a number in
[0, 1]. The scorer combines them. Keeping feature COMPUTATION separate from
feature WEIGHTING matters: it lets Phase 10 learn the weights from labelled data
without touching any of this code, and it makes each signal individually
testable.

THE TWO KINDS OF SIGNAL
-----------------------
  SOFT features are weighted and summed. None of them individually decides.
  HARD constraints veto a match outright regardless of every other score.

Hard constraints are unusual in a scoring system and need justification. We use
exactly one, and it is the single most valuable rule in this module:

    CONFLICTING GIVEN NAMES.  "narendra modi" vs "lalit modi" share a surname
    and differ in the given name. No amount of shared context makes those the
    same human being. A weighted score can be dragged over the threshold by
    five agreeing soft features; a veto cannot.

This is the right shape for the problem because a FALSE MERGE is much worse
than a false split (see the module docstring in resolver.py). Hard-coding the
few things we are CERTAIN about, and scoring everything else, buys precision
where precision matters most.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from rapidfuzz import fuzz

from src.entity_resolution.embeddings import cosine_similarity
from src.entity_resolution.normalize import name_tokens, normalize_name, surname
from src.entity_resolution.profiles import EntityProfile


@dataclass
class PairFeatures:
    """Feature vector for one candidate pair, all values in [0, 1]."""

    # ONE graded name signal rather than separate exact/subset/fuzzy features.
    #
    # Having them separate was a calibration bug: "Modi" vs "Narendra Modi"
    # scored 0 on name_exact AND 0 on a strict-subset test, so a pair that
    # agreed on name, role and country still landed at 0.534 -- below the match
    # threshold. Exact, subset and fuzzy are three strengths of the SAME
    # underlying question ("are these names compatible?"), so they belong on
    # one axis. Splitting one signal across several weighted features
    # double-penalises the cases that score low on only some of them.
    name_compatibility: float = 0.0
    alias_overlap: float = 0.0
    role_similarity: float = 0.0
    country_similarity: float = 0.0
    org_similarity: float = 0.0
    context_similarity: float = 0.0
    temporal_proximity: float = 0.0

    # Hard constraints -- any True forbids the match outright.
    type_mismatch: bool = False
    given_name_conflict: bool = False
    country_conflict: bool = False

    # Diagnostics, so a decision can be explained to a human reviewer.
    notes: list[str] = field(default_factory=list)

    @property
    def vetoed(self) -> bool:
        return self.type_mismatch or self.given_name_conflict or self.country_conflict

    def veto_reason(self) -> str:
        if self.type_mismatch:
            return "entity types differ"
        if self.given_name_conflict:
            return "different given names with the same surname"
        if self.country_conflict:
            return "represent different countries"
        return ""


def _jaccard(a: set[str], b: set[str]) -> float:
    """Set overlap: |intersection| / |union|.

    Returns 0.0 when either side is EMPTY, which is a deliberate choice worth
    stating: an absent attribute is not evidence of agreement. Most documents
    do not state a person's country, so treating "both unknown" as a match
    would make every pair look similar. Absence must be neutral, and in a
    weighted sum the neutral value is 0 -- it simply contributes nothing.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_initial_of(short: str, long: str) -> bool:
    """True when "n" abbreviates "narendra"."""
    return len(short) == 1 and long.startswith(short)


def given_name_conflict(left: str, right: str) -> bool:
    """THE decisive rule for same-surname different-person pairs.

    Both names must be multi-token and share a surname; then the given names
    are compared. Initials are treated as compatible ("N. Modi" may be Narendra
    Modi), but two different spelled-out given names are a conflict.

    Crucially this does NOT fire for "modi" vs "narendra modi": a bare surname
    is compatible with any given name, so that pair stays scoreable.
    """
    left_tokens, right_tokens = name_tokens(left), name_tokens(right)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    if surname(left) != surname(right):
        return False

    left_given, right_given = left_tokens[0], right_tokens[0]
    if left_given == right_given:
        return False
    if _is_initial_of(left_given, right_given) or _is_initial_of(right_given, left_given):
        return False
    return True


# Graded strengths of name compatibility. Ordered, and the gaps matter:
# an exact match is meaningfully stronger than a surname-only reference, which
# is in turn much stronger than mere string similarity.
NAME_EXACT = 1.0
NAME_SUBSET = 0.80   # "modi" inside "narendra modi" -- the news-text pattern
NAME_FUZZY_CAP = 0.70  # ceiling for pairs that only look similar


def name_compatibility(left: EntityProfile, right: EntityProfile) -> float:
    """How compatible are these two names? One graded score in [0, 1].

    Three tiers:
      1.00  normalised strings identical.
      0.80  one name's tokens contain the other's -- "modi" inside "narendra
            modi". This is the surname-only reference that dominates news
            prose, and plain fuzzy ratio scores it poorly (47/100).
      <=0.70 otherwise, scaled by token_set_ratio, which ignores word order so
            "Modi Narendra" still matches "Narendra Modi".

    The cap on the fuzzy tier is deliberate. "Sergey Lavrov" vs "Sergei
    Ryabkov" scores 65/100 on pure string similarity; capping that tier stops
    coincidental spelling similarity from reaching the strength of a real
    containment relationship.
    """
    a, b = left.normalized, right.normalized
    if not a or not b:
        return 0.0

    if a == b:
        return NAME_EXACT

    tokens_a, tokens_b = set(name_tokens(a)), set(name_tokens(b))
    if tokens_a <= tokens_b or tokens_b <= tokens_a:
        return NAME_SUBSET

    # ACRONYM MATCH. Blocking already pairs "BCCI" with "Board of Control for
    # Cricket in India" via an acronym key, but string similarity between them
    # is near zero, so the scorer rejected the pair and evaluation showed it as
    # a false split. An acronym is a genuine equivalence, not a coincidence, so
    # it deserves the same strength as containment.
    #
    # Guarded by a minimum length: 2-letter acronyms collide far too easily to
    # be evidence of anything on their own.
    if _acronym_match(a, b):
        return NAME_SUBSET

    return min(NAME_FUZZY_CAP, fuzz.token_set_ratio(a, b) / 100.0)


MIN_ACRONYM_LENGTH = 3


def _acronym_match(a: str, b: str) -> bool:
    """True when one name is the acronym of the other."""
    for short, long in ((a, b), (b, a)):
        short_tokens, long_tokens = name_tokens(short), name_tokens(long)
        if len(short_tokens) != 1 or len(long_tokens) < 2:
            continue
        if len(short_tokens[0]) < MIN_ACRONYM_LENGTH:
            continue
        if short_tokens[0] == "".join(token[0] for token in long_tokens):
            return True
    return False


def alias_overlap(left: EntityProfile, right: EntityProfile) -> float:
    """Overlap of the surface forms each document used.

    This is ALIAS MATCHING: coreference gave each document a set of surface
    forms for the entity, and a shared alias is direct evidence. "Mr. Modi"
    appearing in both documents is stronger than fuzzy similarity, because it
    is an observed equivalence rather than an inferred one.
    """
    normalise = lambda profile: {  # noqa: E731
        normalize_name(s, profile.entity_type) for s in profile.surfaces
    }
    return _jaccard(normalise(left), normalise(right))


def temporal_proximity(a: datetime, b: datetime, half_life_days: float = 90.0) -> float:
    """Exponential decay on publication-date distance.

    Two articles from the same week are more likely to be about the same person
    than two five years apart -- which is exactly the Narendra/Lalit Modi case
    (Oct 2024 vs Aug 2019).

    WEAK BY DESIGN, and this is important: people appear in the news across
    decades, so a large gap is only mild evidence against a match. It gets a
    small weight, and it is never a veto. Using time as a hard rule would split
    every long-running public figure into per-year entities.
    """
    gap_days = abs((a - b).total_seconds()) / 86400.0
    return float(np.exp(-gap_days / half_life_days))


def compute_features(
    left: EntityProfile,
    right: EntityProfile,
    context_similarity: float = 0.0,
) -> PairFeatures:
    """Compute all features for one candidate pair.

    ``context_similarity`` is passed IN rather than computed here, because it
    must be calibrated against the whole batch of candidate pairs (see
    ``calibrate_context_scores``) and a single pair has no distribution to
    calibrate against.
    """
    features = PairFeatures(context_similarity=context_similarity)

    if left.entity_type != right.entity_type:
        features.type_mismatch = True
        features.notes.append(f"types {left.entity_type} vs {right.entity_type}")
        return features

    features.name_compatibility = name_compatibility(left, right)
    features.alias_overlap = alias_overlap(left, right)

    if given_name_conflict(left.normalized, right.normalized):
        features.given_name_conflict = True
        features.notes.append(
            f"given names differ: {left.normalized!r} vs {right.normalized!r}"
        )

    # Two people who demonstrably represent different countries are different
    # people. Note the asymmetry with roles: a person can hold several roles
    # over time, but "represents" in our schema is a current-state claim.
    if left.countries and right.countries and not (left.countries & right.countries):
        features.country_conflict = True
        features.notes.append(
            f"countries differ: {sorted(left.countries)} vs {sorted(right.countries)}"
        )

    features.role_similarity = _jaccard(left.roles, right.roles)
    features.country_similarity = _jaccard(left.countries, right.countries)
    features.org_similarity = _jaccard(left.orgs, right.orgs)
    features.temporal_proximity = temporal_proximity(left.published_at, right.published_at)

    return features


def calibrate_context_scores(raw_scores: list[float]) -> list[float]:
    """Convert raw cosines into a usable [0, 1] feature by RANK.

    WHY THIS EXISTS -- an empirical finding, not a theoretical nicety.
    Measured on our corpus, raw rescaled cosines between article contexts were:

        Modi-summit vs Modi-AI     0.925   (same person)
        Modi-summit vs Lalit Modi  0.892   (different person)
        Modi-summit vs Putin       0.922   (different person)

    Everything lands in a narrow high band, because a general-purpose encoder
    finds all news prose broadly similar. As an ABSOLUTE feature weighted at
    0.20 this contributes ~0.006 of separation -- effectively nothing.

    The signal is present but only RELATIVELY. So we replace each raw score
    with its percentile rank among the candidate pairs: the most similar pair
    scores 1.0, the least 0.0, and the feature becomes discriminative.

    THE COST, stated plainly: the feature is now relative to the batch, so the
    same pair can score differently in a different corpus. A production system
    calibrates against a FIXED reference distribution computed once over a
    large sample, so scores stay comparable across runs. We note that rather
    than pretending batch-relative scoring is free.
    """
    if not raw_scores:
        return []
    if len(raw_scores) == 1:
        return [0.5]

    order = np.argsort(np.argsort(np.asarray(raw_scores)))
    return (order / (len(raw_scores) - 1)).astype(float).tolist()


def context_similarity_for(
    left_vector: np.ndarray | None, right_vector: np.ndarray | None
) -> float:
    if left_vector is None or right_vector is None:
        return 0.0
    return cosine_similarity(left_vector, right_vector)
