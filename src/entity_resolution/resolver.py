"""Phase 6 orchestration: profiles in, canonical entities out.

THE PIPELINE
------------
    1. NORMALISE   surface forms                     (normalize.py)
    2. PROFILE     group mentions into local entities (profiles.py)
    3. BLOCK       generate candidate pairs           (blocking.py)
    4. FEATURISE   compute pairwise signals           (features.py)
    5. SCORE       weighted sum + hard vetoes         (scoring.py)
    6. DECIDE      match / review / no_match
    7. CLUSTER     transitive closure over matches    (here)
    8. CANONICALISE build the entity records          (here)

WHY TRANSITIVE CLOSURE IS DANGEROUS HERE
----------------------------------------
Union-Find merges A-B and B-C into {A,B,C} even though A and C were never
scored. In coreference that was exactly right. In entity resolution it is a
known hazard, because it lets merges CHAIN:

    "Narendra Modi" ~ "Modi"        (score 0.72, correct)
    "Modi"          ~ "Lalit Modi"  (score 0.58, WRONG)
    => Narendra Modi and Lalit Modi are now the same entity.

One bad link poisons a whole cluster. This is sometimes called the "transitive
closure problem" or cluster drift. Defences, in order of cost:

  * HARD VETOES (what we use). "narendra modi" and "lalit modi" can never be
    linked directly, so the chain has no middle link to travel through --
    provided the ambiguous member joins the right side first.
  * A HIGH THRESHOLD, so weak links never enter the graph.
  * CLUSTER-LEVEL VALIDATION (what we also do): after forming a cluster, check
    that no two MEMBERS of it violate a veto. If they do, the cluster is
    unsound and we refuse it rather than shipping a false merge.
  * Correlational clustering / graph partitioning that optimises globally
    instead of greedily -- the principled fix, and much more expensive.

The cluster-level check is what makes this safe: transitive closure proposes,
veto validation disposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np

from src.config import Config, load_config
from src.entity_resolution.blocking import (
    blocking_report,
    build_blocks,
    candidate_pairs,
)
from src.entity_resolution.features import (
    calibrate_context_scores,
    compute_features,
    context_similarity_for,
    given_name_conflict,
)
from src.entity_resolution.profiles import EntityProfile
from src.entity_resolution.scoring import ScoredPair, decide
from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CanonicalEntity:
    """One real-world entity, assembled from profiles across many articles."""

    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    countries: set[str] = field(default_factory=set)
    orgs: set[str] = field(default_factory=set)
    topics: set[str] = field(default_factory=set)
    events: set[str] = field(default_factory=set)
    profile_ids: list[str] = field(default_factory=list)
    article_ids: list[str] = field(default_factory=list)
    mention_ids: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    # Mean pairwise score of the merges that formed this entity. Low values
    # mark entities assembled from weak evidence, which is what you review.
    merge_confidence: float = 1.0


@dataclass
class ResolutionStats:
    profiles: int = 0
    candidate_pairs: int = 0
    matches: int = 0
    reviews: int = 0
    no_matches: int = 0
    vetoed: int = 0
    entities: int = 0
    unsound_clusters_split: int = 0
    blocking: str = ""

    def summary(self) -> str:
        return (
            f"profiles={self.profiles} pairs={self.candidate_pairs} "
            f"match={self.matches} review={self.reviews} veto={self.vetoed} "
            f"entities={self.entities}"
        )


class _UnionFind:
    """Disjoint sets, as in Phase 4 -- but used far more cautiously here."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
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


def _cluster_is_sound(members: Sequence[EntityProfile]) -> tuple[bool, str]:
    """Reject a cluster whose own members contradict each other.

    Transitive closure can assemble a cluster containing two profiles that
    would never have been linked directly. Checking every pair INSIDE the
    finished cluster catches that chaining, which is the main way false merges
    happen in practice.
    """
    for i, left in enumerate(members):
        for right in members[i + 1 :]:
            if given_name_conflict(left.normalized, right.normalized):
                return False, f"{left.normalized!r} vs {right.normalized!r}"
            if left.countries and right.countries and not (left.countries & right.countries):
                return (
                    False,
                    f"countries {sorted(left.countries)} vs {sorted(right.countries)}",
                )
    return True, ""


def _canonical_name(members: Sequence[EntityProfile]) -> str:
    """Choose the entity's display name.

    Prefers the LONGEST normalised name, because it is the most complete form:
    "Narendra Modi" over "Modi". Among equal lengths, the most frequently
    occurring surface wins, then alphabetical order for determinism. Determinism
    matters because this name feeds the entity ID.
    """
    counts: dict[str, int] = {}
    for profile in members:
        for surface in profile.surfaces:
            counts[surface] = counts.get(surface, 0) + 1

    best = max(
        members,
        key=lambda p: (len(p.normalized.split()), len(p.normalized), p.normalized),
    )
    candidates = [s for s in best.surfaces if s]
    if not candidates:
        return best.canonical_surface
    return max(candidates, key=lambda s: (len(s.split()), counts.get(s, 0), s))


def resolve_entities(
    profiles: Sequence[EntityProfile],
    cfg: Config | None = None,
    embed: bool = True,
) -> tuple[list[CanonicalEntity], list[ScoredPair], ResolutionStats]:
    """Run entity resolution. Returns (entities, pairs_for_review, stats)."""
    cfg = cfg or load_config()
    stats = ResolutionStats(profiles=len(profiles))
    by_id = {p.profile_id: p for p in profiles}

    if not profiles:
        return [], [], stats

    # --- Step 3: blocking ------------------------------------------------
    records = [(p.profile_id, p.canonical_surface, p.entity_type) for p in profiles]
    blocks = build_blocks(records)
    pairs = sorted(candidate_pairs(blocks))
    stats.candidate_pairs = len(pairs)
    stats.blocking = blocking_report(records, pairs)
    logger.info("Blocking: %s", stats.blocking)

    # --- Context embeddings, computed once per profile -------------------
    vectors: dict[str, np.ndarray] = {}
    if embed and cfg.entity_resolution.use_embeddings and pairs:
        try:
            from src.entity_resolution.embeddings import ContextEmbedder

            embedder = ContextEmbedder(cfg.entity_resolution.embedding_model)
            ids = [p.profile_id for p in profiles]
            matrix = embedder.encode([p.context or p.canonical_surface for p in profiles])
            vectors = {pid: matrix[i] for i, pid in enumerate(ids)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embeddings unavailable (%s); continuing without them.", exc)

    raw_context = [
        context_similarity_for(vectors.get(left), vectors.get(right)) for left, right in pairs
    ]
    calibrated = calibrate_context_scores(raw_context)

    # --- Steps 4-6: featurise, score, decide -----------------------------
    scored: list[ScoredPair] = []
    for (left_id, right_id), context_score in zip(pairs, calibrated):
        features = compute_features(by_id[left_id], by_id[right_id], context_score)
        result = decide(
            left_id,
            right_id,
            features,
            cfg.entity_resolution.match_threshold,
            cfg.entity_resolution.review_threshold,
        )
        scored.append(result)
        if features.vetoed:
            stats.vetoed += 1
        if result.decision == "match":
            stats.matches += 1
        elif result.decision == "review":
            stats.reviews += 1
        else:
            stats.no_matches += 1

    # --- Step 7: cluster, then VALIDATE the clusters ---------------------
    union = _UnionFind()
    for profile in profiles:
        union.find(profile.profile_id)
    for result in scored:
        if result.decision == "match":
            union.union(result.left_id, result.right_id)

    score_lookup = {
        (r.left_id, r.right_id): r.score for r in scored if r.decision == "match"
    }

    clusters: list[list[EntityProfile]] = []
    for _, member_ids in sorted(union.groups().items()):
        members = [by_id[m] for m in sorted(member_ids)]
        sound, reason = _cluster_is_sound(members)
        if sound:
            clusters.append(members)
            continue

        # Unsound: transitive closure chained through an ambiguous profile.
        # Rather than ship a false merge, fall back to singletons for this
        # group. Conservative and lossy, and preferable to fusing two people.
        stats.unsound_clusters_split += 1
        logger.warning(
            "Splitting unsound cluster (%s): %s",
            reason,
            [m.canonical_surface for m in members],
        )
        clusters.extend([[member] for member in members])

    # --- Step 8: build canonical entities --------------------------------
    entities: list[CanonicalEntity] = []
    for index, members in enumerate(
        sorted(clusters, key=lambda ms: (-len(ms), ms[0].normalized))
    ):
        entity_type = members[0].entity_type
        name = _canonical_name(members)

        pair_scores = [
            score
            for (left, right), score in score_lookup.items()
            if left in {m.profile_id for m in members} and right in {m.profile_id for m in members}
        ]

        entity = CanonicalEntity(
            entity_id=f"{entity_type}_{index:05d}",
            canonical_name=name,
            entity_type=entity_type,
            merge_confidence=float(np.mean(pair_scores)) if pair_scores else 1.0,
        )
        for profile in members:
            entity.aliases.update(profile.surfaces)
            entity.roles.update(profile.roles)
            entity.countries.update(profile.countries)
            entity.orgs.update(profile.orgs)
            entity.topics.update(profile.topics)
            entity.events.update(profile.events)
            entity.profile_ids.append(profile.profile_id)
            entity.article_ids.append(profile.article_id)
            entity.mention_ids.extend(profile.mention_ids)
            if entity.first_seen is None or profile.published_at < entity.first_seen:
                entity.first_seen = profile.published_at
            if entity.last_seen is None or profile.published_at > entity.last_seen:
                entity.last_seen = profile.published_at

        entity.article_ids = sorted(set(entity.article_ids))
        entities.append(entity)

    stats.entities = len(entities)
    review_queue = [r for r in scored if r.decision == "review"]
    logger.info("Entity resolution complete: %s", stats.summary())
    return entities, review_queue, stats
