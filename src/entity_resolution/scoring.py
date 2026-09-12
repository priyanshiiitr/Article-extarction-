"""Combine pairwise features into a match score, and decide.

THE WEIGHTS BELOW ARE A STARTING POINT, NOT AN ANSWER
-----------------------------------------------------
They are hand-set from reasoning about which signals are trustworthy on this
corpus. That is the honest description. They are NOT optimal, and nobody should
present hand-set weights as if they were tuned.

How you would actually set them, in increasing order of rigour:

  1. Label a few hundred candidate pairs as match / non-match.
  2. Fit LOGISTIC REGRESSION on the feature vectors. The learned coefficients
     ARE the weights, and they come with significance estimates telling you
     which features are pulling their weight at all.
  3. Choose the threshold from the precision/recall curve at the operating
     point your application needs -- NOT by picking a round number. For entity
     resolution that usually means targeting high precision, because a false
     merge is more damaging than a false split.
  4. Re-check on a held-out set, and monitor drift as the corpus changes.

Step 2 is the classical Fellegi-Sunter framework for record linkage, dressed in
modern clothes. Phase 10 builds the labelled data that makes it possible.

WHY A LINEAR WEIGHTED SUM AT ALL
--------------------------------
It is INTERPRETABLE. When a reviewer asks "why did you merge these?", a linear
model answers directly: "name subset 1.0 x 0.20, alias overlap 0.5 x 0.15...".
A gradient-boosted tree would likely score a few points better on F1 and would
be far harder to defend to the person auditing a wrong merge. For a decision
with asymmetric, hard-to-reverse costs, explainability is worth real accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.entity_resolution.features import PairFeatures

# Feature weights. They sum to 1.0 so the total score is directly readable as a
# [0, 1] confidence rather than an arbitrary magnitude.
#
# The reasoning behind the relative sizes:
#   name_compatibility  largest single weight, but deliberately well under 0.5
#                 so that name alone can never reach the match threshold. A
#                 perfect name match still needs corroboration from SOMETHING.
#                 This is the main structural defence against false merges on
#                 common names.
#   alias_overlap observed equivalence beats inferred similarity: two documents
#                 that both used "Mr. Modi" is evidence, not a guess.
#   role/org      precise when present, absent most of the time.
#   context       calibrated by rank; genuinely helpful but topical, not identity.
#   country       small, because a shared country is very weak evidence (a
#                 million Indians) while a CONFLICTING country is handled as a
#                 veto, not as a low score.
#   temporal      smallest; people recur in the news across decades.
WEIGHTS: dict[str, float] = {
    "name_compatibility": 0.35,
    "alias_overlap": 0.18,
    "context_similarity": 0.12,
    "role_similarity": 0.12,
    "org_similarity": 0.10,
    "country_similarity": 0.08,
    "temporal_proximity": 0.05,
}

# Decision bands. Three outcomes, not two -- "I don't know" is a valid and
# valuable answer for a decision this expensive to get wrong.
#
# The two thresholds are set from OPPOSITE directions, and deliberately so:
#   MATCH is conservative, because an automatic merge is hard to undo.
#   REVIEW is generous, because the cost of a review item is a few seconds of
#   human attention, while the cost of dismissing a real match is a permanent
#   false split that nobody ever notices.
# 0.35 is the floor at which a bare surname ("Modi") paired with a full name
# sharing it ("Narendra Modi") reaches the queue even with no corroborating
# attributes -- a case that is genuinely plausible and genuinely uncertain.
MATCH_THRESHOLD = 0.55
REVIEW_THRESHOLD = 0.35

Decision = Literal["match", "review", "no_match"]


@dataclass
class ScoredPair:
    left_id: str
    right_id: str
    score: float
    decision: Decision
    features: PairFeatures
    reason: str = ""

    def explain(self) -> str:
        """Human-readable breakdown, for the review queue and for debugging.

        Every automated merge decision should be explainable to the person who
        has to defend it. This is that explanation.
        """
        if self.features.vetoed:
            return f"VETO: {self.features.veto_reason()}"
        parts = []
        for name, weight in sorted(WEIGHTS.items(), key=lambda kv: -kv[1]):
            value = getattr(self.features, name, 0.0)
            if value > 0:
                parts.append(f"{name}={value:.2f}x{weight:.2f}")
        return " + ".join(parts) or "no positive evidence"


def score_pair(features: PairFeatures) -> float:
    """Weighted sum of the soft features. Vetoes are handled by the caller."""
    return sum(weight * getattr(features, name, 0.0) for name, weight in WEIGHTS.items())


def decide(
    left_id: str,
    right_id: str,
    features: PairFeatures,
    match_threshold: float = MATCH_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
) -> ScoredPair:
    """Turn features into a decision.

    A veto short-circuits to no_match with score 0, regardless of how strong
    the soft evidence is. That asymmetry is intentional: we would rather split
    one entity into two than fuse two real people into one.
    """
    if features.vetoed:
        return ScoredPair(
            left_id=left_id,
            right_id=right_id,
            score=0.0,
            decision="no_match",
            features=features,
            reason=features.veto_reason(),
        )

    score = score_pair(features)
    if score >= match_threshold:
        decision: Decision = "match"
    elif score >= review_threshold:
        decision = "review"
    else:
        decision = "no_match"

    return ScoredPair(
        left_id=left_id,
        right_id=right_id,
        score=score,
        decision=decision,
        features=features,
    )
