"""Evaluation metrics, each with the reasoning for using it.

PRECISION, RECALL, F1
---------------------
    precision = TP / (TP + FP)   "of what I returned, how much was right?"
    recall    = TP / (TP + FN)   "of what existed, how much did I find?"
    F1        = harmonic mean

Why the HARMONIC mean and not the arithmetic one: it punishes imbalance. A
system with precision 1.0 and recall 0.0 has arithmetic mean 0.5, which looks
respectable for a system that returns one correct answer and misses everything
else. Its F1 is 0.0. The harmonic mean refuses to be fooled by one strong half.

WHICH ONE MATTERS DEPENDS ENTIRELY ON THE TASK -- this is the point most
candidates miss:
  * NER feeding a search index -> favour RECALL. A missed entity is invisible;
    a spurious one is a bad search hit someone can ignore.
  * Entity resolution -> favour PRECISION. A false merge corrupts every fact
    about two people and is very hard to undo.
Reporting F1 alone hides which failure you chose.

COREFERENCE NEEDS SPECIAL METRICS
---------------------------------
You cannot score clusters with plain P/R, because the clusters have no
identity -- gold cluster #1 and predicted cluster #3 may be the same set. So
the field uses several link-based metrics, and reports the average:

  MUC        counts the LINKS needed to build each cluster (|cluster| - 1).
             Simple, but biased toward big clusters: merging everything into
             one giant cluster scores well, and singletons score nothing.
  B-CUBED    scores each MENTION individually by the overlap between its
             predicted and gold clusters, then averages. Punishes both
             over-merging and over-splitting, and unlike MUC it notices
             singletons.
  CEAF       finds an optimal 1:1 alignment of clusters. Fairest, most
             expensive (it is an assignment problem).

CoNLL F1, the standard headline number, is the mean of MUC, B-cubed and CEAF-e.
We implement MUC and B-cubed: two metrics with OPPOSITE biases already tell you
most of what a third would.

ENTITY RESOLUTION: PAIRWISE, PLUS THE TWO ERRORS THAT MATTER
------------------------------------------------------------
Pairwise P/R/F1 over "are these two mentions the same entity?" is the standard
measure. But the aggregate hides the distinction that actually matters
operationally, so we also report them separately:

  FALSE MERGE  two different real entities placed in one cluster. Precision
               error. CORRUPTING and hard to reverse.
  FALSE SPLIT  one real entity spread over several clusters. Recall error.
               Annoying, visible, and recoverable by merging later.

They are not symmetric and should never be reported as one number.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, Iterable, Sequence


@dataclass
class PRF:
    """Precision, recall, F1 and the raw counts behind them.

    The counts are kept because rates alone are misleading on small data:
    "precision 1.0" over two predictions is not the same claim as "precision
    1.0" over two thousand.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def row(self, name: str) -> str:
        return (
            f"{name:<26} P={self.precision:.3f}  R={self.recall:.3f}  "
            f"F1={self.f1:.3f}   (tp={self.true_positives} "
            f"fp={self.false_positives} fn={self.false_negatives})"
        )


def prf_from_sets(predicted: set[Hashable], gold: set[Hashable]) -> PRF:
    """Score two sets of items against each other."""
    return PRF(
        true_positives=len(predicted & gold),
        false_positives=len(predicted - gold),
        false_negatives=len(gold - predicted),
    )


# ---------------------------------------------------------------------------
# Coreference metrics
# ---------------------------------------------------------------------------


def _to_lookup(clusters: Sequence[Sequence[Hashable]]) -> dict[Hashable, int]:
    return {item: index for index, cluster in enumerate(clusters) for item in cluster}


def muc(predicted: Sequence[Sequence[Hashable]], gold: Sequence[Sequence[Hashable]]) -> PRF:
    """MUC: link-based coreference scoring.

    For each gold cluster of size n, the minimum number of links needed to
    build it is n-1. Recall asks how many of those links the prediction
    supplies. Precision is the same computation with the roles swapped.

    KNOWN BIAS: MUC ignores singletons entirely (a 1-element cluster needs 0
    links), and it rewards over-merging. Never report it alone.
    """

    def _score(source: Sequence[Sequence[Hashable]], target: Sequence[Sequence[Hashable]]) -> tuple[int, int]:
        lookup = _to_lookup(target)
        correct = total = 0
        for cluster in source:
            if len(cluster) < 2:
                continue
            total += len(cluster) - 1
            # Partition this cluster by which TARGET cluster each item is in.
            partitions: dict[Hashable, int] = {}
            for item in cluster:
                key = lookup.get(item, f"__missing_{item}")
                partitions[key] = partitions.get(key, 0) + 1
            correct += len(cluster) - len(partitions)
        return correct, total

    recall_correct, recall_total = _score(gold, predicted)
    precision_correct, precision_total = _score(predicted, gold)

    return PRF(
        true_positives=recall_correct,
        false_negatives=recall_total - recall_correct,
        false_positives=precision_total - precision_correct,
    )


def b_cubed(predicted: Sequence[Sequence[Hashable]], gold: Sequence[Sequence[Hashable]]) -> tuple[float, float, float]:
    """B-cubed: per-mention precision and recall, then averaged.

    For each mention m:
        precision(m) = |predicted(m) AND gold(m)| / |predicted(m)|
        recall(m)    = |predicted(m) AND gold(m)| / |gold(m)|

    Because every mention contributes equally, over-merging is punished
    (precision of each mention in the bloated cluster falls) and so is
    over-splitting (recall falls). That two-sided behaviour is why B-cubed is
    the metric to look at when MUC looks suspiciously good.

    Returns floats rather than a PRF because the scores are averages of ratios,
    not counts of discrete correct items -- there is no meaningful "tp" here.
    """
    gold_of = {item: set(cluster) for cluster in gold for item in cluster}
    predicted_of = {item: set(cluster) for cluster in predicted for item in cluster}

    mentions = set(gold_of) | set(predicted_of)
    if not mentions:
        return 0.0, 0.0, 0.0

    precision_total = recall_total = 0.0
    for mention in mentions:
        gold_cluster = gold_of.get(mention, {mention})
        predicted_cluster = predicted_of.get(mention, {mention})
        overlap = len(gold_cluster & predicted_cluster)
        precision_total += overlap / len(predicted_cluster)
        recall_total += overlap / len(gold_cluster)

    precision = precision_total / len(mentions)
    recall = recall_total / len(mentions)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Entity resolution metrics
# ---------------------------------------------------------------------------


@dataclass
class ClusteringReport:
    pairwise: PRF
    false_merges: int = 0
    false_splits: int = 0
    false_merge_examples: list[tuple[str, str]] = None  # type: ignore[assignment]
    false_split_examples: list[tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.false_merge_examples = self.false_merge_examples or []
        self.false_split_examples = self.false_split_examples or []


def _pairs(cluster: Iterable[Hashable]) -> set[tuple[Hashable, Hashable]]:
    return {tuple(sorted(pair, key=str)) for pair in combinations(sorted(cluster, key=str), 2)}


def evaluate_clustering(
    predicted: Sequence[Sequence[Hashable]],
    gold: Sequence[Sequence[Hashable]],
    max_examples: int = 5,
) -> ClusteringReport:
    """Pairwise clustering evaluation, splitting the two error types apart.

    A FALSE MERGE is a pair the system put together that gold keeps apart.
    A FALSE SPLIT is a pair gold puts together that the system keeps apart.

    Reporting them separately is the whole point. An aggregate F1 of 0.85 could
    be 15% missed merges (mildly annoying) or 15% wrong merges (a corrupted
    knowledge base), and those demand completely different responses.
    """
    predicted_pairs: set[tuple[Hashable, Hashable]] = set()
    for cluster in predicted:
        predicted_pairs |= _pairs(cluster)

    gold_pairs: set[tuple[Hashable, Hashable]] = set()
    for cluster in gold:
        gold_pairs |= _pairs(cluster)

    merges = predicted_pairs - gold_pairs
    splits = gold_pairs - predicted_pairs

    return ClusteringReport(
        pairwise=prf_from_sets(predicted_pairs, gold_pairs),
        false_merges=len(merges),
        false_splits=len(splits),
        false_merge_examples=[(str(a), str(b)) for a, b in sorted(merges, key=str)[:max_examples]],
        false_split_examples=[(str(a), str(b)) for a, b in sorted(splits, key=str)[:max_examples]],
    )
