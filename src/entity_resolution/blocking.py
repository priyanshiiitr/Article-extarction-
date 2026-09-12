"""Step 3 of entity resolution: candidate generation (blocking).

THE SCALING PROBLEM THIS SOLVES
-------------------------------
Comparing every record to every other is O(N^2):

     1,000 records ->        500,000 pairs  ->  5 seconds
   100,000 records ->  5,000,000,000 pairs  ->  14 hours
 1,000,000 records -> 500,000,000,000 pairs ->  58 days

Blocking computes a cheap KEY per record and only compares records that share a
key. Cost falls to roughly O(N x average block size). Records that share no key
are never compared -- which is the point, and also the risk.

THE CENTRAL TRADEOFF: PAIRS COMPARED vs RECALL
----------------------------------------------
  * Blocking key TOO STRICT (e.g. the full normalised name): few comparisons,
    but "Narendra Modi" and "Modi" never meet, so they can NEVER merge.
  * Blocking key TOO LOOSE (e.g. first letter): huge blocks, back to O(N^2)
    inside each block.

Blocking errors are UNRECOVERABLE: a pair never generated cannot be scored, so
it cannot be matched no matter how good the scorer is. Blocking sets the CEILING
on recall. That is why we use SEVERAL keys and require only ONE to be shared --
a strategy called multi-pass blocking (or, when done with cheap hashes, the
canopy/LSH family).

THE KEYS WE USE, AND WHY EACH EARNS ITS PLACE
---------------------------------------------
  surname        "narendra modi" and "modi" both -> "modi".  The workhorse for
                 news text, where a person is introduced in full and then
                 referred to by surname.
  initial+surname "n modi".  Tighter than surname alone; separates "N. Modi"
                 from "L. Modi" while still matching "Narendra Modi".
  full name      exact normalised string. Catches the easy cases instantly.
  phonetic       Soundex-style code, so spelling variants of transliterated
                 names ("Sergey"/"Sergei", "Mohamed"/"Muhammad") collide.
  acronym        ORG only: "New Development Bank" -> "ndb", so the abbreviation
                 used in body text meets the full name used on first mention.

WHAT PRODUCTION WOULD ADD
-------------------------
At tens of millions of records you move from exact-key blocking to approximate
nearest neighbours over embeddings (FAISS / HNSW / ScaNN), or MinHash LSH over
character n-grams. Same idea -- cheaply restrict the comparison set -- but the
key becomes a learned vector rather than a string.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Sequence

from src.entity_resolution.normalize import (
    initials,
    name_tokens,
    normalize_name,
    surname,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)

# A block bigger than this is almost certainly a degenerate key (a very common
# surname, or a stopword that slipped through). Comparing everything inside it
# would reintroduce the quadratic blow-up we are trying to avoid, so we warn.
MAX_BLOCK_WARN = 200

_VOWELS = re.compile(r"[aeiouyhw]")


def phonetic_code(token: str) -> str:
    """A compact Soundex-style code for a single token.

    Soundex maps letters to digit classes so that similarly-pronounced
    consonants collide, then drops vowels. "sergey" and "sergei" produce the
    same code, which is exactly what we want for transliterated names where
    spelling is unstable but pronunciation is not.

    This is a simplified implementation -- real systems use Double Metaphone,
    which handles non-English phonology far better. Good enough as a BLOCKING
    key, where a false collision only costs one extra comparison.
    """
    token = re.sub(r"[^a-z]", "", token.lower())
    if not token:
        return ""

    mapping = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        **dict.fromkeys("l", "4"),
        **dict.fromkeys("mn", "5"),
        **dict.fromkeys("r", "6"),
    }

    first = token[0]
    digits: list[str] = []
    previous = mapping.get(first, "")
    for char in token[1:]:
        code = mapping.get(char, "")
        # Adjacent identical codes collapse (double letters sound as one);
        # vowels reset the "previous" so "l-a-l" keeps both l's.
        if code and code != previous:
            digits.append(code)
        if not _VOWELS.match(char):
            previous = code
        else:
            previous = ""
    return (first + "".join(digits) + "000")[:4]


def acronym(normalized: str) -> str:
    """First letters of a multi-token name: "new development bank" -> "ndb"."""
    tokens = name_tokens(normalized)
    return "".join(t[0] for t in tokens) if len(tokens) > 1 else ""


def blocking_keys(surface: str, entity_type: str = "PERSON") -> set[str]:
    """Compute every blocking key for one name.

    Returns a SET because a record belongs to several blocks at once; sharing
    any one of them is enough to become a candidate pair.
    """
    normalized = normalize_name(surface, entity_type)
    if not normalized:
        return set()

    keys: set[str] = {f"full:{normalized}"}
    tokens = name_tokens(normalized)

    if entity_type == "PERSON":
        last = surname(normalized)
        if last:
            keys.add(f"sur:{last}")
            keys.add(f"phon:{phonetic_code(last)}")
        if len(tokens) > 1:
            keys.add(f"init:{tokens[0][0]}_{last}")
    else:
        # For organisations the whole name is the identity, and the acronym is
        # how body text refers back to it.
        for token in tokens:
            # Index on distinctive tokens so "New Development Bank" and "the
            # NDB, a development bank" can still meet.
            if len(token) > 3:
                keys.add(f"tok:{token}")
        code = acronym(normalized)
        if code:
            keys.add(f"acr:{code}")
        # An all-caps short surface form IS an acronym: "NDB" -> acr:ndb.
        if len(tokens) == 1 and 2 <= len(normalized) <= 6:
            keys.add(f"acr:{normalized}")

    return keys


def build_blocks(
    records: Sequence[tuple[str, str, str]],
) -> dict[str, list[str]]:
    """Index records into blocks.

    ``records`` is a sequence of (record_id, surface, entity_type).
    Returns {blocking_key: [record_id, ...]}.
    """
    blocks: dict[str, list[str]] = defaultdict(list)
    for record_id, surface, entity_type in records:
        for key in blocking_keys(surface, entity_type):
            blocks[key].append(record_id)

    oversized = {k: len(v) for k, v in blocks.items() if len(v) > MAX_BLOCK_WARN}
    if oversized:
        logger.warning(
            "Oversized blocks (quadratic risk inside them): %s",
            sorted(oversized.items(), key=lambda kv: -kv[1])[:5],
        )
    return dict(blocks)


def candidate_pairs(blocks: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Generate the unique record pairs worth scoring.

    Pairs are stored ordered (smaller id first) in a set, so a pair sharing
    three blocking keys is still scored exactly once. Without that
    deduplication, multi-pass blocking would multiply the work by the number of
    keys -- which is precisely the cost it exists to avoid.
    """
    pairs: set[tuple[str, str]] = set()
    for members in blocks.values():
        if len(members) < 2:
            continue
        ordered = sorted(set(members))
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                pairs.add((left, right))
    return pairs


def blocking_report(
    records: Sequence[tuple[str, str, str]],
    pairs: Iterable[tuple[str, str]],
) -> str:
    """Quantify what blocking bought us. Worth logging on every run."""
    n = len(records)
    all_pairs = n * (n - 1) // 2
    generated = len(list(pairs))
    if all_pairs == 0:
        return "no records"
    reduction = 1 - (generated / all_pairs)
    return (
        f"records={n} all_pairs={all_pairs} generated={generated} "
        f"reduction={reduction:.1%}"
    )
