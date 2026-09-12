"""Step 1 of entity resolution: normalise mention surface forms.

WHY NORMALISE BEFORE COMPARING
------------------------------
Every later step -- blocking, string similarity, alias matching -- compares
strings. If those strings still carry honorifics, role prefixes, punctuation
and casing differences, then every comparison is fighting noise that we could
simply have removed:

    "Mr. Modi"                    -> "modi"
    "Prime Minister Narendra Modi" -> "narendra modi"
    "Reliance Industries Ltd."     -> "reliance industries"

Note the second one: stripping the ROLE PREFIX is the highest-value single
transformation for news text, because news almost always introduces a person
as "<Title> <Name>". Without it, "Prime Minister Narendra Modi" and "Modi"
share almost no tokens and fuzzy matching scores them low.

WHAT NORMALISATION MUST NOT DO
------------------------------
It must not destroy information that distinguishes entities. Specifically we do
NOT strip given names, do NOT reduce to a surname, and do NOT drop middle
tokens -- because "Narendra Modi" and "Lalit Modi" differ ONLY in the given
name. Normalisation that collapsed them to "modi" would make a false merge
inevitable, no matter how good the later scoring is.

Normalisation is lossy by nature; the skill is choosing what is safe to lose.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from src.ner.gazetteer import ROLE_TITLES

# Honorifics carry no identifying information. "Mr. Modi" and "Modi" are the
# same string once this is gone.
_HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "dame",
    "shri", "smt", "sri", "hon", "rev", "fr", "st",
}

# Corporate suffixes: "Reliance Industries Ltd." and "Reliance Industries" are
# the same company. Kept separate from honorifics because they are ORG-specific.
_ORG_SUFFIXES = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation", "co",
    "company", "plc", "llc", "llp", "pvt", "private", "group", "holdings",
    "sa", "ag", "nv", "gmbh", "bv",
}

# Generic words that make an organisation name longer without making it more
# distinctive when comparing. Removed only when other tokens survive.
#
# "in" belongs here: without it, "Board of Control for Cricket in India"
# normalises to "board control cricket in india" whose acronym is "bccii", not
# "bcci", so the acronym feature silently failed to match the real abbreviation.
# Function words are exactly the tokens an acronym drops, so the stopword list
# and the acronym rule have to agree about which words those are.
_ORG_STOPWORDS = {"the", "of", "and", "for", "in", "at", "on", "de", "la", "du"}

# Abbreviated titles, which the ROLE gazetteer does not contain because they
# are not how a role is usually WRITTEN OUT in body text -- but they are
# extremely common in headlines, which we prepend to every document.
# Without these, "PM Modi" normalises to "pm modi" and shares no tokens with
# "narendra modi", so the two never score highly enough to merge.
_TITLE_ABBREVIATIONS = {
    "pm", "cm", "fm", "hm", "dy", "deputy",
    "ceo", "cfo", "cto", "coo", "mp", "mla", "gov", "sen", "rep", "amb",
    "pres", "vp", "sec", "gen", "lt", "col", "capt", "maj",
}
# DELIBERATELY EXCLUDED: "md". It means Managing Director in business copy, but
# in South Asian news -- which this corpus is full of -- "Md" is the standard
# abbreviation of the GIVEN NAME Mohammed. Stripping it turned "Md Salim" into
# "salim", discarding the only token that distinguishes him from every other
# Salim. A domain-specific call: the cost of the false positive here outweighs
# the recall we would gain on "MD Ambani".

# Role titles, lower-cased, longest first so "deputy foreign minister" is
# stripped before "foreign minister" would match inside it.
_ROLE_PREFIXES = sorted((role.lower() for role in ROLE_TITLES), key=len, reverse=True)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Fold accented characters to ASCII: "Lula da Silvá" -> "Lula da Silva".

    NFD splits a character into base + combining mark, then we drop the marks.
    Wire services transliterate inconsistently, so this prevents the same name
    spelled two ways from becoming two entities.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


@lru_cache(maxsize=4096)
def normalize_name(surface: str, entity_type: str = "PERSON") -> str:
    """Normalise a mention's surface form for comparison.

    Cached because the same surface forms recur constantly across a corpus --
    "Modi" appears hundreds of times and the transformation is pure.
    """
    text = _strip_accents(surface).lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    # Strip a leading role title: "prime minister narendra modi" -> "narendra modi".
    if entity_type == "PERSON":
        for role in _ROLE_PREFIXES:
            if text.startswith(role + " "):
                text = text[len(role) + 1 :].strip()
                break
        # Then honorifics, which may sit inside or before the name.
        tokens = [t for t in text.split() if t not in _HONORIFICS]
        # Strip LEADING abbreviated titles only. Leading-only matters: a title
        # abbreviation is a prefix, and removing it anywhere would delete real
        # name tokens -- "Md" is a common given name in South Asia, and "Lt"
        # could plausibly appear inside one.
        while len(tokens) > 1 and tokens[0] in _TITLE_ABBREVIATIONS:
            tokens.pop(0)
        text = " ".join(tokens)
    else:
        tokens = text.split()
        while tokens and tokens[-1] in _ORG_SUFFIXES:
            tokens.pop()
        if len(tokens) > 1:
            filtered = [t for t in tokens if t not in _ORG_STOPWORDS]
            tokens = filtered or tokens
        text = " ".join(tokens)

    return text.strip()


def name_tokens(normalized: str) -> list[str]:
    return [t for t in normalized.split() if t]


def surname(normalized: str) -> str:
    """Last token of a normalised person name.

    A crude but effective proxy for the family name in English-language news.
    It is WRONG for name orders that put the family name first (much Chinese
    and Hungarian usage) and for single-token names. We use it only as a
    BLOCKING key -- a hint about which records to compare -- never as evidence
    that two records match, so an error here costs recall, not precision.
    """
    tokens = name_tokens(normalized)
    return tokens[-1] if tokens else ""


def initials(normalized: str) -> str:
    return "".join(token[0] for token in name_tokens(normalized) if token)


def is_single_token(normalized: str) -> bool:
    """True for bare surnames like "modi" or "putin".

    Worth knowing about a mention: a single-token name is inherently ambiguous
    and should never, on its own, be strong evidence of identity.
    """
    return len(name_tokens(normalized)) == 1
