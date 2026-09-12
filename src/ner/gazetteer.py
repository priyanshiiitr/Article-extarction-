"""Dictionary and rule based extraction for types no general NER model has.

WHY RULES, IN A DEEP-LEARNING PIPELINE
--------------------------------------
This is not nostalgia. Three of the types you need simply do not exist in any
standard NER model's label set:

  COUNTRY  Models emit GPE (spaCy) or LOC (CoNLL), which lump countries with
           cities and states. "Russia" and "Kazan" get the SAME label, and no
           amount of model quality separates them -- the distinction is not in
           the training data. But the set of countries is small, closed, and
           changes about once a decade. A lookup table is not a hack here, it
           is the correct tool.

  ROLE     "Prime Minister", "Finance Minister", "chairman". No mainstream NER
           model has a ROLE type at all. Roles are also highly patterned in
           news prose, which is what makes rules work well.

  EVENT    spaCy has EVENT but applies it inconsistently (it called our "16th
           BRICS Summit" an ORG). CoNLL has no EVENT type whatsoever.

WHEN RULES BEAT MODELS -- the general principle worth stating in an interview:
rules win when the target set is CLOSED, SMALL and STABLE, and when you need
100% recall on a known list. Models win when the set is OPEN and unbounded
(people's names) or when context decides the answer. Production systems use
both, and the interesting engineering is in merging them.

DEMONYMS
--------
We also map nationality adjectives to countries: "Indian" -> India. This is not
a mention type, it is EVIDENCE. "the Indian Prime Minister" is how Phase 6
learns which country Modi represents, and how "the Indian Prime Minister"
eventually resolves to the same entity as "Narendra Modi".

HONEST LIMITATIONS OF THIS APPROACH
-----------------------------------
  * No context: "Turkey" the country vs the bird, "Chad" the country vs the
    name. We mitigate slightly by requiring a capital letter, not by
    understanding. A production system disambiguates against a knowledge base.
  * Maintenance: every list is a thing someone must update.
  * Coverage: a name not in the list is invisible. Recall is bounded by the
    list, which is exactly why rules SUPPLEMENT models rather than replace them.

In production you would load these from Wikidata rather than hand-maintaining
them; the code path is identical, only the source of the data changes.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from src.logging_utils import get_logger
from src.ner.base import build_mention
from src.schemas import Document, Mention

logger = get_logger(__name__)

# Rules are deterministic lookups, not predictions, so a high fixed score is
# honest -- we are certain the string matched. It says nothing about whether
# the match was contextually CORRECT ("Turkey" the bird), which is a different
# kind of error that a confidence score cannot express.
GAZETTEER_SCORE = 0.90
PATTERN_SCORE = 0.80

# --- Countries -------------------------------------------------------------
# Scoped to what a BRICS / global-politics corpus actually needs, plus the
# major economies. Not exhaustive by design: a short curated list beats a
# 250-entry list full of ambiguous short names we would have to special-case.
COUNTRIES: set[str] = {
    "India", "Russia", "China", "Brazil", "South Africa", "Egypt", "Ethiopia",
    "Iran", "United Arab Emirates", "Saudi Arabia", "Indonesia", "Turkey",
    "United States", "United Kingdom", "France", "Germany", "Japan", "Italy",
    "Canada", "Australia", "Mexico", "Argentina", "Nigeria", "Kenya", "Vietnam",
    "Bangladesh", "Pakistan", "Sri Lanka", "Nepal", "Kazakhstan", "Belarus",
    "Ukraine", "Poland", "Spain", "Netherlands", "Singapore", "Malaysia",
    "Thailand", "South Korea", "North Korea", "Israel", "Qatar", "Kuwait",
}

# Common alternative surface forms. Maps alias -> canonical country name.
COUNTRY_ALIASES: dict[str, str] = {
    "USA": "United States",
    "U.S.": "United States",
    "US": "United States",
    "America": "United States",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Britain": "United Kingdom",
    "UAE": "United Arab Emirates",
    "Republic of India": "India",
    "Russian Federation": "Russia",
}

# --- Demonyms: nationality adjective -> country ----------------------------
DEMONYMS: dict[str, str] = {
    "Indian": "India", "Russian": "Russia", "Chinese": "China",
    "Brazilian": "Brazil", "South African": "South Africa", "Egyptian": "Egypt",
    "Ethiopian": "Ethiopia", "Iranian": "Iran", "Emirati": "United Arab Emirates",
    "Saudi": "Saudi Arabia", "Indonesian": "Indonesia", "Turkish": "Turkey",
    "American": "United States", "British": "United Kingdom", "French": "France",
    "German": "Germany", "Japanese": "Japan", "Italian": "Italy",
    "Canadian": "Canada", "Australian": "Australia", "Mexican": "Mexico",
    "Argentine": "Argentina", "Nigerian": "Nigeria", "Kenyan": "Kenya",
    "Ukrainian": "Ukraine", "Kazakh": "Kazakhstan", "Pakistani": "Pakistan",
}

# --- Roles -----------------------------------------------------------------
# Ordered longest-first so that "Deputy Foreign Minister" is matched before
# "Foreign Minister", which would otherwise steal part of the span. Getting
# this ordering wrong is the classic gazetteer bug.
ROLE_TITLES: list[str] = sorted(
    [
        "Prime Minister", "Deputy Prime Minister", "President", "Vice President",
        "Finance Minister", "Foreign Minister", "Deputy Foreign Minister",
        "External Affairs Minister", "Defence Minister", "Defense Minister",
        "Home Minister", "Commerce Minister", "Trade Minister",
        "Minister of State", "Chief Minister", "Chancellor", "Secretary of State",
        "Foreign Secretary", "Chief Executive Officer", "Chief Executive",
        "Managing Director", "Executive Director", "Director General",
        "Chairman", "Chairperson", "Chairwoman", "Chair",
        "Governor", "Ambassador", "Spokesperson", "Secretary General",
        "Chief Economist", "Chief Financial Officer",
    ],
    key=len,
    reverse=True,
)

# --- Events ----------------------------------------------------------------
# Events in news prose are highly patterned: one or more capitalised words (and
# optional ordinal) followed by an event head noun.
EVENT_HEAD_NOUNS = (
    "Summit", "Declaration", "Conference", "Forum", "Assembly", "Session",
    "Plenary", "Olympics", "Games", "Congress", "Convention", "Accord", "Treaty",
)

_EVENT_RE = re.compile(
    r"\b(?:\d{1,3}(?:st|nd|rd|th)\s+)?"      # optional ordinal: "16th "
    r"(?:[A-Z][\w.\-]*\s+){0,3}"              # up to 3 capitalised words
    r"(?:" + "|".join(EVENT_HEAD_NOUNS) + r")\b"
)

# --- Topics ----------------------------------------------------------------
# THE WEAKEST COMPONENT IN THIS FILE, and worth being explicit about.
#
# Countries and roles are closed sets. Topics are NOT -- "what was discussed"
# is unbounded, so a fixed list can never have good recall. It is included
# because Phase 5 needs a TOPIC argument for `PERSON -- discussed --> TOPIC`,
# and because a small curated list gives high PRECISION on the subjects this
# corpus actually covers.
#
# How you would really do this in production, in increasing order of cost:
#   1. Keyphrase extraction (YAKE, KeyBERT) -- unsupervised, no list needed.
#   2. Classify against a fixed taxonomy with a text classifier -- needs labels
#      but gives consistent, queryable categories.
#   3. An LLM with a constrained output schema -- highest quality on open
#      topics, but per-document cost and latency, and it must be validated.
# See the LLM-usage section for when option 3 actually pays for itself.
TOPICS: dict[str, str] = {
    "artificial intelligence": "artificial intelligence",
    "AI": "artificial intelligence",
    "civil nuclear energy": "civil nuclear energy",
    "nuclear energy": "nuclear energy",
    "energy cooperation": "energy cooperation",
    "climate change": "climate change",
    "cybersecurity": "cybersecurity",
    "cross-border payment system": "cross-border payments",
    "payment systems": "cross-border payments",
    "national currencies": "trade in national currencies",
    "crude supply": "crude supply",
    "defence ties": "defence cooperation",
    "global governance": "global governance",
    "grain trading": "grain trading",
    "compute infrastructure": "compute infrastructure",
    "infrastructure projects": "infrastructure",
    "connectivity": "connectivity",
    "skilling": "skilling",
}

# Case-insensitive: "AI" is capitalised but "artificial intelligence" usually
# is not, and both spellings appear mid-sentence.
_TOPIC_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(TOPICS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _compile_alternation(phrases: Iterable[str]) -> re.Pattern[str]:
    """Build one regex that matches any phrase in the list, longest first.

    ONE combined pattern rather than a loop of N patterns: the regex engine
    scans the text a single time instead of N times. With 40 roles over 100,000
    articles that is a 40x difference in this stage.

    ``re.escape`` is essential -- "U.S." contains a dot, which is a regex
    wildcard. Unescaped, it would match "UxSy".
    """
    ordered = sorted(set(phrases), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in ordered) + r")\b")


_COUNTRY_RE = _compile_alternation(list(COUNTRIES) + list(COUNTRY_ALIASES))
_DEMONYM_RE = _compile_alternation(DEMONYMS)

# Roles are matched CASE-INSENSITIVELY. News prose capitalises a title before a
# name ("Prime Minister Modi") but lower-cases it in apposition ("Mukesh
# Ambani, chairman of Reliance"). A case-sensitive pattern silently missed
# every appositive role, which cost us the whole works_for relation family.
#
# Countries and demonyms stay case-SENSITIVE on purpose: lower-casing them
# would match "us" as the United States and "turkey" as the country.
_ROLE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in ROLE_TITLES) + r")\b",
    re.IGNORECASE,
)


def canonical_country(surface: str) -> str | None:
    """Resolve a surface form to a canonical country name, or None."""
    if surface in COUNTRIES:
        return surface
    if surface in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[surface]
    return DEMONYMS.get(surface)


class GazetteerExtractor:
    """Dictionary + pattern extraction for COUNTRY, ROLE and EVENT."""

    name = "gazetteer"
    model_version = "rules-v1"

    def __init__(
        self,
        include_countries: bool = True,
        include_roles: bool = True,
        include_events: bool = True,
        include_demonyms: bool = True,
        include_topics: bool = True,
    ) -> None:
        self.include_countries = include_countries
        self.include_roles = include_roles
        self.include_events = include_events
        self.include_demonyms = include_demonyms
        self.include_topics = include_topics

    def extract(self, document: Document) -> list[Mention]:
        text = document.text
        mentions: list[Mention] = []

        def add(start: int, end: int, label: str, raw_label: str, score: float) -> None:
            mentions.append(
                build_mention(
                    document=document,
                    start=start,
                    end=end,
                    label=label,
                    raw_label=raw_label,
                    score=score,
                    extractor=self.name,
                    model_version=self.model_version,
                )
            )

        if self.include_countries:
            for match in _COUNTRY_RE.finditer(text):
                add(match.start(), match.end(), "COUNTRY", "GAZ_COUNTRY", GAZETTEER_SCORE)

        if self.include_demonyms:
            for match in _DEMONYM_RE.finditer(text):
                # Labelled MISC, not COUNTRY: "Indian" is not the country, it
                # is evidence pointing at it. raw_label carries the resolved
                # country so Phase 5/6 can use it without re-deriving it.
                country = DEMONYMS[match.group(0)]
                add(match.start(), match.end(), "MISC", f"GAZ_DEMONYM:{country}", GAZETTEER_SCORE)

        if self.include_roles:
            for match in _ROLE_RE.finditer(text):
                add(match.start(), match.end(), "ROLE", "GAZ_ROLE", GAZETTEER_SCORE)

        if self.include_events:
            for match in _EVENT_RE.finditer(text):
                start, end = match.start(), match.end()
                # Drop a leading determiner the pattern may have swept up, and
                # require the span to actually start with a capital or digit so
                # we do not emit "the summit" as a named event.
                surface = text[start:end]
                if not (surface[0].isupper() or surface[0].isdigit()):
                    continue
                add(start, end, "EVENT", "PAT_EVENT", PATTERN_SCORE)

        if self.include_topics:
            for match in _TOPIC_RE.finditer(text):
                # The canonical topic goes in raw_label so that "AI" and
                # "artificial intelligence" already agree before Phase 6 -- a
                # tiny bit of normalisation done at extraction time, where we
                # still have the lookup table that knows they are the same.
                canonical = TOPICS.get(match.group(0), TOPICS.get(match.group(0).lower(), ""))
                add(match.start(), match.end(), "TOPIC", f"GAZ_TOPIC:{canonical}", PATTERN_SCORE)

        return mentions

    def extract_batch(self, documents: Sequence[Document]) -> list[list[Mention]]:
        """No model here, so batching is just a loop -- but we keep the same
        interface so the orchestrator treats all extractors identically."""
        return [self.extract(document) for document in documents]
