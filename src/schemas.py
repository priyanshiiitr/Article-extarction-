"""Core data contracts for the pipeline.

Every stage communicates through these objects. Read this file first: if you
understand these shapes you understand how the whole system fits together.

THE CENTRAL DESIGN RULE -- character offsets
--------------------------------------------
Once ``Document.text`` exists, it is immutable. Every annotation produced by
any later stage (sentences, NER mentions, coreference clusters, relation
arguments) identifies its piece of text as a half-open character interval
``[start, end)`` into that exact string, so that ``document.text[start:end]``
always returns the annotated surface form.

This is what lets independent stages compose. spaCy, a HuggingFace NER model
and a coreference model all tokenise differently, and none of their token
indices are comparable -- but character offsets into a shared string are a
common denominator all of them can be converted to.

The price is that ``Document.text`` may never be edited after creation.
``Document.text_sha256`` exists to catch violations of that rule: if the text
changes, the hash changes, and stale annotations can be detected instead of
silently pointing at the wrong words.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator

# ---------------------------------------------------------------------------
# Phase 1 -- ingestion
# ---------------------------------------------------------------------------


class Article(BaseModel):
    """A raw ingested news article: the boundary between the messy outside
    world and our pipeline.

    ``body`` here is still RAW -- it may contain HTML, smart quotes,
    non-breaking spaces and irregular whitespace. Cleaning happens in Phase 2,
    deliberately, so that the raw form stays recoverable.
    """

    article_id: str
    title: str
    source: str
    published_at: datetime
    body: str
    url: str | None = None
    language: str = "en"
    # When WE ingested it, as opposed to when the publisher published it.
    # These differ, and confusing the two corrupts any time-based analysis.
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Free-form extras (section, author, tags) that we do not want to model
    # strictly yet. Keeps readers for new sources from needing a schema change.
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at", "collected_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        """Reject naive datetimes.

        A datetime without a timezone is ambiguous: "2024-10-22 09:00" is a
        different instant in Moscow and in Delhi. News corpora are inherently
        multi-timezone, so we force everything to UTC at the boundary. A small
        validator here prevents a whole class of silent ordering bugs later.
        """
        if value.tzinfo is None:
            raise ValueError("published_at/collected_at must be timezone-aware")
        return value.astimezone(timezone.utc)


def make_article_id(
    source: str,
    published_at: datetime,
    title: str,
    url: str | None = None,
) -> str:
    """Build a deterministic, reproducible article ID.

    Deterministic (content-hash) rather than random (uuid4) so that ingesting
    the same article twice yields the same ID -- which makes the ingestion
    stage IDEMPOTENT and therefore safely retryable.

    We hash ``url`` when present because it is the most stable identifier a
    publisher gives us, and fall back to source+date+title otherwise.
    """
    if url:
        key = url
    else:
        day = published_at.astimezone(timezone.utc).date().isoformat()
        key = f"{source}|{day}|{title}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"art_{digest}"


# ---------------------------------------------------------------------------
# Phase 2 -- preprocessing
# ---------------------------------------------------------------------------


class Sentence(BaseModel):
    """One sentence, located by character offsets into ``Document.text``.

    We store offsets rather than the substring itself so a sentence can always
    be related back to its exact position in the document -- needed to decide
    whether two entity mentions occur in the same sentence (a strong signal for
    relation extraction) and to render highlighted provenance in a UI.
    """

    index: int  # 0-based position of the sentence within the document
    start: int
    end: int

    def text_in(self, document_text: str) -> str:
        return document_text[self.start : self.end]


# ---------------------------------------------------------------------------
# Phase 3 -- named entity recognition
# ---------------------------------------------------------------------------


class Mention(BaseModel):
    """One occurrence of an entity in one document, located by char offsets.

    TERMINOLOGY -- keep these three straight, interviewers test it:
      * MENTION  -- a specific piece of text in a specific document.
                    "PM Modi" at chars 412-419 of article art_e8d8. There can
                    be thousands of mentions of one person.
      * ENTITY   -- the real-world thing. One Narendra Modi exists.
      * SURFACE FORM -- the literal characters of a mention ("PM Modi").

    Phase 3 produces mentions. Phase 6 groups mentions into entities. Confusing
    the two is the single most common conceptual error in this pipeline.
    """

    mention_id: str
    article_id: str

    # Offsets into Document.text -- the universal currency (see module docstring).
    start: int
    end: int

    # The surface form. DENORMALISED: it is derivable as document.text[start:end],
    # so storing it duplicates data. We accept that duplication deliberately,
    # because it makes the stored JSONL readable by a human, and lets the
    # storage and entity-resolution layers work on mentions without having to
    # load every document. The risk of denormalisation is drift -- the copy
    # disagreeing with the source -- which `verify_mentions` checks for.
    text: str

    # Our normalised type, from the shared EntityType vocabulary.
    label: str
    # What the MODEL actually said ("GPE", "NORP", "MISC"). Kept because the
    # mapping to our vocabulary is lossy, and when a downstream result looks
    # wrong the first question is always "what did the model really predict?"
    raw_label: str

    # Model confidence in [0, 1]. See the NER notes: treat as a RANKING signal,
    # not a calibrated probability -- neural nets are systematically overconfident.
    score: float = 1.0

    sentence_index: int = -1

    # Provenance: which component produced this mention, and which model version.
    # Essential once several extractors (model + gazetteer + rules) can each
    # produce mentions and you need to know which one to blame or improve.
    extractor: str = "unknown"
    model_version: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Mention") -> bool:
        """True if the two spans share at least one character.

        Used when merging several extractors: two components will often both
        find "New Development Bank", and one of them must win.
        """
        return self.start < other.end and other.start < self.end

    def contains(self, other: "Mention") -> bool:
        """True if this span fully encloses the other.

        The common case is a model finding "Narendra Modi" while a rule finds
        "Modi". Preferring the longer span is usually right for entity typing.
        """
        return self.start <= other.start and other.end <= self.end


def make_mention_id(article_id: str, start: int, end: int, label: str) -> str:
    """Deterministic mention ID: same document + same span + same label -> same ID.

    Deterministic for the same reason article IDs are: re-running NER must not
    create a second copy of every mention. The label is part of the key because
    two extractors may legitimately assign different types to the same span,
    and we want both to survive until the merge step decides between them.
    """
    return f"{article_id}:{start}-{end}:{label}"


class Document(BaseModel):
    """An article after preprocessing: the annotation container that later
    stages read from and add to.

    Phase 2 fills ``text`` and ``sentences``. Phases 3-5 will add ``mentions``,
    ``coref_clusters`` and ``relations`` to this same object.
    """

    article_id: str
    title: str
    source: str
    published_at: datetime
    url: str | None = None
    language: str = "en"

    # The canonical, cleaned, immutable text. ALL offsets index into this.
    text: str
    sentences: list[Sentence] = Field(default_factory=list)

    # Filled by Phase 3. The Document is an ANNOTATION CONTAINER that grows as
    # the pipeline runs: Phase 4 will add coref_clusters, Phase 5 relations.
    mentions: list[Mention] = Field(default_factory=list)

    # Bookkeeping so we can tell which code produced this artifact.
    pipeline_version: str = "0.1.0"
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def text_sha256(self) -> str:
        """Fingerprint of the canonical text.

        Persisted alongside annotations. If a later run produces a different
        hash for the same article_id, every stored offset for that article is
        stale and must be recomputed rather than trusted.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def sentence_containing(self, char_index: int) -> Sentence | None:
        """Find the sentence that covers a character position.

        Used constantly downstream: given an entity mention at offset 412,
        which sentence is it in? A linear scan is fine at our scale; for very
        long documents you would bisect over sentence start offsets.
        """
        for sentence in self.sentences:
            if sentence.start <= char_index < sentence.end:
                return sentence
        return None


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

# Entity types the pipeline standardises on. Note that each MODEL has its own
# label set (spaCy uses GPE/NORP/ORG, the CoNLL BERT models use LOC/ORG/PER).
# Phase 3 maps model-specific labels onto this shared vocabulary so that the
# rest of the system is not coupled to whichever model we happen to run.
EntityType = Literal[
    "PERSON",
    "ORG",
    "COUNTRY",
    "LOCATION",
    "EVENT",
    "ROLE",
    "TOPIC",
    "DATE",
    "MISC",
]
