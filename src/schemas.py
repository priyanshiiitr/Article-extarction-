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
from typing import Any, Literal, Sequence

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

    # Filled by Phase 4. The back-reference from an entity mention to its
    # within-document coreference cluster.
    #
    # We store the link in BOTH directions (here, and CorefMention.ner_mention_id)
    # because the two stages ask opposite questions:
    #   Phase 5 has a mention and asks "what does this resolve to?"  -> this field
    #   A UI has a cluster and asks "which entities are in it?"      -> the other
    # The cost of bidirectional links is that they can disagree, so they are
    # written in one place (the coref pipeline) and never edited separately.
    coref_cluster_id: str | None = None

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
    # the pipeline runs: Phase 5 will add relations.
    mentions: list[Mention] = Field(default_factory=list)

    # Filled by Phase 4. Note these are WITHIN-DOCUMENT clusters only.
    coref_clusters: list[CorefCluster] = Field(default_factory=list)

    # Filled by Phase 5.
    relations: list[Relation] = Field(default_factory=list)

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
# Phase 4 -- coreference resolution
# ---------------------------------------------------------------------------

# How a mention refers to its entity. Difficulty increases down the list, and
# the distinction drives which mention we pick to REPRESENT a cluster.
MentionForm = Literal[
    "NAMED",        # "Narendra Modi", "Modi"        -- contains the name
    "NOMINAL",      # "The Indian Prime Minister"    -- a description
    "PRONOMINAL",   # "He", "his", "they"            -- carries almost no info
]

PRONOUNS: frozenset[str] = frozenset(
    {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "who", "whom", "whose", "which", "that", "this", "these", "those",
    }
)


def classify_mention_form(surface: str) -> str:
    """Classify a mention's surface form as NAMED, NOMINAL or PRONOMINAL.

    A deliberately simple heuristic. What it can and cannot do:
      * PRONOMINAL is reliable -- pronouns are a small closed set.
      * NAMED vs NOMINAL keys off a LEADING DETERMINER first, and only then on
        capitalisation.

    Why the determiner and not capitalisation alone: news style capitalises
    role titles, so "The Indian Prime Minister" is fully capitalised and a
    capitalisation test calls it a name. But a definite description begins with
    "the" and a personal name essentially never does. Checking the determiner
    first fixes exactly the case this pipeline cares about most.

    Known cost of this rule: organisations whose name includes the article
    ("The Hindu") are classified NOMINAL. That is acceptable here, because the
    only consumer is representative selection, where a determiner-initial
    phrase is a worse representative than the bare name anyway.
    """
    stripped = surface.strip()
    if not stripped:
        return "NOMINAL"
    if stripped.lower() in PRONOUNS:
        return "PRONOMINAL"

    words = stripped.split()
    if words[0].lower() in {"the", "a", "an", "this", "that", "these", "those"}:
        return "NOMINAL"

    # A name is a run of capitalised tokens, allowing internal lowercase
    # particles such as "da" in "Luiz Inacio Lula da Silva".
    capitalised = sum(1 for w in words if w[:1].isupper())
    return "NAMED" if capitalised >= max(1, len(words) - 1) else "NOMINAL"


class CorefMention(BaseModel):
    """One mention inside a coreference cluster.

    WHY THIS IS NOT JUST A ``Mention``
    ----------------------------------
    Coreference generates its OWN candidate spans and needs mentions that NER
    never produces: "He", "the airport", "the two leaders". Reusing ``Mention``
    would force us to invent an entity ``label`` for every pronoun, which is
    meaningless. So this is a lighter object, and ``ner_mention_id`` is the
    optional bridge back to a Phase 3 entity mention when the spans align.
    """

    start: int
    end: int
    text: str
    form: str = "NOMINAL"
    # Set when this coref span aligns with a Phase 3 NER mention. None for
    # pronouns and for nominals that NER did not consider entities.
    ner_mention_id: str | None = None


class CorefCluster(BaseModel):
    """A set of mentions in ONE document that refer to the same thing.

    A cluster is a WITHIN-DOCUMENT entity. It is NOT a real-world entity --
    that is Phase 6's job. Article A's "Modi" cluster and Article B's "Modi"
    cluster are two separate clusters until entity resolution merges them.
    """

    cluster_id: str
    article_id: str
    mentions: list[CorefMention] = Field(default_factory=list)
    # Index into ``mentions`` of the span that best NAMES this cluster. This is
    # the value downstream stages actually consume -- never a rewritten string.
    representative_index: int = 0
    score: float = 1.0
    method: str = "unknown"

    @property
    def representative(self) -> CorefMention | None:
        if not self.mentions:
            return None
        index = min(self.representative_index, len(self.mentions) - 1)
        return self.mentions[index]

    @property
    def representative_text(self) -> str:
        rep = self.representative
        return rep.text if rep else ""

    def spans(self) -> list[tuple[int, int]]:
        return [(m.start, m.end) for m in self.mentions]


def choose_representative(mentions: Sequence[CorefMention]) -> int:
    """Pick the mention that best names a cluster.

    Preference order, and the reasoning behind it:
      1. NAMED over NOMINAL over PRONOMINAL -- "Narendra Modi" identifies the
         referent; "He" identifies nothing outside this document.
      2. Within the same form, the LONGEST span -- "Narendra Modi" carries more
         identifying information than "Modi", which matters enormously in
         Phase 6 where a full name is far easier to resolve than a surname.
      3. Earliest position, as a deterministic tiebreak. Determinism is not
         cosmetic: a non-deterministic representative would make the whole
         pipeline produce different entity IDs on identical input.
    """
    if not mentions:
        return 0
    rank = {"NAMED": 0, "NOMINAL": 1, "PRONOMINAL": 2}
    best = min(
        range(len(mentions)),
        key=lambda i: (
            rank.get(mentions[i].form, 1),
            -(mentions[i].end - mentions[i].start),
            mentions[i].start,
        ),
    )
    return best


# ---------------------------------------------------------------------------
# Phase 5 -- relation extraction
# ---------------------------------------------------------------------------

# The relation schema. A CLOSED, controlled vocabulary, and that is deliberate:
# if extractors were free to emit any verb they found, the graph would contain
# "met", "meets", "had met", "held talks with" and "sat down with" as five
# different edge types, and no query could find them all. Normalising to a
# fixed set is what makes the graph queryable.
#
# The cost of a closed schema is coverage: a relation not in this list is
# simply not extracted. That is the standard trade, and it is why an LLM (open
# schema) is attractive when you genuinely cannot enumerate your relations.
RelationType = Literal[
    "holds_position",   # PERSON -> ROLE     "Prime Minister Narendra Modi"
    "represents",       # PERSON -> COUNTRY  "the Indian Prime Minister"
    "works_for",        # PERSON -> ORG      "Rousseff, who chairs the NDB"
    "attended",         # PERSON -> EVENT    "Modi arrived for the BRICS Summit"
    "met",              # PERSON -> PERSON|ORG
    "discussed",        # PERSON -> TOPIC
    "said",             # PERSON -> TOPIC
    "located_in",       # EVENT|ORG -> LOCATION|COUNTRY
    "signed",           # PERSON|ORG -> EVENT (agreements, declarations)
]


class Relation(BaseModel):
    """One extracted fact: (subject) --predicate--> (object).

    ARGUMENTS ARE MENTION IDS, NOT STRINGS
    --------------------------------------
    ``subject_mention_id`` points at a Mention, which points at a character
    span, which points into the immutable document text. So every relation is
    traceable to the exact words that produced it. If we stored plain strings
    we would have a fact we could not defend -- and "Modi met Putin" sourced
    from nowhere is not usable evidence.

    The ``*_text`` fields are denormalised copies, kept for the same reason
    Mention.text is: readable output, and the storage layer can work without
    loading documents. Same drift risk, same verification.
    """

    relation_id: str
    article_id: str

    subject_mention_id: str
    subject_text: str
    subject_label: str

    predicate: str

    object_mention_id: str
    object_text: str
    object_label: str

    # Where the fact was stated. Provenance is not optional: a knowledge graph
    # whose facts cannot be traced back to a sentence is not auditable, and
    # "show me why you believe this" is the first question anyone asks.
    sentence_index: int = -1
    evidence_start: int = -1
    evidence_end: int = -1

    confidence: float = 0.5
    # How many times this same fact was stated in the article. Repetition is
    # corroboration: a role asserted in three sentences is better supported
    # than one mentioned in passing, and Phase 6 uses this when weighing
    # conflicting claims about the same person.
    evidence_count: int = 1
    # Which component produced this: "pattern", "dependency", "llm", ...
    extractor: str = "unknown"
    # The surface verb/trigger that licensed the relation, before normalisation
    # ("held talks with" -> predicate "met"). Kept for the same reason NER keeps
    # raw_label: when output looks wrong, you need to see what actually matched.
    trigger: str = ""

    # True when an argument came from a coreference cluster representative
    # rather than the literal text ("He" -> "Narendra Modi"). Worth flagging
    # because these inherit the coref model's errors ON TOP of the parser's,
    # so their confidence should be treated as lower.
    subject_via_coref: bool = False
    object_via_coref: bool = False

    def evidence_in(self, document_text: str) -> str:
        if self.evidence_start < 0:
            return ""
        return document_text[self.evidence_start : self.evidence_end]

    def as_triple(self) -> str:
        return f"({self.subject_text}) -[{self.predicate}]-> ({self.object_text})"


def make_relation_id(
    article_id: str,
    subject_mention_id: str,
    predicate: str,
    object_mention_id: str,
) -> str:
    """Deterministic relation ID.

    Same document + same argument spans + same predicate -> same ID, so
    re-running extraction is idempotent and duplicate facts collapse instead of
    accumulating. Hashed because the component IDs are long and the raw
    concatenation would be unreadable in logs.
    """
    key = f"{article_id}|{subject_mention_id}|{predicate}|{object_mention_id}"
    return f"rel_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


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
