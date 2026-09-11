"""Sentence segmentation: canonical text -> ``Sentence`` spans.

WHAT SENTENCE SEGMENTATION IS
-----------------------------
Deciding where one sentence ends and the next begins. Also called "sentence
boundary disambiguation" (SBD), and the word *disambiguation* is the whole
problem: the character "." is overloaded. It ends sentences, but it also marks
abbreviations ("Ltd."), initials ("S."), decimals ("3.5"), ordinals ("No. 1")
and acronyms ("U.S."). Roughly 10-20% of periods in news text do not end a
sentence.

WHY EVERY LATER STAGE NEEDS IT
------------------------------
  * Relation extraction operates mostly within a sentence: two entities in one
    sentence are far more likely to be related than two in distant paragraphs.
  * Coreference uses sentence distance as a feature -- a pronoun usually refers
    to something one or two sentences back.
  * Provenance: the knowledge store shows the user the SENTENCE a fact came
    from. "Source: article 4,312" is not evidence; the sentence is.
  * Transformer models have a bounded context window, so long documents must be
    chunked -- and chunking at sentence boundaries avoids cutting an entity in
    half.

TWO BACKENDS, AND WHY
---------------------
``spacy``  (default) -- statistical. The dependency parser assigns sentence
    roots, so boundaries come from learned syntax rather than a rule list. It
    handles abbreviations it was never explicitly told about, because it judges
    from context. This is the right default.

``regex``  (fallback) -- an abbreviation-aware rule system. Included for three
    honest reasons: it lets the pipeline run with zero heavy dependencies, it
    makes the tradeoff concrete and measurable rather than hand-waved, and it
    is roughly 50x faster, which matters if you ever need a cheap first pass
    over millions of documents. It is a BASELINE and will lose on unusual text.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, Protocol, Sequence, runtime_checkable

from src.logging_utils import get_logger
from src.nlp_resources import SpacyUnavailableError, load_spacy
from src.schemas import Sentence

logger = get_logger(__name__)


@runtime_checkable
class SentenceSegmenter(Protocol):
    """Every segmenter returns ``Sentence`` spans over the SAME input string."""

    def segment(self, text: str) -> list[Sentence]:
        ...

    def segment_batch(self, texts: Sequence[str]) -> list[list[Sentence]]:
        ...


# ---------------------------------------------------------------------------
# Span hygiene -- shared by all backends
# ---------------------------------------------------------------------------


def _finalize_spans(text: str, raw_spans: Iterable[tuple[int, int]]) -> list[Sentence]:
    """Trim, validate and index candidate spans.

    Invariants enforced here, once, for every backend:
      * no leading/trailing whitespace inside a span (otherwise a "sentence"
        that starts with a newline shifts every offset the UI highlights);
      * empty spans dropped;
      * spans sorted and re-indexed contiguously from 0.

    Centralising this is deliberate. If each backend trimmed its own spans they
    would drift apart, and a bug would show up as "coreference works with spaCy
    but not with the regex segmenter" -- a miserable thing to debug.
    """
    sentences: list[Sentence] = []
    for start, end in sorted(raw_spans):
        # Walk the boundaries inward past whitespace.
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            sentences.append(Sentence(index=len(sentences), start=start, end=end))
    return sentences


# ---------------------------------------------------------------------------
# Backend 1: spaCy (statistical, default)
# ---------------------------------------------------------------------------


class SpacySegmenter:
    """Sentence segmentation via spaCy's dependency parser."""

    # The parser needs the tokenizer and tok2vec. Everything else is dead
    # weight for THIS task, so we refuse to construct it. NER is excluded here
    # on purpose -- Phase 3 loads its own pipeline for that.
    DEFAULT_EXCLUDE = ("ner", "lemmatizer", "attribute_ruler", "tagger")

    def __init__(self, model_name: str = "en_core_web_sm", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._nlp = load_spacy(model_name, self.DEFAULT_EXCLUDE)

    def segment(self, text: str) -> list[Sentence]:
        if not text.strip():
            return []
        doc = self._nlp(text)
        return _finalize_spans(text, ((s.start_char, s.end_char) for s in doc.sents))

    def segment_batch(self, texts: Sequence[str]) -> list[list[Sentence]]:
        """Segment many documents at once.

        ``nlp.pipe`` batches internally so the neural components run over a
        batch of documents per forward pass instead of one at a time. On CPU
        this is a solid win; on GPU it is the difference between using the
        device and wasting it. Always prefer ``pipe`` over a Python loop.
        """
        results: list[list[Sentence]] = []
        for text, doc in zip(texts, self._nlp.pipe(texts, batch_size=self.batch_size)):
            results.append(_finalize_spans(text, ((s.start_char, s.end_char) for s in doc.sents)))
        return results


# ---------------------------------------------------------------------------
# Backend 2: abbreviation-aware regex (rule-based fallback)
# ---------------------------------------------------------------------------

# Known abbreviations whose trailing period is usually NOT a sentence end.
# Note "usually": "He joined Reliance Industries Ltd. The company then..." is a
# genuine boundary after "Ltd.". That irreducible ambiguity is exactly why the
# statistical backend is the default -- no list can settle it, only context can.
_ABBREVIATIONS = {
    # Titles
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon", "gen",
    "capt", "col", "lt", "sgt", "gov", "sen", "rep", "amb",
    # Organisational
    "inc", "ltd", "co", "corp", "plc", "pvt", "llc", "llp", "assn", "bros",
    # Reference / measure
    "no", "vs", "etc", "eg", "ie", "al", "fig", "vol", "pp", "approx", "est",
    "rs", "usd", "dept", "univ", "min", "max",
    # Temporal
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sept", "sep", "oct",
    "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    # Geographic / political
    "u.s", "u.k", "u.n", "e.u", "d.c", "govt", "natl",
}

# A candidate boundary: terminal punctuation, then whitespace, then something
# that can start a sentence (capital letter, digit, or an opening quote).
_CANDIDATE_RE = re.compile(r'([.!?]+)(["\')\]]*)(\s+)(?=["\'(\[]*[A-Z0-9])')

# The word immediately preceding the punctuation.
_PRECEDING_WORD_RE = re.compile(r'([A-Za-z][A-Za-z.]*)$')


class RegexSegmenter:
    """Rule-based sentence segmentation. A documented BASELINE, not production.

    Known to be wrong on: abbreviations outside the list, sentences that begin
    with a lowercase word ("iPhone sales rose."), ellipses mid-sentence, and
    quoted dialogue where the attribution follows the closing quote.
    """

    def __init__(self, abbreviations: set[str] | None = None) -> None:
        self.abbreviations = abbreviations or _ABBREVIATIONS

    def _is_true_boundary(self, text: str, punct_start: int) -> bool:
        """Decide whether the punctuation at ``punct_start`` ends a sentence."""
        preceding = text[:punct_start]
        match = _PRECEDING_WORD_RE.search(preceding)
        if not match:
            return True

        word = match.group(1).lower().rstrip(".")

        # Single letter before a period is an initial: "Dr. S. Jaishankar".
        if len(word) == 1:
            return False
        # Dotted acronym: "U.S", "e.g" -- strip internal dots and re-check.
        if word in self.abbreviations or word.replace(".", "") in self.abbreviations:
            return False
        return True

    def segment(self, text: str) -> list[Sentence]:
        if not text.strip():
            return []

        boundaries: list[int] = []

        # Paragraph breaks are HARD boundaries regardless of punctuation --
        # headlines and list items often carry no terminal punctuation at all.
        for match in re.finditer(r"\n\s*\n", text):
            boundaries.append(match.start())

        for match in _CANDIDATE_RE.finditer(text):
            if self._is_true_boundary(text, match.start(1)):
                # End the sentence after the punctuation and any closing quote,
                # i.e. at the start of the whitespace run.
                boundaries.append(match.start(3))

        spans: list[tuple[int, int]] = []
        cursor = 0
        for boundary in sorted(set(boundaries)):
            if boundary > cursor:
                spans.append((cursor, boundary))
            cursor = boundary
        if cursor < len(text):
            spans.append((cursor, len(text)))

        return _finalize_spans(text, spans)

    def segment_batch(self, texts: Sequence[str]) -> list[list[Sentence]]:
        return [self.segment(text) for text in texts]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_segmenter(backend: str = "spacy", model_name: str = "en_core_web_sm") -> SentenceSegmenter:
    """Construct the configured segmenter, degrading gracefully.

    If spaCy is requested but unavailable we log a clear WARNING and fall back
    to the regex backend rather than crashing. Deliberate choice: a degraded
    result the operator is TOLD about beats a dead pipeline. The inverse choice
    -- silently falling back with no log line -- would be indefensible, because
    quality would drop with no visible cause.
    """
    backend = backend.lower()
    if backend == "regex":
        return RegexSegmenter()
    if backend == "spacy":
        try:
            return SpacySegmenter(model_name=model_name)
        except SpacyUnavailableError as exc:
            logger.warning("spaCy unavailable (%s). Falling back to regex segmenter.", exc)
            return RegexSegmenter()
    raise ValueError(f"Unknown segmenter backend: {backend!r}. Use 'spacy' or 'regex'.")


def iter_sentence_texts(text: str, sentences: Iterable[Sentence]) -> Iterator[str]:
    """Convenience: yield the surface text of each sentence span."""
    for sentence in sentences:
        yield text[sentence.start : sentence.end]
