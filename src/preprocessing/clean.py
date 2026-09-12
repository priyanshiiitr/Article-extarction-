"""Text cleaning: raw article body -> canonical text.

THE CONTRACT
------------
The string this module returns becomes ``Document.text`` and is IMMUTABLE from
that point on. Every character offset produced by every later stage indexes
into it. So this is the last moment at which the text may change -- which is
exactly why cleaning is its own stage with its own tests rather than something
each model does privately.

WHY CLEAN AT ALL
----------------
Transformer models are trained on ordinary prose. Markup costs you three ways:
  * Tokenisation waste -- "<p>" becomes tokens that consume context length.
  * Attention noise -- the model spends capacity modelling markup structure.
  * Spurious entities -- NER models will happily tag "Ltd" inside an <a href>
    URL, or treat a CSS class name as an organisation.

WHAT WE DELIBERATELY DO NOT DO
------------------------------
No lowercasing, no stopword removal, no stemming, no punctuation stripping.
Those are habits from the bag-of-words era and they actively destroy signal for
modern NLP:
  * Casing IS the signal for NER -- "modi" vs "Modi", "us" vs "US".
  * Stopwords carry relations -- "Modi met Putin" vs "Modi met with Putin's
    aide"; removing "with"/"'s" changes who met whom.
  * Punctuation marks sentence and clause boundaries, which the parser needs.
A 2010 preprocessing pipeline and a 2024 one look almost opposite.
"""

from __future__ import annotations

import html
import re
import unicodedata

from src.logging_utils import get_logger

logger = get_logger(__name__)

# Elements whose *contents* are not prose. Removing only the tags would leave
# JavaScript source or CSS rules sitting in the text as if they were sentences.
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|template)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

# Block-level elements imply a line break. If we deleted these tags without
# substituting whitespace, "</p><p>" would weld the last word of one paragraph
# to the first word of the next -- creating fake tokens like "Wednesday.Dilma".
_BLOCK_TAGS_RE = re.compile(
    r"</?(p|div|br|li|ul|ol|h[1-6]|section|article|blockquote|tr|td|th|table)\b[^>]*>",
    re.IGNORECASE,
)

_ANY_TAG_RE = re.compile(r"<[^>]+>")

# Characters that are invisible but occupy a character position. They shift
# every offset and silently break string equality between two visually
# identical mentions -- a nasty class of entity-resolution bug.
_INVISIBLE_CHARS = {
    "​": "",  # zero-width space
    "‌": "",  # zero-width non-joiner
    "‍": "",  # zero-width joiner
    "﻿": "",  # byte-order mark
    "­": "",  # soft hyphen
}

# Typographic punctuation -> ASCII. This is a deliberate TRADEOFF: we lose
# typographic fidelity, but we gain that "India's" is one string rather than
# two depending on which publisher's CMS produced it. Since aliases are matched
# as strings in Phase 6, that consistency is worth more than the curly quotes.
_PUNCTUATION_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-",   # en dash
    "—": " - ", # em dash: used as a clause separator, needs spacing
    "−": "-",   # minus sign
    "…": "...", # ellipsis
    " ": " ",   # non-breaking space
    " ": " ",   # narrow no-break space
    " ": " ",   # thin space
}

# Reference markers: "[1]", "[a]", "[citation needed]", "[note 2]".
# Encyclopaedic sources and some news sites embed these inline, and they attach
# themselves to entity spans -- we extracted "Narendra Damodardas Modi[a" as a
# PERSON from a live Wikipedia page. Removed BEFORE whitespace collapsing so the
# gap they leave behind is tidied in the same pass.
_CITATION_RE = re.compile(
    r"\[(?:\d{1,3}|[a-z]|citation needed|note \d+)\]", re.IGNORECASE
)

_MULTI_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SPACE_AROUND_NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*")


def strip_html(text: str) -> str:
    """Remove HTML markup, preserving paragraph structure.

    A regex HTML stripper is a BASELINE, not production-grade. It is fast and
    dependency-free, and adequate for the already-extracted article bodies we
    receive. For real scraped pages you want a parser -- ``selectolax`` or
    ``lxml`` for speed, or ``trafilatura``, which additionally solves *boiler-
    plate removal*: identifying which part of the page is the article as
    opposed to nav bars, related-links and cookie notices. Regexes cannot do
    that, because it requires reasoning over the DOM structure.
    """
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _BLOCK_TAGS_RE.sub("\n", text)
    text = _ANY_TAG_RE.sub("", text)
    # Entity decoding happens AFTER tag removal so that an encoded "&lt;p&gt;"
    # inside the article text is not resurrected into a real tag and stripped.
    return html.unescape(text)


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Apply Unicode normalisation.

    The problem: "e" + combining-acute and the single character "e-acute" look
    identical but are different strings, so ``==`` says they differ. NFKC picks
    one canonical spelling. The K ("compatibility") also folds ligatures and
    full-width forms -- useful for scraped text.

    Caveat worth knowing: NFKC is lossy. It turns superscript "2" into "2" and
    can alter mathematical notation. For news prose that is a fine trade; for
    scientific text it is not.
    """
    return unicodedata.normalize(form, text)


def fold_punctuation(text: str) -> str:
    """Map typographic punctuation and invisible characters to plain ASCII."""
    for source, target in {**_INVISIBLE_CHARS, **_PUNCTUATION_FOLD}.items():
        if source in text:
            text = text.replace(source, target)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks.

    Paragraph breaks are kept (as exactly one blank line) rather than collapsed
    to a single space because they are a real signal: sentence segmenters use
    them as hard boundaries, and a paragraph break is strong evidence that a
    sentence ended even when punctuation is missing (common in headlines).
    """
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_text(raw: str, strip_html_markup: bool = True, unicode_form: str = "NFKC") -> str:
    """Full cleaning pipeline: raw body -> canonical text.

    Order is not arbitrary:
      1. HTML first, because tags may contain characters the other steps would
         otherwise normalise into something that looks like prose.
      2. Unicode normalisation next, so that the punctuation fold below sees a
         single canonical spelling of each character.
      3. Punctuation/invisible folding.
      4. Whitespace last, because every earlier step can introduce whitespace
         (stripped tags leave gaps, em dashes expand to " - ").
    """
    text = raw
    if strip_html_markup:
        text = strip_html(text)
    text = normalize_unicode(text, unicode_form)
    text = fold_punctuation(text)
    text = _CITATION_RE.sub("", text)
    text = normalize_whitespace(text)
    return text


def build_document_text(title: str, body: str, **clean_kwargs) -> str:
    """Compose the canonical document text from title and body.

    We PREPEND the title, for a reason that matters downstream: headlines are
    entity-dense ("Modi and Putin discuss energy") and often introduce the full
    form of a name that the body then refers to only by surname. Coreference
    and entity resolution both do better when the title is inside the same
    document context rather than held in a separate metadata field.

    The title is terminated with a period if it lacks end punctuation, so the
    sentence segmenter treats it as its own sentence instead of running it into
    the first body sentence.
    """
    clean_title = clean_text(title, **clean_kwargs)
    clean_body = clean_text(body, **clean_kwargs)
    if clean_title and clean_title[-1] not in ".!?":
        clean_title += "."
    if not clean_title:
        return clean_body
    return f"{clean_title}\n\n{clean_body}"
