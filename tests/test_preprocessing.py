"""Tests for Phase 2 -- cleaning and sentence segmentation.

The most valuable test in this file is ``test_every_sentence_span_is_exact``.
Offset bugs do not crash -- they silently hand the wrong words to every later
stage, so entities and relations come out quietly wrong. The only defence is to
assert the invariant directly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config import Config
from src.preprocessing.clean import (
    build_document_text,
    clean_text,
    fold_punctuation,
    normalize_whitespace,
    strip_html,
)
from src.preprocessing.processor import preprocess_articles, verify_offsets
from src.preprocessing.segment import RegexSegmenter, build_segmenter
from src.schemas import Article


def _article(title: str, body: str) -> Article:
    return Article(
        article_id="art_test",
        title=title,
        source="Test Wire",
        published_at=datetime(2024, 10, 22, tzinfo=timezone.utc),
        body=body,
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def test_script_contents_are_removed_not_just_tags() -> None:
    """Removing <script> tags but keeping their contents would leave JavaScript
    sitting in the text as if it were a sentence."""
    cleaned = strip_html("<p>Real prose.</p><script>track('pageview');</script>")
    assert "Real prose." in cleaned
    assert "track" not in cleaned
    assert "pageview" not in cleaned


def test_block_tags_become_line_breaks() -> None:
    """Without substituting whitespace for block tags, '</p><p>' welds the last
    word of one paragraph onto the first word of the next."""
    cleaned = clean_text("<p>Ends here.</p><p>Starts here.</p>")
    assert "here.Starts" not in cleaned
    assert "Ends here." in cleaned and "Starts here." in cleaned


def test_html_entities_are_decoded() -> None:
    assert "$1.2 billion" in clean_text("<p>worth &#36;1.2&nbsp;billion</p>")


def test_invisible_characters_are_stripped() -> None:
    """Zero-width characters occupy an offset and break string equality between
    two visually identical mentions -- a nasty entity-resolution bug."""
    assert fold_punctuation("Modi​ said") == "Modi said"
    assert fold_punctuation("New Delhi") == "New Delhi"


def test_smart_quotes_are_folded_to_ascii() -> None:
    assert fold_punctuation("“Hello,” he said") == '"Hello," he said'
    assert fold_punctuation("India’s growth") == "India's growth"


def test_paragraph_breaks_survive_whitespace_normalisation() -> None:
    """Paragraph breaks are a real signal for segmentation, so they collapse to
    exactly one blank line rather than disappearing."""
    assert normalize_whitespace("One.\n\n\n\n\nTwo.") == "One.\n\nTwo."
    assert normalize_whitespace("One.   Two.") == "One. Two."


def test_casing_and_stopwords_are_preserved() -> None:
    """Modern NLP needs both. Casing IS the signal for NER ('modi' vs 'Modi'),
    and stopwords carry relations ('met' vs 'met with')."""
    cleaned = clean_text("<p>Modi met with the US President.</p>")
    assert cleaned == "Modi met with the US President."


def test_title_is_prepended_and_terminated() -> None:
    """The headline is entity-dense and often carries the full form of a name
    the body only uses a surname for, so it belongs inside the document text."""
    text = build_document_text("Modi arrives in Kazan", "He was received at the airport.")
    assert text.startswith("Modi arrives in Kazan.")
    assert "He was received" in text


def test_title_that_already_ends_in_punctuation_is_not_double_terminated() -> None:
    assert build_document_text("Is Modi visiting?", "Yes.").startswith("Is Modi visiting?")


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spacy_segmenter():
    """Load the spaCy model once for the whole module, not once per test.

    ``scope="module"`` matters: loading spaCy takes about a second, and there
    are several segmentation tests below.
    """
    return build_segmenter("spacy")


ABBREVIATION_TEXT = (
    "Reliance Industries Ltd. is in talks worth about Rs. 4,500 crore a month. "
    "Reliance is the No. 1 private refiner in India. "
    "Shares rose 3.5 per cent on Friday. "
    "Analysts at Morgan Stanley Inc. said the deal reduces exposure to U.S. dollar pricing."
)


def test_abbreviations_do_not_split_sentences_spacy(spacy_segmenter) -> None:
    """A naive text.split('.') produces 12 fragments from this paragraph."""
    assert len(spacy_segmenter.segment(ABBREVIATION_TEXT)) == 4


def test_abbreviations_do_not_split_sentences_regex() -> None:
    assert len(RegexSegmenter().segment(ABBREVIATION_TEXT)) == 4


def test_decimals_are_never_sentence_boundaries() -> None:
    """'3.5' has no whitespace after the period, so it can never be a boundary."""
    sentences = RegexSegmenter().segment("Shares rose 3.5 per cent today.")
    assert len(sentences) == 1


def test_spans_are_trimmed_of_whitespace(spacy_segmenter) -> None:
    text = "First sentence.\n\n   Second sentence."
    for sentence in spacy_segmenter.segment(text):
        surface = text[sentence.start : sentence.end]
        assert surface == surface.strip()


def test_spans_do_not_overlap_and_are_ordered(spacy_segmenter) -> None:
    sentences = spacy_segmenter.segment(ABBREVIATION_TEXT)
    for previous, current in zip(sentences, sentences[1:]):
        assert previous.end <= current.start
        assert previous.index < current.index


def test_empty_text_yields_no_sentences(spacy_segmenter) -> None:
    assert spacy_segmenter.segment("") == []
    assert spacy_segmenter.segment("   \n  ") == []
    assert RegexSegmenter().segment("") == []


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown segmenter backend"):
        build_segmenter("magic")


def test_regex_segmenter_documented_weakness_is_real() -> None:
    """Pin the KNOWN failure so it is a documented limitation, not a surprise.

    'Ltd.' really does end this sentence, but no abbreviation list can tell.
    Only a statistical model that reads context gets this right -- which is
    exactly why spaCy is the default backend.
    """
    text = "He joined Reliance Industries Ltd. The company later expanded."
    assert len(RegexSegmenter().segment(text)) == 1  # wrong, and known to be


# ---------------------------------------------------------------------------
# The offset invariant -- the most important test here
# ---------------------------------------------------------------------------


def test_every_sentence_span_is_exact() -> None:
    """document.text[start:end] must return exactly the sentence.

    If this ever fails, every entity offset in the system is wrong while
    nothing crashes -- the worst failure mode an offset pipeline has.
    """
    articles = [
        _article("Modi arrives in Kazan", "He met Putin. The Indian Prime Minister spoke."),
        _article("NDB lending", "<p>Dilma Rousseff chairs the NDB.</p><p>She spoke on Wednesday.</p>"),
        _article("Jaishankar meets Lavrov", "Dr. S. Jaishankar met Sergey Lavrov in Kazan."),
    ]
    documents = preprocess_articles(articles, cfg=Config())

    for document in documents:
        assert verify_offsets(document) == []
        for sentence in document.sentences:
            surface = document.text[sentence.start : sentence.end]
            assert surface.strip() == surface
            assert len(surface) > 0
        # Concatenating the sentences must recover the text minus whitespace.
        joined = "".join(
            document.text[s.start : s.end] for s in document.sentences
        )
        assert joined.replace(" ", "") == document.text.replace(" ", "").replace("\n", "")


def test_text_hash_changes_when_text_changes() -> None:
    """The tripwire for stale annotations."""
    documents = preprocess_articles([_article("T", "One sentence here.")], cfg=Config())
    original = documents[0].text_sha256
    mutated = documents[0].model_copy(update={"text": documents[0].text + " Added."})
    assert mutated.text_sha256 != original


def test_verify_offsets_detects_corruption() -> None:
    """Feed it a deliberately broken span and confirm it complains."""
    documents = preprocess_articles([_article("T", "One. Two.")], cfg=Config())
    document = documents[0]
    broken = document.model_copy(
        update={"sentences": [s.model_copy(update={"end": len(document.text) + 50}) for s in document.sentences[:1]]}
    )
    assert verify_offsets(broken) != []
