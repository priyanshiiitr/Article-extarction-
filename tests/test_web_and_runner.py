"""Tests for URL ingestion and the pipeline runner.

Network calls are NOT made here. Tests that hit the live web are slow, flaky,
and fail when someone else's site is down — which teaches you nothing about
your own code. We test the pure functions and the error paths; the fetching
itself was verified manually against real URLs.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config import Config
from src.ingestion.web import clean_page_title, fetch_article, fetch_articles
from src.pipeline.runner import apply_profile, run_pipeline
from src.preprocessing.clean import clean_text
from src.schemas import Article, make_article_id

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Page title cleaning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,source,expected",
    [
        ("Narendra Modi - Wikipedia", "en.wikipedia.org", "Narendra Modi"),
        ("India, Russia sign deal | Reuters", "Reuters", "India, Russia sign deal"),
        ("Modi in Kazan — The Hindu", "The Hindu", "Modi in Kazan"),
    ],
)
def test_site_name_is_stripped_from_title(title, source, expected):
    """Regression: the page title is PREPENDED to the document text, so
    "- Wikipedia" became a sentence and we extracted "Narendra Modi - Wikipedia"
    as a PERSON entity from a live page."""
    assert clean_page_title(title, source) == expected


def test_genuine_headline_dash_is_preserved():
    """Only strip when the trailing segment really looks like the site name."""
    assert (
        clean_page_title("Modi in Kazan - what it means", "The Hindu")
        == "Modi in Kazan - what it means"
    )


def test_title_without_a_separator_is_untouched():
    assert clean_page_title("Modi arrives in Kazan", "Reuters") == "Modi arrives in Kazan"


# ---------------------------------------------------------------------------
# Citation markers
# ---------------------------------------------------------------------------


def test_reference_markers_are_removed():
    """Regression: "Narendra Damodardas Modi[a" was extracted as a PERSON."""
    cleaned = clean_text("Narendra Damodardas Modi[a] is the PM[1] of India[citation needed].")
    assert cleaned == "Narendra Damodardas Modi is the PM of India."


def test_non_citation_brackets_survive():
    """[sic] and [Reuters] are editorial content, not reference markers."""
    assert "[sic]" in clean_text("The bracket [sic] stays.")
    assert "[Reuters]" in clean_text("Attributed [Reuters] here.")


# ---------------------------------------------------------------------------
# Fetch error handling
# ---------------------------------------------------------------------------


def test_invalid_urls_produce_readable_errors_not_exceptions():
    """One bad URL must never lose the rest of the batch."""
    for bad in ["", "   ", "not-a-url", "ftp://example.com/x"]:
        result = fetch_article(bad)
        assert not result.ok
        assert result.error


def test_fetch_articles_deduplicates_and_skips_blanks():
    results = fetch_articles(["not-a-url", "not-a-url", "", "   "])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Pipeline profiles
# ---------------------------------------------------------------------------


def test_lite_profile_swaps_every_heavy_component():
    """The whole deployment story: lite is a CONFIG change, not a code path."""
    lite = apply_profile(Config(), "lite")
    assert lite.ner.extractors == ["spacy", "gazetteer"]
    assert lite.coreference.backend == "rules"
    assert lite.entity_resolution.use_embeddings is False


def test_full_profile_keeps_the_heavy_components():
    full = apply_profile(Config(), "full")
    assert "transformer" in full.ner.extractors
    assert full.coreference.backend == "fastcoref"
    assert full.entity_resolution.use_embeddings is True


def test_apply_profile_does_not_mutate_the_shared_config():
    """load_config is lru_cached, so mutating it would silently change
    behaviour for every other caller in the process."""
    base = Config()
    original = list(base.ner.extractors)
    apply_profile(base, "lite")
    assert base.ner.extractors == original


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _article(title: str, body: str, day: int = 22) -> Article:
    published = datetime(2024, 10, day, tzinfo=UTC)
    return Article(
        article_id=make_article_id("Test Wire", published, title),
        title=title, source="Test Wire", published_at=published, body=body,
    )


def test_runner_produces_a_complete_result():
    articles = [
        _article(
            "Modi meets Putin in Kazan",
            "Prime Minister Narendra Modi met President Vladimir Putin in Kazan. "
            "He discussed energy cooperation with the Russian leader.",
        ),
        _article(
            "PM Modi reviews AI mission",
            "Prime Minister Modi reviewed the artificial intelligence mission in New Delhi.",
            day=25,
        ),
    ]
    result = run_pipeline(articles, cfg=Config(), profile="lite")

    assert result.counts["articles"] == 2
    assert result.counts["mentions"] > 0
    assert result.counts["entities"] > 0
    assert set(result.timings) == {
        "preprocess", "ner", "coreference", "relations", "entity_resolution"
    }
    # The two "Modi" mentions across articles should resolve to one entity.
    names = {e.canonical_name for e in result.entities}
    assert any("Modi" in n for n in names)


def test_runner_reports_progress_in_order():
    seen: list[str] = []
    run_pipeline(
        [_article("T", "Narendra Modi met Vladimir Putin in Kazan on Tuesday.")],
        cfg=Config(), profile="lite",
        progress=lambda stage, fraction, detail: seen.append(stage),
    )
    assert seen[0] == "preprocess"
    assert seen[-1] == "done"
    assert "entity_resolution" in seen


def test_runner_handles_empty_input():
    result = run_pipeline([], cfg=Config(), profile="lite")
    assert result.documents == []
    assert result.warnings
