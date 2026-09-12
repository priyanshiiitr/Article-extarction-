"""Tests that a degraded run is REPORTED, not silently fast.

WHY THIS MATTERS
----------------
Every heavy stage falls back to a lighter component when its model cannot be
loaded -- the transformer NER drops out, and neural coreference becomes the
rule-based resolver. That is correct behaviour for a batch job, where an
operator reads the logs.

In a web UI nobody reads the logs. The only symptom is that a "full" run
finishes suspiciously fast, which is indistinguishable from it simply being
efficient. These tests pin the behaviour that makes the difference visible.

The failure is simulated by monkeypatching the builders, so the tests need
none of the 4.5 GB of models the real full profile requires.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config import Config
from src.coreference.rule_resolver import RuleBasedCorefResolver
from src.ner.gazetteer import GazetteerExtractor
from src.ner.spacy_ner import SpacyEntityExtractor
from src.pipeline import runner
from src.schemas import Article, make_article_id

UTC = timezone.utc


def _articles() -> list[Article]:
    published = datetime(2024, 10, 22, tzinfo=UTC)
    return [
        Article(
            article_id=make_article_id("Wire", published, "Modi meets Putin"),
            title="Modi meets Putin",
            source="Wire",
            published_at=published,
            body=(
                "Prime Minister Narendra Modi met President Vladimir Putin in Kazan. "
                "He discussed energy cooperation with the Russian leader."
            ),
        )
    ]


def test_healthy_run_reports_its_backends() -> None:
    """Even a good run must say what it used, so 'fast' is never ambiguous."""
    result = runner.run_pipeline(_articles(), cfg=Config(), profile="lite")
    assert result.backends["coreference"] == "rules"
    assert "spacy" in result.backends["ner"]
    assert result.backends["entity_resolution"] == "attributes only (no embeddings)"


def test_lite_profile_is_not_flagged_as_degraded() -> None:
    """Lite is a deliberate choice, not a failure -- it must not cry wolf."""
    result = runner.run_pipeline(_articles(), cfg=Config(), profile="lite")
    assert result.degraded == []


def test_neural_coref_fallback_is_reported(monkeypatch) -> None:
    """THE case that prompted this: a 'full' run where the 590M coreference
    model fails to load (commonly out of memory) silently becomes a lite run,
    and coreference is ~94% of full-profile runtime."""
    monkeypatch.setattr(
        runner, "build_coref_resolver", lambda cfg: RuleBasedCorefResolver()
    )
    monkeypatch.setattr(
        runner,
        "build_ner_extractors",
        lambda cfg: [SpacyEntityExtractor(), GazetteerExtractor()],
    )

    result = runner.run_pipeline(_articles(), cfg=Config(), profile="full")

    assert result.degraded, "a degraded full run must not look like a healthy one"
    assert any("coreference" in note for note in result.degraded)
    assert result.backends["coreference"] == "rules"


def test_missing_ner_extractor_is_reported(monkeypatch) -> None:
    """If the transformer drops out, the run is still useful but weaker --
    and the user has to be able to see that."""
    monkeypatch.setattr(
        runner,
        "build_ner_extractors",
        lambda cfg: [SpacyEntityExtractor(), GazetteerExtractor()],
    )
    monkeypatch.setattr(
        runner, "build_coref_resolver", lambda cfg: RuleBasedCorefResolver()
    )

    result = runner.run_pipeline(_articles(), cfg=Config(), profile="full")
    assert any("transformer" in note for note in result.degraded)


def test_degraded_message_explains_the_consequence(monkeypatch) -> None:
    """A warning nobody understands is barely better than no warning. The
    message must say what capability was actually lost."""
    monkeypatch.setattr(
        runner, "build_coref_resolver", lambda cfg: RuleBasedCorefResolver()
    )
    monkeypatch.setattr(
        runner,
        "build_ner_extractors",
        lambda cfg: [SpacyEntityExtractor(), GazetteerExtractor()],
    )
    result = runner.run_pipeline(_articles(), cfg=Config(), profile="full")
    coref_note = next(n for n in result.degraded if "coreference" in n)
    assert "nominal" in coref_note.lower()
