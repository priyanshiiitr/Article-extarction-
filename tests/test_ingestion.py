"""Tests for Phase 1 -- ingestion.

Testing philosophy for this project: assert on BEHAVIOUR at the stage boundary,
not on internal implementation details. Each test below corresponds to a rule
we claimed the ingestion stage enforces. If someone later "optimises" the
loader and silently stops deduplicating, one of these fails.

Most tests use a fake in-memory reader rather than the corpus file. That keeps
them fast and independent of the sample data -- a test that breaks whenever you
add an article to the fixture is a test that will get deleted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator

import pytest
from pydantic import ValidationError

from src.config import Config
from src.ingestion.loader import ingest, read_articles, write_articles
from src.ingestion.readers import JsonlArticleReader
from src.schemas import Article, make_article_id


class FakeReader:
    """In-memory reader satisfying the ArticleReader protocol."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def read_raw(self) -> Iterator[dict[str, Any]]:
        yield from self._records


def _record(**overrides: Any) -> dict[str, Any]:
    base = {
        "title": "Modi arrives in Kazan",
        "source": "The Hindu",
        "published_at": "2024-10-22T09:15:00+05:30",
        "url": "https://example.test/a1",
        "body": "Prime Minister Narendra Modi arrived in Kazan on Tuesday. " * 4,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def test_article_id_is_deterministic() -> None:
    """The same inputs must always produce the same ID -- this is what makes
    re-running ingestion idempotent instead of duplicate-generating."""
    published = datetime(2024, 10, 22, 9, 15, tzinfo=timezone.utc)
    first = make_article_id("The Hindu", published, "T", "https://example.test/a1")
    second = make_article_id("The Hindu", published, "T", "https://example.test/a1")
    assert first == second
    assert first.startswith("art_")


def test_article_id_differs_for_different_urls() -> None:
    published = datetime(2024, 10, 22, 9, 15, tzinfo=timezone.utc)
    assert make_article_id("S", published, "T", "https://a.test/1") != make_article_id(
        "S", published, "T", "https://a.test/2"
    )


def test_article_id_is_timezone_independent() -> None:
    """The same instant expressed in two timezones must yield one ID.

    Guards the ordering bug where validation happens after hashing: +05:30 and
    the equivalent Z time are the same moment and must not split into two IDs.
    """
    delhi = datetime.fromisoformat("2024-10-22T09:15:00+05:30")
    utc = datetime.fromisoformat("2024-10-22T03:45:00+00:00")
    assert make_article_id("S", delhi, "T") == make_article_id("S", utc, "T")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_naive_datetime_is_rejected() -> None:
    """Timezone-naive timestamps are ambiguous across a multi-timezone corpus."""
    with pytest.raises(ValidationError):
        Article(
            article_id="a",
            title="t",
            source="s",
            published_at=datetime(2024, 1, 1),  # no tzinfo
            body="b",
        )


def test_published_at_is_normalised_to_utc() -> None:
    article = Article(
        article_id="a",
        title="t",
        source="s",
        published_at=datetime.fromisoformat("2024-10-22T09:15:00+05:30"),
        body="b",
    )
    assert article.published_at.tzinfo == timezone.utc
    assert article.published_at.hour == 3 and article.published_at.minute == 45


def test_invalid_record_is_skipped_not_fatal() -> None:
    """One bad record must not abort the batch. This is the single most
    important robustness property of a batch ingestion stage."""
    cfg = Config()
    reader = FakeReader([_record(), {"title": "no date", "source": "s", "body": "x" * 200}])
    articles, stats = ingest(reader=reader, cfg=cfg)
    assert stats.seen == 2
    assert stats.invalid == 1
    assert stats.kept == 1
    assert "published_at" in stats.invalid_reasons[0]


# --------------------------------------------------------------------------
# Deduplication and filtering
# --------------------------------------------------------------------------


def test_duplicate_article_is_dropped() -> None:
    reader = FakeReader([_record(), _record()])
    articles, stats = ingest(reader=reader, cfg=Config())
    assert stats.duplicates == 1
    assert stats.kept == 1
    assert len(articles) == 1


def test_short_body_is_dropped() -> None:
    cfg = Config()
    cfg.ingestion.min_body_chars = 120
    reader = FakeReader([_record(body="Too short.", url="https://example.test/stub")])
    _, stats = ingest(reader=reader, cfg=cfg)
    assert stats.too_short == 1
    assert stats.kept == 0


def test_min_body_chars_is_configurable() -> None:
    """Behaviour must follow config, not a hardcoded constant."""
    cfg = Config()
    cfg.ingestion.min_body_chars = 5
    reader = FakeReader([_record(body="Short but allowed now.")])
    _, stats = ingest(reader=reader, cfg=cfg)
    assert stats.kept == 1


# --------------------------------------------------------------------------
# Readers and round-tripping
# --------------------------------------------------------------------------


def test_jsonl_reader_skips_malformed_lines(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"title": "ok"}\n' "{ this is not json }\n" "\n" '["not", "an", "object"]\n',
        encoding="utf-8",
    )
    records = list(JsonlArticleReader(path).read_raw())
    assert len(records) == 1
    assert records[0]["title"] == "ok"
    # Provenance is attached from the first moment we touch the data.
    assert records[0]["_provenance"]["line"] == 1


def test_jsonl_reader_missing_file_raises(tmp_path) -> None:
    """A missing corpus is a configuration error, not a data error: fail loudly
    with an actionable message rather than silently yielding zero articles."""
    reader = JsonlArticleReader(tmp_path / "does_not_exist.jsonl")
    with pytest.raises(FileNotFoundError, match="make_sample_data"):
        list(reader.read_raw())


def test_write_then_read_round_trip(tmp_path) -> None:
    """Serialising to the raw store and back must preserve the article exactly.
    If this breaks, every downstream offset computed from a re-read article is
    computed against different text."""
    articles, _ = ingest(reader=FakeReader([_record()]), cfg=Config())
    out = tmp_path / "raw.jsonl"
    assert write_articles(articles, out) == 1

    restored = read_articles(out)
    assert len(restored) == 1
    assert restored[0].article_id == articles[0].article_id
    assert restored[0].body == articles[0].body
    assert restored[0].published_at == articles[0].published_at
