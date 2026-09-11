"""Ingestion orchestration: raw records in, validated ``Article`` objects out.

The pipeline stage boundary. Everything upstream is "the outside world";
everything downstream may assume a well-formed ``Article``.

Five steps, in this order and for these reasons:

1. READ        -- pull raw dicts from whichever source reader is configured.
2. VALIDATE    -- coerce into the ``Article`` schema. Bad records are logged
                  and SKIPPED, never fatal.
3. IDENTIFY    -- assign a deterministic ``article_id`` if the source did not
                  supply one, so re-ingestion is idempotent.
4. DEDUPLICATE -- drop repeats of an ``article_id`` already seen. Wire services
                  genuinely do republish the same story.
5. FILTER      -- drop stubs below a minimum length: paywall teasers and failed
                  fetches produce short bodies that generate garbage entities.

Steps 4 and 5 run AFTER validation so that the statistics we report are honest
about why each record was dropped. "We ingested 13 and kept 11" is only useful
if you can say which two went where.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from src.config import Config, load_config
from src.ingestion.readers import ArticleReader, build_reader
from src.logging_utils import get_logger
from src.schemas import Article, make_article_id

logger = get_logger(__name__)


@dataclass
class IngestionStats:
    """Counters describing one ingestion run.

    Observability is not optional in batch processing. Without these numbers a
    silent drop from 10,000 articles to 4,000 looks exactly like success.
    """

    seen: int = 0
    invalid: int = 0
    duplicates: int = 0
    too_short: int = 0
    kept: int = 0
    invalid_reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"seen={self.seen} kept={self.kept} "
            f"invalid={self.invalid} duplicates={self.duplicates} too_short={self.too_short}"
        )


def _coerce_to_article(record: dict[str, Any]) -> Article:
    """Turn one raw dict into a validated ``Article``.

    Assigns a deterministic ID when the source does not provide one. Note the
    ordering: we must validate ``published_at`` into a real datetime *before*
    hashing it, otherwise "2024-10-22T09:15:00+05:30" and "2024-10-22T03:45:00Z"
    -- the same instant written two ways -- would produce two different IDs.
    """
    payload = {k: v for k, v in record.items() if not k.startswith("_")}

    # Validate with a placeholder ID first so pydantic normalises published_at.
    staged = Article(article_id="pending", **payload)
    if not record.get("article_id"):
        staged = staged.model_copy(
            update={
                "article_id": make_article_id(
                    source=staged.source,
                    published_at=staged.published_at,
                    title=staged.title,
                    url=staged.url,
                )
            }
        )
    if record.get("_provenance"):
        staged.extra.setdefault("_provenance", record["_provenance"])
    return staged


def ingest(
    reader: ArticleReader | None = None,
    cfg: Config | None = None,
) -> tuple[list[Article], IngestionStats]:
    """Run the ingestion stage.

    Returns both the articles and the run statistics. Returning stats rather
    than only logging them lets tests assert on behaviour ("the duplicate was
    dropped") instead of scraping log output.
    """
    cfg = cfg or load_config()
    reader = reader or build_reader(cfg.path(cfg.ingestion.source_file))

    stats = IngestionStats()
    seen_ids: set[str] = set()
    articles: list[Article] = []

    for record in reader.read_raw():
        stats.seen += 1

        try:
            article = _coerce_to_article(record)
        except ValidationError as exc:
            stats.invalid += 1
            # Report the field that failed, not the whole pydantic dump: in a
            # 100k-article run the readable one-line reason is what you need.
            reasons = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            stats.invalid_reasons.append(reasons)
            logger.warning(
                "Invalid record (%s) from %s -- skipped",
                reasons,
                record.get("url") or record.get("title") or "<unknown>",
            )
            continue

        if article.article_id in seen_ids:
            stats.duplicates += 1
            logger.info("Duplicate article_id=%s (%s) -- skipped", article.article_id, article.url)
            continue

        if len(article.body) < cfg.ingestion.min_body_chars:
            stats.too_short += 1
            logger.info(
                "Body too short (%d < %d) for article_id=%s -- skipped",
                len(article.body),
                cfg.ingestion.min_body_chars,
                article.article_id,
            )
            continue

        seen_ids.add(article.article_id)
        articles.append(article)
        stats.kept += 1

    logger.info("Ingestion complete: %s", stats.summary())
    return articles, stats


def write_articles(articles: Iterable[Article], out_path: Path) -> int:
    """Persist validated articles to the RAW store as JSONL.

    This is the immutable landing zone. Preprocessing reads from here and
    writes elsewhere; nothing ever edits this file in place. If a downstream
    bug is found, you fix the code and re-derive -- you never have to re-fetch.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for article in articles:
            fh.write(json.dumps(article.model_dump(mode="json"), ensure_ascii=False) + "\n")
            count += 1
    logger.info("Wrote %d articles -> %s", count, out_path)
    return count


def read_articles(path: Path) -> list[Article]:
    """Load validated articles back from the raw store."""
    articles: list[Article] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                articles.append(Article(**json.loads(line)))
    return articles
