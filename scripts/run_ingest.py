"""Phase 1 entry point: ingest the configured corpus into the raw store.

    python scripts/run_ingest.py
"""

from __future__ import annotations

from src.config import load_config
from src.ingestion.loader import ingest, write_articles
from src.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    articles, stats = ingest(cfg=cfg)
    out_path = cfg.path(cfg.paths.raw_dir) / "articles.jsonl"
    write_articles(articles, out_path)

    print("\n" + "=" * 78)
    print("INGESTION REPORT")
    print("=" * 78)
    print(f"  records seen in source : {stats.seen}")
    print(f"  invalid (schema)       : {stats.invalid}")
    print(f"  duplicates dropped     : {stats.duplicates}")
    print(f"  too short dropped      : {stats.too_short}")
    print(f"  kept                   : {stats.kept}")
    print("-" * 78)
    for article in articles:
        print(
            f"  {article.article_id}  {article.published_at.date()}  "
            f"{article.source:<18.18}  {article.title[:44]}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
