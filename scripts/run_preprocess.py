"""Phase 2 entry point: clean and segment the ingested corpus.

    python scripts/run_preprocess.py
"""

from __future__ import annotations

from src.config import load_config
from src.ingestion.loader import read_articles
from src.logging_utils import configure_logging, get_logger
from src.preprocessing.processor import (
    preprocess_articles,
    verify_offsets,
    write_documents,
)

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    raw_path = cfg.path(cfg.paths.raw_dir) / "articles.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found. Run `python scripts/run_ingest.py` first.")

    articles = read_articles(raw_path)
    documents = preprocess_articles(articles, cfg=cfg)

    out_path = cfg.path(cfg.paths.processed_dir) / "documents.jsonl"
    write_documents(documents, out_path)

    # Fail loudly on offset corruption rather than propagating bad spans.
    total_problems = 0
    for document in documents:
        problems = verify_offsets(document)
        total_problems += len(problems)
        for problem in problems:
            logger.error("%s: %s", document.article_id, problem)

    print("\n" + "=" * 82)
    print("PREPROCESSING REPORT")
    print("=" * 82)
    print(f"  documents      : {len(documents)}")
    print(f"  sentences      : {sum(len(d.sentences) for d in documents)}")
    print(f"  offset problems: {total_problems}")
    print("-" * 82)
    print(f"  {'article_id':<18} {'chars':>6} {'sents':>6}  title")
    for document in documents:
        print(
            f"  {document.article_id:<18} {len(document.text):>6} "
            f"{len(document.sentences):>6}  {document.title[:40]}"
        )

    sample = documents[0]
    print("-" * 82)
    print(f"  SAMPLE DOCUMENT: {sample.article_id}")
    print(f"  text_sha256    : {sample.text_sha256[:32]}...")
    for sentence in sample.sentences:
        surface = sample.text[sentence.start : sentence.end]
        print(f"    [{sentence.index}] ({sentence.start:>4},{sentence.end:>4})  {surface[:66]}")
    print("=" * 82)


if __name__ == "__main__":
    main()
