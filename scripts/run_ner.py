"""Phase 3 entry point: run named entity recognition over processed documents.

    python scripts/run_ner.py
"""

from __future__ import annotations

from collections import Counter

from src.config import load_config
from src.logging_utils import configure_logging, get_logger
from src.ner.pipeline import run_ner
from src.preprocessing.processor import read_documents, write_documents

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    in_path = cfg.path(cfg.paths.processed_dir) / "documents.jsonl"
    if not in_path.exists():
        raise SystemExit(f"{in_path} not found. Run `python scripts/run_preprocess.py` first.")

    documents = read_documents(in_path)
    stats = run_ner(documents, cfg=cfg)

    out_path = cfg.path(cfg.paths.processed_dir) / "documents_ner.jsonl"
    write_documents(documents, out_path)

    by_label = Counter(m.label for d in documents for m in d.mentions)
    total = sum(by_label.values())

    print("\n" + "=" * 84)
    print("NER REPORT")
    print("=" * 84)
    print(f"  candidate mentions (all extractors) : {stats.input_mentions}")
    print(f"  overlap conflicts resolved          : {stats.conflicts}")
    print(f"  final mentions                      : {total}")
    print(f"  winners by extractor                : {dict(stats.winners_by_extractor)}")
    print(f"  dropped by extractor                : {dict(stats.dropped_by_extractor)}")
    print("-" * 84)
    print("  FINAL MENTIONS BY TYPE")
    for label, count in by_label.most_common():
        bar = "#" * min(count, 45)
        print(f"    {label:<9} {count:>4}  {bar}")

    sample = documents[0]
    print("-" * 84)
    print(f"  SAMPLE: {sample.title}")
    for sentence in sample.sentences:
        print(f"\n    S{sentence.index}: {sample.text[sentence.start:sentence.end]}")
        for mention in sample.mentions:
            if mention.sentence_index == sentence.index:
                print(
                    f"         {mention.text:<26} -> {mention.label:<9} "
                    f"[{mention.extractor}/{mention.raw_label}] {mention.score:.2f}"
                )
    print("=" * 84)


if __name__ == "__main__":
    main()
