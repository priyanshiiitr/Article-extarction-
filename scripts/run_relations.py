"""Phase 5 entry point: extract relations from coref-annotated documents.

    python scripts/run_relations.py
"""

from __future__ import annotations

from src.config import load_config
from src.logging_utils import configure_logging, get_logger
from src.preprocessing.processor import read_documents, write_documents
from src.relation_extraction.pipeline import run_relation_extraction

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    in_path = cfg.path(cfg.paths.processed_dir) / "documents_coref.jsonl"
    if not in_path.exists():
        raise SystemExit(f"{in_path} not found. Run `python scripts/run_coref.py` first.")

    documents = read_documents(in_path)
    stats = run_relation_extraction(documents, cfg=cfg)

    out_path = cfg.path(cfg.paths.processed_dir) / "documents_relations.jsonl"
    write_documents(documents, out_path)

    print("\n" + "=" * 92)
    print("RELATION EXTRACTION REPORT")
    print("=" * 92)
    print(f"  raw candidates          : {stats.raw}")
    print(f"  after deduplication     : {stats.after_dedup}")
    print(f"  dropped below threshold : {stats.below_threshold}")
    print(f"  used coreference        : {stats.via_coref}")
    print(f"  by extractor            : {dict(stats.by_extractor)}")
    print("-" * 92)
    print("  BY PREDICATE")
    for predicate, count in stats.by_predicate.most_common():
        print(f"    {predicate:<16} {count:>3}  {'#' * min(count, 40)}")
    print("-" * 92)

    for document in documents:
        if not document.relations:
            continue
        print(f"\n  {document.title[:64]}")
        for relation in document.relations:
            flags = ""
            if relation.subject_via_coref or relation.object_via_coref:
                flags += " [coref]"
            if relation.evidence_count > 1:
                flags += f" [x{relation.evidence_count}]"
            print(
                f"    {relation.as_triple():<62} {relation.confidence:.2f} "
                f"{relation.extractor[:4]}/{relation.trigger[:18]}{flags}"
            )
    print("=" * 92)


if __name__ == "__main__":
    main()
