"""Phase 4 entry point: resolve coreference over NER-annotated documents.

    python scripts/run_coref.py

NOTE THE ``if __name__ == "__main__":`` GUARD AT THE BOTTOM. It is not optional
here. fastcoref tokenises via HuggingFace ``datasets``, which uses
multiprocessing; on Windows each child process re-imports this module, and
without the guard the children re-run main() and the whole thing exits silently
with code 0 and no output.
"""

from __future__ import annotations

from src.config import load_config
from src.coreference.pipeline import run_coreference
from src.logging_utils import configure_logging, get_logger
from src.preprocessing.processor import read_documents, write_documents

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    in_path = cfg.path(cfg.paths.processed_dir) / "documents_ner.jsonl"
    if not in_path.exists():
        raise SystemExit(f"{in_path} not found. Run `python scripts/run_ner.py` first.")

    documents = read_documents(in_path)
    stats = run_coreference(documents, cfg=cfg)

    out_path = cfg.path(cfg.paths.processed_dir) / "documents_coref.jsonl"
    write_documents(documents, out_path)

    print("\n" + "=" * 86)
    print(f"COREFERENCE REPORT   (backend={cfg.coreference.backend})")
    print("=" * 86)
    print(f"  documents                  : {stats.documents}")
    print(f"  clusters found             : {stats.clusters}")
    print(f"  mentions inside clusters   : {stats.cluster_mentions}")
    print(f"  pronouns resolved          : {stats.pronouns_resolved}")
    print(f"  NER mentions linked        : {stats.linked_ner_mentions}")
    print(f"  mention forms              : {stats.by_form}")
    print("-" * 86)

    for document in documents:
        if not document.coref_clusters:
            continue
        print(f"\n  {document.article_id}  {document.title[:56]}")
        for cluster in document.coref_clusters:
            rep = cluster.representative
            print(f"    [{cluster.cluster_id.split(':')[-1]}] -> {rep.text!r}")
            for mention in cluster.mentions:
                marker = " *" if mention is rep else "  "
                linked = f"  ({mention.ner_mention_id.split(':')[-1]})" if mention.ner_mention_id else ""
                print(
                    f"      {marker} ({mention.start:>4},{mention.end:>4}) "
                    f"{mention.form:<11} {mention.text!r}{linked}"
                )
    print("=" * 86)
    print("  * = representative mention chosen for this cluster")


if __name__ == "__main__":
    main()
