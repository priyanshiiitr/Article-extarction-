"""Phases 7-8 entry point: load the pipeline output into the knowledge graph.

    python scripts/build_graph.py
"""

from __future__ import annotations

from src.config import load_config
from src.entity_resolution.profiles import build_all_profiles
from src.entity_resolution.resolver import resolve_entities
from src.logging_utils import configure_logging, get_logger
from src.preprocessing.processor import read_documents
from src.storage.store import (
    build_entity_relations,
    connect,
    database_stats,
    load_documents,
    load_entities,
    load_review_queue,
)

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    in_path = cfg.path(cfg.paths.processed_dir) / "documents_relations.jsonl"
    if not in_path.exists():
        raise SystemExit(f"{in_path} not found. Run `python scripts/run_relations.py` first.")

    documents = read_documents(in_path)
    profiles = build_all_profiles(documents)
    entities, review_queue, _ = resolve_entities(profiles, cfg=cfg)

    db_path = cfg.path(cfg.paths.db_path)
    with connect(db_path) as connection:
        load_documents(connection, documents)
        load_entities(connection, entities)
        edges = build_entity_relations(connection)
        load_review_queue(connection, review_queue)
        stats = database_stats(connection)

    print("\n" + "=" * 70)
    print(f"KNOWLEDGE GRAPH BUILT  ->  {db_path}")
    print("=" * 70)
    for table, count in stats.items():
        print(f"  {table:<26} {count:>6}")
    print(f"  {'graph edges built':<26} {edges:>6}")
    print("=" * 70)


if __name__ == "__main__":
    main()
