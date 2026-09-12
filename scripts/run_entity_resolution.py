"""Phase 6 entry point: resolve mentions into canonical entities.

    python scripts/run_entity_resolution.py
"""

from __future__ import annotations

import json

from src.config import load_config
from src.entity_resolution.profiles import build_all_profiles
from src.entity_resolution.resolver import resolve_entities
from src.logging_utils import configure_logging, get_logger
from src.preprocessing.processor import read_documents

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    in_path = cfg.path(cfg.paths.processed_dir) / "documents_relations.jsonl"
    if not in_path.exists():
        raise SystemExit(f"{in_path} not found. Run `python scripts/run_relations.py` first.")

    documents = read_documents(in_path)
    profiles = build_all_profiles(documents)
    entities, review_queue, stats = resolve_entities(profiles, cfg=cfg)

    out_dir = cfg.path(cfg.paths.processed_dir)
    with (out_dir / "entities.jsonl").open("w", encoding="utf-8") as fh:
        for entity in entities:
            payload = {
                "entity_id": entity.entity_id,
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "aliases": sorted(entity.aliases),
                "roles": sorted(entity.roles),
                "countries": sorted(entity.countries),
                "orgs": sorted(entity.orgs),
                "topics": sorted(entity.topics),
                "events": sorted(entity.events),
                "article_ids": entity.article_ids,
                "mention_ids": entity.mention_ids,
                "first_seen": entity.first_seen.isoformat() if entity.first_seen else None,
                "last_seen": entity.last_seen.isoformat() if entity.last_seen else None,
                "merge_confidence": round(entity.merge_confidence, 3),
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with (out_dir / "review_queue.jsonl").open("w", encoding="utf-8") as fh:
        for pair in review_queue:
            fh.write(
                json.dumps(
                    {
                        "left": pair.left_id,
                        "right": pair.right_id,
                        "score": round(pair.score, 3),
                        "explanation": pair.explain(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\n" + "=" * 94)
    print("ENTITY RESOLUTION REPORT")
    print("=" * 94)
    print(f"  local entity profiles     : {stats.profiles}")
    print(f"  blocking                  : {stats.blocking}")
    print(f"  candidate pairs scored    : {stats.candidate_pairs}")
    print(f"  matched                   : {stats.matches}")
    print(f"  sent to review            : {stats.reviews}")
    print(f"  vetoed (hard constraint)  : {stats.vetoed}")
    print(f"  unsound clusters split    : {stats.unsound_clusters_split}")
    print(f"  CANONICAL ENTITIES        : {stats.entities}")
    print("-" * 94)

    for entity in entities:
        if len(entity.article_ids) < 2 and entity.entity_type == "ORG":
            continue
        print(f"\n  {entity.entity_id}  {entity.canonical_name}   [{entity.entity_type}]")
        print(f"     aliases   : {sorted(entity.aliases)}")
        if entity.roles:
            print(f"     roles     : {sorted(entity.roles)}")
        if entity.countries:
            print(f"     countries : {sorted(entity.countries)}")
        if entity.orgs:
            print(f"     orgs      : {sorted(entity.orgs)}")
        print(
            f"     articles  : {len(entity.article_ids)}   "
            f"seen {entity.first_seen.date()} .. {entity.last_seen.date()}   "
            f"merge_conf={entity.merge_confidence:.2f}"
        )

    if review_queue:
        print("\n" + "-" * 94)
        print(f"  REVIEW QUEUE ({len(review_queue)} pairs need a human)")
        for pair in sorted(review_queue, key=lambda p: -p.score)[:10]:
            print(f"    {pair.score:.3f}  {pair.left_id}  <->  {pair.right_id}")
            print(f"           {pair.explain()}")
    print("=" * 94)


if __name__ == "__main__":
    main()
