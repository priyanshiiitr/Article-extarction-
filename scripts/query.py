"""Phase 9 entry point: run the demonstration queries against the graph.

    python scripts/query.py
    python scripts/query.py "Narendra Modi"
"""

from __future__ import annotations

import sys

from src.config import load_config
from src.logging_utils import configure_logging
from src.storage.queries import (
    articles_for_entity,
    co_occurring_entities,
    entities_by_attribute,
    entity_graph,
    find_entity,
    low_confidence_relations,
    people_in_articles_about,
    pending_reviews,
    relation_evidence,
    relations_for_entity,
)
from src.storage.store import connect


def _rule(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    db_path = cfg.path(cfg.paths.db_path)
    if not db_path.exists():
        raise SystemExit(f"{db_path} not found. Run `python scripts/build_graph.py` first.")

    target = sys.argv[1] if len(sys.argv) > 1 else "PM Modi"

    with connect(db_path, create=False) as connection:
        # --- Alias lookup: the payoff of Phase 6 --------------------------
        _rule(f'ALIAS LOOKUP: searching for "{target}"')
        found = find_entity(connection, target)
        if not found:
            print(f"  no entity matched {target!r}")
            return
        entity = found[0]
        entity_id = entity["entity_id"]
        print(f"  {target!r}  ->  {entity['entity_id']}  {entity['canonical_name']}")
        print(f"  appears in {entity['article_count']} articles, "
              f"{entity['mention_count']} mentions, merge_conf={entity['merge_confidence']:.2f}")

        # --- Q1 -----------------------------------------------------------
        _rule("Q1. What people appear in BRICS articles?")
        for row in people_in_articles_about(connection, "BRICS"):
            print(f"  {row['canonical_name']:<30} {row['article_count']} article(s)")

        # --- Q2 -----------------------------------------------------------
        _rule(f"Q2. Which organizations did {entity['canonical_name']} interact with?")
        orgs = [
            row
            for row in relations_for_entity(connection, entity_id)
            if row["object_label"] == "ORG" or row["predicate"] == "works_for"
        ]
        works = []
        for row in orgs:
            print(f"  -[{row['predicate']}]-> {row['object_value']:<28} "
                  f"conf={row['confidence']:.2f} support={row['support_count']}")
        if not (orgs or works):
            print("  (none extracted)")

        # --- Q3 -----------------------------------------------------------
        _rule(f"Q3. What topics did {entity['canonical_name']} discuss?")
        for row in relations_for_entity(connection, entity_id, predicate="discussed"):
            print(f"  {row['object_value']:<36} conf={row['confidence']:.2f} "
                  f"support={row['support_count']}")

        # --- The knowledge graph for this entity --------------------------
        _rule(f"KNOWLEDGE GRAPH: {entity['canonical_name']}")
        graph = entity_graph(connection, entity_id)
        print(f"  {graph['canonical_name']}")
        print(f"  aliases: {graph['aliases']}")
        for attr_type, values in graph["attributes"].items():
            print(f"  {attr_type}: {values}")
        print()
        predicates = list(graph["relations"].items())
        for i, (predicate, edges) in enumerate(predicates):
            last_group = i == len(predicates) - 1
            for j, edge in enumerate(edges):
                last = last_group and j == len(edges) - 1
                branch = "`--" if last else "|--"
                print(f"  {branch} {predicate} -> {edge['object_value']}  "
                      f"({edge['confidence']:.2f}, {edge['support_count']}x)")

        # --- Q4 -----------------------------------------------------------
        _rule(f"Q4. Which articles mention {entity['canonical_name']}?")
        for row in articles_for_entity(connection, entity_id):
            print(f"  {row['published_at'][:10]}  {row['source']:<18} {row['title'][:44]}")
            print(f"               surfaces used: {row['surfaces']}")

        _rule(f"Who co-occurs with {entity['canonical_name']}?")
        for row in co_occurring_entities(connection, entity_id, limit=8):
            print(f"  {row['canonical_name']:<30} {row['shared_articles']} shared article(s)")

        # --- Q5 -----------------------------------------------------------
        _rule("Q5. Which extracted relationships have LOW confidence?")
        for row in low_confidence_relations(connection, threshold=0.6, limit=8):
            flags = ""
            if row["subject_via_coref"] or row["object_via_coref"]:
                flags = " [coref-resolved]"
            print(f"  {row['confidence']:.2f}  ({row['subject_text']}) -[{row['predicate']}]-> "
                  f"({row['object_text']}){flags}")
            print(f"         trigger={row['trigger']!r}  source={row['source']}")
            print(f"         evidence: {row['evidence'][:88]}")

        # --- Provenance: why do we believe an edge? -----------------------
        met = relations_for_entity(connection, entity_id, predicate="met")
        if met:
            _rule(f"PROVENANCE: why do we believe {entity['canonical_name']} met "
                  f"{met[0]['object_value']}?")
            for row in relation_evidence(connection, met[0]["edge_id"]):
                print(f"  {row['published_at'][:10]}  {row['source']}  (conf {row['confidence']:.2f}, "
                      f"{row['extractor']}/{row['trigger']})")
                print(f"     \"{row['evidence'][:84]}\"")

        # --- Operational --------------------------------------------------
        _rule("OPERATIONS: entities by attribute, and the review queue")
        print("  Entities representing India:")
        for row in entities_by_attribute(connection, "country", "India"):
            print(f"    {row['canonical_name']:<30} {row['article_count']} article(s)")
        print(f"\n  Pending review pairs: {len(pending_reviews(connection))}")
        for row in pending_reviews(connection, limit=4):
            print(f"    {row['score']:.3f}  {row['left_profile']} <-> {row['right_profile']}")
        print("=" * 88)


if __name__ == "__main__":
    main()
