"""Write the pipeline's output into the knowledge graph.

IDEMPOTENCY, AGAIN
------------------
Loading must be safe to repeat. Every insert uses INSERT OR REPLACE on a
deterministic primary key, so re-running the loader over the same corpus
produces the same database rather than duplicating every row. That property is
inherited all the way from Phase 1's content-hash article IDs -- this is what
those were for.

AGGREGATION: MENTION RELATIONS -> GRAPH EDGES
---------------------------------------------
Mention-level relations are per-sentence facts. The graph wants one edge per
(entity, predicate, object) with evidence attached:

    (Modi[art_a]) -[met]-> (Putin[art_a])  \\
    (Modi[art_b]) -[met]-> (Putin[art_b])   >-- ONE edge, support_count = 3
    (Modi[art_e]) -[met]-> (Putin[art_e])  /

Confidence is combined with a NOISY-OR rather than a max or a mean:

    combined = 1 - PRODUCT(1 - c_i)

Two independent sources at 0.7 give 0.91, not 0.7. That is the right shape:
independent corroboration should INCREASE belief, which a max cannot express
and a mean actively contradicts (averaging two 0.7s stays 0.7, and adding a
weak third would LOWER it). The assumption being made is independence, which is
imperfect for syndicated wire copy -- the same story republished is not
independent evidence. We cap the result to keep that from running away.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from src.entity_resolution.normalize import normalize_name
from src.entity_resolution.resolver import CanonicalEntity
from src.entity_resolution.scoring import ScoredPair
from src.logging_utils import get_logger
from src.schemas import Document
from src.storage.schema import PRAGMAS, SCHEMA

logger = get_logger(__name__)

PIPELINE_VERSION = "0.1.0"

# Ceiling on noisy-OR aggregation. Sources are not truly independent (wire copy
# gets syndicated), so unbounded accumulation would report near-certainty from
# what is really one story repeated. Capping is cruder than modelling source
# dependence, and it is honest about the fact that we have not modelled it.
MAX_COMBINED_CONFIDENCE = 0.99


@contextmanager
def connect(db_path: Path, create: bool = True) -> Iterator[sqlite3.Connection]:
    """Open a connection with the right pragmas, always closed afterwards."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    # Rows behave like dicts, so queries read as row["canonical_name"] instead
    # of row[3] -- which silently breaks whenever a column is added.
    connection.row_factory = sqlite3.Row
    try:
        for pragma in PRAGMAS:
            connection.execute(pragma)
        if create:
            connection.executescript(SCHEMA)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _edge_id(subject_entity_id: str, predicate: str, object_key: str) -> str:
    key = f"{subject_entity_id}|{predicate}|{object_key}"
    return f"edge_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def _noisy_or(confidences: Sequence[float]) -> float:
    """Combine independent evidence. See the module docstring."""
    product = 1.0
    for confidence in confidences:
        product *= 1.0 - max(0.0, min(1.0, confidence))
    return min(MAX_COMBINED_CONFIDENCE, 1.0 - product)


# ---------------------------------------------------------------------------
# Document level
# ---------------------------------------------------------------------------


def load_documents(connection: sqlite3.Connection, documents: Sequence[Document]) -> dict[str, int]:
    counts = {"articles": 0, "sentences": 0, "mentions": 0, "clusters": 0, "relations": 0}

    for document in documents:
        connection.execute(
            """INSERT OR REPLACE INTO articles
               (article_id, title, source, url, published_at, language,
                text_sha256, text, pipeline_version, processed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                document.article_id, document.title, document.source, document.url,
                document.published_at.isoformat(), document.language,
                document.text_sha256, document.text,
                document.pipeline_version, document.processed_at.isoformat(),
            ),
        )
        counts["articles"] += 1

        connection.executemany(
            "INSERT OR REPLACE INTO sentences (article_id, idx, start_char, end_char) VALUES (?,?,?,?)",
            [(document.article_id, s.index, s.start, s.end) for s in document.sentences],
        )
        counts["sentences"] += len(document.sentences)

        connection.executemany(
            """INSERT OR REPLACE INTO coref_clusters
               (cluster_id, article_id, representative_text, mention_count, method, score)
               VALUES (?,?,?,?,?,?)""",
            [
                (c.cluster_id, document.article_id, c.representative_text,
                 len(c.mentions), c.method, c.score)
                for c in document.coref_clusters
            ],
        )
        counts["clusters"] += len(document.coref_clusters)

        connection.executemany(
            """INSERT OR REPLACE INTO mentions
               (mention_id, article_id, start_char, end_char, text, label, raw_label,
                score, sentence_index, extractor, model_version, coref_cluster_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (m.mention_id, m.article_id, m.start, m.end, m.text, m.label,
                 m.raw_label, m.score, m.sentence_index, m.extractor,
                 m.model_version, m.coref_cluster_id)
                for m in document.mentions
            ],
        )
        counts["mentions"] += len(document.mentions)

        connection.executemany(
            """INSERT OR REPLACE INTO relations
               (relation_id, article_id, subject_mention_id, subject_text, subject_label,
                predicate, object_mention_id, object_text, object_label, confidence,
                evidence_count, sentence_index, evidence_start, evidence_end,
                extractor, trigger, subject_via_coref, object_via_coref)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (r.relation_id, r.article_id, r.subject_mention_id, r.subject_text,
                 r.subject_label, r.predicate, r.object_mention_id, r.object_text,
                 r.object_label, r.confidence, r.evidence_count, r.sentence_index,
                 r.evidence_start, r.evidence_end, r.extractor, r.trigger,
                 int(r.subject_via_coref), int(r.object_via_coref))
                for r in document.relations
            ],
        )
        counts["relations"] += len(document.relations)

    logger.info("Loaded documents: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# Entity level
# ---------------------------------------------------------------------------


def load_entities(
    connection: sqlite3.Connection, entities: Sequence[CanonicalEntity]
) -> dict[str, int]:
    counts = {"entities": 0, "aliases": 0, "attributes": 0, "provenance": 0}
    now = _now()

    for entity in entities:
        connection.execute(
            """INSERT OR REPLACE INTO entities
               (entity_id, canonical_name, entity_type, first_seen, last_seen,
                merge_confidence, article_count, mention_count, pipeline_version, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                entity.entity_id, entity.canonical_name, entity.entity_type,
                entity.first_seen.isoformat() if entity.first_seen else None,
                entity.last_seen.isoformat() if entity.last_seen else None,
                entity.merge_confidence, len(entity.article_ids),
                len(entity.mention_ids), PIPELINE_VERSION, now,
            ),
        )
        counts["entities"] += 1

        connection.executemany(
            "INSERT OR REPLACE INTO entity_aliases (entity_id, alias, normalized) VALUES (?,?,?)",
            [
                (entity.entity_id, alias, normalize_name(alias, entity.entity_type))
                for alias in sorted(entity.aliases)
            ],
        )
        counts["aliases"] += len(entity.aliases)

        attribute_rows = (
            [(entity.entity_id, "role", v) for v in sorted(entity.roles)]
            + [(entity.entity_id, "country", v) for v in sorted(entity.countries)]
            + [(entity.entity_id, "org", v) for v in sorted(entity.orgs)]
            + [(entity.entity_id, "topic", v) for v in sorted(entity.topics)]
            + [(entity.entity_id, "event", v) for v in sorted(entity.events)]
        )
        connection.executemany(
            "INSERT OR REPLACE INTO entity_attributes (entity_id, attr_type, value) VALUES (?,?,?)",
            attribute_rows,
        )
        counts["attributes"] += len(attribute_rows)

        # PROVENANCE: which mentions support this entity, and how we decided.
        provenance_rows = []
        for mention_id in entity.mention_ids:
            article_id = mention_id.split(":", 1)[0]
            provenance_rows.append(
                (entity.entity_id, mention_id, article_id,
                 entity.merge_confidence, "entity_resolution_v1", now)
            )
        connection.executemany(
            """INSERT OR REPLACE INTO entity_mentions
               (entity_id, mention_id, article_id, confidence, method, created_at)
               VALUES (?,?,?,?,?,?)""",
            provenance_rows,
        )
        counts["provenance"] += len(provenance_rows)

    logger.info("Loaded entities: %s", counts)
    return counts


def build_entity_relations(connection: sqlite3.Connection) -> int:
    """Aggregate mention-level relations into entity-level graph edges.

    Runs in SQL rather than Python because the join from relation -> mention ->
    entity is exactly what a relational database is good at, and doing it in
    Python would mean pulling every relation into memory.
    """
    connection.execute("DELETE FROM entity_relation_evidence")
    connection.execute("DELETE FROM entity_relations")

    rows = connection.execute(
        """
        SELECT r.relation_id, r.predicate, r.confidence, r.object_text, r.object_label,
               subj_em.entity_id      AS subject_entity_id,
               obj_em.entity_id       AS object_entity_id,
               a.published_at         AS published_at
        FROM relations r
        JOIN entity_mentions subj_em ON subj_em.mention_id = r.subject_mention_id
        LEFT JOIN entity_mentions obj_em ON obj_em.mention_id = r.object_mention_id
        JOIN articles a ON a.article_id = r.article_id
        """
    ).fetchall()

    grouped: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
    for row in rows:
        # Group key uses the object ENTITY when the object resolved to one, and
        # the normalised object TEXT otherwise. That is what lets "met Putin"
        # and "met Vladimir Putin" collapse into a single edge while "discussed
        # AI" stays keyed on the literal topic.
        object_key = row["object_entity_id"] or row["object_text"].strip().lower()
        grouped.setdefault(
            (row["subject_entity_id"], row["predicate"], object_key), []
        ).append(row)

    edges = 0
    for (subject_entity_id, predicate, object_key), members in grouped.items():
        edge_id = _edge_id(subject_entity_id, predicate, object_key)
        confidence = _noisy_or([m["confidence"] for m in members])
        dates = sorted(m["published_at"] for m in members)

        # Prefer the longest object surface as the display value: "Vladimir
        # Putin" reads better than "Putin" on a graph edge.
        object_value = max((m["object_text"] for m in members), key=len)
        object_entity_id = next((m["object_entity_id"] for m in members if m["object_entity_id"]), None)

        connection.execute(
            """INSERT OR REPLACE INTO entity_relations
               (edge_id, subject_entity_id, predicate, object_entity_id, object_value,
                object_label, confidence, support_count, first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                edge_id, subject_entity_id, predicate, object_entity_id, object_value,
                members[0]["object_label"], confidence, len(members), dates[0], dates[-1],
            ),
        )
        connection.executemany(
            "INSERT OR REPLACE INTO entity_relation_evidence (edge_id, relation_id) VALUES (?,?)",
            [(edge_id, m["relation_id"]) for m in members],
        )
        edges += 1

    logger.info("Built %d entity-level graph edges from %d mention relations", edges, len(rows))
    return edges


def load_review_queue(connection: sqlite3.Connection, pairs: Iterable[ScoredPair]) -> int:
    now = _now()
    count = 0
    for pair in pairs:
        review_id = f"rev_{hashlib.sha1((pair.left_id + pair.right_id).encode()).hexdigest()[:12]}"
        connection.execute(
            """INSERT OR REPLACE INTO review_queue
               (review_id, left_profile, right_profile, left_name, right_name,
                score, explanation, status, created_at)
               VALUES (?,?,?,?,?,?,?,
                       COALESCE((SELECT status FROM review_queue WHERE review_id = ?), 'pending'),
                       ?)""",
            (
                review_id, pair.left_id, pair.right_id, pair.left_id, pair.right_id,
                pair.score, pair.explain(), review_id, now,
            ),
        )
        count += 1
    return count


def database_stats(connection: sqlite3.Connection) -> dict[str, int]:
    tables = [
        "articles", "sentences", "mentions", "coref_clusters", "relations",
        "entities", "entity_aliases", "entity_attributes", "entity_mentions",
        "entity_relations", "review_queue",
    ]
    return {
        table: connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        for table in tables
    }
