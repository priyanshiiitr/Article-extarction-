"""Phase 9 -- the query layer.

Every function answers one question a user would actually ask, and every one
returns PROVENANCE alongside the answer. That is the design rule here: a
knowledge graph that returns "Modi met Putin" without saying which sentence,
from which article, with what confidence, is not auditable -- and an
unauditable fact extracted by a statistical pipeline is not worth much.

WHY A PYTHON API RATHER THAN RAW SQL
------------------------------------
Three reasons, in order of importance:
  1. Parameter binding. Every value goes through "?" placeholders, so a user
     searching for an entity named  O'Brien  or  '; DROP TABLE --  is handled
     as data, never as SQL. String-formatting queries is the classic injection
     bug and it is entirely avoidable.
  2. The queries become testable units with names.
  3. Callers depend on a stable function signature, not on the schema, so the
     schema can change underneath them.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Entity lookup
# ---------------------------------------------------------------------------


def find_entity(connection: sqlite3.Connection, name: str) -> list[dict[str, Any]]:
    """Find entities by canonical name OR by any alias.

    Searching aliases is the whole point of Phase 6: a user who types "PM Modi"
    must find the entity whose canonical name is "Narendra Modi". We match on
    the NORMALISED alias so the user's capitalisation and honorifics do not
    matter either.
    """
    from src.entity_resolution.normalize import normalize_name

    normalized = normalize_name(name, "PERSON")
    return _rows(
        connection.execute(
            """
            SELECT DISTINCT e.entity_id, e.canonical_name, e.entity_type,
                   e.article_count, e.mention_count, e.merge_confidence,
                   e.first_seen, e.last_seen
            FROM entities e
            LEFT JOIN entity_aliases a ON a.entity_id = e.entity_id
            WHERE LOWER(e.canonical_name) = LOWER(?)
               OR a.normalized = ?
               OR LOWER(a.alias) = LOWER(?)
            ORDER BY e.article_count DESC
            """,
            (name, normalized, name),
        )
    )


def entity_profile(connection: sqlite3.Connection, entity_id: str) -> dict[str, Any]:
    """Everything known about one entity, with its attributes and aliases."""
    entity = connection.execute(
        "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
    ).fetchone()
    if entity is None:
        return {}

    profile = dict(entity)
    profile["aliases"] = [
        r["alias"]
        for r in connection.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias", (entity_id,)
        )
    ]
    profile["attributes"] = {}
    for row in connection.execute(
        "SELECT attr_type, value FROM entity_attributes WHERE entity_id = ? ORDER BY attr_type, value",
        (entity_id,),
    ):
        profile["attributes"].setdefault(row["attr_type"], []).append(row["value"])
    return profile


# ---------------------------------------------------------------------------
# Q1. What people appear in BRICS articles?
# ---------------------------------------------------------------------------


def people_in_articles_about(
    connection: sqlite3.Connection, keyword: str, entity_type: str = "PERSON"
) -> list[dict[str, Any]]:
    """Entities of a type appearing in articles that mention a keyword.

    Two joins deep: keyword -> articles -> mentions -> entities. The
    entity_mentions provenance table is what makes the last hop possible, and
    it is why the answer is a list of ENTITIES rather than a list of strings.
    """
    return _rows(
        connection.execute(
            """
            SELECT e.entity_id, e.canonical_name, e.entity_type,
                   COUNT(DISTINCT em.article_id) AS article_count,
                   GROUP_CONCAT(DISTINCT a.title) AS titles
            FROM articles a
            JOIN entity_mentions em ON em.article_id = a.article_id
            JOIN entities e ON e.entity_id = em.entity_id
            WHERE (a.text LIKE '%' || ? || '%' OR a.title LIKE '%' || ? || '%')
              AND e.entity_type = ?
            GROUP BY e.entity_id
            ORDER BY article_count DESC, e.canonical_name
            """,
            (keyword, keyword, entity_type),
        )
    )


# ---------------------------------------------------------------------------
# Q2. Which organizations did Narendra Modi interact with?
# Q3. What topics did Narendra Modi discuss?
# ---------------------------------------------------------------------------


def relations_for_entity(
    connection: sqlite3.Connection,
    entity_id: str,
    predicate: str | None = None,
    object_label: str | None = None,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Outgoing graph edges for an entity, optionally filtered.

    One function serves "who did X meet", "what did X discuss" and "where does
    X work", because they are the same query with a different predicate. Adding
    a function per question would triple the surface area for no benefit.
    """
    sql = """
        SELECT er.edge_id, er.predicate, er.object_value, er.object_label,
               er.object_entity_id, er.confidence, er.support_count,
               er.first_seen, er.last_seen
        FROM entity_relations er
        WHERE er.subject_entity_id = ? AND er.confidence >= ?
    """
    params: list[Any] = [entity_id, min_confidence]
    if predicate:
        sql += " AND er.predicate = ?"
        params.append(predicate)
    if object_label:
        sql += " AND er.object_label = ?"
        params.append(object_label)
    sql += " ORDER BY er.confidence DESC, er.support_count DESC"
    return _rows(connection.execute(sql, params))


def entity_graph(connection: sqlite3.Connection, entity_id: str) -> dict[str, Any]:
    """The full outgoing sub-graph for one entity, grouped by predicate.

    This is the "Narendra Modi -> holds_position -> Prime Minister" tree from
    the project brief.
    """
    profile = entity_profile(connection, entity_id)
    if not profile:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for edge in relations_for_entity(connection, entity_id):
        grouped.setdefault(edge["predicate"], []).append(edge)
    profile["relations"] = grouped
    return profile


# ---------------------------------------------------------------------------
# Q4. Which articles mention the same canonical entity?
# ---------------------------------------------------------------------------


def articles_for_entity(connection: sqlite3.Connection, entity_id: str) -> list[dict[str, Any]]:
    """Articles mentioning an entity, with the surface forms each one used.

    The surfaces column is the visible payoff of entity resolution: one row per
    article showing that "Modi", "PM Modi" and "Narendra Modi" all resolved
    here.
    """
    return _rows(
        connection.execute(
            """
            SELECT a.article_id, a.title, a.source, a.published_at,
                   COUNT(em.mention_id) AS mention_count,
                   GROUP_CONCAT(DISTINCT m.text) AS surfaces
            FROM entity_mentions em
            JOIN articles a ON a.article_id = em.article_id
            JOIN mentions m ON m.mention_id = em.mention_id
            WHERE em.entity_id = ?
            GROUP BY a.article_id
            ORDER BY a.published_at
            """,
            (entity_id,),
        )
    )


def co_occurring_entities(
    connection: sqlite3.Connection, entity_id: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Entities that appear in the same articles.

    A self-join on entity_mentions through article_id. Note this is
    CO-OCCURRENCE, not a relation: appearing in the same article does not mean
    two people interacted. It is a useful exploration signal and a bad
    inference, and conflating the two is a common analytical mistake.
    """
    return _rows(
        connection.execute(
            """
            SELECT e.entity_id, e.canonical_name, e.entity_type,
                   COUNT(DISTINCT mine.article_id) AS shared_articles
            FROM entity_mentions mine
            JOIN entity_mentions theirs ON theirs.article_id = mine.article_id
            JOIN entities e ON e.entity_id = theirs.entity_id
            WHERE mine.entity_id = ? AND theirs.entity_id != ?
            GROUP BY e.entity_id
            ORDER BY shared_articles DESC, e.canonical_name
            LIMIT ?
            """,
            (entity_id, entity_id, limit),
        )
    )


# ---------------------------------------------------------------------------
# Q5. Which extracted relationships have low confidence?
# ---------------------------------------------------------------------------


def low_confidence_relations(
    connection: sqlite3.Connection, threshold: float = 0.6, limit: int = 25
) -> list[dict[str, Any]]:
    """Mention-level relations below a confidence threshold, with evidence.

    This is the AUDIT query, and it is the most operationally useful one here.
    It answers "what should a human look at first?" -- and because it returns
    the evidence sentence, a reviewer can judge without opening the article.
    """
    return _rows(
        connection.execute(
            """
            SELECT r.relation_id, r.subject_text, r.predicate, r.object_text,
                   r.confidence, r.extractor, r.trigger,
                   r.subject_via_coref, r.object_via_coref,
                   a.title, a.source,
                   SUBSTR(a.text, r.evidence_start + 1, r.evidence_end - r.evidence_start) AS evidence
            FROM relations r
            JOIN articles a ON a.article_id = r.article_id
            WHERE r.confidence < ?
            ORDER BY r.confidence ASC
            LIMIT ?
            """,
            (threshold, limit),
        )
    )


def relation_evidence(connection: sqlite3.Connection, edge_id: str) -> list[dict[str, Any]]:
    """Every sentence supporting one graph edge.

    The answer to "why do you believe this?". Without this, the graph is a set
    of assertions; with it, it is a set of citations.
    """
    return _rows(
        connection.execute(
            """
            SELECT r.relation_id, r.confidence, r.extractor, r.trigger,
                   a.article_id, a.title, a.source, a.published_at,
                   SUBSTR(a.text, r.evidence_start + 1, r.evidence_end - r.evidence_start) AS evidence
            FROM entity_relation_evidence ere
            JOIN relations r ON r.relation_id = ere.relation_id
            JOIN articles a ON a.article_id = r.article_id
            WHERE ere.edge_id = ?
            ORDER BY a.published_at
            """,
            (edge_id,),
        )
    )


# ---------------------------------------------------------------------------
# Operational queries
# ---------------------------------------------------------------------------


def pending_reviews(connection: sqlite3.Connection, limit: int = 25) -> list[dict[str, Any]]:
    return _rows(
        connection.execute(
            """SELECT * FROM review_queue WHERE status = 'pending'
               ORDER BY score DESC LIMIT ?""",
            (limit,),
        )
    )


def weakly_supported_entities(
    connection: sqlite3.Connection, threshold: float = 0.65
) -> list[dict[str, Any]]:
    """Entities assembled from weak merge evidence -- the audit list."""
    return _rows(
        connection.execute(
            """SELECT entity_id, canonical_name, entity_type, article_count,
                      merge_confidence
               FROM entities
               WHERE merge_confidence < ? AND article_count > 1
               ORDER BY merge_confidence ASC""",
            (threshold,),
        )
    )


def entities_by_attribute(
    connection: sqlite3.Connection, attr_type: str, value: str
) -> list[dict[str, Any]]:
    """e.g. every entity whose country is India, or whose role is president."""
    return _rows(
        connection.execute(
            """SELECT e.entity_id, e.canonical_name, e.entity_type, e.article_count
               FROM entity_attributes ea
               JOIN entities e ON e.entity_id = ea.entity_id
               WHERE ea.attr_type = ? AND LOWER(ea.value) = LOWER(?)
               ORDER BY e.article_count DESC""",
            (attr_type, value),
        )
    )
