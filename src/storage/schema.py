"""SQLite schema for the knowledge graph.

WHY A RELATIONAL DATABASE FOR A GRAPH
-------------------------------------
A graph is just nodes and edges, and a relational database stores those in two
tables. The question is not "is it a graph?" but "what queries do you run?":

  * 1-2 HOP queries ("who did Modi meet?", "what did he discuss?") are plain
    joins. SQL is excellent at these, and SQLite needs no server, no ops, and
    produces a single file you can copy or commit.
  * VARIABLE-LENGTH PATH queries ("everyone within three hops of Modi",
    "shortest path between two entities", community detection) are where SQL
    needs recursive CTEs that are awkward to write and slow to run. That is
    when a graph database -- Neo4j, Memgraph -- starts paying for itself.

At 22 entities and 35 relations, adding a graph database would be infrastructure
for its own sake. The switch is justified by QUERY SHAPE and scale, not by the
word "graph" appearing in the design.

THE TWO-LEVEL DESIGN
--------------------
Facts are stored at TWO levels, and keeping both is the key decision here:

    MENTION LEVEL   (mentions, relations)
        What a specific document literally said, at specific character offsets.
        Immutable evidence. Never rewritten.

    ENTITY LEVEL    (entities, entity_relations)
        What we BELIEVE about the world after resolving mentions into entities.
        Derived, and therefore re-derivable.

If you keep only the entity level you cannot audit a fact or undo a bad merge.
If you keep only the mention level you cannot answer "what do we know about
Narendra Modi?" without re-running resolution. The provenance table
``entity_mentions`` is the bridge, and it is what makes a wrong merge
REVERSIBLE -- which matters enormously, because a false merge is the most
damaging error this pipeline can make.
"""

from __future__ import annotations

# Foreign keys are OFF by default in SQLite -- a historical default that
# surprises people. We enable them per connection so a relation can never point
# at a mention that does not exist.
PRAGMAS = [
    "PRAGMA foreign_keys = ON",
    # WAL lets readers proceed while a writer is active, which matters as soon
    # as a query layer runs alongside a pipeline job.
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
]

SCHEMA = """
-- ===========================================================================
-- DOCUMENT LEVEL -- the immutable evidence
-- ===========================================================================

CREATE TABLE IF NOT EXISTS articles (
    article_id       TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    source           TEXT NOT NULL,
    url              TEXT,
    published_at     TEXT NOT NULL,     -- ISO-8601 UTC
    language         TEXT NOT NULL DEFAULT 'en',
    -- Fingerprint of the canonical text. If this changes, every character
    -- offset stored below is stale. Persisted so that can be DETECTED rather
    -- than silently trusted.
    text_sha256      TEXT NOT NULL,
    text             TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    processed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sentences (
    article_id  TEXT NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL,
    PRIMARY KEY (article_id, idx)
);

CREATE TABLE IF NOT EXISTS mentions (
    mention_id       TEXT PRIMARY KEY,
    article_id       TEXT NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    start_char       INTEGER NOT NULL,
    end_char         INTEGER NOT NULL,
    text             TEXT NOT NULL,
    label            TEXT NOT NULL,
    raw_label        TEXT NOT NULL,
    score            REAL NOT NULL,
    sentence_index   INTEGER NOT NULL,
    extractor        TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    coref_cluster_id TEXT
);

CREATE TABLE IF NOT EXISTS coref_clusters (
    cluster_id          TEXT PRIMARY KEY,
    article_id          TEXT NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    representative_text TEXT NOT NULL,
    mention_count       INTEGER NOT NULL,
    method              TEXT NOT NULL,
    score               REAL NOT NULL
);

-- Mention-level relations: what a specific sentence said.
CREATE TABLE IF NOT EXISTS relations (
    relation_id        TEXT PRIMARY KEY,
    article_id         TEXT NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    subject_mention_id TEXT NOT NULL REFERENCES mentions(mention_id) ON DELETE CASCADE,
    subject_text       TEXT NOT NULL,
    subject_label      TEXT NOT NULL,
    predicate          TEXT NOT NULL,
    object_mention_id  TEXT NOT NULL REFERENCES mentions(mention_id) ON DELETE CASCADE,
    object_text        TEXT NOT NULL,
    object_label       TEXT NOT NULL,
    confidence         REAL NOT NULL,
    evidence_count     INTEGER NOT NULL DEFAULT 1,
    sentence_index     INTEGER NOT NULL,
    evidence_start     INTEGER NOT NULL,
    evidence_end       INTEGER NOT NULL,
    extractor          TEXT NOT NULL,
    trigger            TEXT NOT NULL,
    subject_via_coref  INTEGER NOT NULL DEFAULT 0,
    object_via_coref   INTEGER NOT NULL DEFAULT 0
);

-- ===========================================================================
-- ENTITY LEVEL -- what we believe, and why
-- ===========================================================================

CREATE TABLE IF NOT EXISTS entities (
    entity_id        TEXT PRIMARY KEY,
    canonical_name   TEXT NOT NULL,
    entity_type      TEXT NOT NULL,
    first_seen       TEXT,
    last_seen        TEXT,
    -- Mean pairwise score of the merges that formed this entity. Entities
    -- assembled from weak evidence are exactly what a reviewer should look at,
    -- so the number is stored rather than discarded.
    merge_confidence REAL NOT NULL DEFAULT 1.0,
    article_count    INTEGER NOT NULL DEFAULT 0,
    mention_count    INTEGER NOT NULL DEFAULT 0,
    -- Which version of the code produced this entity. Without it, you cannot
    -- tell whether a strange entity came from today's model or last quarter's.
    pipeline_version TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id  TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    alias      TEXT NOT NULL,
    normalized TEXT NOT NULL,
    PRIMARY KEY (entity_id, alias)
);

-- Attributes as (type, value) rows rather than columns.
-- WHY: an entity can hold several roles, several countries, several
-- organisations, and the set of attribute types will grow. Columns would mean
-- a migration per new attribute and a lot of NULLs; rows mean neither.
-- The cost is that querying one attribute needs a join, which is what the
-- index below is for.
CREATE TABLE IF NOT EXISTS entity_attributes (
    entity_id  TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    attr_type  TEXT NOT NULL,       -- role | country | org | topic | event
    value      TEXT NOT NULL,
    PRIMARY KEY (entity_id, attr_type, value)
);

-- THE PROVENANCE TABLE. The bridge between belief and evidence.
-- Every (entity, mention) link records how and when it was made, so that:
--   * any fact can be traced to the exact characters that produced it;
--   * a wrong merge can be UNDONE by deleting rows here, without touching
--     the immutable mention-level evidence.
CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id   TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    mention_id  TEXT NOT NULL REFERENCES mentions(mention_id) ON DELETE CASCADE,
    article_id  TEXT NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    confidence  REAL NOT NULL,
    method      TEXT NOT NULL,       -- how this link was decided
    created_at  TEXT NOT NULL,
    PRIMARY KEY (entity_id, mention_id)
);

-- Entity-level relations: the actual knowledge graph edges.
--
-- object_entity_id is NULLABLE by design. Many objects are not resolvable
-- entities: a ROLE ("Prime Minister"), a COUNTRY ("India") or a TOPIC
-- ("artificial intelligence") is already canonical and has no entity record.
-- Rather than invent entities for them, we store object_value and leave
-- object_entity_id NULL. The alternative -- forcing everything into the entity
-- table -- would fill the graph with thousands of junk nodes.
CREATE TABLE IF NOT EXISTS entity_relations (
    edge_id           TEXT PRIMARY KEY,
    subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    predicate         TEXT NOT NULL,
    object_entity_id  TEXT REFERENCES entities(entity_id) ON DELETE CASCADE,
    object_value      TEXT NOT NULL,
    object_label      TEXT NOT NULL,
    -- Aggregate confidence, and how many independent mention-level relations
    -- support this edge. support_count is the corroboration signal: a fact
    -- asserted in five articles is stronger than one asserted once.
    confidence        REAL NOT NULL,
    support_count     INTEGER NOT NULL DEFAULT 1,
    first_seen        TEXT,
    last_seen         TEXT
);

-- Which mention-level relations support which graph edge. Provenance again:
-- this is how "show me why you believe Modi met Putin" is answered.
CREATE TABLE IF NOT EXISTS entity_relation_evidence (
    edge_id     TEXT NOT NULL REFERENCES entity_relations(edge_id) ON DELETE CASCADE,
    relation_id TEXT NOT NULL REFERENCES relations(relation_id) ON DELETE CASCADE,
    PRIMARY KEY (edge_id, relation_id)
);

-- Pairs the resolver could not decide. A queue, not an error log: these are
-- waiting for a human, and "I don't know" is a valid pipeline output.
CREATE TABLE IF NOT EXISTS review_queue (
    review_id     TEXT PRIMARY KEY,
    left_profile  TEXT NOT NULL,
    right_profile TEXT NOT NULL,
    left_name     TEXT NOT NULL,
    right_name    TEXT NOT NULL,
    score         REAL NOT NULL,
    explanation   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending|merged|rejected
    created_at    TEXT NOT NULL
);

-- ===========================================================================
-- INDEXES
-- ===========================================================================
-- Each of these exists for a SPECIFIC query in queries.py. Indexes are not
-- free -- they cost write time and disk -- so adding them speculatively is a
-- mistake. These correspond to the access paths the query layer actually uses.

CREATE INDEX IF NOT EXISTS idx_mentions_article  ON mentions(article_id);
CREATE INDEX IF NOT EXISTS idx_mentions_label    ON mentions(label);
CREATE INDEX IF NOT EXISTS idx_mentions_text     ON mentions(text);
CREATE INDEX IF NOT EXISTS idx_mentions_cluster  ON mentions(coref_cluster_id);

CREATE INDEX IF NOT EXISTS idx_relations_article ON relations(article_id);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_mention_id);
CREATE INDEX IF NOT EXISTS idx_relations_pred    ON relations(predicate);
-- Partial index for the "low confidence" audit query. A partial index is much
-- smaller than a full one because it only covers the rows that query touches.
CREATE INDEX IF NOT EXISTS idx_relations_lowconf ON relations(confidence)
    WHERE confidence < 0.6;

CREATE INDEX IF NOT EXISTS idx_entities_name     ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_type     ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_aliases_norm      ON entity_aliases(normalized);
CREATE INDEX IF NOT EXISTS idx_attributes_lookup ON entity_attributes(attr_type, value);

CREATE INDEX IF NOT EXISTS idx_entmen_entity     ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entmen_mention    ON entity_mentions(mention_id);
CREATE INDEX IF NOT EXISTS idx_entmen_article    ON entity_mentions(article_id);

CREATE INDEX IF NOT EXISTS idx_edges_subject     ON entity_relations(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_edges_object      ON entity_relations(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_edges_predicate   ON entity_relations(predicate);
"""
