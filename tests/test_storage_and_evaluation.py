"""Tests for Phases 7-10: storage, query layer and evaluation metrics."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.entity_resolution.features import _acronym_match, name_compatibility
from src.entity_resolution.resolver import CanonicalEntity
from src.evaluation.metrics import (
    PRF,
    b_cubed,
    evaluate_clustering,
    muc,
    prf_from_sets,
)
from src.ner.base import build_mention
from src.schemas import Document, Relation, Sentence, make_relation_id
from src.storage.queries import (
    articles_for_entity,
    entities_by_attribute,
    entity_graph,
    find_entity,
    low_confidence_relations,
    relation_evidence,
    relations_for_entity,
)
from src.storage.store import (
    _noisy_or,
    build_entity_relations,
    connect,
    database_stats,
    load_documents,
    load_entities,
)

UTC = timezone.utc


@pytest.fixture
def populated_db(tmp_path):
    """A tiny end-to-end graph: two articles, one shared entity."""
    text_a = "Narendra Modi met Vladimir Putin in Kazan."
    document_a = Document(
        article_id="art_a", title="Modi meets Putin", source="Reuters",
        published_at=datetime(2024, 10, 22, tzinfo=UTC), text=text_a,
        sentences=[Sentence(index=0, start=0, end=len(text_a))],
    )
    modi_a = build_mention(document_a, 0, 13, "PERSON", "PER", 0.99, "transformer")
    putin_a = build_mention(document_a, 18, 32, "PERSON", "PER", 0.99, "transformer")
    document_a.mentions = [modi_a, putin_a]
    document_a.relations = [
        Relation(
            relation_id=make_relation_id("art_a", modi_a.mention_id, "met", putin_a.mention_id),
            article_id="art_a",
            subject_mention_id=modi_a.mention_id, subject_text="Narendra Modi", subject_label="PERSON",
            predicate="met",
            object_mention_id=putin_a.mention_id, object_text="Vladimir Putin", object_label="PERSON",
            confidence=0.75, sentence_index=0, evidence_start=0, evidence_end=len(text_a),
            extractor="dependency", trigger="met",
        )
    ]

    text_b = "PM Modi met Vladimir Putin again."
    document_b = Document(
        article_id="art_b", title="Second meeting", source="PTI",
        published_at=datetime(2024, 11, 5, tzinfo=UTC), text=text_b,
        sentences=[Sentence(index=0, start=0, end=len(text_b))],
    )
    modi_b = build_mention(document_b, 0, 7, "PERSON", "PER", 0.99, "transformer")
    putin_b = build_mention(document_b, 12, 26, "PERSON", "PER", 0.99, "transformer")
    document_b.mentions = [modi_b, putin_b]
    document_b.relations = [
        Relation(
            relation_id=make_relation_id("art_b", modi_b.mention_id, "met", putin_b.mention_id),
            article_id="art_b",
            subject_mention_id=modi_b.mention_id, subject_text="PM Modi", subject_label="PERSON",
            predicate="met",
            object_mention_id=putin_b.mention_id, object_text="Vladimir Putin", object_label="PERSON",
            confidence=0.75, sentence_index=0, evidence_start=0, evidence_end=len(text_b),
            extractor="dependency", trigger="met",
        )
    ]

    entities = [
        CanonicalEntity(
            entity_id="PERSON_00000", canonical_name="Narendra Modi", entity_type="PERSON",
            aliases={"Narendra Modi", "PM Modi"}, roles={"prime minister"}, countries={"India"},
            mention_ids=[modi_a.mention_id, modi_b.mention_id],
            article_ids=["art_a", "art_b"],
            first_seen=datetime(2024, 10, 22, tzinfo=UTC),
            last_seen=datetime(2024, 11, 5, tzinfo=UTC), merge_confidence=0.8,
        ),
        CanonicalEntity(
            entity_id="PERSON_00001", canonical_name="Vladimir Putin", entity_type="PERSON",
            aliases={"Vladimir Putin"},
            mention_ids=[putin_a.mention_id, putin_b.mention_id],
            article_ids=["art_a", "art_b"],
            first_seen=datetime(2024, 10, 22, tzinfo=UTC),
            last_seen=datetime(2024, 11, 5, tzinfo=UTC), merge_confidence=0.9,
        ),
    ]

    db_path = tmp_path / "kg.db"
    with connect(db_path) as connection:
        load_documents(connection, [document_a, document_b])
        load_entities(connection, entities)
        build_entity_relations(connection)
    return db_path


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_loading_is_idempotent(populated_db, tmp_path):
    """Re-running the loader must not duplicate rows -- the property that makes
    a crashed pipeline safe to re-run."""
    with connect(populated_db, create=False) as connection:
        before = database_stats(connection)
    with connect(populated_db) as connection:
        rows = connection.execute("SELECT * FROM articles").fetchall()
    assert before["articles"] == 2
    assert len(rows) == 2


def test_foreign_keys_are_enforced(populated_db):
    """SQLite disables foreign keys by DEFAULT. If the pragma is not applied, a
    relation can point at a mention that does not exist."""
    with connect(populated_db, create=False) as connection:
        with pytest.raises(Exception):
            connection.execute(
                """INSERT INTO relations
                   (relation_id, article_id, subject_mention_id, subject_text, subject_label,
                    predicate, object_mention_id, object_text, object_label, confidence,
                    evidence_count, sentence_index, evidence_start, evidence_end,
                    extractor, trigger, subject_via_coref, object_via_coref)
                   VALUES ('bad','art_a','no_such_mention','x','PERSON','met',
                           'also_missing','y','PERSON',0.5,1,0,0,1,'test','test',0,0)"""
            )


def test_noisy_or_increases_with_corroboration():
    """Independent evidence must RAISE confidence. A max would ignore the
    second source; a mean would be dragged down by a weaker third."""
    single = _noisy_or([0.7])
    double = _noisy_or([0.7, 0.7])
    triple = _noisy_or([0.7, 0.7, 0.7])
    assert single == pytest.approx(0.7)
    assert double > single
    assert triple > double
    assert triple <= 0.99  # capped: sources are not truly independent


def test_entity_relations_aggregate_mention_relations(populated_db):
    """Two articles asserting the same fact become ONE edge with support 2."""
    with connect(populated_db, create=False) as connection:
        edges = relations_for_entity(connection, "PERSON_00000", predicate="met")
    assert len(edges) == 1
    assert edges[0]["support_count"] == 2
    assert edges[0]["confidence"] > 0.75  # noisy-OR of two 0.75s


# ---------------------------------------------------------------------------
# Query layer
# ---------------------------------------------------------------------------


def test_alias_lookup_finds_the_canonical_entity(populated_db):
    """The payoff of Phase 6: a user typing "PM Modi" must find Narendra Modi."""
    with connect(populated_db, create=False) as connection:
        assert find_entity(connection, "PM Modi")[0]["canonical_name"] == "Narendra Modi"
        assert find_entity(connection, "Narendra Modi")[0]["entity_id"] == "PERSON_00000"


def test_query_parameters_are_bound_not_interpolated(populated_db):
    """A name containing SQL syntax must be treated as data."""
    with connect(populated_db, create=False) as connection:
        assert find_entity(connection, "'; DROP TABLE entities; --") == []
        assert database_stats(connection)["entities"] == 2


def test_entity_graph_groups_relations_by_predicate(populated_db):
    with connect(populated_db, create=False) as connection:
        graph = entity_graph(connection, "PERSON_00000")
    assert graph["canonical_name"] == "Narendra Modi"
    assert "PM Modi" in graph["aliases"]
    assert graph["attributes"]["country"] == ["India"]
    assert "met" in graph["relations"]


def test_articles_for_entity_shows_surface_forms(populated_db):
    """One row per article, showing which alias that article used."""
    with connect(populated_db, create=False) as connection:
        rows = articles_for_entity(connection, "PERSON_00000")
    assert len(rows) == 2
    surfaces = {s for row in rows for s in row["surfaces"].split(",")}
    assert surfaces == {"Narendra Modi", "PM Modi"}


def test_every_edge_can_be_traced_to_its_evidence(populated_db):
    """The auditability requirement: 'show me why you believe this'."""
    with connect(populated_db, create=False) as connection:
        edge = relations_for_entity(connection, "PERSON_00000", predicate="met")[0]
        evidence = relation_evidence(connection, edge["edge_id"])
    assert len(evidence) == 2
    for row in evidence:
        assert "Modi" in row["evidence"] and "Putin" in row["evidence"]


def test_low_confidence_audit_query(populated_db):
    with connect(populated_db, create=False) as connection:
        assert low_confidence_relations(connection, threshold=0.9)
        assert low_confidence_relations(connection, threshold=0.1) == []


def test_entities_by_attribute(populated_db):
    with connect(populated_db, create=False) as connection:
        rows = entities_by_attribute(connection, "country", "india")  # case-insensitive
    assert [r["canonical_name"] for r in rows] == ["Narendra Modi"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_f1_punishes_imbalance():
    """The reason to use the harmonic mean: perfect precision with zero recall
    must not look like a 0.5 system."""
    lopsided = PRF(true_positives=1, false_positives=0, false_negatives=99)
    assert lopsided.precision == 1.0
    assert lopsided.f1 < 0.05


def test_prf_from_sets():
    result = prf_from_sets({"a", "b", "c"}, {"b", "c", "d"})
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_perfect_clustering_scores_one():
    gold = [["a", "b", "c"], ["x", "y"]]
    assert muc(gold, gold).f1 == pytest.approx(1.0)
    assert b_cubed(gold, gold)[2] == pytest.approx(1.0)


def test_muc_is_fooled_by_over_merging_but_b_cubed_is_not():
    """The documented bias, demonstrated. This is WHY both are reported."""
    gold = [["a", "b"], ["x", "y"]]
    everything_merged = [["a", "b", "x", "y"]]

    assert muc(everything_merged, gold).recall == pytest.approx(1.0)
    assert b_cubed(everything_merged, gold)[0] < 0.6  # B-cubed precision drops


def test_false_merge_and_false_split_are_reported_separately():
    """An aggregate F1 cannot distinguish a corrupted knowledge base from a
    merely incomplete one."""
    gold = [["narendra", "modi_a"], ["lalit", "modi_b"]]
    merged = [["narendra", "modi_a", "lalit", "modi_b"]]
    split = [["narendra"], ["modi_a"], ["lalit"], ["modi_b"]]

    merge_report = evaluate_clustering(merged, gold)
    split_report = evaluate_clustering(split, gold)

    assert merge_report.false_merges > 0 and merge_report.false_splits == 0
    assert split_report.false_splits > 0 and split_report.false_merges == 0


def test_clustering_report_gives_examples():
    gold = [["a"], ["b"]]
    report = evaluate_clustering([["a", "b"]], gold)
    assert report.false_merge_examples == [("a", "b")]


# ---------------------------------------------------------------------------
# The acronym feature added in response to evaluation
# ---------------------------------------------------------------------------


def test_acronym_match():
    assert _acronym_match("bcci", "board control cricket india")
    assert _acronym_match("ndb", "new development bank")
    assert not _acronym_match("bcci", "new development bank")


def test_short_acronyms_are_rejected():
    """Two-letter acronyms collide far too easily to be evidence."""
    assert not _acronym_match("un", "united nations")


def test_acronym_raises_name_compatibility():
    """Regression for a false split found by evaluation: blocking paired BCCI
    with its expansion, but the scorer had no notion of an acronym."""
    from src.entity_resolution.profiles import EntityProfile

    def org(name: str) -> EntityProfile:
        from src.entity_resolution.normalize import normalize_name

        return EntityProfile(
            profile_id=name, article_id="a", source="s",
            published_at=datetime(2024, 1, 1, tzinfo=UTC), entity_type="ORG",
            canonical_surface=name, normalized=normalize_name(name, "ORG"),
        )

    assert name_compatibility(org("BCCI"), org("Board of Control for Cricket in India")) >= 0.8
    assert name_compatibility(org("NDB"), org("New Development Bank")) >= 0.8
