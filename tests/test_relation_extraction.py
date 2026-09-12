"""Tests for Phase 5 -- relation extraction.

The dependency tests DO load spaCy (it is small and fast, unlike the coref
model) because the parse is the thing under test. Pattern and dedup logic is
tested with hand-built data.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.coreference.base import build_cluster
from src.ner.base import build_mention
from src.relation_extraction.base import (
    resolve_argument,
    resolve_arguments,
    types_are_valid,
    verify_relations,
)
from src.relation_extraction.dependency import DependencyRelationExtractor
from src.relation_extraction.patterns import PatternRelationExtractor
from src.relation_extraction.pipeline import deduplicate
from src.schemas import Document, Sentence, make_relation_id


def _document(text: str, spans: list[tuple[int, int, str]], raw: dict[int, str] | None = None) -> Document:
    """Build a document with sentences and hand-placed entity mentions."""
    document = Document(
        article_id="art_test",
        title="T",
        source="S",
        published_at=datetime(2024, 10, 22, tzinfo=timezone.utc),
        text=text,
        sentences=[Sentence(index=0, start=0, end=len(text))],
    )
    document.mentions = [
        build_mention(
            document, s, e, label, (raw or {}).get(i, label), 0.9, "test"
        )
        for i, (s, e, label) in enumerate(spans)
    ]
    return document


def _find(text: str, needle: str) -> tuple[int, int]:
    start = text.index(needle)
    return start, start + len(needle)


# ---------------------------------------------------------------------------
# Identity and type constraints
# ---------------------------------------------------------------------------


def test_relation_id_is_deterministic() -> None:
    """Re-running extraction must not duplicate every fact."""
    assert make_relation_id("a", "m1", "met", "m2") == make_relation_id("a", "m1", "met", "m2")
    assert make_relation_id("a", "m1", "met", "m2") != make_relation_id("a", "m2", "met", "m1")


def test_argument_type_constraints() -> None:
    """The parse says X is an object; it does not say X is a topic."""
    assert types_are_valid("discussed", "PERSON", "TOPIC")
    assert not types_are_valid("discussed", "LOCATION", "DATE")
    assert types_are_valid("holds_position", "PERSON", "ROLE")
    assert not types_are_valid("holds_position", "ORG", "ROLE")
    assert types_are_valid("met", "PERSON", "ORG")


def test_unknown_predicate_is_not_constrained() -> None:
    assert types_are_valid("some_new_predicate", "PERSON", "DATE")


# ---------------------------------------------------------------------------
# Pattern extractor
# ---------------------------------------------------------------------------


def test_role_before_person_gives_holds_position() -> None:
    text = "Prime Minister Narendra Modi arrived."
    document = _document(text, [(*_find(text, "Prime Minister"), "ROLE"),
                                (*_find(text, "Narendra Modi"), "PERSON")])
    relations = PatternRelationExtractor().extract(document)
    assert any(
        r.predicate == "holds_position"
        and r.subject_text == "Narendra Modi"
        and r.object_text == "Prime Minister"
        for r in relations
    )


def test_demonym_role_person_gives_represents() -> None:
    """'Indian Prime Minister Narendra Modi' tells us who Modi represents."""
    text = "Indian Prime Minister Narendra Modi arrived."
    document = _document(
        text,
        [(*_find(text, "Indian"), "MISC"),
         (*_find(text, "Prime Minister"), "ROLE"),
         (*_find(text, "Narendra Modi"), "PERSON")],
        raw={0: "GAZ_DEMONYM:India"},
    )
    relations = PatternRelationExtractor().extract(document)
    represents = [r for r in relations if r.predicate == "represents"]
    assert len(represents) == 1
    # The mention span is "Indian"; the RELATION's object is the country.
    assert represents[0].object_text == "India"
    assert represents[0].object_label == "COUNTRY"


def test_role_of_org_gives_works_for() -> None:
    text = "Mukesh Ambani, chairman of Reliance, met executives."
    document = _document(text, [(*_find(text, "Mukesh Ambani"), "PERSON"),
                                (*_find(text, "chairman"), "ROLE"),
                                (*_find(text, "Reliance"), "ORG")])
    relations = PatternRelationExtractor().extract(document)
    assert any(
        r.predicate == "works_for" and r.subject_text == "Mukesh Ambani"
        and r.object_text == "Reliance"
        for r in relations
    )


def test_distant_mentions_are_not_treated_as_adjacent() -> None:
    """Adjacency means nothing but whitespace or a short connector between."""
    text = "Prime Minister Abiy Ahmed spoke at length before Narendra Modi replied."
    document = _document(text, [(*_find(text, "Prime Minister"), "ROLE"),
                                (*_find(text, "Narendra Modi"), "PERSON")])
    relations = PatternRelationExtractor().extract(document)
    assert not any(r.subject_text == "Narendra Modi" for r in relations)


# ---------------------------------------------------------------------------
# Dependency extractor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dependency_extractor():
    return DependencyRelationExtractor()


def test_simple_subject_verb_object(dependency_extractor) -> None:
    text = "Narendra Modi met Vladimir Putin."
    document = _document(text, [(*_find(text, "Narendra Modi"), "PERSON"),
                                (*_find(text, "Vladimir Putin"), "PERSON")])
    relations = dependency_extractor.extract(document)
    assert any(
        r.predicate == "met" and r.subject_text == "Narendra Modi"
        and r.object_text == "Vladimir Putin"
        for r in relations
    )


def test_intervening_clause_does_not_break_extraction(dependency_extractor) -> None:
    """The whole reason to use a parse rather than string adjacency."""
    text = "Narendra Modi, who arrived on Tuesday, met Vladimir Putin."
    document = _document(text, [(*_find(text, "Narendra Modi"), "PERSON"),
                                (*_find(text, "Vladimir Putin"), "PERSON")])
    relations = dependency_extractor.extract(document)
    assert any(r.predicate == "met" and r.subject_text == "Narendra Modi" for r in relations)


def test_coordinated_object_yields_both_relations(dependency_extractor) -> None:
    text = "Narendra Modi met Vladimir Putin and Xi Jinping."
    document = _document(text, [(*_find(text, "Narendra Modi"), "PERSON"),
                                (*_find(text, "Vladimir Putin"), "PERSON"),
                                (*_find(text, "Xi Jinping"), "PERSON")])
    objects = {r.object_text for r in dependency_extractor.extract(document) if r.predicate == "met"}
    assert {"Vladimir Putin", "Xi Jinping"} <= objects


def test_negation_is_not_extracted_as_fact(dependency_extractor) -> None:
    """Extracting a negated sentence asserts the OPPOSITE of the article --
    the worst possible error in a fact store."""
    text = "Narendra Modi did not meet Vladimir Putin."
    document = _document(text, [(*_find(text, "Narendra Modi"), "PERSON"),
                                (*_find(text, "Vladimir Putin"), "PERSON")])
    assert not [r for r in dependency_extractor.extract(document) if r.predicate == "met"]


def test_modal_hedging_lowers_confidence(dependency_extractor) -> None:
    """'may meet' did not happen yet; it should not look as certain as 'met'."""
    certain = _document(
        "Narendra Modi met Vladimir Putin.",
        [(0, 13, "PERSON"), (18, 32, "PERSON")],
    )
    hedged_text = "Narendra Modi will meet Vladimir Putin."
    hedged = _document(hedged_text, [(*_find(hedged_text, "Narendra Modi"), "PERSON"),
                                     (*_find(hedged_text, "Vladimir Putin"), "PERSON")])

    certain_conf = max(r.confidence for r in dependency_extractor.extract(certain))
    hedged_conf = max(r.confidence for r in dependency_extractor.extract(hedged))
    assert hedged_conf < certain_conf


def test_wrong_argument_types_are_rejected(dependency_extractor) -> None:
    """'Kazan discussed Wednesday' parses fine and is nonsense."""
    text = "Kazan discussed Wednesday."
    document = _document(text, [(*_find(text, "Kazan"), "LOCATION"),
                                (*_find(text, "Wednesday"), "DATE")])
    assert dependency_extractor.extract(document) == []


# ---------------------------------------------------------------------------
# Coreference-driven argument resolution
# ---------------------------------------------------------------------------


def test_pronoun_subject_resolves_via_coreference() -> None:
    """The payoff of Phase 4: without it this sentence yields nothing usable."""
    text = "Narendra Modi arrived. He met Vladimir Putin."
    document = _document(text, [(*_find(text, "Narendra Modi"), "PERSON"),
                                (*_find(text, "Vladimir Putin"), "PERSON")])
    cluster = build_cluster(document, [_find(text, "Narendra Modi"), _find(text, "He")], 0, "test")
    assert cluster is not None
    document.coref_clusters = [cluster]
    from src.coreference.base import align_clusters_to_mentions

    align_clusters_to_mentions(document)

    start, end = _find(text, "He")
    resolved = resolve_argument(document, start, end)
    assert resolved is not None
    assert resolved.text == "Narendra Modi"
    assert resolved.via_coref is True


def test_unresolved_pronoun_is_not_a_usable_argument() -> None:
    text = "He met Vladimir Putin."
    document = _document(text, [(*_find(text, "Vladimir Putin"), "PERSON")])
    assert resolve_argument(document, 0, 2) is None


def test_plural_antecedent_yields_every_member() -> None:
    """'The two leaders discussed energy' is a fact about BOTH of them.
    Collapsing it to one argument silently drops half the facts."""
    text = "Modi and Putin arrived. The two leaders discussed energy cooperation."
    document = _document(text, [(*_find(text, "Modi"), "PERSON"),
                                (*_find(text, "Putin"), "PERSON"),
                                (*_find(text, "energy cooperation"), "TOPIC")])
    cluster = build_cluster(
        document, [_find(text, "Modi and Putin"), _find(text, "The two leaders")], 0, "test"
    )
    assert cluster is not None
    document.coref_clusters = [cluster]
    from src.coreference.base import align_clusters_to_mentions

    align_clusters_to_mentions(document)

    start, end = _find(text, "The two leaders")
    resolved = resolve_arguments(document, start, end)
    assert {r.text for r in resolved} == {"Modi", "Putin"}


def test_named_mentions_are_not_redirected_through_coreference() -> None:
    """'Narendra Modi' IS the entity -- replacing it with a cluster
    representative would be pointless indirection that can only add error."""
    text = "Narendra Modi arrived. Modi met Putin."
    document = _document(text, [(*_find(text, "Narendra Modi"), "PERSON")])
    cluster = build_cluster(
        document, [_find(text, "Narendra Modi"), (23, 27)], 0, "test"
    )
    assert cluster is not None
    document.coref_clusters = [cluster]
    resolved = resolve_argument(document, *_find(text, "Narendra Modi"))
    assert resolved is not None and resolved.via_coref is False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_repeated_fact_is_collapsed_and_counted() -> None:
    """Our documents prepend the title, so headline facts repeat in the body.
    Repetition is corroboration, so we count it rather than discard it."""
    text = "Prime Minister Modi arrived. Prime Minister Modi spoke."
    document = _document(text, [(0, 14, "ROLE"), (15, 19, "PERSON"),
                                (29, 43, "ROLE"), (44, 48, "PERSON")])
    relations = PatternRelationExtractor().extract(document)
    merged = deduplicate(relations)
    holds = [r for r in merged if r.predicate == "holds_position"]
    assert len(holds) == 1
    assert holds[0].evidence_count == 2


def test_deduplicate_keeps_highest_confidence() -> None:
    text = "Prime Minister Narendra Modi arrived."
    document = _document(text, [(*_find(text, "Prime Minister"), "ROLE"),
                                (*_find(text, "Narendra Modi"), "PERSON")])
    relations = PatternRelationExtractor().extract(document)
    low = relations[0].model_copy(update={"confidence": 0.2})
    high = relations[0].model_copy(update={"confidence": 0.95})
    assert deduplicate([low, high])[0].confidence == 0.95


def test_self_relations_are_rejected() -> None:
    """Coreference makes 'Modi said he would go' resolve both arguments to
    Modi; a self-loop is never a useful fact."""
    text = "Narendra Modi met Narendra Modi."
    document = _document(text, [(0, 13, "PERSON")])
    relations = DependencyRelationExtractor().extract(document)
    assert all(r.subject_mention_id != r.object_mention_id for r in relations)


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_verify_relations_detects_dangling_arguments() -> None:
    text = "Prime Minister Narendra Modi arrived."
    document = _document(text, [(*_find(text, "Prime Minister"), "ROLE"),
                                (*_find(text, "Narendra Modi"), "PERSON")])
    document.relations = PatternRelationExtractor().extract(document)
    assert verify_relations(document) == []

    document.mentions = []
    assert verify_relations(document) != []


def test_every_relation_carries_evidence() -> None:
    """A fact whose source sentence cannot be shown is not auditable."""
    text = "Prime Minister Narendra Modi arrived."
    document = _document(text, [(*_find(text, "Prime Minister"), "ROLE"),
                                (*_find(text, "Narendra Modi"), "PERSON")])
    for relation in PatternRelationExtractor().extract(document):
        assert relation.evidence_start >= 0
        assert relation.evidence_in(document.text)
