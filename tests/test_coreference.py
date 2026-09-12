"""Tests for Phase 4 -- coreference resolution.

As in Phase 3, OUR logic is tested deterministically with hand-built data; the
neural model is not exercised here because a 1.7 GB, 13-second-per-document
model makes a test suite useless, and asserting on a model's exact output makes
tests fail whenever the model is updated, which teaches you nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.coreference.base import (
    align_clusters_to_mentions,
    build_cluster,
    verify_clusters,
)
from src.coreference.pipeline import resolve_mention, run_coreference
from src.coreference.rule_resolver import (
    RuleBasedCorefResolver,
    _UnionFind,
    normalize_name,
)
from src.ner.base import build_mention
from src.schemas import (
    CorefMention,
    Document,
    Sentence,
    choose_representative,
    classify_mention_form,
)


def _document(text: str, sentence_spans: list[tuple[int, int]] | None = None) -> Document:
    spans = sentence_spans or [(0, len(text))]
    return Document(
        article_id="art_test",
        title="T",
        source="S",
        published_at=datetime(2024, 10, 22, tzinfo=timezone.utc),
        text=text,
        sentences=[Sentence(index=i, start=s, end=e) for i, (s, e) in enumerate(spans)],
    )


# ---------------------------------------------------------------------------
# Mention form classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "surface,expected",
    [
        ("He", "PRONOMINAL"),
        ("his", "PRONOMINAL"),
        ("They", "PRONOMINAL"),
        ("Narendra Modi", "NAMED"),
        ("Modi", "NAMED"),
        ("New Development Bank", "NAMED"),
        ("The Indian Prime Minister", "NOMINAL"),
        ("the lender", "NOMINAL"),
        ("the two leaders", "NOMINAL"),
    ],
)
def test_classify_mention_form(surface: str, expected: str) -> None:
    assert classify_mention_form(surface) == expected


def test_determiner_beats_capitalisation() -> None:
    """News style capitalises role titles, so capitalisation alone calls
    'The Indian Prime Minister' a name. The leading determiner is the fix."""
    assert classify_mention_form("The Indian Prime Minister") == "NOMINAL"
    assert classify_mention_form("Indian Prime Minister") == "NAMED"


# ---------------------------------------------------------------------------
# Representative selection
# ---------------------------------------------------------------------------


def test_named_mention_beats_pronoun_as_representative() -> None:
    mentions = [
        CorefMention(start=0, end=2, text="He", form="PRONOMINAL"),
        CorefMention(start=10, end=14, text="Modi", form="NAMED"),
    ]
    assert mentions[choose_representative(mentions)].text == "Modi"


def test_longest_named_mention_wins() -> None:
    """A full name resolves far more easily in Phase 6 than a surname."""
    mentions = [
        CorefMention(start=0, end=4, text="Modi", form="NAMED"),
        CorefMention(start=10, end=23, text="Narendra Modi", form="NAMED"),
    ]
    assert mentions[choose_representative(mentions)].text == "Narendra Modi"


def test_nominal_beats_pronoun_when_no_name_present() -> None:
    mentions = [
        CorefMention(start=0, end=2, text="He", form="PRONOMINAL"),
        CorefMention(start=10, end=20, text="the lender", form="NOMINAL"),
    ]
    assert mentions[choose_representative(mentions)].text == "the lender"


def test_representative_choice_is_deterministic() -> None:
    """Non-determinism here would make the pipeline emit different entity IDs
    for identical input, which breaks idempotency downstream."""
    mentions = [
        CorefMention(start=5, end=9, text="Modi", form="NAMED"),
        CorefMention(start=0, end=4, text="Modi", form="NAMED"),
    ]
    assert choose_representative(mentions) == choose_representative(mentions)
    assert mentions[choose_representative(mentions)].start == 0  # earliest wins


# ---------------------------------------------------------------------------
# Cluster construction
# ---------------------------------------------------------------------------


def test_singleton_clusters_are_dropped() -> None:
    """A 'cluster' of one is just a mention -- it carries no coreference
    information. Phase 10 must know we drop them, since some metrics count them."""
    document = _document("Narendra Modi arrived.")
    assert build_cluster(document, [(0, 13)], 0, "test") is None


def test_duplicate_spans_are_deduplicated() -> None:
    document = _document("Narendra Modi met Modi again.")
    cluster = build_cluster(document, [(0, 13), (0, 13), (18, 22)], 0, "test")
    assert cluster is not None
    assert len(cluster.mentions) == 2


def test_out_of_bounds_spans_are_dropped_not_fatal() -> None:
    document = _document("Short.")
    assert build_cluster(document, [(0, 5), (100, 200)], 0, "test") is None


def test_cluster_mentions_are_sorted_by_position() -> None:
    document = _document("Narendra Modi arrived. He spoke. Modi left.")
    cluster = build_cluster(document, [(33, 37), (0, 13), (23, 25)], 0, "test")
    assert cluster is not None
    assert [m.start for m in cluster.mentions] == [0, 23, 33]


def test_verify_clusters_detects_span_drift() -> None:
    document = _document("Narendra Modi arrived. He spoke.")
    cluster = build_cluster(document, [(0, 13), (23, 25)], 0, "test")
    assert cluster is not None
    document.coref_clusters = [cluster]
    assert verify_clusters(document) == []

    drifted = document.model_copy(update={"text": "Someone entirely different here."})
    assert verify_clusters(drifted) != []


# ---------------------------------------------------------------------------
# Union-Find (transitive closure)
# ---------------------------------------------------------------------------


def test_union_find_computes_transitive_closure() -> None:
    """Coreference is an equivalence relation: A~B and B~C implies A~C even
    though A and C were never directly compared."""
    union = _UnionFind()
    union.union("a", "b")
    union.union("b", "c")
    union.union("x", "y")
    groups = {frozenset(v) for v in union.groups().values()}
    assert groups == {frozenset({"a", "b", "c"}), frozenset({"x", "y"})}


# ---------------------------------------------------------------------------
# Rule-based resolver
# ---------------------------------------------------------------------------


def _document_with_mentions(text: str, spans: list[tuple[int, int, str]]) -> Document:
    document = _document(text)
    document.mentions = [
        build_mention(document, s, e, label, label, 0.9, "test") for s, e, label in spans
    ]
    return document


def test_normalize_name_strips_honorifics() -> None:
    assert normalize_name("Mr. Modi") == "modi"
    assert normalize_name("Modi") == "modi"
    assert normalize_name("Dr. S. Jaishankar") == "s jaishankar"


def test_surname_links_to_full_name() -> None:
    text = "Narendra Modi arrived today. Modi then left."
    document = _document_with_mentions(text, [(0, 13, "PERSON"), (29, 33, "PERSON")])
    clusters = RuleBasedCorefResolver().resolve(document)
    assert len(clusters) == 1
    assert clusters[0].representative_text == "Narendra Modi"


def test_pronoun_links_to_nearest_preceding_person() -> None:
    text = "Narendra Modi arrived. He was received at the airport."
    document = _document_with_mentions(text, [(0, 13, "PERSON")])
    document.sentences = [Sentence(index=0, start=0, end=21), Sentence(index=1, start=23, end=54)]
    clusters = RuleBasedCorefResolver().resolve(document)
    assert len(clusters) == 1
    assert {m.text for m in clusters[0].mentions} == {"Narendra Modi", "He"}


def test_pronoun_does_not_link_to_wrong_entity_type() -> None:
    """'He' must not attach to an ORG."""
    text = "Reliance Industries reported profit. He declined to comment."
    document = _document_with_mentions(text, [(0, 19, "ORG")])
    document.sentences = [Sentence(index=0, start=0, end=35), Sentence(index=1, start=37, end=60)]
    assert RuleBasedCorefResolver().resolve(document) == []


def test_gender_consistency_prevents_mixing_he_and_she() -> None:
    """Not gender KNOWLEDGE -- a consistency constraint. One cluster must not
    absorb both 'he' and 'she'."""
    text = "Narendra Modi spoke. He agreed. She disagreed."
    document = _document_with_mentions(text, [(0, 13, "PERSON")])
    document.sentences = [
        Sentence(index=0, start=0, end=19),
        Sentence(index=1, start=21, end=30),
        Sentence(index=2, start=32, end=46),
    ]
    clusters = RuleBasedCorefResolver().resolve(document)
    texts = {m.text for c in clusters for m in c.mentions}
    assert not ({"He", "She"} <= texts)


def test_pronoun_beyond_distance_limit_is_not_linked() -> None:
    text = "Narendra Modi spoke. A. B. C. D. He left."
    document = _document_with_mentions(text, [(0, 13, "PERSON")])
    document.sentences = [
        Sentence(index=0, start=0, end=19),
        Sentence(index=1, start=21, end=23),
        Sentence(index=2, start=24, end=26),
        Sentence(index=3, start=27, end=29),
        Sentence(index=4, start=30, end=32),
        Sentence(index=5, start=33, end=41),
    ]
    clusters = RuleBasedCorefResolver(max_sentence_distance=2).resolve(document)
    assert all("He" not in {m.text for m in c.mentions} for c in clusters)


def test_documented_weakness_recency_ignores_syntax() -> None:
    """PIN THE KNOWN FAILURE so it is a limitation, not a surprise.

    In "Modi met Putin. He said...", "He" is far more likely to be Modi (the
    SUBJECT). A recency rule picks Putin, the nearest. Only a model that
    represents syntax and discourse salience gets this right.
    """
    text = "Modi met Putin. He said the talks were productive."
    document = _document_with_mentions(text, [(0, 4, "PERSON"), (9, 14, "PERSON")])
    document.sentences = [Sentence(index=0, start=0, end=15), Sentence(index=1, start=16, end=50)]
    clusters = RuleBasedCorefResolver().resolve(document)
    with_pronoun = [c for c in clusters if any(m.text == "He" for m in c.mentions)]
    assert with_pronoun, "the rule should link the pronoun to something"
    assert with_pronoun[0].representative_text == "Putin"  # wrong, and known to be


# ---------------------------------------------------------------------------
# Alignment to NER mentions, and the downstream interface
# ---------------------------------------------------------------------------


def test_alignment_handles_span_mismatch_between_stages() -> None:
    """Coref generates its own spans, so exact equality is too strict:
    NER   'Narendra Modi'                 (15, 28)
    coref 'Prime Minister Narendra Modi'  ( 0, 28)
    """
    text = "Prime Minister Narendra Modi arrived. He spoke."
    document = _document_with_mentions(text, [(15, 28, "PERSON")])
    cluster = build_cluster(document, [(0, 28), (37, 39)], 0, "test")
    assert cluster is not None
    document.coref_clusters = [cluster]

    assert align_clusters_to_mentions(document) == 1
    assert document.mentions[0].coref_cluster_id == cluster.cluster_id
    assert cluster.mentions[0].ner_mention_id == document.mentions[0].mention_id


def test_touching_spans_are_not_linked() -> None:
    """Adjacent-but-not-containing spans must not be joined."""
    text = "Narendra Modi arrived. He spoke."
    document = _document_with_mentions(text, [(0, 13, "PERSON")])
    cluster = build_cluster(document, [(14, 21), (23, 25)], 0, "test")
    assert cluster is not None
    document.coref_clusters = [cluster]
    assert align_clusters_to_mentions(document) == 0


def test_resolve_mention_returns_representative_not_rewritten_text() -> None:
    """The Phase 5 interface. The document text must be untouched."""
    text = "Narendra Modi arrived. He was received at the airport."
    document = _document_with_mentions(text, [(0, 13, "PERSON")])
    document.sentences = [Sentence(index=0, start=0, end=21), Sentence(index=1, start=23, end=54)]

    original_text = document.text
    run_coreference([document], resolver=RuleBasedCorefResolver())

    assert document.text == original_text  # NEVER rewritten
    mention_id = document.mentions[0].mention_id
    assert resolve_mention(document, mention_id) == "Narendra Modi"


def test_resolve_mention_for_unclustered_mention_returns_its_own_text() -> None:
    text = "Kazan hosted the summit."
    document = _document_with_mentions(text, [(0, 5, "LOCATION")])
    assert resolve_mention(document, document.mentions[0].mention_id) == "Kazan"


def test_resolve_mention_unknown_id_returns_none() -> None:
    document = _document_with_mentions("Kazan hosted.", [(0, 5, "LOCATION")])
    assert resolve_mention(document, "no-such-mention") is None


def test_alignment_prefers_referent_over_modifier_label() -> None:
    """Regression test for a real bug found on the corpus.

    "Prime Minister Narendra Modi" contains both a ROLE ("Prime Minister", 14
    chars) and a PERSON ("Narendra Modi", 13 chars). Largest-overlap picked the
    ROLE by one character. A person is the REFERENT; a role merely describes
    one, so label priority must win.
    """
    text = "Prime Minister Narendra Modi arrived. He spoke."
    document = _document(text)
    document.mentions = [
        build_mention(document, 0, 14, "ROLE", "GAZ_ROLE", 0.9, "gazetteer"),
        build_mention(document, 15, 28, "PERSON", "PER", 0.99, "transformer"),
    ]
    cluster = build_cluster(document, [(0, 28), (37, 39)], 0, "test")
    assert cluster is not None
    document.coref_clusters = [cluster]
    align_clusters_to_mentions(document)

    linked_id = cluster.mentions[0].ner_mention_id
    linked = next(m for m in document.mentions if m.mention_id == linked_id)
    assert linked.label == "PERSON"


def test_alignment_prefers_head_final_person_over_longer_org() -> None:
    """The Lalit Modi case: "Former Indian Premier League chairman Lalit Modi"
    linked to the ORG (21 chars) rather than the PERSON (10 chars)."""
    text = "Former Indian Premier League chairman Lalit Modi lost an appeal."
    document = _document(text)
    document.mentions = [
        build_mention(document, 7, 28, "ORG", "ORG", 0.95, "transformer"),
        build_mention(document, 38, 48, "PERSON", "PER", 0.99, "transformer"),
    ]
    cluster = build_cluster(document, [(0, 48), (49, 53)], 0, "test")
    assert cluster is not None
    document.coref_clusters = [cluster]
    align_clusters_to_mentions(document)

    linked_id = cluster.mentions[0].ner_mention_id
    linked = next(m for m in document.mentions if m.mention_id == linked_id)
    assert linked.text == "Lalit Modi"
