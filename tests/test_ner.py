"""Tests for Phase 3 -- named entity recognition.

Structure note: the merge logic is tested with HAND-BUILT mentions, not by
running models. Model-dependent tests are slow and, worse, they break whenever
a model is updated -- which teaches you nothing about your own code. We test
OUR logic deterministically, and test the models only for the contract we rely
on (offsets are valid, labels are mapped).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.ner.base import OffsetError, build_mention, trim_span, verify_mentions
from src.ner.gazetteer import GazetteerExtractor, canonical_country
from src.ner.labels import map_label, strip_bio_prefix
from src.ner.merge import MergeStats, merge_mentions
from src.schemas import Document, Mention, Sentence


def _document(text: str) -> Document:
    return Document(
        article_id="art_test",
        title="T",
        source="S",
        published_at=datetime(2024, 10, 22, tzinfo=timezone.utc),
        text=text,
        sentences=[Sentence(index=0, start=0, end=len(text))],
    )


def _mention(start: int, end: int, label: str, extractor: str, score: float = 0.9) -> Mention:
    return Mention(
        mention_id=f"m{start}-{end}-{label}",
        article_id="art_test",
        start=start,
        end=end,
        text="x" * (end - start),
        label=label,
        raw_label=label,
        score=score,
        extractor=extractor,
    )


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def test_model_labels_map_to_shared_vocabulary() -> None:
    """The whole point of the mapping layer: two models, one vocabulary."""
    assert map_label("PERSON", "spacy") == "PERSON"
    assert map_label("PER", "transformer") == "PERSON"
    assert map_label("GPE", "spacy") == "LOCATION"
    assert map_label("LOC", "transformer") == "LOCATION"


def test_uninteresting_labels_are_dropped() -> None:
    assert map_label("MONEY", "spacy") is None
    assert map_label("PERCENT", "spacy") is None


def test_unknown_label_degrades_to_misc_rather_than_raising() -> None:
    """A model update that adds a label must not crash a 100k-article batch."""
    assert map_label("SOME_NEW_LABEL", "spacy") == "MISC"


def test_unknown_source_raises() -> None:
    """An unknown SOURCE is a programming error, unlike an unknown label."""
    with pytest.raises(ValueError, match="No label map"):
        map_label("PER", "not_a_real_extractor")


def test_strip_bio_prefix() -> None:
    assert strip_bio_prefix("B-PER") == "PER"
    assert strip_bio_prefix("I-ORG") == "ORG"
    assert strip_bio_prefix("O") == "O"


# ---------------------------------------------------------------------------
# Mention construction and offset safety
# ---------------------------------------------------------------------------


def test_build_mention_stores_the_actual_surface_text() -> None:
    document = _document("Narendra Modi met Putin.")
    mention = build_mention(document, 0, 13, "PERSON", "PER", 0.99, "test")
    assert mention.text == "Narendra Modi"
    assert mention.sentence_index == 0


def test_out_of_bounds_span_raises_rather_than_clamping() -> None:
    """A silently clamped span points at the wrong words and produces a wrong
    entity with nothing in the logs. Fail loudly instead."""
    document = _document("Short text.")
    with pytest.raises(OffsetError):
        build_mention(document, 0, 999, "PERSON", "PER", 0.9, "test")
    with pytest.raises(OffsetError):
        build_mention(document, 5, 5, "PERSON", "PER", 0.9, "test")


def test_score_is_clamped_to_unit_interval() -> None:
    document = _document("Narendra Modi met Putin.")
    assert build_mention(document, 0, 13, "PERSON", "PER", 1.7, "test").score == 1.0
    assert build_mention(document, 0, 13, "PERSON", "PER", -3.0, "test").score == 0.0


def test_trim_span_removes_trailing_whitespace_and_punctuation() -> None:
    """'Modi ' and 'Modi' must not become two different candidate entities."""
    text = "Modi , and Reliance ."
    assert text[slice(*trim_span(text, 0, 6))] == "Modi"
    assert text[slice(*trim_span(text, 11, 21))] == "Reliance"


def test_trim_span_keeps_abbreviation_periods() -> None:
    """'U.S.' and 'Inc.' need their trailing period."""
    text = "Morgan Stanley Inc. said"
    start, end = trim_span(text, 0, 19)
    assert text[start:end] == "Morgan Stanley Inc."


def test_verify_mentions_detects_drift() -> None:
    """Mention.text is a denormalised copy; this is the tripwire for drift."""
    document = _document("Narendra Modi met Putin.")
    document.mentions = [build_mention(document, 0, 13, "PERSON", "PER", 0.9, "test")]
    assert verify_mentions(document) == []

    drifted = document.model_copy(update={"text": "Someone else met Putin.."})
    assert verify_mentions(drifted) != []


# ---------------------------------------------------------------------------
# Gazetteer
# ---------------------------------------------------------------------------


def test_gazetteer_finds_countries_and_roles() -> None:
    document = _document("Prime Minister Narendra Modi represents India.")
    mentions = GazetteerExtractor().extract(document)
    by_label = {m.label: m.text for m in mentions}
    assert by_label["ROLE"] == "Prime Minister"
    assert by_label["COUNTRY"] == "India"


def test_longest_role_wins_over_its_own_substring() -> None:
    """'Deputy Foreign Minister' must not be matched as 'Foreign Minister'.
    This is the classic gazetteer bug, prevented by longest-first ordering."""
    document = _document("Deputy Foreign Minister Sergei Ryabkov attended.")
    roles = [m.text for m in GazetteerExtractor().extract(document) if m.label == "ROLE"]
    assert roles == ["Deputy Foreign Minister"]


def test_demonyms_carry_the_resolved_country() -> None:
    """'Indian' is not a country -- it is EVIDENCE pointing at one."""
    document = _document("The Indian Prime Minister spoke.")
    demonyms = [m for m in GazetteerExtractor().extract(document) if "DEMONYM" in m.raw_label]
    assert demonyms[0].text == "Indian"
    assert demonyms[0].raw_label == "GAZ_DEMONYM:India"
    assert demonyms[0].label == "MISC"


def test_country_aliases_resolve() -> None:
    assert canonical_country("UAE") == "United Arab Emirates"
    assert canonical_country("Britain") == "United Kingdom"
    assert canonical_country("Indian") == "India"
    assert canonical_country("Kazan") is None


def test_event_pattern_matches_summit_names() -> None:
    document = _document("Leaders met at the 16th BRICS Summit in Kazan.")
    events = [m.text for m in GazetteerExtractor().extract(document) if m.label == "EVENT"]
    assert "16th BRICS Summit" in events


def test_topics_are_case_insensitive_and_canonicalised() -> None:
    document = _document("They discussed artificial intelligence and AI policy.")
    topics = [m for m in GazetteerExtractor().extract(document) if m.label == "TOPIC"]
    canonicals = {m.raw_label for m in topics}
    assert canonicals == {"GAZ_TOPIC:artificial intelligence"}


# ---------------------------------------------------------------------------
# Merge -- the decision logic, tested deterministically
# ---------------------------------------------------------------------------


def test_non_overlapping_mentions_all_survive() -> None:
    merged = merge_mentions([[_mention(0, 5, "PERSON", "spacy")], [_mention(10, 15, "ORG", "spacy")]])
    assert len(merged) == 2


def test_transformer_person_beats_spacy_org() -> None:
    """The real 'Sitharaman' case: spaCy called a person an organisation."""
    merged = merge_mentions([
        [_mention(0, 10, "ORG", "spacy")],
        [_mention(0, 10, "PERSON", "transformer")],
    ])
    assert len(merged) == 1
    assert merged[0].label == "PERSON"
    assert merged[0].extractor == "transformer"


def test_gazetteer_country_beats_model_location() -> None:
    """The 'India' case: GPE cannot distinguish a country from a city, so the
    closed-set lookup wins."""
    merged = merge_mentions([
        [_mention(0, 5, "LOCATION", "spacy")],
        [_mention(0, 5, "COUNTRY", "gazetteer")],
    ])
    assert merged[0].label == "COUNTRY"


def test_gazetteer_topic_beats_spacy_location() -> None:
    """The 'AI' case that made us correct the trust matrix."""
    merged = merge_mentions([
        [_mention(0, 2, "LOCATION", "spacy")],
        [_mention(0, 2, "TOPIC", "gazetteer")],
    ])
    assert merged[0].label == "TOPIC"


def test_misc_never_beats_a_specific_label() -> None:
    """MISC means 'we could not tell' and must lose to a real answer."""
    merged = merge_mentions([
        [_mention(0, 12, "MISC", "transformer", score=1.0)],
        [_mention(0, 12, "EVENT", "gazetteer", score=0.8)],
    ])
    assert merged[0].label == "EVENT"


def test_longer_span_wins_when_trust_is_equal() -> None:
    """'Narendra Modi' should beat 'Modi' -- longer spans carry more info."""
    merged = merge_mentions([
        [_mention(0, 4, "PERSON", "transformer"), _mention(0, 13, "PERSON", "transformer")],
    ])
    assert len(merged) == 1
    assert merged[0].length == 13


def test_merged_output_has_no_overlaps_and_is_sorted() -> None:
    merged = merge_mentions([
        [_mention(0, 10, "ORG", "spacy"), _mention(20, 30, "PERSON", "spacy")],
        [_mention(5, 15, "PERSON", "transformer"), _mention(25, 35, "ORG", "transformer")],
    ])
    for previous, current in zip(merged, merged[1:]):
        assert previous.end <= current.start
        assert previous.start <= current.start


def test_merge_stats_are_recorded() -> None:
    """Observability: the merge must be inspectable, not a black box."""
    stats = MergeStats()
    merge_mentions([
        [_mention(0, 10, "ORG", "spacy")],
        [_mention(0, 10, "PERSON", "transformer")],
    ], stats)
    assert stats.input_mentions == 2
    assert stats.output_mentions == 1
    assert stats.conflicts == 1
    assert stats.winners_by_extractor["transformer"] == 1
    assert stats.dropped_by_extractor["spacy"] == 1


def test_empty_input_is_handled() -> None:
    assert merge_mentions([]) == []
    assert merge_mentions([[], []]) == []
