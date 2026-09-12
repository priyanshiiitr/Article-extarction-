"""Tests for Phase 6 -- entity resolution.

The most important tests here are the FALSE MERGE tests. A false split costs
recall and is recoverable; a false merge permanently fuses two real people and
corrupts every fact attributed to them. The suite is weighted accordingly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.entity_resolution.blocking import (
    blocking_keys,
    build_blocks,
    candidate_pairs,
    phonetic_code,
)
from src.entity_resolution.features import (
    calibrate_context_scores,
    compute_features,
    given_name_conflict,
    temporal_proximity,
)
from src.entity_resolution.normalize import (
    initials,
    is_single_token,
    normalize_name,
    surname,
)
from src.entity_resolution.profiles import EntityProfile, attach_bare_surnames, is_resolvable
from src.entity_resolution.resolver import resolve_entities
from src.entity_resolution.scoring import decide, score_pair

UTC = timezone.utc


def _profile(
    profile_id: str,
    surface: str,
    *,
    article: str = "art_a",
    entity_type: str = "PERSON",
    roles: set[str] | None = None,
    countries: set[str] | None = None,
    orgs: set[str] | None = None,
    surfaces: set[str] | None = None,
    published: datetime | None = None,
    context: str = "",
) -> EntityProfile:
    return EntityProfile(
        profile_id=profile_id,
        article_id=article,
        source="Test Wire",
        published_at=published or datetime(2024, 10, 22, tzinfo=UTC),
        entity_type=entity_type,
        canonical_surface=surface,
        normalized=normalize_name(surface, entity_type),
        surfaces=surfaces or {surface},
        roles=roles or set(),
        countries=countries or set(),
        orgs=orgs or set(),
        context=context,
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "surface,expected",
    [
        ("Prime Minister Narendra Modi", "narendra modi"),
        ("Mr. Modi", "modi"),
        ("PM Modi", "modi"),
        ("Narendra Modi", "narendra modi"),
        ("Lalit Modi", "lalit modi"),
        ("Dr. S. Jaishankar", "s jaishankar"),
        ("Finance Minister Nirmala Sitharaman", "nirmala sitharaman"),
    ],
)
def test_person_normalisation(surface: str, expected: str) -> None:
    assert normalize_name(surface, "PERSON") == expected


def test_org_normalisation_strips_suffixes() -> None:
    assert normalize_name("Reliance Industries Ltd.", "ORG") == "reliance industries"
    assert normalize_name("The New Development Bank", "ORG") == "new development bank"


def test_normalisation_preserves_the_distinguishing_given_name() -> None:
    """THE critical property. If normalisation collapsed these to "modi", a
    false merge would be unavoidable no matter how good the scorer is."""
    assert normalize_name("Narendra Modi") != normalize_name("Lalit Modi")


def test_md_is_not_treated_as_a_title() -> None:
    """'Md' is the abbreviation of the given name Mohammed in South Asian news,
    not 'Managing Director'. Stripping it would discard the only distinguishing
    token."""
    assert normalize_name("Md Salim") == "md salim"


def test_surname_and_initials() -> None:
    assert surname("narendra modi") == "modi"
    assert initials("narendra modi") == "nm"
    assert is_single_token("modi") is True
    assert is_single_token("narendra modi") is False


def test_unresolvable_names_are_filtered() -> None:
    """NER produced "Rs" (rupees) as a PERSON. A currency abbreviation must not
    become an entity."""
    assert is_resolvable("rs") is False
    assert is_resolvable("ai") is False
    assert is_resolvable("modi") is True


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------


def test_phonetic_code_collides_for_transliteration_variants() -> None:
    assert phonetic_code("sergey") == phonetic_code("sergei")
    assert phonetic_code("mohamed") == phonetic_code("muhammad")
    assert phonetic_code("modi") != phonetic_code("putin")


def test_surname_blocking_brings_variants_together() -> None:
    """The whole point: "Modi" and "Narendra Modi" must share a block or they
    can NEVER be compared, and blocking errors are unrecoverable."""
    assert blocking_keys("Narendra Modi") & blocking_keys("Modi")
    assert blocking_keys("Narendra Modi") & blocking_keys("PM Modi")


def test_acronym_blocking_for_organisations() -> None:
    assert blocking_keys("New Development Bank", "ORG") & blocking_keys("NDB", "ORG")


def test_blocking_generates_the_dangerous_pair_too() -> None:
    """Blocking PROPOSES; scoring DISPOSES. Narendra/Lalit Modi must be
    compared -- and then rejected -- not silently never considered."""
    assert blocking_keys("Narendra Modi") & blocking_keys("Lalit Modi")


def test_blocking_reduces_the_comparison_space() -> None:
    records = [(f"r{i}", name, "PERSON") for i, name in enumerate(
        ["Narendra Modi", "PM Modi", "Modi", "Lalit Modi", "Vladimir Putin",
         "Putin", "Sergey Lavrov", "Sergei Ryabkov", "Xi Jinping", "Abiy Ahmed"]
    )]
    pairs = candidate_pairs(build_blocks(records))
    all_pairs = len(records) * (len(records) - 1) // 2
    assert len(pairs) < all_pairs * 0.5


def test_different_surnames_never_become_candidates() -> None:
    """Lavrov and Ryabkov share a first name but are different people; they
    must not even be compared."""
    records = [("a", "Sergey Lavrov", "PERSON"), ("b", "Sergei Ryabkov", "PERSON")]
    assert candidate_pairs(build_blocks(records)) == set()


# ---------------------------------------------------------------------------
# Hard constraints -- the false-merge defences
# ---------------------------------------------------------------------------


def test_given_name_conflict_blocks_same_surname_different_person() -> None:
    assert given_name_conflict("narendra modi", "lalit modi") is True


def test_bare_surname_is_compatible_with_any_given_name() -> None:
    """"Modi" might be either of them, so it must stay scoreable rather than
    being vetoed outright."""
    assert given_name_conflict("modi", "narendra modi") is False


def test_initials_are_compatible_with_full_given_names() -> None:
    assert given_name_conflict("n modi", "narendra modi") is False


def test_different_surnames_are_not_a_given_name_conflict() -> None:
    assert given_name_conflict("sergey lavrov", "sergei ryabkov") is False


def test_veto_produces_no_match_regardless_of_other_evidence() -> None:
    """A veto must not be outvoted by agreeing soft features."""
    left = _profile("p1", "Narendra Modi", roles={"chairman"}, countries={"India"})
    right = _profile("p2", "Lalit Modi", roles={"chairman"}, countries={"India"})
    result = decide("p1", "p2", compute_features(left, right, context_similarity=1.0))
    assert result.decision == "no_match"
    assert result.score == 0.0
    assert "given names" in result.reason


def test_type_mismatch_is_vetoed() -> None:
    left = _profile("p1", "Reliance", entity_type="ORG")
    right = _profile("p2", "Reliance", entity_type="PERSON")
    assert decide("p1", "p2", compute_features(left, right)).decision == "no_match"


def test_conflicting_countries_are_vetoed() -> None:
    left = _profile("p1", "Modi", countries={"India"})
    right = _profile("p2", "Modi", countries={"Ethiopia"})
    assert decide("p1", "p2", compute_features(left, right)).decision == "no_match"


# ---------------------------------------------------------------------------
# Soft features
# ---------------------------------------------------------------------------


def test_identical_names_score_above_the_match_threshold() -> None:
    """Regression: a STRICT subset test scored identical names 0.0 on the
    highest-weighted feature, so "Jaishankar" and "Jaishankar" fell below
    threshold and split into two entities."""
    left = _profile("p1", "Jaishankar", article="art_a")
    right = _profile("p2", "Jaishankar", article="art_b")
    assert decide("p1", "p2", compute_features(left, right)).decision == "match"


def test_surname_matches_full_name_with_corroboration() -> None:
    left = _profile("p1", "Narendra Modi", roles={"prime minister"}, countries={"India"})
    right = _profile(
        "p2", "Modi", article="art_b", roles={"prime minister"}, countries={"India"},
        surfaces={"Modi", "Narendra Modi"},
    )
    assert decide("p1", "p2", compute_features(left, right, 0.8)).decision == "match"


def test_absent_attributes_are_neutral_not_agreeing() -> None:
    """Most documents do not state a country. Treating "both unknown" as a
    match would make every pair look similar."""
    left = _profile("p1", "Modi")
    right = _profile("p2", "Modi", article="art_b")
    features = compute_features(left, right)
    assert features.country_similarity == 0.0
    assert features.role_similarity == 0.0


def test_temporal_proximity_decays() -> None:
    now = datetime(2024, 10, 22, tzinfo=UTC)
    assert temporal_proximity(now, now) == pytest.approx(1.0)
    assert temporal_proximity(now, now + timedelta(days=90)) < 0.4
    assert temporal_proximity(now, now + timedelta(days=1900)) < 0.01


def test_context_calibration_spreads_scores_over_the_full_range() -> None:
    """Raw cosines between news contexts cluster in a narrow high band
    (0.88-0.93 measured), making them useless as an absolute feature.
    Rank calibration restores discriminative power."""
    raw = [0.892, 0.925, 0.894, 0.922, 0.885]
    calibrated = calibrate_context_scores(raw)
    assert min(calibrated) == 0.0
    assert max(calibrated) == 1.0
    # Order must be preserved.
    assert calibrated[1] > calibrated[0]


def test_score_is_bounded() -> None:
    left = _profile("p1", "Narendra Modi", roles={"pm"}, countries={"India"}, orgs={"X"})
    right = _profile("p2", "Narendra Modi", roles={"pm"}, countries={"India"}, orgs={"X"})
    assert 0.0 <= score_pair(compute_features(left, right, 1.0)) <= 1.0


# ---------------------------------------------------------------------------
# Bare-surname attachment (the chaining defence)
# ---------------------------------------------------------------------------


def test_bare_surname_attaches_to_full_name_in_same_document() -> None:
    """Fixes the chaining bridge: an unattached bare "Modi" in Lalit's article
    matched a bare "Modi" in Narendra's, fusing the two men."""
    profiles = [
        _profile("p0", "Lalit Modi", article="art_lalit"),
        _profile("p1", "Modi", article="art_lalit"),
    ]
    remaining = attach_bare_surnames(profiles)
    assert len(remaining) == 1
    assert remaining[0].canonical_surface == "Lalit Modi"
    assert remaining[0].surfaces == {"Lalit Modi", "Modi"}


def test_bare_surname_is_not_attached_when_ambiguous() -> None:
    """A document containing BOTH Modis is genuinely ambiguous, so we attach
    nothing rather than guess."""
    profiles = [
        _profile("p0", "Lalit Modi", article="art_x"),
        _profile("p1", "Narendra Modi", article="art_x"),
        _profile("p2", "Modi", article="art_x"),
    ]
    assert len(attach_bare_surnames(profiles)) == 3


def test_bare_surname_does_not_cross_document_boundaries() -> None:
    profiles = [
        _profile("p0", "Lalit Modi", article="art_a"),
        _profile("p1", "Modi", article="art_b"),
    ]
    assert len(attach_bare_surnames(profiles)) == 2


# ---------------------------------------------------------------------------
# End-to-end resolution
# ---------------------------------------------------------------------------


def test_aliases_across_articles_resolve_to_one_entity() -> None:
    """The question this whole project started from."""
    profiles = [
        _profile("a:p0", "Narendra Modi", article="art_a",
                 roles={"prime minister"}, countries={"India"}),
        _profile("b:p0", "PM Modi", article="art_b",
                 roles={"prime minister"}, countries={"India"},
                 published=datetime(2024, 10, 23, tzinfo=UTC)),
        _profile("c:p0", "Prime Minister Modi", article="art_c",
                 roles={"prime minister"}, countries={"India"},
                 published=datetime(2024, 11, 5, tzinfo=UTC)),
    ]
    entities, _, stats = resolve_entities(profiles, embed=False)
    assert stats.entities == 1
    assert entities[0].canonical_name == "Narendra Modi"
    assert len(entities[0].article_ids) == 3


def test_same_surname_different_people_stay_separate() -> None:
    """THE false-merge test. Everything else in this file exists to make this
    assertion hold."""
    profiles = [
        _profile("a:p0", "Narendra Modi", article="art_a",
                 roles={"prime minister"}, countries={"India"}),
        _profile("b:p0", "Lalit Modi", article="art_b", roles={"chairman"},
                 published=datetime(2019, 8, 14, tzinfo=UTC)),
    ]
    entities, _, stats = resolve_entities(profiles, embed=False)
    assert stats.entities == 2
    assert stats.vetoed >= 1
    names = {e.canonical_name for e in entities}
    assert names == {"Narendra Modi", "Lalit Modi"}


def test_chaining_through_a_bare_surname_is_prevented() -> None:
    """Integration test for the exact bug we hit: the direct pair is vetoed,
    but a bare surname bridged around the veto via transitive closure."""
    profiles = [
        _profile("a:p0", "Narendra Modi", article="art_a",
                 roles={"prime minister"}, countries={"India"}),
        _profile("b:p0", "Lalit Modi", article="art_b", roles={"chairman"},
                 surfaces={"Lalit Modi", "Modi"},
                 published=datetime(2019, 8, 14, tzinfo=UTC)),
        _profile("c:p0", "Modi", article="art_c",
                 roles={"prime minister"}, countries={"India"},
                 published=datetime(2024, 10, 23, tzinfo=UTC)),
    ]
    entities, _, _ = resolve_entities(profiles, embed=False)
    lalit = next(e for e in entities if e.canonical_name == "Lalit Modi")
    narendra = next(e for e in entities if "Narendra" in e.canonical_name)
    assert "art_b" not in narendra.article_ids
    assert "art_a" not in lalit.article_ids


def test_uncertain_pairs_go_to_review_not_to_a_decision() -> None:
    """'I don't know' is a valid and valuable output when a wrong merge is
    expensive and hard to undo."""
    profiles = [
        _profile("a:p0", "Modi", article="art_a"),
        _profile("b:p0", "Narendra Modi", article="art_b",
                 published=datetime(2024, 12, 1, tzinfo=UTC)),
    ]
    _, review_queue, stats = resolve_entities(profiles, embed=False)
    # A bare surname against a full name with NO corroborating attribute is
    # genuinely uncertain: not confident enough to merge, not weak enough to
    # dismiss. That is exactly what the review band is for.
    assert stats.matches == 0
    assert stats.reviews == 1
    assert review_queue[0].explain()


def test_canonical_name_prefers_the_most_complete_form() -> None:
    profiles = [
        _profile("a:p0", "Modi", article="art_a", roles={"prime minister"}, countries={"India"}),
        _profile("b:p0", "Narendra Modi", article="art_b",
                 roles={"prime minister"}, countries={"India"}),
    ]
    entities, _, _ = resolve_entities(profiles, embed=False)
    assert entities[0].canonical_name == "Narendra Modi"


def test_resolution_is_deterministic() -> None:
    """Non-determinism would give different entity IDs on identical input,
    breaking every downstream reference."""
    profiles = [
        _profile("a:p0", "Narendra Modi", article="art_a", roles={"prime minister"}),
        _profile("b:p0", "Modi", article="art_b", roles={"prime minister"}),
    ]
    first, _, _ = resolve_entities(profiles, embed=False)
    second, _, _ = resolve_entities(profiles, embed=False)
    assert [e.entity_id for e in first] == [e.entity_id for e in second]
    assert [e.canonical_name for e in first] == [e.canonical_name for e in second]


def test_empty_input_is_handled() -> None:
    entities, review, stats = resolve_entities([], embed=False)
    assert entities == [] and review == [] and stats.entities == 0
