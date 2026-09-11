"""Translate model-specific NER labels into our shared vocabulary.

WHY THIS FILE EXISTS
--------------------
A model's label set is a property of its TRAINING DATA, not of language:

    spaCy en_core_web_sm  (OntoNotes 5)  -> 18 types: PERSON ORG GPE LOC NORP
                                            EVENT FAC DATE MONEY PERCENT ...
    dslim/bert-base-NER   (CoNLL-2003)   ->  4 types: PER ORG LOC MISC

If those raw labels leaked into the rest of the pipeline, then swapping the
model would break every downstream rule, query and test. So this module is the
single choke point where foreign vocabularies become ours.

THE MAPPING IS LOSSY, AND THAT IS THE POINT
-------------------------------------------
Some information is genuinely destroyed here:

  * GPE ("geo-political entity") covers BOTH countries and cities. We cannot
    tell "Russia" from "Kazan" by label alone, so both map to LOCATION and a
    gazetteer (see gazetteer.py) promotes the real countries to COUNTRY.
  * MISC is a dustbin. CoNLL labels nationalities, events and product names all
    as MISC, so "BRICS Summit" is MISC with no way to recover "this is an
    event" from the label.

Because the mapping loses information, every Mention also stores ``raw_label``
-- what the model actually said. When a downstream result looks wrong, the
first question is always "what did the model really predict?", and without
raw_label you cannot answer it.
"""

from __future__ import annotations

# --- spaCy / OntoNotes -----------------------------------------------------
# None means "drop this mention": the type is real but not useful to a
# knowledge graph of people, organisations and events. Dropping at the mapping
# layer keeps the noise out of every later stage.
SPACY_LABEL_MAP: dict[str, str | None] = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    # GPE = countries, cities, states. Mapped to LOCATION, then refined to
    # COUNTRY by the gazetteer -- a label alone cannot make that distinction.
    "GPE": "LOCATION",
    "LOC": "LOCATION",      # non-GPE locations: rivers, mountain ranges
    "FAC": "LOCATION",      # facilities: airports, buildings, bridges
    "EVENT": "EVENT",
    # NORP = Nationalities, Religious or Political groups: "Indian", "Russian".
    # Kept (not dropped) because a nationality adjective is strong evidence of
    # a COUNTRY link -- "the Indian Prime Minister" tells us who Modi
    # represents. Phase 5 converts these demonyms to countries.
    "NORP": "MISC",
    "DATE": "DATE",
    "LAW": None,
    "LANGUAGE": None,
    "PRODUCT": None,
    "WORK_OF_ART": None,
    "MONEY": None,
    "PERCENT": None,
    "QUANTITY": None,
    "CARDINAL": None,
    "ORDINAL": None,
    "TIME": None,
}

# --- HuggingFace / CoNLL-2003 ----------------------------------------------
TRANSFORMER_LABEL_MAP: dict[str, str | None] = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    # MISC is genuinely ambiguous in CoNLL: nationalities, events and products
    # share it. We keep it rather than drop it, because in our corpus MISC is
    # where "BRICS" lands -- but we keep it at LOW trust in the merge step.
    "MISC": "MISC",
}

LABEL_MAPS: dict[str, dict[str, str | None]] = {
    "spacy": SPACY_LABEL_MAP,
    "transformer": TRANSFORMER_LABEL_MAP,
}


def map_label(raw_label: str, source: str) -> str | None:
    """Map one model label to our vocabulary.

    Returns None when the mention should be dropped entirely.

    Unknown labels are passed through as MISC rather than raising. Deliberate:
    a model update that adds a new label should degrade the output slightly,
    not crash a 100,000-article batch at 3am.
    """
    mapping = LABEL_MAPS.get(source)
    if mapping is None:
        raise ValueError(f"No label map for source {source!r}. Known: {list(LABEL_MAPS)}")
    if raw_label not in mapping:
        return "MISC"
    return mapping[raw_label]


def strip_bio_prefix(tag: str) -> str:
    """Turn a BIO/BIOES tag into its bare entity type.

        "B-PER" -> "PER"      "I-ORG" -> "ORG"
        "E-LOC" -> "LOC"      "O"     -> "O"

    Needed because token-classification models emit position-tagged labels,
    while our vocabulary only describes the entity type.
    """
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag
