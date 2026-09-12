"""Relation extraction from the dependency parse.

WHAT A DEPENDENCY PARSE IS
--------------------------
A tree over the words of a sentence. Every word points to its syntactic HEAD
with a labelled arc:

    "PM Modi met Vladimir Putin in Kazan on Wednesday."

                    met                     <- ROOT
        ┌────────────┼──────────┬────────┐
      nsubj        dobj       prep     prep
        │            │          │        │
      Modi         Putin       in       on
                               pobj     pobj
                                │        │
                              Kazan   Wednesday

    nsubj = nominal subject   dobj = direct object
    prep/pobj = preposition and its object

WHY THIS BEATS STRING PATTERNS
------------------------------
Word order is unreliable; grammatical structure is not. All of these have the
same (Modi, met, Putin) triple, and only the parse finds all four:

    "Modi met Putin."
    "Modi, who arrived on Tuesday, met Putin."       <- 5 words in between
    "Putin was met by Modi."                         <- reversed word order
    "Modi met Putin and Xi Jinping."                 <- coordinated object

THREE THINGS THAT MATTER FOR PRECISION
--------------------------------------
1. NEGATION. "Modi did NOT meet Putin" parses identically to the positive
   sentence apart from a `neg` arc. Ignore it and you assert the opposite of
   what the article said -- the worst possible error in a fact store.
2. PASSIVE VOICE. "The declaration was signed by the leaders" has the
   declaration as `nsubjpass` and the leaders as the agent. The semantic
   subject is the agent, so the arguments must be swapped.
3. TYPE CHECKING. The parse says "X is the object of discussed". It does not
   say X is a topic. `(Kazan) -[discussed]-> (Wednesday)` is grammatically
   fine and factually nonsense, so the ARGUMENT_TYPES table rejects it.

WHAT IT STILL CANNOT DO
-----------------------
  * Paraphrase beyond the trigger list. "sat down with" means "met", but only
    if someone puts it in the list. A supervised model learns this from data.
  * Implicit relations. "Modi's Moscow visit" implies travel with no verb.
  * Cross-sentence relations. "Modi arrived. He signed the accord." only works
    because coreference supplies the subject.
  * Modality and hedging. "Modi MAY meet Putin" is extracted as fact. We flag
    modality below but a proper treatment needs factuality classification.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from src.logging_utils import get_logger
from src.nlp_resources import load_spacy
from src.relation_extraction.base import (
    ResolvedArgument,
    build_relation,
    resolve_arguments,
    types_are_valid,
)
from src.schemas import Document, Relation

logger = get_logger(__name__)

# Verb lemma -> our normalised predicate. This is the CLOSED relation schema in
# action: many surface verbs collapse onto one edge type, so a query for "met"
# finds "met", "meets", "held talks with" and "sat down with" alike.
#
# Lemmatisation is what makes a lemma list workable: "met", "meets", "meeting"
# and "had met" all lemmatise to "meet", so one entry covers the paradigm.
PREDICATE_MAP: dict[str, str] = {
    "meet": "met",
    "greet": "met",
    "receive": "met",
    "host": "met",
    "discuss": "discussed",
    "review": "discussed",
    "address": "discussed",
    "say": "said",
    "tell": "said",
    "state": "said",
    "add": "said",
    "attend": "attended",
    "arrive": "attended",
    "join": "attended",
    "sign": "signed",
    "adopt": "signed",
    "chair": "works_for",
    "head": "works_for",
    "lead": "works_for",
    "represent": "represents",
}

# Multi-word triggers where the verb alone is too generic. "hold" means nothing
# on its own, but "hold talks" means "met". Checked against the verb's direct
# object before the plain lemma map.
NOUN_TRIGGERS: dict[tuple[str, str], str] = {
    ("hold", "talk"): "met",
    ("hold", "meeting"): "met",
    ("have", "talk"): "met",
    ("hold", "discussion"): "discussed",
}

# Parser output is structural, not semantic, so confidence is lower than a
# deterministic pattern match. A direct object is a more reliable argument than
# a prepositional one, which is why they differ.
DIRECT_OBJECT_CONFIDENCE = 0.75
PREP_OBJECT_CONFIDENCE = 0.65
# Coreference-resolved arguments inherit the coref model's errors ON TOP of the
# parser's, so they are penalised multiplicatively.
COREF_PENALTY = 0.85
# "may meet", "is expected to meet" -- reported as fact by the parse, but the
# article did not assert it happened.
MODAL_PENALTY = 0.70
MODAL_LEMMAS = {"may", "might", "could", "would", "should", "will", "plan", "expect", "schedule"}

SUBJECT_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "dative"}


class DependencyRelationExtractor:
    """Extract verb-mediated relations from spaCy's dependency parse."""

    name = "dependency"

    # We exclude spaCy's NER because Phase 3 already produced better mentions by
    # merging three extractors. We KEEP the tagger, parser, attribute_ruler and
    # lemmatizer: the parse is the point, and lemmas are what let one entry in
    # PREDICATE_MAP cover a whole verb paradigm.
    KEEP_EXCLUDE = ("ner",)

    def __init__(self, model_name: str = "en_core_web_sm", batch_size: int = 16) -> None:
        self._nlp = load_spacy(model_name, self.KEEP_EXCLUDE)
        self.batch_size = batch_size

    # -- parse navigation ---------------------------------------------------

    @staticmethod
    def _is_negated(verb: Any) -> bool:
        """True when the verb carries a negation arc.

        Missing this makes the pipeline assert the OPPOSITE of the article.
        """
        return any(child.dep_ == "neg" for child in verb.children)

    @staticmethod
    def _is_hedged(verb: Any) -> bool:
        """True when a modal or planning verb governs this one."""
        if any(child.lemma_.lower() in MODAL_LEMMAS for child in verb.children):
            return True
        head = verb.head
        return head is not verb and head.lemma_.lower() in MODAL_LEMMAS

    @staticmethod
    def _with_conjuncts(token: Any) -> Iterator[Any]:
        """Yield a token and anything coordinated with it.

        "Modi met Putin and Xi Jinping" gives Putin as `dobj` and Xi as `conj`
        hanging off Putin. Without this we would silently drop half the facts in
        every coordinated sentence.
        """
        yield token
        for child in token.children:
            if child.dep_ == "conj":
                yield from DependencyRelationExtractor._with_conjuncts(child)

    def _subjects(self, verb: Any) -> list[Any]:
        subjects = [c for c in verb.children if c.dep_ in SUBJECT_DEPS]
        if not subjects and verb.dep_ in {"conj", "xcomp", "advcl", "ccomp"}:
            # "Modi arrived and met Putin": the second verb has no subject of
            # its own; it inherits the first verb's.
            subjects = [c for c in verb.head.children if c.dep_ in SUBJECT_DEPS]
        expanded: list[Any] = []
        for subject in subjects:
            expanded.extend(self._with_conjuncts(subject))
        return expanded

    def _objects(self, verb: Any) -> list[tuple[Any, float, str]]:
        """Return (token, confidence, preposition) for each candidate object."""
        found: list[tuple[Any, float, str]] = []

        for child in verb.children:
            if child.dep_ in OBJECT_DEPS:
                for token in self._with_conjuncts(child):
                    found.append((token, DIRECT_OBJECT_CONFIDENCE, ""))
            elif child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        for token in self._with_conjuncts(grandchild):
                            found.append((token, PREP_OBJECT_CONFIDENCE, child.lemma_.lower()))
            elif child.dep_ == "agent":
                # Passive "signed BY the leaders": handled in _extract_sentence.
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        found.append((grandchild, DIRECT_OBJECT_CONFIDENCE, "by"))
        return found

    def _predicate_for(self, verb: Any) -> tuple[str, str] | None:
        """Resolve a verb to (predicate, trigger surface), or None."""
        lemma = verb.lemma_.lower()

        # Multi-word triggers first: "hold talks" must beat bare "hold".
        for child in verb.children:
            if child.dep_ in OBJECT_DEPS:
                key = (lemma, child.lemma_.lower())
                if key in NOUN_TRIGGERS:
                    return NOUN_TRIGGERS[key], f"{verb.text} {child.text}"

        predicate = PREDICATE_MAP.get(lemma)
        return (predicate, verb.text) if predicate else None

    # -- extraction ---------------------------------------------------------

    def _extract_from_doc(self, document: Document, spacy_doc: Any) -> list[Relation]:
        relations: list[Relation] = []

        for token in spacy_doc:
            if token.pos_ not in {"VERB", "AUX"}:
                continue

            resolved = self._predicate_for(token)
            if resolved is None:
                continue
            predicate, trigger = resolved

            if self._is_negated(token):
                # Correct behaviour is to skip. A production system would store
                # it as a NEGATIVE fact instead, because "Modi did not meet
                # Putin" is itself information worth keeping.
                logger.debug("%s: skipping negated '%s'", document.article_id, trigger)
                continue

            confidence_scale = MODAL_PENALTY if self._is_hedged(token) else 1.0

            subjects = self._subjects(token)
            objects = self._objects(token)
            if not subjects or not objects:
                continue

            # Passive voice: "The declaration was signed by the leaders".
            # The grammatical subject is the semantic OBJECT, and the agent is
            # the semantic subject, so the two must be swapped.
            is_passive = any(s.dep_ == "nsubjpass" for s in subjects)

            for subject_token in subjects:
                for object_token, base_confidence, preposition in objects:
                    pair = (subject_token, object_token)
                    if is_passive and preposition == "by":
                        pair = (object_token, subject_token)
                    elif is_passive:
                        continue  # a passive with no agent has no usable subject

                    relations.extend(
                        self._build(
                            document=document,
                            subject_token=pair[0],
                            object_token=pair[1],
                            predicate=predicate,
                            trigger=trigger,
                            confidence=base_confidence * confidence_scale,
                        )
                    )

        return relations

    def _build(
        self,
        document: Document,
        subject_token: Any,
        object_token: Any,
        predicate: str,
        trigger: str,
        confidence: float,
    ) -> list[Relation]:
        """Build every relation licensed by this (subject, object) token pair.

        Returns a LIST because a plural antecedent has several referents:
        "The two leaders discussed energy" is a fact about Modi AND about
        Putin, so it yields two relations, not one arbitrary one.
        """
        subjects = self._arguments(document, subject_token)
        objects = self._arguments(document, object_token)

        built: list[Relation] = []
        for subject in subjects:
            for obj in objects:
                if not types_are_valid(predicate, subject.label, obj.label):
                    continue

                scaled = confidence
                if subject.via_coref:
                    scaled *= COREF_PENALTY
                if obj.via_coref:
                    scaled *= COREF_PENALTY

                relation = build_relation(
                    document=document,
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    confidence=scaled,
                    extractor=self.name,
                    trigger=trigger,
                )
                if relation is not None:
                    built.append(relation)
        return built

    @staticmethod
    def _arguments(document: Document, token: Any) -> list[ResolvedArgument]:
        """Map a parse token to a relation argument.

        ``token.idx`` is already a character offset into the string we parsed,
        which is ``document.text`` -- the same string every other stage indexes.
        That is exactly why the offset-based architecture lets us bolt a parser
        onto NER and coreference output that tokenised completely differently.
        """
        return resolve_arguments(document, token.idx, token.idx + len(token.text))

    def extract(self, document: Document) -> list[Relation]:
        return self._extract_from_doc(document, self._nlp(document.text))

    def extract_batch(self, documents: Sequence[Document]) -> list[list[Relation]]:
        texts = [d.text for d in documents]
        return [
            self._extract_from_doc(document, spacy_doc)
            for document, spacy_doc in zip(documents, self._nlp.pipe(texts, batch_size=self.batch_size))
        ]
