"""NER using spaCy's statistical model.

WHY SPACY AS THE DEFAULT
------------------------
  * 18 entity types vs the transformer's 4 -- it is the only one of our
    extractors that knows EVENT at all, which matters for "BRICS Summit".
  * ~20x faster on CPU, which decides whether a 100k-article batch takes an
    hour or a day.
  * It also gives us the dependency parse, which Phase 5 needs for relation
    extraction -- so we are loading it regardless.

WHAT IT COSTS
-------------
``en_core_web_sm`` is a small CNN model, noticeably less accurate than a
transformer, especially on names it never saw in training. We quantify that in
Phase 10 rather than guessing, and we run both extractors so the merge step can
prefer whichever is more trustworthy for a given type.

A NOTE ON CONFIDENCE SCORES
---------------------------
spaCy's default NER pipe does NOT expose per-entity probabilities -- it does a
greedy transition-based decode and throws the scores away. So we assign a fixed
prior per extractor instead of inventing a number. Being honest that we have no
real score is better than fabricating a plausible-looking 0.87.
"""

from __future__ import annotations

from typing import Sequence

from src.logging_utils import get_logger
from src.ner.base import build_mention, trim_span
from src.ner.labels import map_label
from src.nlp_resources import load_spacy
from src.schemas import Document, Mention

logger = get_logger(__name__)

# A fixed prior standing in for a real probability. Chosen below the
# transformer's typical scores so that, on a tie, the more accurate model wins
# in the merge step. This is a documented heuristic, NOT a calibrated value.
SPACY_DEFAULT_SCORE = 0.75


class SpacyEntityExtractor:
    """Wraps spaCy's NER pipe and emits Mentions in our schema."""

    name = "spacy"

    # Unlike the segmenter we obviously keep "ner", but still drop components
    # we do not use, because every component costs time per document.
    DEFAULT_EXCLUDE = ("lemmatizer",)

    def __init__(self, model_name: str = "en_core_web_sm", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._nlp = load_spacy(model_name, self.DEFAULT_EXCLUDE)
        self.model_version = f"{model_name}-{self._nlp.meta.get('version', '?')}"

    def _mentions_from_doc(self, document: Document, spacy_doc) -> list[Mention]:
        """Convert spaCy entity spans into our Mentions.

        ``ent.start_char`` / ``ent.end_char`` are already CHARACTER offsets into
        the same string we passed in, which is exactly why we can compose spaCy
        with a transformer that tokenises completely differently.
        """
        mentions: list[Mention] = []
        for ent in spacy_doc.ents:
            label = map_label(ent.label_, "spacy")
            if label is None:
                continue  # a type we deliberately drop (MONEY, PERCENT, ...)

            start, end = trim_span(document.text, ent.start_char, ent.end_char)
            if start >= end:
                continue

            mentions.append(
                build_mention(
                    document=document,
                    start=start,
                    end=end,
                    label=label,
                    raw_label=ent.label_,
                    score=SPACY_DEFAULT_SCORE,
                    extractor=self.name,
                    model_version=self.model_version,
                )
            )
        return mentions

    def extract(self, document: Document) -> list[Mention]:
        return self._mentions_from_doc(document, self._nlp(document.text))

    def extract_batch(self, documents: Sequence[Document]) -> list[list[Mention]]:
        """Batch inference via ``nlp.pipe``.

        The neural components run over a batch per forward pass instead of one
        document at a time. Always prefer this over a Python loop.
        """
        texts = [document.text for document in documents]
        spacy_docs = self._nlp.pipe(texts, batch_size=self.batch_size)
        return [
            self._mentions_from_doc(document, spacy_doc)
            for document, spacy_doc in zip(documents, spacy_docs)
        ]
