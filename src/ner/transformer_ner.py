"""NER using a HuggingFace transformer token-classification model.

WHAT ACTUALLY HAPPENS INSIDE
----------------------------
1. TOKENIZE. The text is split into subword units by WordPiece:
       "Jaishankar" -> ["Ja", "##ish", "##ank", "##ar"]
   Subwords exist so a fixed 30k vocabulary can represent any word, including
   names the model never saw in training. This is why transformers degrade
   gracefully on rare Indian names where a word-level model would emit <UNK>.

2. FORWARD PASS. Self-attention produces one CONTEXTUAL vector per token --
   the representation of "Modi" differs between a cricket sentence and a summit
   sentence. That context-sensitivity is the whole advantage over word2vec-era
   embeddings, which had one fixed vector per word.

3. CLASSIFICATION HEAD. A linear layer maps each token vector to 9 scores
   (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC), then softmax
   turns those into probabilities.

4. DECODE. Consecutive B-/I- tags of the same type are merged into one span,
   and subword pieces are glued back into words.

5. MAP TO CHARACTERS. Token indices are meaningless outside this tokenizer, so
   we use the offset mapping to convert back to character positions in the
   original string. This is the step that lets us compose this model with spaCy.

THE 512-TOKEN LIMIT
-------------------
BERT has a fixed maximum input length. A long article exceeds it, and the
naive behaviours are both bad: truncation silently loses the end of the
document, and blind fixed-size chunking cuts entities in half. We chunk on
SENTENCE boundaries (which Phase 2 already computed) and shift the returned
offsets back into document coordinates.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.logging_utils import get_logger
from src.ner.base import build_mention, trim_span
from src.ner.labels import map_label
from src.schemas import Document, Mention

logger = get_logger(__name__)


class TransformerUnavailableError(RuntimeError):
    pass


class TransformerEntityExtractor:
    """Token-classification NER via the HuggingFace ``pipeline`` API."""

    name = "transformer"

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        device: int = -1,          # -1 = CPU, 0 = first GPU
        max_chars_per_chunk: int = 1200,
        batch_size: int = 8,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise TransformerUnavailableError(
                "transformers is not installed. `pip install transformers torch`"
            ) from exc

        self.model_name = model_name
        self.model_version = model_name
        self.max_chars_per_chunk = max_chars_per_chunk
        self.batch_size = batch_size

        # aggregation_strategy decides how subword predictions become entities,
        # and the choice is NOT cosmetic. We learned this empirically:
        #
        #   "simple"  merges only consecutive tags of the same type, respecting
        #             B-/I- boundaries AT THE SUBWORD LEVEL. When a name is not
        #             in BERT's 30k vocabulary it is split -- "Modi" becomes
        #             "Mo" + "##di" -- and if the model emits B-PER for BOTH
        #             pieces, "simple" returns TWO people named "Mo" and "di".
        #             We observed exactly this on our corpus.
        #
        #   "first"   groups subwords back into WORDS first, then gives each
        #             word the label of its first subword. A word can no longer
        #             be split in half. This matches how CoNLL was annotated,
        #             which is why it is the usual choice for CoNLL models.
        #
        #   "max"     word takes the label of its highest-scoring subword.
        #   "average" averages subword scores, then argmax.
        #
        # We use "first": deterministic, conventional, and it fixes the split.
        # This failure mode is worth remembering -- it hits hardest on exactly
        # the names a 2003-vintage vocabulary never saw, i.e. modern Indian and
        # non-Western names, which is our whole corpus.
        self._pipe = pipeline(
            task="token-classification",
            model=model_name,
            aggregation_strategy="first",
            device=device,
        )
        logger.info("Loaded transformer NER model '%s' (device=%s)", model_name, device)

    # -- chunking -----------------------------------------------------------

    def _chunks(self, document: Document) -> list[tuple[int, int]]:
        """Split the document into (start, end) chunks on sentence boundaries.

        Greedy packing: keep adding whole sentences until the next one would
        exceed the character budget, then start a new chunk. Because we never
        split inside a sentence, we never cut an entity in half.

        Character budget rather than token count is an approximation -- English
        averages roughly 4 characters per BERT token, so 1200 chars is ~300
        tokens, comfortably inside the 512 limit even for dense text. Being
        conservative here is cheap; being wrong means silent truncation.
        """
        if not document.sentences:
            return [(0, len(document.text))]

        chunks: list[tuple[int, int]] = []
        chunk_start = document.sentences[0].start
        chunk_end = chunk_start

        for sentence in document.sentences:
            if sentence.end - chunk_start > self.max_chars_per_chunk and chunk_end > chunk_start:
                chunks.append((chunk_start, chunk_end))
                chunk_start = sentence.start
            chunk_end = sentence.end

        if chunk_end > chunk_start:
            chunks.append((chunk_start, chunk_end))
        return chunks

    # -- extraction ---------------------------------------------------------

    def _mentions_from_predictions(
        self,
        document: Document,
        predictions: list[dict[str, Any]],
        offset: int,
    ) -> list[Mention]:
        """Convert raw pipeline output into Mentions, shifting offsets.

        ``offset`` is the chunk's start position in the document. The pipeline
        reports positions relative to the CHUNK, so every span must be shifted
        back into document coordinates. Forgetting this shift is the classic
        chunking bug: entities land at plausible-looking but wrong positions.
        """
        mentions: list[Mention] = []
        for prediction in predictions:
            raw_label = prediction["entity_group"]
            label = map_label(raw_label, "transformer")
            if label is None:
                continue

            start = int(prediction["start"]) + offset
            end = int(prediction["end"]) + offset
            start, end = trim_span(document.text, start, end)
            if start >= end:
                continue

            mentions.append(
                build_mention(
                    document=document,
                    start=start,
                    end=end,
                    label=label,
                    raw_label=raw_label,
                    # A REAL model probability, unlike spaCy's fixed prior.
                    score=float(prediction["score"]),
                    extractor=self.name,
                    model_version=self.model_version,
                )
            )
        return mentions

    def extract(self, document: Document) -> list[Mention]:
        mentions: list[Mention] = []
        for chunk_start, chunk_end in self._chunks(document):
            chunk_text = document.text[chunk_start:chunk_end]
            if not chunk_text.strip():
                continue
            predictions = self._pipe(chunk_text)
            mentions.extend(self._mentions_from_predictions(document, predictions, chunk_start))
        return mentions

    def extract_batch(self, documents: Sequence[Document]) -> list[list[Mention]]:
        """Batch across ALL chunks of ALL documents in one call.

        Flattening first matters: a GPU (or even a CPU with threads) is far
        better used on one batch of 64 chunks than on 11 separate calls. We
        keep an index so results can be scattered back to the right document.
        """
        flat_texts: list[str] = []
        owners: list[tuple[int, int]] = []  # (document index, chunk offset)

        for doc_index, document in enumerate(documents):
            for chunk_start, chunk_end in self._chunks(document):
                chunk_text = document.text[chunk_start:chunk_end]
                if chunk_text.strip():
                    flat_texts.append(chunk_text)
                    owners.append((doc_index, chunk_start))

        if not flat_texts:
            return [[] for _ in documents]

        all_predictions = self._pipe(flat_texts, batch_size=self.batch_size)

        results: list[list[Mention]] = [[] for _ in documents]
        for (doc_index, chunk_start), predictions in zip(owners, all_predictions):
            results[doc_index].extend(
                self._mentions_from_predictions(documents[doc_index], predictions, chunk_start)
            )
        return results
