"""Preprocessing orchestration: ``Article`` -> ``Document``.

This is the stage that FREEZES the text. After ``build_document`` returns,
``Document.text`` must never change, because from here on every annotation in
the system is a pair of character offsets into it.

Batch-oriented by design: ``preprocess_articles`` cleans every document first,
then segments them all in one batched call. Running clean+segment per document
in a Python loop would forfeit spaCy's internal batching, which is the main
throughput lever available on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from src.config import Config, load_config
from src.logging_utils import get_logger
from src.preprocessing.clean import build_document_text
from src.preprocessing.segment import SentenceSegmenter, build_segmenter
from src.schemas import Article, Document

logger = get_logger(__name__)

PIPELINE_VERSION = "0.1.0"


def build_document(article: Article, segmenter: SentenceSegmenter, cfg: Config) -> Document:
    """Clean one article and segment it into sentences."""
    text = build_document_text(
        article.title,
        article.body,
        strip_html_markup=cfg.preprocessing.strip_html,
        unicode_form=cfg.preprocessing.unicode_form,
    )
    sentences = segmenter.segment(text)
    return _assemble(article, text, sentences)


def _assemble(article: Article, text: str, sentences) -> Document:
    return Document(
        article_id=article.article_id,
        title=article.title,
        source=article.source,
        published_at=article.published_at,
        url=article.url,
        language=article.language,
        text=text,
        sentences=sentences,
        pipeline_version=PIPELINE_VERSION,
    )


def preprocess_articles(
    articles: Sequence[Article],
    cfg: Config | None = None,
    segmenter: SentenceSegmenter | None = None,
) -> list[Document]:
    """Clean and segment a batch of articles."""
    cfg = cfg or load_config()
    segmenter = segmenter or build_segmenter(
        cfg.preprocessing.sentence_segmenter, cfg.preprocessing.spacy_model
    )

    texts = [
        build_document_text(
            article.title,
            article.body,
            strip_html_markup=cfg.preprocessing.strip_html,
            unicode_form=cfg.preprocessing.unicode_form,
        )
        for article in articles
    ]
    all_sentences = segmenter.segment_batch(texts)

    documents = [
        _assemble(article, text, sentences)
        for article, text, sentences in zip(articles, texts, all_sentences)
    ]

    total_sentences = sum(len(d.sentences) for d in documents)
    logger.info(
        "Preprocessed %d articles -> %d sentences (mean %.1f per document)",
        len(documents),
        total_sentences,
        total_sentences / len(documents) if documents else 0.0,
    )
    return documents


def verify_offsets(document: Document) -> list[str]:
    """Self-check that sentence spans are consistent with the document text.

    Cheap, and it catches the single most corrosive bug class in an offset-based
    pipeline: spans that no longer line up with the text. Such a bug does not
    crash -- it silently returns the WRONG WORDS, so every entity and relation
    downstream is quietly wrong. Better to assert the invariant loudly here.
    """
    problems: list[str] = []
    previous_end = -1
    for sentence in document.sentences:
        if sentence.start < 0 or sentence.end > len(document.text):
            problems.append(f"sentence {sentence.index}: span outside text bounds")
            continue
        if sentence.start >= sentence.end:
            problems.append(f"sentence {sentence.index}: empty or inverted span")
        if sentence.start < previous_end:
            problems.append(f"sentence {sentence.index}: overlaps the previous sentence")
        surface = document.text[sentence.start : sentence.end]
        if surface != surface.strip():
            problems.append(f"sentence {sentence.index}: span has untrimmed whitespace")
        previous_end = sentence.end
    return problems


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_documents(documents: Iterable[Document], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for document in documents:
            fh.write(json.dumps(document.model_dump(mode="json"), ensure_ascii=False) + "\n")
            count += 1
    logger.info("Wrote %d documents -> %s", count, out_path)
    return count


def read_documents(path: Path) -> list[Document]:
    """Load documents, re-verifying the text hash.

    ``text_sha256`` is a computed field, so pydantic recomputes it from the text
    on load. Comparing it with the stored value detects a document whose text
    was edited after its annotations were produced -- the stale-offset scenario.
    """
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            stored_hash = payload.pop("text_sha256", None)
            document = Document(**payload)
            if stored_hash and stored_hash != document.text_sha256:
                logger.error(
                    "Text hash mismatch for article_id=%s: stored annotations are STALE",
                    document.article_id,
                )
            documents.append(document)
    return documents
