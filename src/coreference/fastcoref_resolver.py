"""Neural coreference resolution via fastcoref (LingMess / F-Coref).

WHY THIS LIBRARY
----------------
Coreference tooling ages badly. The options considered, and why they were
rejected or chosen:

  * ``neuralcoref``      -- abandoned, pinned to spaCy 2.x. Dead end.
  * ``allennlp`` coref   -- AllenNLP is deprecated and pulls a heavy, conflicting
                            dependency tree.
  * ``spacy-experimental`` coref -- requires spaCy 3.4/3.5; conflicts with the
                            3.8 we need for the rest of the pipeline.
  * ``maverick-coref``   -- newer and strong, but fails to build in this
                            environment (we tried; the install errors out).
  * ``fastcoref``        -- CHOSEN. Installs with ZERO dependency changes to an
                            existing transformers/torch environment, which
                            matters enormously in a real project where the
                            coref model must not dictate your torch version.

Two models are available:
  LingMess -- higher accuracy (~81 F1 on OntoNotes). Its insight is that
              different mention-pair TYPES (pronoun-pronoun, name-name,
              pronoun-name, ...) deserve different scoring functions, so it
              uses several specialised pairwise scorers instead of one.
  FCoref   -- distilled and much faster, a few F1 points lower.

WHAT THE MODEL RETURNS
----------------------
Character-offset spans, grouped into clusters, over the exact string we passed
in. That is precisely the interface our offset-based architecture needs -- no
token-index translation required.

WHAT IT DOES NOT DO
-------------------
It resolves within ONE document only. It has no mechanism, even in principle,
to know that another article exists. Cross-document identity is Phase 6.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.coreference.base import build_cluster
from src.logging_utils import get_logger
from src.schemas import CorefCluster, Document

logger = get_logger(__name__)


class CorefUnavailableError(RuntimeError):
    pass


class FastCorefResolver:
    """Document-level neural coreference."""

    name = "fastcoref"

    def __init__(
        self,
        model_name: str = "lingmess",
        device: str = "cpu",
        batch_size: int = 4,
        max_chars: int = 20000,
    ) -> None:
        try:
            from fastcoref import FCoref, LingMessCoref
        except ImportError as exc:  # pragma: no cover
            raise CorefUnavailableError(
                "fastcoref is not installed. Install it with: pip install fastcoref"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        # Coref is O(mentions^2); a pathologically long document can blow up
        # both memory and latency. We truncate with a loud warning rather than
        # letting one document stall an entire batch.
        self.max_chars = max_chars

        try:
            if model_name.lower() in {"lingmess", "lingmess-coref"}:
                self._model: Any = LingMessCoref(device=device)
            elif model_name.lower() in {"fcoref", "f-coref"}:
                self._model = FCoref(device=device)
            else:
                raise ValueError(
                    f"Unknown coref model {model_name!r}. Use 'lingmess' or 'fcoref'."
                )
        except Exception as exc:  # noqa: BLE001 -- download/load can fail many ways
            raise CorefUnavailableError(f"Could not load coref model: {exc}") from exc

        logger.info("Loaded coreference model '%s' on %s", model_name, device)

    def _texts(self, documents: Sequence[Document]) -> list[str]:
        texts = []
        for document in documents:
            text = document.text
            if len(text) > self.max_chars:
                logger.warning(
                    "%s: truncating to %d chars for coreference (was %d)",
                    document.article_id,
                    self.max_chars,
                    len(text),
                )
                text = text[: self.max_chars]
            texts.append(text)
        return texts

    def _to_clusters(self, document: Document, prediction: Any) -> list[CorefCluster]:
        """Convert one fastcoref prediction into our CorefCluster objects.

        ``as_strings=False`` gives CHARACTER offsets into the text we passed in,
        which is exactly the coordinate system the rest of the pipeline uses.
        """
        raw_clusters = prediction.get_clusters(as_strings=False)

        clusters: list[CorefCluster] = []
        for index, spans in enumerate(raw_clusters):
            cluster = build_cluster(
                document=document,
                spans=[(int(s), int(e)) for s, e in spans],
                index=index,
                method=f"{self.name}:{self.model_name}",
            )
            if cluster is not None:
                clusters.append(cluster)
        return clusters

    def resolve(self, document: Document) -> list[CorefCluster]:
        return self.resolve_batch([document])[0]

    def resolve_batch(self, documents: Sequence[Document]) -> list[list[CorefCluster]]:
        if not documents:
            return []
        predictions = self._model.predict(
            texts=self._texts(documents), max_tokens_in_batch=10000
        )
        return [
            self._to_clusters(document, prediction)
            for document, prediction in zip(documents, predictions)
        ]
