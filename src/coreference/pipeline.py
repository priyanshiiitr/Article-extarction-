"""Phase 4 orchestration: resolve coreference and link clusters to mentions.

IMPORTANT -- WINDOWS / MULTIPROCESSING
--------------------------------------
``fastcoref`` tokenises through HuggingFace ``datasets``, which uses
multiprocessing. On Windows (spawn, not fork) every child process RE-IMPORTS
the entry-point module. A script that calls this code at import time therefore
forks children that re-run the script and exit silently -- the symptom is a
process that terminates with exit code 0 and produces NO OUTPUT AT ALL.

The fix is the standard Python guard in every entry point:

    if __name__ == "__main__":
        main()

We hit exactly this while building the stage, which is why it is documented
here rather than left as folklore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.config import Config, load_config
from src.coreference.base import (
    CoreferenceResolver,
    align_clusters_to_mentions,
    verify_clusters,
)
from src.coreference.rule_resolver import RuleBasedCorefResolver
from src.logging_utils import get_logger
from src.schemas import Document

logger = get_logger(__name__)


@dataclass
class CorefStats:
    documents: int = 0
    clusters: int = 0
    cluster_mentions: int = 0
    linked_ner_mentions: int = 0
    pronouns_resolved: int = 0
    by_form: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"docs={self.documents} clusters={self.clusters} "
            f"mentions_in_clusters={self.cluster_mentions} "
            f"linked_to_ner={self.linked_ner_mentions} forms={self.by_form}"
        )


def build_resolver(cfg: Config | None = None) -> CoreferenceResolver:
    """Construct the configured resolver, degrading loudly if unavailable.

    Same policy as Phase 3: if the neural model cannot load we log a WARNING
    and fall back to rules. A degraded result the operator is TOLD about beats
    a dead pipeline; silently degrading with no log line would be indefensible.
    """
    cfg = cfg or load_config()
    backend = cfg.coreference.backend.lower()

    if backend == "rules":
        return RuleBasedCorefResolver(
            max_sentence_distance=cfg.coreference.max_sentence_distance
        )

    if backend == "fastcoref":
        try:
            from src.coreference.fastcoref_resolver import FastCorefResolver

            return FastCorefResolver(
                model_name=cfg.coreference.model,
                device=cfg.coreference.device,
                batch_size=cfg.coreference.batch_size,
                max_chars=cfg.coreference.max_chars,
            )
        except Exception as exc:  # noqa: BLE001 -- load can fail many ways
            logger.warning(
                "Neural coreference unavailable (%s). Falling back to rules.", exc
            )
            return RuleBasedCorefResolver(
                max_sentence_distance=cfg.coreference.max_sentence_distance
            )

    raise ValueError(f"Unknown coreference backend {backend!r}. Use 'fastcoref' or 'rules'.")


def run_coreference(
    documents: Sequence[Document],
    cfg: Config | None = None,
    resolver: CoreferenceResolver | None = None,
) -> CorefStats:
    """Annotate documents in place with coreference clusters."""
    cfg = cfg or load_config()
    resolver = resolver or build_resolver(cfg)

    all_clusters = resolver.resolve_batch(documents)

    stats = CorefStats(documents=len(documents))
    for document, clusters in zip(documents, all_clusters):
        document.coref_clusters = clusters

        # Connect clusters to Phase 3 entity mentions in both directions.
        stats.linked_ner_mentions += align_clusters_to_mentions(document)

        stats.clusters += len(clusters)
        for cluster in clusters:
            stats.cluster_mentions += len(cluster.mentions)
            for mention in cluster.mentions:
                stats.by_form[mention.form] = stats.by_form.get(mention.form, 0) + 1
                if mention.form == "PRONOMINAL":
                    stats.pronouns_resolved += 1

    # The usual tripwire: cluster spans must still match the document text.
    problems = 0
    for document in documents:
        for problem in verify_clusters(document):
            logger.error("%s: %s", document.article_id, problem)
            problems += 1
    if problems:
        raise RuntimeError(f"{problems} coreference span problems -- aborting")

    logger.info("Coreference complete: %s", stats.summary())
    return stats


def resolve_mention(document: Document, mention_id: str) -> str | None:
    """Return the representative surface form for an entity mention.

    THIS is the interface Phase 5 uses, and it is the reason we never rewrote
    the text. Given a mention, answer "what does this actually refer to?" --
    returning a STRING chosen from the document, with the original text and all
    offsets left completely intact.

    Returns None when the mention is not in any cluster, which is the correct
    answer for a mention that appears exactly once.
    """
    for mention in document.mentions:
        if mention.mention_id != mention_id:
            continue
        if not mention.coref_cluster_id:
            return mention.text
        for cluster in document.coref_clusters:
            if cluster.cluster_id == mention.coref_cluster_id:
                return cluster.representative_text or mention.text
        return mention.text
    return None
