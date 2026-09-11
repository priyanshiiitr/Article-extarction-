"""Source readers: the only part of the system that knows where articles come from.

WHY A READER ABSTRACTION
------------------------
Today the corpus is a local JSONL file. Tomorrow it might be an RSS feed, an S3
bucket of HTML, a Postgres table, or a Kafka topic. If the loader called
``json.loads`` directly, adding a source would mean editing the loader, the
tests, and anything else that touched input.

Instead every reader yields the same thing -- raw ``dict`` records -- and the
loader handles validation and cleanup for all of them identically. Adding a
source becomes: write one class, change one config value.

Readers deliberately do NOT validate. They yield whatever they found, warts and
all. Validation is the loader's job, in one place, so the rules cannot drift
between sources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from src.logging_utils import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ArticleReader(Protocol):
    """Structural interface every source reader must satisfy.

    A ``Protocol`` (PEP 544) rather than an abstract base class: a reader does
    not need to inherit from anything, it just needs a ``read_raw`` method of
    the right shape. This is "duck typing with type checking" and keeps source
    implementations decoupled from our class hierarchy.
    """

    def read_raw(self) -> Iterator[dict[str, Any]]:
        """Yield raw, unvalidated article records."""
        ...


class JsonlArticleReader:
    """Read articles from a JSON Lines file (one JSON object per line).

    JSONL over a single JSON array because it is *streamable* (constant memory
    regardless of corpus size), *appendable* (add an article without rewriting
    the file), and *robust* (one malformed line does not invalidate the file).
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def read_raw(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Corpus not found: {self.path}. Run `python scripts/make_sample_data.py` first."
            )

        with self.path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    # Log and skip. A single unparseable line must never abort a
                    # batch of 100,000 articles -- that is the difference between
                    # a pipeline that finishes and one that dies at 3am.
                    logger.warning(
                        "Skipping unparseable JSON on line %d of %s: %s",
                        line_number,
                        self.path.name,
                        exc,
                    )
                    continue

                if not isinstance(record, dict):
                    logger.warning(
                        "Skipping non-object record on line %d of %s", line_number, self.path.name
                    )
                    continue

                # Carry provenance from the very first moment we touch the data:
                # if anything downstream looks wrong we can point at the exact
                # source line that produced it.
                record.setdefault("_provenance", {"file": str(self.path), "line": line_number})
                yield record


def build_reader(source_file: Path) -> ArticleReader:
    """Select a reader based on the source file's extension.

    A deliberately tiny factory. It exists so that ``loader.py`` never contains
    an ``if`` over source types, and so a future ``RssArticleReader`` plugs in
    here rather than in the orchestration logic.
    """
    suffix = source_file.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return JsonlArticleReader(source_file)
    raise ValueError(
        f"No reader registered for '{suffix}' files. Supported: .jsonl, .ndjson"
    )
