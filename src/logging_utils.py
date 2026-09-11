"""Project-wide logging setup.

Why logging and not ``print``: an NLP pipeline is a long-running batch job over
thousands of documents. When article 4,317 fails you need to know *which*
article, *which* stage, and *when* -- with severity levels you can filter on.
``print`` gives you none of that and cannot be turned off or redirected.

Two formats are supported deliberately:
  * ``rich``  -- colourised, human-readable. For local development.
  * ``json``  -- one JSON object per line. This is what you want in production,
                 because log aggregators (CloudWatch, Datadog, Loki) index
                 structured fields and let you query ``stage="ner" AND
                 level="ERROR"``. Grepping prose does not scale.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Render each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via logger.info("...", extra={"article_id": ...})
        # is attached to the record and becomes a queryable field.
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__ and key != "message":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "rich") -> None:
    """Install a single root handler. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    # Remove existing handlers so repeated calls (e.g. in tests) do not emit
    # every line two or three times.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if fmt == "json":
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
    else:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
        except ImportError:  # pragma: no cover
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
            )
    root.addHandler(handler)

    # Third-party libraries are chatty at INFO. Keep our own logs readable.
    for noisy in ("urllib3", "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Call as ``get_logger(__name__)``."""
    return logging.getLogger(name)
