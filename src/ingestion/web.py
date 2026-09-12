"""Fetch articles from URLs.

WHY TRAFILATURA AND NOT OUR REGEX STRIPPER
------------------------------------------
Phase 2's ``strip_html`` was explicitly documented as a baseline that "cannot
distinguish article prose from navigation links" -- and we saw that fail: the
link text "More BRICS coverage" survived cleaning and sat in the document as if
it were a sentence.

That limitation is fatal for real web pages, where the article is typically 10%
of the HTML and the rest is nav bars, cookie notices, related-story teasers,
share buttons and comment forms. The task is called BOILERPLATE REMOVAL, and it
cannot be done with regexes because it requires reasoning over DOM structure and
text density: which subtree looks like prose, and which looks like chrome.

``trafilatura`` does this properly and also extracts metadata -- title, author,
publication date -- which we would otherwise have to guess. It consistently
tops the benchmarks for this specific task.

FETCHING IS THE PART THAT WILL FAIL
-----------------------------------
Network code fails constantly and in boring ways: timeouts, 403s from bot
protection, paywalls that return a teaser, redirects to consent pages, and
pages that are entirely JavaScript. Every one of those must degrade into a
skipped article with a readable reason, never a crash that loses the other 49
URLs in the batch. Same rule as Phase 1's malformed-record handling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence
from urllib.parse import urlparse

from src.logging_utils import get_logger
from src.schemas import Article, make_article_id

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 20
# A browser-like agent. Many news sites return 403 to obvious bots. This is not
# evasion -- it is how a normal HTTP client identifies itself -- but it is worth
# being explicit that scraping has terms-of-service implications and should
# respect robots.txt and rate limits in anything beyond personal use.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class FetchResult:
    url: str
    article: Article | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.article is not None


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc
        return host[4:] if host.startswith("www.") else host or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# Page <title> tags almost always append the site name: "Narendra Modi -
# Wikipedia", "India, Russia sign deal | Reuters". Because Phase 2 PREPENDS the
# title to the document text, that suffix becomes a sentence, and we extracted
# "Narendra Modi - Wikipedia" as a PERSON entity. Stripping it is a small fix
# for a visible quality bug.
_TITLE_SUFFIX_RE = re.compile(r"\s*[-|–—]\s*[^-|–—]{1,40}$")


def clean_page_title(title: str, source: str) -> str:
    """Remove a trailing site name from a scraped page title.

    Only strips when the trailing segment actually looks like the site name, so
    a genuine headline such as "Modi in Kazan - what it means" keeps its dash.
    """
    title = title.strip()
    match = _TITLE_SUFFIX_RE.search(title)
    if not match:
        return title
    tail = match.group(0).lstrip(" -|–—").strip().lower()
    source_tokens = {t for t in re.split(r"[^a-z0-9]+", source.lower()) if t}
    tail_tokens = {t for t in re.split(r"[^a-z0-9]+", tail) if t}
    if tail_tokens and tail_tokens & source_tokens:
        stripped = title[: match.start()].strip()
        return stripped or title
    return title


def _parse_date(value: str | None) -> datetime:
    """Parse trafilatura's date, falling back to now.

    Falling back to NOW rather than dropping the article is deliberate: a
    missing publication date should not lose us the content. But it does
    corrupt time-based signals, so it is logged -- and Phase 6's temporal
    feature has a deliberately small weight partly for this reason.
    """
    if value:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                parsed = datetime.strptime(value[: len(fmt) + 5], fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                continue
    return datetime.now(timezone.utc)


def fetch_article(url: str, timeout: int = DEFAULT_TIMEOUT) -> FetchResult:
    """Download one URL and extract the article, or return a readable error."""
    url = url.strip()
    if not url:
        return FetchResult(url=url, error="empty URL")
    if not url.startswith(("http://", "https://")):
        return FetchResult(url=url, error="URL must start with http:// or https://")

    try:
        import trafilatura
        from trafilatura.settings import use_config
    except ImportError as exc:  # pragma: no cover
        # Report the ACTUAL import error, not a guess about it.
        #
        # This message originally read "trafilatura is not installed", which was
        # wrong and cost real debugging time: trafilatura WAS installed, but its
        # dependency chain failed because `lxml.html.clean` moved into a
        # separate `lxml_html_clean` package. A generic message swallowed an
        # error that named its own fix.
        #
        # The rule: when you catch an exception to produce a friendly message,
        # include the original. Friendly and uninformative is worse than blunt.
        return FetchResult(
            url=url,
            error=(
                f"web extraction unavailable: {exc}. "
                "Install with: pip install trafilatura lxml_html_clean"
            ),
        )

    try:
        config = use_config()
        config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(timeout))
        config.set("DEFAULT", "USER_AGENTS", USER_AGENT)

        downloaded = trafilatura.fetch_url(url, config=config)
        if not downloaded:
            return FetchResult(
                url=url,
                error="could not download (timeout, 403, or blocked by the site)",
            )

        # with_metadata gives us title and date; include_comments=False keeps
        # reader comments out of the article, which would otherwise pollute
        # entity extraction with names of commenters.
        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            with_metadata=True,
            output_format="json",
        )
        if not extracted:
            return FetchResult(
                url=url,
                error="no article text found (JavaScript-rendered page or paywall?)",
            )

        import json

        payload = json.loads(extracted)
        body = (payload.get("text") or "").strip()
        if not body:
            return FetchResult(url=url, error="extracted article body was empty")

        source = (payload.get("sitename") or "").strip() or _domain(url)
        title = clean_page_title((payload.get("title") or "").strip() or url, source)
        published = _parse_date(payload.get("date"))

        article = Article(
            article_id=make_article_id(source, published, title, url),
            title=title,
            source=source,
            published_at=published,
            body=body,
            url=url,
            extra={
                "author": payload.get("author"),
                "hostname": payload.get("hostname"),
                "fetched_by": "trafilatura",
            },
        )
        return FetchResult(url=url, article=article)

    except Exception as exc:  # noqa: BLE001 -- network code fails many ways
        logger.warning("Failed to fetch %s: %s", url, exc)
        return FetchResult(url=url, error=f"{type(exc).__name__}: {exc}")


def fetch_articles(
    urls: Sequence[str], timeout: int = DEFAULT_TIMEOUT, progress=None
) -> list[FetchResult]:
    """Fetch many URLs, skipping failures.

    ``progress`` is an optional callback ``(index, total, url)`` so a UI can
    report where it is. Passing a callback rather than printing keeps this
    module usable from a script, a notebook and Streamlit alike.
    """
    results: list[FetchResult] = []
    seen: set[str] = set()

    for index, url in enumerate(urls):
        url = url.strip()
        if not url or url in seen:
            continue
        seen.add(url)
        if progress:
            progress(index, len(urls), url)
        results.append(fetch_article(url, timeout=timeout))

    ok = sum(1 for r in results if r.ok)
    logger.info("Fetched %d/%d URLs successfully", ok, len(results))
    return results
