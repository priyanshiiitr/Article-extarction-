"""Generate the reproducible sample corpus at ``data/sample/articles.jsonl``.

WHY A SCRIPT INSTEAD OF A SCRAPER
---------------------------------
Live scraping makes a project unreproducible: sites change, articles get
paywalled, and you cannot write a test whose expected output is stable. A
small, committed, version-controlled corpus means `pytest` produces the same
result on any machine on any day. The ingestion layer is still written against
a pluggable reader interface, so swapping in a real scraper later touches one
file.

WHY THESE PARTICULAR ARTICLES
-----------------------------
Each record carries a ``_why`` key documenting the pipeline behaviour it is
designed to exercise. The corpus deliberately contains TRAPS, because a corpus
where everything works teaches nothing:

  * alias variety      -- "Narendra Modi" / "PM Modi" / "Mr. Modi" / "the
                          Indian Prime Minister" must collapse to one entity.
  * a name collision   -- "Lalit Modi" is a DIFFERENT person. A system that
                          merges him with Narendra Modi has made a false merge,
                          the most dangerous entity-resolution error.
  * role ambiguity     -- one article contains two prime ministers, so "the
                          Prime Minister" is genuinely ambiguous.
  * dirty HTML         -- tags, entities and a <script> block to strip.
  * unicode damage     -- non-breaking spaces, smart quotes, zero-width spaces.
  * abbreviations      -- "Dr.", "U.S.", "Rs.", "No. 1", "Ltd." break naive
                          sentence splitters that cut on every full stop.
  * an exact duplicate -- same URL ingested twice, to prove idempotency.
  * a too-short stub   -- to exercise the minimum-length filter.
  * a malformed record -- missing a required field, to exercise error handling.

The ``_why`` keys are stripped before writing: they document the fixture, they
are not data.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

# Note: bodies are RAW on purpose -- HTML and broken unicode are left in so
# that Phase 2 (preprocessing) has something real to clean.
ARTICLES: list[dict] = [
    {
        "_why": "Alias-rich and pronoun-rich. The anchor document for coreference.",
        "title": "Modi arrives in Kazan for BRICS Summit",
        "source": "The Hindu",
        "published_at": "2024-10-22T09:15:00+05:30",
        "url": "https://example-thehindu.test/news/modi-kazan-brics",
        "body": (
            "Prime Minister Narendra Modi arrived in Kazan on Tuesday for the 16th BRICS "
            "Summit. He was received by senior Russian officials at the airport. The Indian "
            "Prime Minister is scheduled to hold bilateral talks with President Vladimir "
            "Putin later in the day. Mr. Modi will also meet Chinese President Xi Jinping on "
            "the sidelines of the summit. Officials said the visit underlines the importance "
            "New Delhi attaches to the grouping."
        ),
        "extra": {"section": "International"},
    },
    {
        "_why": "Dense relation source: met / discussed / said. Feeds Phase 5.",
        "title": "Modi and Putin discuss energy, defence and trade",
        "source": "Reuters",
        "published_at": "2024-10-23T14:40:00+03:00",
        "url": "https://example-reuters.test/world/modi-putin-kazan",
        "body": (
            "PM Modi met Vladimir Putin in Kazan on Wednesday. The two leaders discussed "
            "energy cooperation, defence ties and trade settled in national currencies. "
            "Putin said bilateral trade between Russia and India had crossed 65 billion "
            "dollars. Modi told reporters that India and Russia would expand cooperation on "
            "artificial intelligence and civil nuclear energy. He described the partnership "
            "as a special and privileged strategic partnership."
        ),
        "extra": {"section": "World"},
    },
    {
        "_why": "Dirty HTML: tags, <script>, &entities, &nbsp;. Exercises the cleaner.",
        "title": "New Development Bank approves fresh lending",
        "source": "The Economic Times",
        "published_at": "2024-10-23T18:05:00+05:30",
        "url": "https://example-et.test/markets/ndb-lending-kazan",
        "body": (
            '<div class="article-body">\n'
            "  <p>The <b>New Development Bank</b> (NDB), headquartered in Shanghai, approved "
            'loans worth &#36;1.2&nbsp;billion for infrastructure projects across member '
            "states, the lender said on Wednesday.</p>\n"
            "  <p>Dilma Rousseff, who chairs the NDB, said the bank would increase lending in "
            "local currencies. She added that the institution was in talks with several "
            "countries about membership.</p>\n"
            '  <p><a href="https://example-et.test/tag/brics">More BRICS coverage</a></p>\n'
            "  <script>window.analytics.track('pageview');</script>\n"
            "</div>"
        ),
        "extra": {"section": "Markets"},
    },
    {
        "_why": (
            "Unicode damage (NBSP, smart quotes, zero-width space, em dash) plus the "
            "'Dr.' abbreviation, and two similar Russian first names (Sergey/Sergei) "
            "that naive fuzzy matching will want to merge."
        ),
        "title": "Jaishankar meets Lavrov on summit sidelines",
        "source": "PTI",
        "published_at": "2024-10-24T11:20:00+05:30",
        "url": "https://example-pti.test/diplomacy/jaishankar-lavrov",
        "body": (
            "External Affairs Minister Dr. S. Jaishankar met Russian Foreign Minister "
            "Sergey Lavrov in Kazan on Thursday.\n\n\n"
            "“We reviewed the full range of our bilateral ties — trade, energy "
            "and connectivity,” Jaishankar said in a post on X.​\n\n"
            "Deputy Foreign Minister Sergei Ryabkov also attended the meeting. He is expected "
            "to travel to New Delhi next month."
        ),
        "extra": {"section": "Diplomacy"},
    },
    {
        "_why": (
            "Abbreviation minefield for sentence segmentation: Rs., No. 1, Ltd., "
            "U.S., Inc., and the decimal '3.5 per cent'. A regex that splits on "
            "every period shatters this article."
        ),
        "title": "Reliance in talks with Rosneft over crude supply",
        "source": "Business Standard",
        "published_at": "2024-10-25T08:00:00+05:30",
        "url": "https://example-bs.test/companies/reliance-rosneft",
        "body": (
            "Reliance Industries Ltd. is in advanced talks with Rosneft over a long-term "
            "crude supply deal worth about Rs. 4,500 crore a month. Mukesh Ambani, chairman "
            "of Reliance, met Rosneft executives in Moscow last week. Reliance is the No. 1 "
            "private refiner in India. Shares of the company rose 3.5 per cent on Friday. "
            "Analysts at Morgan Stanley Inc. said the deal would reduce exposure to U.S. "
            "dollar pricing."
        ),
        "extra": {"section": "Companies"},
    },
    {
        "_why": "Female subject, so coreference must handle 'she'/'her', not just 'he'.",
        "title": "Sitharaman backs BRICS cross-border payment system",
        "source": "The Hindu",
        "published_at": "2024-10-26T10:30:00+05:30",
        "url": "https://example-thehindu.test/business/sitharaman-brics-payments",
        "body": (
            "Finance Minister Nirmala Sitharaman said India supports work on a BRICS "
            "cross-border payment system. She told reporters in New Delhi that the initiative "
            "was about efficiency rather than replacing any currency. The Finance Minister "
            "added that India would continue to engage with the New Development Bank. Her "
            "remarks came days after the Kazan summit concluded."
        ),
        "extra": {"section": "Business"},
    },
    {
        "_why": (
            "THE FALSE-MERGE TRAP. 'Lalit Modi' shares a surname with Narendra Modi "
            "but is a different person, a different domain and five years earlier. "
            "Phase 6 must NOT merge them; Phase 10 measures whether it did."
        ),
        "title": "Lalit Modi loses appeal in London court",
        "source": "Mint",
        "published_at": "2019-08-14T16:45:00+05:30",
        "url": "https://example-mint.test/sports/lalit-modi-appeal",
        "body": (
            "Former Indian Premier League chairman Lalit Modi has lost an appeal in a London "
            "court over a long-running dispute with the Board of Control for Cricket in "
            "India. Mr. Modi, who has lived in London since 2010, said he would consider "
            "further legal options. The BCCI welcomed the ruling. Lalit Modi founded the IPL "
            "in 2008."
        ),
        "extra": {"section": "Sports Business"},
    },
    {
        "_why": (
            "ROLE AMBIGUITY. Two prime ministers appear, so the nominal mention "
            "'the Prime Minister' cannot be resolved by role lookup alone -- it "
            "needs local context. Directly exercises a classic interview question."
        ),
        "title": "Leaders gather as Kazan summit opens",
        "source": "Reuters",
        "published_at": "2024-10-24T09:00:00+03:00",
        "url": "https://example-reuters.test/world/kazan-summit-opens",
        "body": (
            "Ethiopian Prime Minister Abiy Ahmed arrived in Kazan on Thursday, joining Indian "
            "Prime Minister Narendra Modi and other leaders at the BRICS Summit. Abiy Ahmed "
            "addressed the plenary session in the morning. The Prime Minister called for "
            "greater representation of African economies in global financial institutions. "
            "Brazilian President Luiz Inacio Lula da Silva joined the session by video link."
        ),
        "extra": {"section": "World"},
    },
    {
        "_why": "Chinese leader, with 'the Chinese President' as a nominal alias.",
        "title": "Xi calls for deeper BRICS cooperation",
        "source": "Xinhua",
        "published_at": "2024-10-23T20:10:00+08:00",
        "url": "https://example-xinhua.test/world/xi-brics-cooperation",
        "body": (
            "Chinese President Xi Jinping called for deeper cooperation among BRICS members "
            "in a speech in Kazan on Wednesday. The Chinese President said the grouping "
            "should push forward reform of global governance. Xi met Narendra Modi on the "
            "sidelines, the first formal meeting between the two leaders in five years."
        ),
        "extra": {"section": "World"},
    },
    {
        "_why": "Event-centric: the Kazan Declaration as a named EVENT/document.",
        "title": "BRICS leaders adopt Kazan Declaration",
        "source": "TASS",
        "published_at": "2024-10-23T21:30:00+03:00",
        "url": "https://example-tass.test/politics/kazan-declaration",
        "body": (
            "BRICS leaders adopted the Kazan Declaration at the close of the 16th BRICS "
            "Summit on Wednesday. The document covers payment systems, grain trading and "
            "cooperation on artificial intelligence. Vladimir Putin chaired the session. "
            "Russia holds the BRICS chairmanship in 2024."
        ),
        "extra": {"section": "Politics"},
    },
    {
        "_why": (
            "Later, separate article. Cross-article entity resolution must link "
            "'Prime Minister Modi' here to 'Narendra Modi' in the October articles."
        ),
        "title": "Prime Minister Modi reviews national AI mission",
        "source": "Financial Express",
        "published_at": "2024-11-05T12:00:00+05:30",
        "url": "https://example-fe.test/tech/modi-ai-mission-review",
        "body": (
            "Prime Minister Modi reviewed the progress of the national artificial "
            "intelligence mission at a meeting in New Delhi on Tuesday. The Indian Prime "
            "Minister asked officials to accelerate work on compute infrastructure and "
            "skilling. He said artificial intelligence would be central to India's growth. "
            "Officials from the Ministry of Electronics and Information Technology attended."
        ),
        "extra": {"section": "Technology"},
    },
    {
        "_why": "EXACT DUPLICATE of the first article (same URL) -> must be deduplicated.",
        "title": "Modi arrives in Kazan for BRICS Summit",
        "source": "The Hindu",
        "published_at": "2024-10-22T09:15:00+05:30",
        "url": "https://example-thehindu.test/news/modi-kazan-brics",
        "body": (
            "Prime Minister Narendra Modi arrived in Kazan on Tuesday for the 16th BRICS "
            "Summit. He was received by senior Russian officials at the airport. The Indian "
            "Prime Minister is scheduled to hold bilateral talks with President Vladimir "
            "Putin later in the day. Mr. Modi will also meet Chinese President Xi Jinping on "
            "the sidelines of the summit. Officials said the visit underlines the importance "
            "New Delhi attaches to the grouping."
        ),
        "extra": {"section": "International"},
    },
    {
        "_why": "Too short (< min_body_chars) -> dropped as a stub/failed fetch.",
        "title": "Summit briefing",
        "source": "Reuters",
        "published_at": "2024-10-24T07:00:00+03:00",
        "url": "https://example-reuters.test/world/summit-briefing-stub",
        "body": "Leaders meet in Kazan.",
        "extra": {"section": "World"},
    },
]

# Written verbatim as a raw line: it is missing `published_at`, so pydantic
# validation must reject it and the loader must skip it WITHOUT crashing the
# whole batch. One bad record out of 100,000 should never kill a pipeline run.
MALFORMED_LINE = json.dumps(
    {
        "title": "Broken record with no publication date",
        "source": "Unknown Wire",
        "url": "https://example-unknown.test/broken",
        "body": "This record is intentionally malformed to exercise error handling "
        "in the ingestion loader. It has no published_at field at all.",
    }
)


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    out_path: Path = cfg.path(cfg.ingestion.source_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for record in ARTICLES:
            payload = {k: v for k, v in record.items() if not k.startswith("_")}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fh.write(MALFORMED_LINE + "\n")

    logger.info(
        "Wrote sample corpus: %d good records + 1 malformed -> %s",
        len(ARTICLES),
        out_path,
    )


if __name__ == "__main__":
    main()
