"""News Knowledge Graph -- Gradio dashboard (HuggingFace Spaces entry point).

    python app.py

WHY THIS FILE IS AT THE REPOSITORY ROOT AND NAMED app.py
---------------------------------------------------------
HuggingFace Spaces looks for ``app_file`` from the README frontmatter, and the
convention is ``app.py`` at the root. Keeping it there avoids one class of
"the Space builds but shows nothing" confusion.

WHY GRADIO RATHER THAN STREAMLIT
--------------------------------
Streamlit is no longer a supported Spaces SDK -- the config reference allows
only ``gradio``, ``docker`` or ``static``, and Docker requires a paid plan.
Nothing in ``src/`` changed for this port: the pipeline never knew what was
rendering it, which is the payoff of keeping the UI as a thin layer over an
orchestrator (``src/pipeline/runner.py``).

TWO GRADIO SPECIFICS WORTH KNOWING
----------------------------------
1. STATE IS EXPLICIT. Gradio does not re-run the script on every interaction
   the way Streamlit does; instead, per-session values live in ``gr.State`` and
   are passed in and out of event handlers. That is more verbose but easier to
   reason about -- there is no hidden rerun.
2. MODELS ARE MODULE-LEVEL AND LAZY. Loading happens on first use inside the
   pipeline and is cached by ``functools.lru_cache`` in ``nlp_resources`` and
   the embedder, so a 590M-parameter model is loaded once per process rather
   than once per request.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import gradio as gr  # noqa: E402

from src.config import load_config  # noqa: E402
from src.entity_resolution.normalize import normalize_name  # noqa: E402
from src.ingestion.loader import ingest  # noqa: E402
from src.ingestion.web import fetch_articles  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.pipeline.runner import run_pipeline  # noqa: E402
from src.schemas import Article, make_article_id  # noqa: E402

# --- HuggingFace ZeroGPU integration ---------------------------------------
# A Space running on ZeroGPU hardware REFUSES TO START unless at least one
# function is decorated with @spaces.GPU. The runtime scans the app at import
# time and fails with "No @spaces.GPU function detected during startup" if it
# finds none. That check is on the HARDWARE you selected, not on whether your
# code actually needs a GPU -- and this pipeline is CPU-bound by design.
#
# The `spaces` package only exists inside a Space container, so importing it
# unconditionally would break every local run. We fall back to an identity
# decorator, which is exactly how HF documents `@spaces.GPU` behaving outside
# a ZeroGPU environment: effect-free.
try:
    import spaces  # noqa: E402

    gpu_task = spaces.GPU
except ImportError:  # running locally, or on non-ZeroGPU hardware

    def gpu_task(*decorator_args, **decorator_kwargs):
        """No-op stand-in for spaces.GPU outside a ZeroGPU Space."""
        # Support both @gpu_task and @gpu_task(duration=...) spellings.
        if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
            return decorator_args[0]

        def wrap(function):
            return function

        return wrap


@gpu_task(duration=15)
def gpu_status() -> str:
    """Report whether a real GPU is attached right now.

    This is the function that satisfies ZeroGPU's startup requirement, and it
    is deliberately a real, honest feature rather than a dead stub: ZeroGPU
    grants a physical GPU only for the duration of a decorated call, so this
    genuinely answers "is one attached at this moment?".

    It is NOT on the main pipeline path. The pipeline stays on CPU, so normal
    use consumes none of the 5-minutes-per-day free ZeroGPU quota.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return (
                f"**GPU attached:** {torch.cuda.get_device_name(0)} "
                f"(torch {torch.__version__})"
            )
        return (
            f"**No GPU attached** — running on CPU (torch {torch.__version__}). "
            "This is expected: the pipeline is CPU-bound by design, so it uses "
            "none of the ZeroGPU daily quota."
        )
    except ImportError:
        return "**torch is not installed** — this is the `lite` profile, which needs no GPU."

configure_logging("WARNING", "rich")
CFG = load_config()

PROFILE_NOTE = {
    "lite": (
        "**LITE** — spaCy + gazetteer NER, rule-based coreference, no embeddings. "
        "~0.6 GB RAM, under 1 s per article. Resolves **no nominal coreference**, "
        "so \"The Indian Prime Minister\" will not link to Modi."
    ),
    "full": (
        "**FULL** — transformer NER, LingMess neural coreference (590M), context "
        "embeddings. ~4.5 GB RAM, ~14 s per article on CPU. First run downloads "
        "~2.5 GB of models."
    ),
}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def load_sample() -> tuple[list, str, list]:
    articles, stats = ingest(cfg=CFG)
    message = (
        f"Loaded **{stats.kept}** articles from the sample corpus "
        f"(dropped {stats.invalid} invalid, {stats.duplicates} duplicate, "
        f"{stats.too_short} too short).\n\n"
        "This corpus contains deliberate traps: alias variety, a name collision "
        "(Lalit Modi vs Narendra Modi), role ambiguity, dirty HTML and broken unicode."
    )
    return articles, message, _article_rows(articles)


def load_urls(urls_text: str, progress=gr.Progress()) -> tuple[list, str, list]:
    urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
    if not urls:
        return [], "Paste at least one URL.", []

    progress(0, desc="Fetching…")
    results = fetch_articles(urls)
    ok = [r.article for r in results if r.ok]
    failed = [(r.url, r.error) for r in results if not r.ok]

    lines = [f"Fetched **{len(ok)}** of {len(results)} URL(s)."]
    if failed:
        lines.append("\n**Failures**")
        for url, error in failed:
            lines.append(f"- `{url[:70]}` — {error}")
        lines.append(
            "\n*Large publishers (Reuters, Bloomberg) block scrapers, and "
            "JavaScript-rendered pages have nothing to extract. Use the "
            "**Paste text** box for those.*"
        )
    return ok, "\n".join(lines), _article_rows(ok)


def add_text(existing: list, title: str, source: str, body: str) -> tuple[list, str, list]:
    if not (body or "").strip():
        return existing, "Article text is required.", _article_rows(existing)
    now = datetime.now(timezone.utc)
    headline = (title or "").strip() or body.strip()[:60]
    source = (source or "").strip() or "Manual input"
    article = Article(
        article_id=make_article_id(source, now, headline),
        title=headline, source=source, published_at=now, body=body.strip(),
    )
    updated = list(existing) + [article]
    return updated, f"Added. **{len(updated)}** article(s) loaded.", _article_rows(updated)


def _article_rows(articles: list) -> list[list]:
    return [
        [a.published_at.strftime("%Y-%m-%d"), a.source, a.title[:70], len(a.body)]
        for a in articles
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(articles: list, profile: str, progress=gr.Progress()) -> tuple:
    if not articles:
        return None, "Load articles on the **Ingest** tab first.", [], []

    def report(stage: str, fraction: float, detail: str) -> None:
        progress(min(fraction, 1.0), desc=f"{stage} — {detail}")

    result = run_pipeline(articles, cfg=CFG, profile=profile, progress=report)

    counts = result.counts
    timing_rows = sorted(
        ([stage, round(seconds, 2)] for stage, seconds in result.timings.items()),
        key=lambda row: -row[1],
    )
    slowest = timing_rows[0][0] if timing_rows else "n/a"

    # Report what ACTUALLY ran, not what was requested. A "full" run that
    # finishes in seconds means a model failed to load and the pipeline fell
    # back to lighter components -- correct behaviour, but it must be visible.
    backend_lines = "\n".join(
        f"- **{stage}**: {backend}" for stage, backend in result.backends.items()
    )
    if result.degraded:
        banner = (
            "> ⚠️ **This run was DEGRADED — you did not get the full pipeline.**\n>\n"
            + "\n".join(f"> - {d}" for d in result.degraded)
            + "\n>\n> That is why it finished quickly.\n\n"
        )
    else:
        banner = ""

    summary = (
        banner
        + f"### Done in {result.total_seconds:.1f}s\n\n"
        + f"**Components that actually ran**\n{backend_lines}\n\n"
        f"| | |\n|---|---:|\n"
        f"| Articles | {counts.get('articles', 0)} |\n"
        f"| Sentences | {counts.get('sentences', 0)} |\n"
        f"| Entity mentions | {counts.get('mentions', 0)} |\n"
        f"| Coreference clusters | {counts.get('coref_clusters', 0)} |\n"
        f"| Relations | {counts.get('relations', 0)} |\n"
        f"| **Canonical entities** | **{counts.get('entities', 0)}** |\n"
        f"| Merges blocked by veto | {counts.get('vetoed', 0)} |\n"
        f"| Pairs sent to review | {counts.get('review_pairs', 0)} |\n\n"
        f"Slowest stage: **{slowest}**."
    )
    return result, summary, timing_rows, _entity_rows(result)


def _entity_rows(result) -> list[list]:
    if not result:
        return []
    return [
        [
            e.entity_id, e.canonical_name, e.entity_type, len(e.article_ids),
            ", ".join(sorted(e.aliases)),
            ", ".join(sorted(e.roles)) or "—",
            ", ".join(sorted(e.countries)) or "—",
            round(e.merge_confidence, 2),
        ]
        for e in sorted(result.entities, key=lambda e: -len(e.article_ids))
    ]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query_entity(result, name: str) -> str:
    if not result:
        return "Run the pipeline first."
    if not (name or "").strip():
        return "Type an entity name or alias."

    needle = normalize_name(name, "PERSON")
    matches = [
        e for e in result.entities
        if needle == normalize_name(e.canonical_name, e.entity_type)
        or any(needle == normalize_name(a, e.entity_type) for a in e.aliases)
        or name.strip().lower() in e.canonical_name.lower()
    ]
    if not matches:
        return f"No entity matched **{name}**."

    out: list[str] = []
    for entity in matches[:4]:
        out.append(f"## {entity.canonical_name}\n")
        out.append(
            f"`{entity.entity_id}` · {entity.entity_type} · "
            f"{len(entity.article_ids)} article(s)  \n"
            f"**Aliases:** {', '.join(sorted(entity.aliases))}"
        )
        if entity.roles:
            out.append(f"**Roles:** {', '.join(sorted(entity.roles))}")
        if entity.countries:
            out.append(f"**Countries:** {', '.join(sorted(entity.countries))}")

        grouped: dict[str, list] = {}
        for document in result.documents:
            for relation in document.relations:
                if relation.subject_mention_id in entity.mention_ids:
                    grouped.setdefault(relation.predicate, []).append((relation, document))

        if grouped:
            tree = [entity.canonical_name]
            items = list(grouped.items())
            for i, (predicate, pairs) in enumerate(items):
                for j, (relation, _) in enumerate(pairs):
                    last = i == len(items) - 1 and j == len(pairs) - 1
                    tree.append(
                        f"{'`--' if last else '|--'} {predicate} -> "
                        f"{relation.object_text}  ({relation.confidence:.2f})"
                    )
            out.append("\n```\n" + "\n".join(tree) + "\n```")

            out.append("\n**Why we believe these — provenance**\n")
            for predicate, pairs in grouped.items():
                for relation, document in pairs:
                    out.append(
                        f"- **{relation.subject_text} → {predicate} → "
                        f"{relation.object_text}** "
                        f"({relation.confidence:.2f}, `{relation.extractor}`/"
                        f"`{relation.trigger}`)  \n"
                        f"  *{document.source}* — "
                        f"“{relation.evidence_in(document.text)[:160]}”"
                    )
        else:
            out.append("\n*No relations extracted for this entity.*")

        surfaces_by_article = []
        for document in result.documents:
            used = sorted({m.text for m in document.mentions
                           if m.mention_id in entity.mention_ids})
            if used:
                surfaces_by_article.append(
                    f"- {document.published_at.date()} · *{document.source}* · "
                    f"{document.title[:56]} — used: **{', '.join(used)}**"
                )
        if surfaces_by_article:
            out.append("\n**Articles mentioning this entity**\n")
            out.extend(surfaces_by_article)
        out.append("\n---\n")
    return "\n".join(out)


def low_confidence(result, threshold: float) -> list[list]:
    if not result:
        return []
    rows = []
    for document in result.documents:
        for relation in document.relations:
            if relation.confidence < threshold:
                rows.append([
                    round(relation.confidence, 2),
                    relation.subject_text, relation.predicate, relation.object_text,
                    "yes" if (relation.subject_via_coref or relation.object_via_coref) else "no",
                    relation.evidence_in(document.text)[:110],
                ])
    return sorted(rows, key=lambda row: row[0])


def review_rows(result) -> tuple[str, list[list]]:
    if not result:
        return "Run the pipeline first.", []
    profiles = {p.profile_id: p for p in result.profiles}
    rows = []
    for pair in sorted(result.review_queue, key=lambda p: -p.score):
        left, right = profiles.get(pair.left_id), profiles.get(pair.right_id)
        rows.append([
            round(pair.score, 3),
            left.canonical_surface if left else pair.left_id,
            right.canonical_surface if right else pair.right_id,
            ", ".join(sorted(left.roles)) if left and left.roles else "—",
            ", ".join(sorted(right.roles)) if right and right.roles else "—",
            pair.explain()[:120],
        ])
    note = (
        f"**{len(rows)}** pair(s) the resolver would not decide automatically.\n\n"
        "“I don't know” is a valid output here. A **false merge** fuses two real "
        "people permanently and corrupts every fact about them; a **false split** "
        "only costs recall and is recoverable. So uncertain pairs wait for a human."
    )
    return note, rows


def evaluate_run(result) -> str:
    if not result:
        return "Run the pipeline first."
    gold_path = CFG.path(CFG.paths.sample_dir) / "gold" / "annotations.json"
    if not gold_path.exists():
        return "Gold annotations not found."

    from src.evaluation.evaluate import evaluate, load_gold

    report = evaluate(result.documents, load_gold(gold_path), result.entities)
    if report.articles_evaluated == 0:
        return (
            "None of the gold-labelled articles are in this run. "
            "Load the **sample corpus** and re-run the pipeline."
        )

    lines = [
        f"### Scored on {report.articles_evaluated} hand-labelled article(s)\n",
        "| Metric | P | R | F1 |",
        "|---|---:|---:|---:|",
        f"| NER (strict span+label) | {report.ner_strict.precision:.3f} | "
        f"{report.ner_strict.recall:.3f} | **{report.ner_strict.f1:.3f}** |",
        f"| NER (relaxed overlap) | {report.ner_relaxed.precision:.3f} | "
        f"{report.ner_relaxed.recall:.3f} | {report.ner_relaxed.f1:.3f} |",
        f"| Relations | {report.relations.precision:.3f} | "
        f"{report.relations.recall:.3f} | **{report.relations.f1:.3f}** |",
        f"| Coreference (MUC) | {report.coref_muc.precision:.3f} | "
        f"{report.coref_muc.recall:.3f} | {report.coref_muc.f1:.3f} |",
    ]
    p, r, f = report.coref_b3
    lines.append(f"| Coreference (B-cubed) | {p:.3f} | {r:.3f} | {f:.3f} |")

    if report.entity_resolution:
        er = report.entity_resolution
        lines.append(
            f"| Entity resolution (pairwise) | {er.pairwise.precision:.3f} | "
            f"{er.pairwise.recall:.3f} | {er.pairwise.f1:.3f} |"
        )
        lines.append(
            f"\n**False merges: {er.false_merges}** (corrupting, hard to undo) · "
            f"**False splits: {er.false_splits}** (costs recall, recoverable)\n\n"
            "Reported separately on purpose — an aggregate F1 cannot distinguish "
            "a corrupted knowledge base from a merely incomplete one."
        )

    lines.append("\n**NER by entity type**\n")
    lines.append("| label | P | R | F1 |")
    lines.append("|---|---:|---:|---:|")
    for label, prf in sorted(report.ner_by_label.items(), key=lambda kv: -kv[1].f1):
        lines.append(f"| {label} | {prf.precision:.3f} | {prf.recall:.3f} | {prf.f1:.3f} |")

    lines.append(
        "\n> **Caveat:** 3 articles, single annotator, no inter-annotator "
        "agreement score. A development signal, not a benchmark."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="News Knowledge Graph") as demo:
    articles_state = gr.State([])
    result_state = gr.State(None)

    gr.Markdown(
        "# 🕸️ News Knowledge Graph\n"
        "Paste news article URLs → entities, relations and a queryable knowledge "
        "graph, with every fact traceable to the sentence it came from.\n\n"
        "`articles → clean & segment → NER → coreference → relations → entity resolution`"
    )

    with gr.Tabs():
        # --- Ingest ------------------------------------------------------
        with gr.Tab("📥 Ingest"):
            with gr.Row():
                with gr.Column(scale=3):
                    url_box = gr.Textbox(
                        label="Article URLs (one per line)",
                        lines=6,
                        placeholder="https://en.wikipedia.org/wiki/BRICS\nhttps://...",
                    )
                    with gr.Row():
                        fetch_btn = gr.Button("Fetch URLs", variant="primary")
                        sample_btn = gr.Button("Use sample corpus")
                with gr.Column(scale=2):
                    gr.Markdown(
                        "**Extraction uses trafilatura**, which removes navigation, "
                        "cookie banners and related-story boilerplate — a regex HTML "
                        "stripper cannot do that reliably.\n\n"
                        "Big publishers block scrapers. Wikipedia and regional outlets "
                        "work well; for anything else use the box below."
                    )

            with gr.Accordion("Paste raw text instead (for paywalls / blocked sites)", open=False):
                text_title = gr.Textbox(label="Title")
                text_source = gr.Textbox(label="Source", value="Manual input")
                text_body = gr.Textbox(label="Article text", lines=8)
                add_btn = gr.Button("Add article")

            ingest_status = gr.Markdown()
            article_table = gr.Dataframe(
                headers=["published", "source", "title", "chars"],
                label="Loaded articles", interactive=False, wrap=True,
            )

        # --- Run ---------------------------------------------------------
        with gr.Tab("⚙️ Run pipeline"):
            profile_radio = gr.Radio(
                ["lite", "full"], value="lite", label="Resource profile",
                info="Profiles swap heavy models for light ones. Config only, no code change.",
            )
            profile_note = gr.Markdown(PROFILE_NOTE["lite"])
            run_btn = gr.Button("▶ Run pipeline", variant="primary", size="lg")
            run_summary = gr.Markdown()
            timing_table = gr.Dataframe(
                headers=["stage", "seconds"], label="Where the time went",
                interactive=False,
            )

        # --- Entities ----------------------------------------------------
        with gr.Tab("🧩 Entities"):
            gr.Markdown(
                "Canonical entities resolved **across** articles. The aliases column "
                "is the point: several surface forms, one entity."
            )
            entity_table = gr.Dataframe(
                headers=["entity_id", "name", "type", "articles", "aliases",
                         "roles", "countries", "merge_conf"],
                label="Canonical entities", interactive=False, wrap=True,
            )

        # --- Query -------------------------------------------------------
        with gr.Tab("🔎 Query"):
            gr.Markdown(
                "Search by **any alias** — typing `PM Modi` finds the canonical "
                "entity `Narendra Modi`. That is what entity resolution bought us."
            )
            with gr.Row():
                query_box = gr.Textbox(label="Entity name or alias",
                                       placeholder="PM Modi", scale=4)
                query_btn = gr.Button("Search", variant="primary", scale=1)
            query_out = gr.Markdown()

            gr.Markdown("---\n### Low-confidence relations — the audit view")
            conf_slider = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                    label="Show relations with confidence below")
            lowconf_table = gr.Dataframe(
                headers=["conf", "subject", "predicate", "object", "via coref", "evidence"],
                interactive=False, wrap=True,
            )

        # --- Review ------------------------------------------------------
        with gr.Tab("⚖️ Review queue"):
            review_note = gr.Markdown()
            review_table = gr.Dataframe(
                headers=["score", "left", "right", "left roles", "right roles", "why"],
                interactive=False, wrap=True,
            )
            review_btn = gr.Button("Refresh review queue")

        # --- Evaluation --------------------------------------------------
        with gr.Tab("📊 Evaluation"):
            gr.Markdown(
                "Scores this run against a hand-labelled gold standard. "
                "Requires the **sample corpus** (that is what was annotated)."
            )
            eval_btn = gr.Button("Run evaluation", variant="primary")
            eval_out = gr.Markdown()

            with gr.Accordion("Runtime info", open=False):
                gr.Markdown(
                    "This Space runs on **ZeroGPU** hardware, which requires an app "
                    "to expose at least one GPU-capable function. The pipeline "
                    "itself is CPU-bound, so normal use consumes none of the "
                    "daily GPU quota."
                )
                gpu_btn = gr.Button("Check GPU availability")
                gpu_out = gr.Markdown()

    # --- Wiring ----------------------------------------------------------
    fetch_btn.click(load_urls, [url_box],
                    [articles_state, ingest_status, article_table])
    sample_btn.click(load_sample, None,
                     [articles_state, ingest_status, article_table])
    add_btn.click(add_text, [articles_state, text_title, text_source, text_body],
                  [articles_state, ingest_status, article_table])

    profile_radio.change(lambda p: PROFILE_NOTE[p], [profile_radio], [profile_note])

    run_btn.click(
        run, [articles_state, profile_radio],
        [result_state, run_summary, timing_table, entity_table],
    ).then(review_rows, [result_state], [review_note, review_table]
    ).then(low_confidence, [result_state, conf_slider], [lowconf_table])

    query_btn.click(query_entity, [result_state, query_box], [query_out])
    query_box.submit(query_entity, [result_state, query_box], [query_out])
    conf_slider.change(low_confidence, [result_state, conf_slider], [lowconf_table])
    review_btn.click(review_rows, [result_state], [review_note, review_table])
    eval_btn.click(evaluate_run, [result_state], [eval_out])
    gpu_btn.click(gpu_status, None, [gpu_out])


if __name__ == "__main__":
    # 7860 is the Spaces default for Gradio. 0.0.0.0 so the container is
    # reachable from outside; on a laptop it is still localhost.
    #
    # `theme` moved from the Blocks constructor to launch() in Gradio 6.
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
