"""News Knowledge Graph -- Streamlit dashboard.

    streamlit run app/streamlit_app.py

DESIGN NOTES
------------
1. MODELS ARE CACHED WITH @st.cache_resource, NOT @st.cache_data.
   Streamlit re-runs this entire script on every interaction -- every button
   click, every slider drag. Without caching, a 590M-parameter coreference
   model would reload on each one. The distinction matters:
       cache_data     -> for serialisable RESULTS (dataframes, dicts). Copied.
       cache_resource -> for unserialisable OBJECTS (models, DB connections).
                         Shared by reference across reruns and sessions.
   Using cache_data for a torch model would try to pickle it on every call.

2. STATE LIVES IN st.session_state.
   Because the script re-runs top to bottom, ordinary Python variables are
   destroyed constantly. Anything that must survive a click goes in
   session_state.

3. THE PIPELINE IS NOT RE-RUN ON EVERY INTERACTION.
   It runs once on an explicit button press and the result is stored. Browsing
   the entity list must never retrigger a 14-second-per-document job.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Streamlit executes this file as a script, so the project root is not
# automatically importable the way it is under `pip install -e .` + pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tokenizer parallelism warns loudly and pointlessly inside a web server; and
# the datasets library's progress bars are noise in this context.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config  # noqa: E402
from src.ingestion.loader import ingest  # noqa: E402
from src.ingestion.web import fetch_articles  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.pipeline.runner import run_pipeline  # noqa: E402
from src.schemas import Article, make_article_id  # noqa: E402

st.set_page_config(
    page_title="News Knowledge Graph",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_logging("WARNING", "rich")

PROFILE_HELP = {
    "full": "Transformer NER + neural coreference + embeddings. Best quality. "
            "~4.5 GB RAM, ~14 s per article on CPU.",
    "lite": "spaCy NER + gazetteer + rule-based coreference, no embeddings. "
            "~0.6 GB RAM, under 1 s per article. Lower recall on aliases.",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {
        "articles": [],
        "result": None,
        "fetch_errors": [],
        "profile": "lite",
        "persisted": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()
cfg = load_config()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🕸️ News KG")
    st.caption("Articles → entities → relations → knowledge graph")

    st.subheader("Pipeline profile")
    profile = st.radio(
        "Resource profile",
        options=["lite", "full"],
        index=0 if st.session_state.profile == "lite" else 1,
        format_func=lambda p: p.upper(),
        help="Profiles swap heavy models for light ones. Config only, no code change.",
    )
    st.session_state.profile = profile
    st.info(PROFILE_HELP[profile], icon="⚙️")

    if profile == "full":
        st.warning(
            "First run downloads ~2.5 GB of models and takes several minutes. "
            "Needs ~4.5 GB RAM — will not fit Streamlit Community Cloud's 1 GB.",
            icon="⚠️",
        )

    st.divider()
    st.subheader("Loaded")
    st.metric("Articles", len(st.session_state.articles))
    if st.session_state.result:
        counts = st.session_state.result.counts
        col_a, col_b = st.columns(2)
        col_a.metric("Entities", counts.get("entities", 0))
        col_b.metric("Relations", counts.get("relations", 0))

    st.divider()
    if st.button("Reset everything", use_container_width=True):
        for key in ["articles", "result", "fetch_errors", "persisted"]:
            st.session_state[key] = [] if key in {"articles", "fetch_errors"} else None
        st.session_state.persisted = False
        st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ingest, tab_run, tab_entities, tab_query, tab_review, tab_eval = st.tabs(
    ["📥 Ingest", "⚙️ Run pipeline", "🧩 Entities", "🔎 Query", "⚖️ Review queue", "📊 Evaluation"]
)


# --- 1. Ingest --------------------------------------------------------------

with tab_ingest:
    st.header("Load articles")

    source_mode = st.radio(
        "Source",
        ["Paste URLs", "Paste raw text", "Use the sample corpus"],
        horizontal=True,
    )

    if source_mode == "Paste URLs":
        st.caption(
            "One URL per line. Extraction uses **trafilatura**, which removes "
            "navigation, cookie banners and related-story boilerplate — a regex "
            "HTML stripper cannot do that reliably."
        )
        urls_text = st.text_area(
            "Article URLs",
            height=160,
            placeholder="https://www.reuters.com/world/...\nhttps://www.thehindu.com/news/...",
        )
        if st.button("Fetch articles", type="primary"):
            urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
            if not urls:
                st.error("Paste at least one URL.")
            else:
                bar = st.progress(0.0, text="Starting…")
                results = fetch_articles(
                    urls,
                    progress=lambda i, total, url: bar.progress(
                        i / max(total, 1), text=f"Fetching {url[:70]}…"
                    ),
                )
                bar.empty()
                ok = [r.article for r in results if r.ok]
                errors = [(r.url, r.error) for r in results if not r.ok]
                st.session_state.articles = ok
                st.session_state.fetch_errors = errors
                st.session_state.result = None
                if ok:
                    st.success(f"Fetched {len(ok)} of {len(results)} URLs.")
                if errors:
                    st.warning(f"{len(errors)} URL(s) failed — see below.")

    elif source_mode == "Paste raw text":
        st.caption("For pages that block scrapers or sit behind a paywall.")
        title = st.text_input("Title", placeholder="Modi arrives in Kazan for BRICS Summit")
        source_name = st.text_input("Source", value="Manual input")
        body = st.text_area("Article text", height=220)
        if st.button("Add article", type="primary"):
            if not body.strip():
                st.error("Article text is required.")
            else:
                now = datetime.now(timezone.utc)
                headline = title.strip() or body.strip()[:60]
                article = Article(
                    article_id=make_article_id(source_name, now, headline),
                    title=headline,
                    source=source_name or "Manual input",
                    published_at=now,
                    body=body.strip(),
                )
                st.session_state.articles = st.session_state.articles + [article]
                st.session_state.result = None
                st.success(f"Added. {len(st.session_state.articles)} article(s) loaded.")

    else:
        st.caption(
            "11 articles about BRICS, politics and business, containing deliberate "
            "traps: alias variety, a name collision (Lalit vs Narendra Modi), role "
            "ambiguity, dirty HTML and broken unicode."
        )
        if st.button("Load sample corpus", type="primary"):
            articles, stats = ingest(cfg=cfg)
            st.session_state.articles = articles
            st.session_state.result = None
            st.success(f"Loaded {stats.kept} articles (dropped {stats.invalid} invalid, "
                       f"{stats.duplicates} duplicate, {stats.too_short} too short).")

    if st.session_state.fetch_errors:
        with st.expander(f"⚠️ {len(st.session_state.fetch_errors)} fetch failure(s)"):
            for url, error in st.session_state.fetch_errors:
                st.write(f"**{url}**")
                st.caption(error)

    if st.session_state.articles:
        st.divider()
        st.subheader(f"{len(st.session_state.articles)} article(s) ready")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "published": a.published_at.strftime("%Y-%m-%d"),
                        "source": a.source,
                        "title": a.title[:70],
                        "chars": len(a.body),
                    }
                    for a in st.session_state.articles
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


# --- 2. Run pipeline --------------------------------------------------------

with tab_run:
    st.header("Run the extraction pipeline")

    if not st.session_state.articles:
        st.info("Load articles on the **Ingest** tab first.", icon="👈")
    else:
        stages = (
            "clean & segment → NER → coreference → relations → entity resolution"
        )
        st.caption(f"Profile **{st.session_state.profile.upper()}** · {stages}")

        if st.session_state.profile == "full":
            estimate = len(st.session_state.articles) * 14
            st.caption(f"Estimated runtime: ~{estimate // 60}m {estimate % 60}s "
                       f"(coreference is ~94% of it).")

        if st.button("▶ Run pipeline", type="primary"):
            bar = st.progress(0.0, text="Starting…")
            status = st.empty()

            def progress(stage: str, fraction: float, detail: str) -> None:
                bar.progress(min(fraction, 1.0), text=f"{stage} — {detail}")
                status.caption(f"Stage: **{stage}** · {detail}")

            with st.spinner("Running…"):
                result = run_pipeline(
                    st.session_state.articles,
                    cfg=cfg,
                    profile=st.session_state.profile,
                    progress=progress,
                )
            bar.empty()
            status.empty()
            st.session_state.result = result
            st.session_state.persisted = False
            st.success(f"Done in {result.total_seconds:.1f}s.")

    result = st.session_state.result
    if result:
        st.divider()
        counts = result.counts
        cols = st.columns(6)
        for col, (label, key) in zip(
            cols,
            [("Articles", "articles"), ("Sentences", "sentences"), ("Mentions", "mentions"),
             ("Coref clusters", "coref_clusters"), ("Relations", "relations"),
             ("Entities", "entities")],
        ):
            col.metric(label, counts.get(key, 0))

        st.subheader("Where the time went")
        timings = pd.DataFrame(
            [{"stage": k, "seconds": round(v, 2)} for k, v in result.timings.items()]
        ).sort_values("seconds", ascending=False)
        st.bar_chart(timings.set_index("stage"), horizontal=True)
        st.caption(
            "In FULL profile coreference dominates (~94% of wall clock). That is "
            "the number every scaling decision follows from — profile before you "
            "parallelise."
        )

        if result.warnings:
            with st.expander(f"⚠️ {len(result.warnings)} warning(s)"):
                for warning in result.warnings[:30]:
                    st.caption(warning)

        st.divider()
        if st.button("💾 Save to knowledge graph (SQLite)"):
            from src.pipeline.runner import persist

            with st.spinner("Writing…"):
                stats = persist(result, cfg=cfg)
            st.session_state.persisted = True
            st.success("Saved.")
            st.json(stats)


# --- 3. Entities ------------------------------------------------------------

with tab_entities:
    st.header("Canonical entities")
    result = st.session_state.result

    if not result:
        st.info("Run the pipeline first.", icon="👈")
    else:
        entities = result.entities
        types = sorted({e.entity_type for e in entities})
        chosen = st.multiselect("Entity type", types, default=types)
        visible = [e for e in entities if e.entity_type in chosen]

        st.caption(
            f"{len(visible)} entities resolved from {result.counts.get('profiles', 0)} "
            f"local profiles across {result.counts.get('articles', 0)} articles. "
            f"{result.counts.get('vetoed', 0)} candidate merges were blocked by a hard veto."
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "entity_id": e.entity_id,
                        "name": e.canonical_name,
                        "type": e.entity_type,
                        "articles": len(e.article_ids),
                        "aliases": ", ".join(sorted(e.aliases)),
                        "roles": ", ".join(sorted(e.roles)),
                        "countries": ", ".join(sorted(e.countries)),
                        "merge_conf": round(e.merge_confidence, 2),
                    }
                    for e in sorted(visible, key=lambda e: -len(e.article_ids))
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("Entity detail")
        names = [f"{e.canonical_name}  ({e.entity_id})" for e in visible]
        if names:
            picked = st.selectbox("Entity", names)
            entity = visible[names.index(picked)]

            left, right = st.columns([1, 2])
            with left:
                st.markdown(f"### {entity.canonical_name}")
                st.caption(f"`{entity.entity_id}` · {entity.entity_type}")
                st.write("**Aliases**:", ", ".join(sorted(entity.aliases)) or "—")
                if entity.roles:
                    st.write("**Roles**:", ", ".join(sorted(entity.roles)))
                if entity.countries:
                    st.write("**Countries**:", ", ".join(sorted(entity.countries)))
                if entity.orgs:
                    st.write("**Organisations**:", ", ".join(sorted(entity.orgs)))
                if entity.first_seen:
                    st.write(f"**Seen**: {entity.first_seen.date()} → {entity.last_seen.date()}")
                st.write(f"**Merge confidence**: {entity.merge_confidence:.2f}")

            with right:
                st.markdown("**Relations extracted for this entity**")
                rows = []
                for document in result.documents:
                    for relation in document.relations:
                        if relation.subject_mention_id in entity.mention_ids:
                            rows.append(
                                {
                                    "predicate": relation.predicate,
                                    "object": relation.object_text,
                                    "conf": round(relation.confidence, 2),
                                    "via coref": relation.subject_via_coref or relation.object_via_coref,
                                    "evidence": relation.evidence_in(document.text)[:110],
                                }
                            )
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No relations extracted for this entity.")


# --- 4. Query ---------------------------------------------------------------

with tab_query:
    st.header("Query the knowledge graph")
    result = st.session_state.result

    if not result:
        st.info("Run the pipeline first.", icon="👈")
    else:
        st.caption(
            "Search by **any alias**. Typing 'PM Modi' finds the canonical entity "
            "'Narendra Modi' — that is what Phase 6 bought us."
        )
        query = st.text_input("Entity name or alias", placeholder="PM Modi")

        if query:
            from src.entity_resolution.normalize import normalize_name

            needle = normalize_name(query, "PERSON")
            matches = [
                e
                for e in result.entities
                if needle == normalize_name(e.canonical_name, e.entity_type)
                or any(needle == normalize_name(a, e.entity_type) for a in e.aliases)
                or query.lower() in e.canonical_name.lower()
            ]

            if not matches:
                st.warning(f"No entity matched “{query}”.")
            for entity in matches[:5]:
                st.markdown(f"### {entity.canonical_name}  ·  `{entity.entity_id}`")
                st.caption(f"{entity.entity_type} · {len(entity.article_ids)} article(s) · "
                           f"aliases: {', '.join(sorted(entity.aliases))}")

                grouped: dict[str, list] = {}
                for document in result.documents:
                    for relation in document.relations:
                        if relation.subject_mention_id in entity.mention_ids:
                            grouped.setdefault(relation.predicate, []).append(
                                (relation, document)
                            )

                if grouped:
                    lines = [entity.canonical_name]
                    items = list(grouped.items())
                    for i, (predicate, pairs) in enumerate(items):
                        for j, (relation, _) in enumerate(pairs):
                            last = (i == len(items) - 1) and (j == len(pairs) - 1)
                            branch = "└──" if last else "├──"
                            lines.append(
                                f"{branch} {predicate} → {relation.object_text}  "
                                f"({relation.confidence:.2f})"
                            )
                    st.code("\n".join(lines), language=None)

                    with st.expander("Why do we believe these? (provenance)"):
                        for predicate, pairs in grouped.items():
                            for relation, document in pairs:
                                st.markdown(
                                    f"**{relation.subject_text} → {predicate} → "
                                    f"{relation.object_text}**  ·  conf "
                                    f"{relation.confidence:.2f}  ·  `{relation.extractor}`"
                                    f"/`{relation.trigger}`"
                                )
                                st.caption(f"{document.source} — {document.title}")
                                st.info(relation.evidence_in(document.text))
                else:
                    st.caption("No relations extracted for this entity.")

                articles = [d for d in result.documents
                            if any(m.mention_id in entity.mention_ids for m in d.mentions)]
                st.markdown("**Articles mentioning this entity**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "published": d.published_at.strftime("%Y-%m-%d"),
                                "source": d.source,
                                "title": d.title[:60],
                                "surfaces used": ", ".join(
                                    sorted({m.text for m in d.mentions
                                            if m.mention_id in entity.mention_ids})
                                ),
                            }
                            for d in articles
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.divider()

        st.subheader("Low-confidence relations (the audit view)")
        st.caption("What a human should check first.")
        threshold = st.slider("Confidence below", 0.0, 1.0, 0.6, 0.05)
        rows = [
            {
                "subject": r.subject_text,
                "predicate": r.predicate,
                "object": r.object_text,
                "conf": round(r.confidence, 2),
                "via coref": r.subject_via_coref or r.object_via_coref,
                "evidence": r.evidence_in(d.text)[:100],
            }
            for d in result.documents
            for r in d.relations
            if r.confidence < threshold
        ]
        if rows:
            st.dataframe(
                pd.DataFrame(rows).sort_values("conf"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("Nothing below that threshold.")


# --- 5. Review queue --------------------------------------------------------

with tab_review:
    st.header("Entity resolution review queue")
    result = st.session_state.result

    if not result:
        st.info("Run the pipeline first.", icon="👈")
    else:
        st.caption(
            "Pairs the resolver would not decide automatically. **“I don't know” "
            "is a valid output**: a false merge fuses two real people permanently "
            "and corrupts every fact about them, while a false split only costs "
            "recall and is recoverable. So uncertain pairs wait for a human."
        )
        queue = result.review_queue
        if not queue:
            st.success("Nothing pending — every candidate pair was decided confidently.")
        else:
            st.metric("Pairs awaiting a human", len(queue))
            profiles = {p.profile_id: p for p in result.profiles}
            for pair in sorted(queue, key=lambda p: -p.score)[:20]:
                left = profiles.get(pair.left_id)
                right = profiles.get(pair.right_id)
                label = (
                    f"{left.canonical_surface} ⟷ {right.canonical_surface}"
                    if left and right else f"{pair.left_id} ⟷ {pair.right_id}"
                )
                with st.expander(f"{pair.score:.3f}   {label}"):
                    st.code(pair.explain(), language=None)
                    if left and right:
                        col_l, col_r = st.columns(2)
                        for col, side in ((col_l, left), (col_r, right)):
                            col.markdown(f"**{side.canonical_surface}**")
                            col.caption(f"{side.article_id} · {side.published_at.date()}")
                            col.write(f"roles: {sorted(side.roles) or '—'}")
                            col.write(f"countries: {sorted(side.countries) or '—'}")
                            col.caption(side.context[:220])


# --- 6. Evaluation ----------------------------------------------------------

with tab_eval:
    st.header("Evaluation against the gold standard")
    st.caption(
        "Scores the pipeline on 3 hand-labelled articles. **Single annotator, "
        "no agreement score** — a development signal, not a benchmark."
    )

    gold_path = cfg.path(cfg.paths.sample_dir) / "gold" / "annotations.json"
    result = st.session_state.result

    if not gold_path.exists():
        st.error("Gold annotations not found.")
    elif not result:
        st.info("Run the pipeline on the **sample corpus** to evaluate.", icon="👈")
    else:
        if st.button("Run evaluation", type="primary"):
            from src.evaluation.evaluate import evaluate, load_gold

            report = evaluate(result.documents, load_gold(gold_path), result.entities)

            if report.articles_evaluated == 0:
                st.warning(
                    "None of the gold articles are in the current run. "
                    "Load the sample corpus and re-run the pipeline."
                )
            else:
                st.caption(f"{report.articles_evaluated} labelled article(s) scored.")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("NER F1 (strict)", f"{report.ner_strict.f1:.3f}")
                col_b.metric("NER F1 (relaxed)", f"{report.ner_relaxed.f1:.3f}")
                col_c.metric("Relations F1", f"{report.relations.f1:.3f}")

                st.subheader("NER by entity type")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"label": label, "P": round(prf.precision, 3),
                             "R": round(prf.recall, 3), "F1": round(prf.f1, 3),
                             "tp": prf.true_positives, "fp": prf.false_positives,
                             "fn": prf.false_negatives}
                            for label, prf in sorted(
                                report.ner_by_label.items(), key=lambda kv: -kv[1].f1
                            )
                        ]
                    ),
                    use_container_width=True, hide_index=True,
                )

                st.subheader("Coreference")
                p, r, f = report.coref_b3
                col_a, col_b = st.columns(2)
                col_a.metric("MUC F1", f"{report.coref_muc.f1:.3f}")
                col_b.metric("B-cubed F1", f"{f:.3f}")
                st.caption(
                    "MUC rewards over-merging and ignores singletons; B-cubed "
                    "punishes both over-merging and over-splitting. Read together."
                )

                if report.entity_resolution:
                    er = report.entity_resolution
                    st.subheader("Entity resolution")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Pairwise F1", f"{er.pairwise.f1:.3f}")
                    col_b.metric("False merges", er.false_merges,
                                 help="Corrupting and hard to undo")
                    col_c.metric("False splits", er.false_splits,
                                 help="Costs recall, recoverable")
                    st.caption(
                        "These are reported separately on purpose: an aggregate "
                        "F1 cannot distinguish a corrupted knowledge base from a "
                        "merely incomplete one."
                    )
                    if er.false_merge_examples:
                        st.error("False merges: " + "; ".join(
                            f"{a} == {b}" for a, b in er.false_merge_examples))
                    if er.false_split_examples:
                        st.warning("False splits: " + "; ".join(
                            f"{a} != {b}" for a, b in er.false_split_examples))
