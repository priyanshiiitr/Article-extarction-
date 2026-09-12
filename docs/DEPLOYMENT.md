# Deploying the dashboard

## The constraint that decides everything

The full model stack does not fit on a 1 GB host:

```
LingMess coreference   590M params  →  ~2.4 GB RAM   (4.4 GB cached on disk)
BERT NER               110M params  →  ~1.5 GB RAM
e5-small embeddings    118M params  →  ~0.5 GB RAM
spaCy en_core_web_sm                →  ~0.1 GB RAM
                                       ─────────────
                                        ~4.5 GB
```

So there are two profiles, selectable in the sidebar. They are **config only** —
no code differs between them, because every stage was built with a
config-selectable backend and a documented fallback.

| | `lite` | `full` |
|---|---|---|
| NER | spaCy + gazetteer | + BERT transformer |
| Coreference | rule-based (union-find + recency) | LingMess neural |
| ER context embeddings | off | e5-small |
| Install size | ~200 MB | ~2.7 GB |
| RAM | **~0.6 GB** | ~4.5 GB |
| Speed | **<1 s/article** | ~14 s/article (CPU) |
| Quality | lower alias recall, no nominal coreference | best |

**The measured quality cost of lite:** it keeps `NDB` and `New Development Bank`
as separate entities (embeddings off drops the score under threshold), and
resolves **zero** nominal coreference — "The Indian Prime Minister" never links
to Modi, so the relations that depend on it are lost. Be honest about this in a
demo rather than letting someone assume lite is the real system.

---

## Option A — HuggingFace Spaces (recommended, free, runs `full`)

Free CPU basic tier: **16 GB RAM, 2 vCPU, 50 GB disk.** Enough for the full
stack, and the models come from HF's own CDN.

1. Create a Space at https://huggingface.co/new-space → SDK: **Streamlit**.
2. Push this repository to it.
3. The Space's `README.md` must begin with this frontmatter (HF reads it as
   config — that's why it is kept here rather than in the project README):

```yaml
---
title: News Knowledge Graph
emoji: 🕸️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.63.0
app_file: app/streamlit_app.py
pinned: false
---
```

4. Rename `requirements-full.txt` → `requirements.txt` in the Space (HF installs
   the file with that exact name).
5. Add a persistent-storage upgrade **or** accept that models re-download on
   each cold start (~3 minutes).

**Optional secret:** `HF_TOKEN` under Settings → Variables and secrets. It is
*not required* — all three models are public. It only helps if you hit
anonymous download rate limits.

---

## Option B — Streamlit Community Cloud (free, `lite` only)

1. Push to GitHub.
2. https://share.streamlit.io → New app → `app/streamlit_app.py`.
3. It installs `requirements.txt` (the lite one) automatically.
4. **Force lite mode** so nobody can select `full` and OOM the container:

   Settings → Secrets:
   ```toml
   NEWSKG__NER__EXTRACTORS = '["spacy","gazetteer"]'
   NEWSKG__COREFERENCE__BACKEND = "rules"
   NEWSKG__ENTITY_RESOLUTION__USE_EMBEDDINGS = "false"
   ```

Selecting `full` on this host will crash the container with an out-of-memory
error, not a friendly message. That is the platform's behaviour, not ours.

---

## Option C — Local (best for development)

```bash
pip install -r requirements-full.txt
python -m spacy download en_core_web_sm
streamlit run app/streamlit_app.py
```

Opens on http://localhost:8501. Models cache to `HF_HOME` after the first run.

---

## Option D — Docker (any host)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-full.txt \
 && python -m spacy download en_core_web_sm
COPY . .
# Bake the models into the image so cold starts do not re-download 2.5 GB.
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    [ (AutoTokenizer.from_pretrained(m), AutoModel.from_pretrained(m)) \
      for m in ['dslim/bert-base-NER','intfloat/multilingual-e5-small'] ]"
EXPOSE 8501
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit","run","app/streamlit_app.py", \
            "--server.port=8501","--server.address=0.0.0.0"]
```

Baking models into the image trades image size (~6 GB) for cold-start time.
Worth it when instances scale up and down; not worth it for a single long-lived
container.

---

## Host comparison

| Host | Free RAM | `full`? | Notes |
|---|---:|:---:|---|
| **🤗 HF Spaces (CPU basic)** | **16 GB** | ✅ | Best free option |
| Streamlit Community Cloud | 1 GB | ❌ | `lite` only |
| Render (free) | 512 MB | ❌ | Too small for either |
| Railway | 8 GB | ✅ | Paid after trial |
| Fly.io | configurable | ✅ | Paid |
| Any VM with ≥6 GB | — | ✅ | Docker option above |

---

## A caveat about scraping

`src/ingestion/web.py` sends a browser-like User-Agent, because many news sites
return 403 to obvious bots. In testing:

- ✅ Wikipedia, most blogs, many regional outlets
- ❌ Reuters, Bloomberg and other large publishers — bot protection
- ❌ JavaScript-rendered pages — nothing in the served HTML to extract
- ❌ Paywalls — returns the teaser, not the article

The **Paste raw text** tab exists for exactly these cases.

Scraping has terms-of-service implications. For anything beyond personal
evaluation, respect `robots.txt`, rate-limit requests, and prefer a licensed
news API (NewsAPI, GDELT, Common Crawl) over scraping publishers directly.
