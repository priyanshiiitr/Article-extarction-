# News Knowledge Graph

Extract structured information — people, organizations, countries, events, roles,
relations — from news articles, resolve mentions into canonical entities across
articles, and store the result as a queryable knowledge graph.

Built as a teaching project: every design decision is documented in the module
docstrings, including the tradeoffs and the things that are deliberately only a
baseline.

---

## Pipeline

```
articles → ingestion → preprocessing → NER → coreference → relations
                                                               ↓
              query layer ← knowledge graph ← canonical entities
```

| Phase | Status | Module |
|-------|--------|--------|
| 1. Ingestion | done | `src/ingestion/` |
| 2. Preprocessing (clean + sentence segmentation) | done | `src/preprocessing/` |
| 3. Named entity recognition | done | `src/ner/` |
| 4. Coreference resolution | next | `src/coreference/` |
| 5. Relation extraction | planned | `src/relation_extraction/` |
| 6. Entity resolution | planned | `src/entity_resolution/` |
| 7–9. Storage + query | planned | `src/storage/` |
| 10. Evaluation | planned | `tests/`, `scripts/` |

### The two ideas that hold it together

**1. Character offsets are the universal currency.** Once `Document.text`
exists it is immutable, and every annotation — sentences, entity mentions,
coreference clusters, relation arguments — is a `(start, end)` pair indexing
into that exact string. spaCy, BERT and a coreference model tokenize
differently and their token indices are not comparable, but character offsets
are a common denominator all of them convert to. `Document.text_sha256` is the
tripwire: if the text ever changes, every stored offset is stale.

**2. Within-document and cross-document are different problems.** Coreference
(Phase 4) links "He" to "Narendra Modi" *inside one article*. Entity resolution
(Phase 6) decides that "PM Modi" in a different article three weeks later is the
same human being. Different inputs, different algorithms, different failure
modes — so they are separate modules.

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv --system-site-packages
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # macOS / Linux

pip install -e ".[nlp,dev]"
python -m spacy download en_core_web_sm
```

`--system-site-packages` lets the venv reuse a heavy `torch` install that is
already on the machine instead of downloading gigabytes again. Drop it for a
fully isolated environment.

The transformer NER model (~430 MB) downloads automatically on first run and is
cached afterwards. To run fully offline, remove `transformer` from
`ner.extractors` in `config/config.yaml`.

### Secrets

```bash
cp .env.example .env
```

Nothing in Phases 1–5 needs an API key — the local pipeline is fully offline
once models are cached. `.env` is gitignored and must never be committed.

---

## Running it

```bash
python scripts/make_sample_data.py     # generate the reproducible corpus
python scripts/run_ingest.py           # → data/raw/articles.jsonl
python scripts/run_preprocess.py       # → data/processed/documents.jsonl
python scripts/run_ner.py              # → data/processed/documents_ner.jsonl
pytest -q
```

---

## Configuration

All settings live in `config/config.yaml`. No module hardcodes a path or a
model name.

Override any value with an environment variable using
`NEWSKG__<SECTION>__<KEY>` (a double underscore means one level deeper):

```bash
NEWSKG__NER__MIN_SCORE=0.8 python scripts/run_ner.py
NEWSKG__PREPROCESSING__SENTENCE_SEGMENTER=regex python scripts/run_preprocess.py
```

Precedence: `config.yaml` < `.env` < real environment variables.

---

## Layout

```
config/config.yaml        all settings
data/raw/                 immutable landing zone — never edited in place
data/processed/           derived artifacts — safe to delete and regenerate
data/sample/              the committed sample corpus (with deliberate traps)

src/config.py             typed config loading with env overrides
src/logging_utils.py      rich (dev) and JSON (production) logging
src/schemas.py            Article, Sentence, Mention, Document — the contracts
src/nlp_resources.py      cached model loaders

src/ingestion/readers.py  source readers (JSONL today, pluggable)
src/ingestion/loader.py   validate → identify → deduplicate → filter

src/preprocessing/clean.py      HTML, unicode and whitespace normalization
src/preprocessing/segment.py    sentence segmentation (spaCy or regex)
src/preprocessing/processor.py  Article → Document, freezes the text

src/ner/labels.py         model label sets → one shared vocabulary
src/ner/base.py           extractor Protocol + the single Mention builder
src/ner/spacy_ner.py      spaCy statistical NER (18 types, fast)
src/ner/transformer_ner.py  BERT token classification (better PERSON/ORG)
src/ner/gazetteer.py      COUNTRY, ROLE, EVENT, TOPIC — types no model has
src/ner/merge.py          trust matrix that resolves extractor disagreements
src/ner/pipeline.py       Phase 3 orchestration

scripts/                  runnable entry points
tests/                    58 tests
```

---

## Sample corpus

`data/sample/articles.jsonl` is generated by `scripts/make_sample_data.py` and
contains deliberate traps, each documented with a `_why` note in that script:

- **alias variety** — "Narendra Modi" / "PM Modi" / "Mr. Modi" / "the Indian
  Prime Minister" must collapse to one entity in Phase 6
- **a name collision** — "Lalit Modi" is a *different person*; merging him with
  Narendra Modi is a false merge, the most dangerous entity-resolution error
- **role ambiguity** — one article contains two prime ministers, so "the Prime
  Minister" cannot be resolved by role lookup alone
- **dirty HTML**, **unicode damage**, **abbreviation minefields**, an **exact
  duplicate**, a **too-short stub**, and a **malformed record**

---

## Known limitations

Stated explicitly rather than discovered later:

- **Boilerplate removal.** The regex HTML stripper cannot distinguish article
  prose from navigation links (`trafilatura` would). Link text survives.
- **spaCy `en_core_web_sm` on Indian names.** It labelled "Sitharaman" as ORG
  and "Kazan Declaration" as PERSON. The transformer fixes both, which is why
  both run and are merged.
- **Trust weights in `merge.py` are hand-set, not learned.** They encode
  observed behaviour on this corpus. Phase 10 builds the labelled data needed
  to measure them instead of asserting them.
- **The TOPIC gazetteer is the weakest component.** Topics are an open set, so
  a fixed list cannot have good recall. It buys precision on this corpus only.
- **Confidence scores are not calibrated.** Neural networks are systematically
  overconfident; treat scores as a ranking signal, not as probabilities.
