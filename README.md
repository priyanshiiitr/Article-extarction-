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
| 4. Coreference resolution | done | `src/coreference/` |
| 5. Relation extraction | done | `src/relation_extraction/` |
| 6. Entity resolution | done | `src/entity_resolution/` |
| 7–9. Storage + query | done | `src/storage/` |
| 10. Evaluation | done | `src/evaluation/` |
| 11. Production architecture | done | [`docs/PRODUCTION.md`](docs/PRODUCTION.md) |

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
python scripts/run_coref.py            # → data/processed/documents_coref.jsonl
python scripts/run_relations.py        # → data/processed/documents_relations.jsonl
python scripts/run_entity_resolution.py # → data/processed/entities.jsonl + review_queue.jsonl
python scripts/build_graph.py          # → data/knowledge_graph.db
python scripts/query.py                # answers the five demo questions
python scripts/query.py "Lalit Modi"    # any entity, by name or alias
python scripts/evaluate.py             # scores against the gold set
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

src/coreference/base.py               resolver Protocol + cluster builder + alignment
src/coreference/fastcoref_resolver.py neural coref (LingMess, 590M params)
src/coreference/rule_resolver.py      heuristic baseline (union-find + recency)
src/coreference/pipeline.py           Phase 4 orchestration

src/relation_extraction/base.py        Protocol, Relation builder, argument resolution
src/relation_extraction/patterns.py    juxtaposition relations (role/demonym/of)
src/relation_extraction/dependency.py  verb relations from the dependency parse
src/relation_extraction/pipeline.py    Phase 5 orchestration + deduplication

src/entity_resolution/normalize.py   surface form -> comparable string
src/entity_resolution/profiles.py    mentions -> local entities (+ bare-surname attachment)
src/entity_resolution/blocking.py    candidate generation, O(N^2) -> O(N x block)
src/entity_resolution/embeddings.py  context vectors (e5-small, masked mean pooling)
src/entity_resolution/features.py    pairwise signals + hard vetoes
src/entity_resolution/scoring.py     weight table + decision bands
src/entity_resolution/resolver.py    clustering + cluster soundness validation

src/storage/schema.py     SQLite DDL: two-level design + indexes
src/storage/store.py      loaders, noisy-OR edge aggregation
src/storage/queries.py    the query layer (parameterised, with provenance)
src/evaluation/metrics.py P/R/F1, MUC, B-cubed, false merge vs split
src/evaluation/evaluate.py scoring against the gold set

data/sample/gold/         manually labelled gold standard (3 articles)
docs/PRODUCTION.md        Phase 11: scaling, LLM usage, what to build next

scripts/                  runnable entry points
tests/                    175 tests
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
- **Coreference is slow.** LingMess is 590M parameters and takes roughly 13
  seconds per document on CPU. On a large corpus this dominates runtime; the
  `fcoref` distilled model or a GPU is the answer. Set
  `coreference.backend: rules` for a fast, clearly worse baseline.
- **The rule-based coref backend ignores syntax.** In "Modi met Putin. He
  said..." it links "He" to Putin (nearest) rather than Modi (the subject).
  This failure is pinned by a test so it stays a known limitation.
- **ER weights and thresholds are hand-set.** They encode reasoning, not
  measurement. The correct method is logistic regression on labelled pairs
  (the Fellegi-Sunter framework) with the threshold chosen from a
  precision/recall curve. Phase 10 builds that labelled data.
- **Context embeddings are only weakly discriminative.** Measured raw
  cosines between news contexts sit in 0.88-0.93 whether the people are the
  same or not, so the feature is calibrated by RANK within the candidate
  set. That makes it batch-relative; production would calibrate against a
  fixed reference distribution.
- **Transitive closure can chain merges.** Mitigated by hard vetoes,
  intra-document bare-surname attachment, and post-hoc cluster soundness
  validation — not eliminated. Correlational clustering is the principled fix.
- **The relation schema is closed.** Nine predicates, and a relation outside
  that set is simply not extracted. An open schema is the main argument for
  LLM-based extraction.
- **Verb triggers are a fixed list.** "sat down with" means "met" only if
  someone adds it. A supervised model learns paraphrase from data; this is
  the clearest upgrade path once labelled data exists.
- **Modality is penalised, not modelled.** "Modi will meet Putin" is stored
  with reduced confidence rather than flagged as an unrealised event.
  Proper handling needs factuality classification.
- **Duplicate facts across mention forms survive.** `(Jaishankar) met
  (Lavrov)` and `(Jaishankar) met (Sergey Lavrov)` are the same fact, but
  dedup compares surface strings. Phase 6 is what fixes this.
- **Windows + fastcoref needs a `__main__` guard.** fastcoref tokenises via
  HuggingFace `datasets` multiprocessing; without the guard, child processes
  re-import the entry point and the run exits silently with code 0.
