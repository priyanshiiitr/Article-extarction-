# Phase 11 — Scaling to 100,000+ articles

What changes between the local pipeline and a production system, and — just as
important — what does **not**.

The guiding rule: **make the local system excellent first**. Most of what
follows is infrastructure that adds operational cost. Each item below states
the *trigger* that justifies it, so you can tell when it's needed rather than
adding it because it sounds serious.

---

## 1. Where the time actually goes

Measured on this machine, per document, CPU only:

| Stage | Time/doc | 100k articles | Share |
|---|---:|---:|---:|
| Ingestion + cleaning | ~1 ms | 2 min | 0.0% |
| Sentence segmentation (spaCy) | ~10 ms | 17 min | 0.3% |
| NER — spaCy | ~10 ms | 17 min | 0.3% |
| NER — BERT transformer | ~170 ms | 4.7 hours | 4.5% |
| **Coreference (LingMess, 590M)** | **~13 s** | **15 days** | **94%** |
| Relation extraction | ~15 ms | 25 min | 0.4% |
| Entity resolution (whole corpus) | — | ~1 hour | 0.7% |

**Coreference is 94% of the runtime.** Every optimisation decision follows from
that single number, and this is the lesson: *profile before you parallelise*.
Sharding the whole pipeline across 10 machines to fix a problem that lives in
one stage is how teams waste a quarter.

### Fixing the bottleneck, cheapest first

1. **Skip work that cannot matter.** Run coref only on documents with ≥2 person
   mentions. On a typical news corpus that's ~40% of documents — a 2.5× win for
   zero quality loss and about five lines of code.
2. **Swap the model.** `fcoref` (distilled) is several times faster for a few
   F1 points. Make it a config value and measure the trade on your gold set.
3. **GPU.** A single T4 takes LingMess from ~13 s to ~0.3 s/doc. This is the
   largest single lever and costs less than the engineering time spent avoiding it.
4. **Batch properly.** Group documents of similar length so padding doesn't
   dominate — a batch of one 4,000-token and seven 200-token documents wastes
   ~80% of the compute on padding.
5. **Cache by content hash.** Key results on `text_sha256`. Re-runs after a
   code change in *another* stage become free.

---

## 2. Batch vs streaming

**Use batch.** News is not latency-sensitive: nobody needs an entity graph
updated within 200 ms of publication.

| | Batch | Streaming |
|---|---|---|
| Throughput | High (large GPU batches) | Low (batch size ~1) |
| Cost/doc | Low | 5–20× higher |
| Latency | Minutes–hours | Seconds |
| Complexity | A cron job | A cluster |

**The trigger for streaming:** a product requirement for sub-minute freshness —
a trading signal, a breaking-news alert. "It feels more modern" is not a trigger.

A **hybrid** is usually right: micro-batches every 15 minutes. You get most of
batch's efficiency with acceptable freshness, and it's still a cron job.

### The real reason entity resolution wants batch

ER is **global**: to decide that a new "PM Modi" is the existing Narendra Modi,
you need the existing entities. Streaming forces *incremental* resolution:

```
new profile → block against existing entities → score → attach or create
```

which is doable and is what you'd build — but note it makes the result
**order-dependent**. A different arrival order can produce different clusters.
Batch re-clustering periodically (nightly) corrects the drift. Say this in an
interview and you demonstrate you've thought past the happy path.

---

## 3. Orchestration: queues and workers

```
                  ┌──────────────┐
  sources ───────►│ ingest queue │
                  └──────┬───────┘
                         ▼
              ┌────────────────────┐
              │  CPU worker pool   │  clean, segment, spaCy NER, relations
              │  (n = cores)       │
              └──────────┬─────────┘
                         ▼
              ┌────────────────────┐
              │  GPU worker pool   │  transformer NER, coreference, embeddings
              │  (n = GPUs)        │
              └──────────┬─────────┘
                         ▼
              ┌────────────────────┐
              │  ER (single batch) │  global; not parallel per-document
              └──────────┬─────────┘
                         ▼
                    ┌─────────┐
                    │ graph DB│
                    └─────────┘
```

**Separate CPU and GPU pools.** They have different scaling limits, and mixing
them means your expensive GPU sits idle doing regex work.

**Size the GPU pool by memory, not cores.** Each worker holds its own model
copy (models aren't safely shared across processes — see the note in
`src/nlp_resources.py`). With a 590M coref model plus a BERT NER model, that's
~3 GB/worker.

**Which queue:** Celery + Redis is enough to ~1M docs/day and takes an afternoon.
Kafka earns its place when you need replay, multiple independent consumers, or
ordering guarantees. Don't start there.

---

## 4. Idempotency and retries

This is already built in, from Phase 1 onward — and it's what makes everything
above safe:

| Mechanism | Where | Effect |
|---|---|---|
| Content-hash IDs | `make_article_id` | Re-ingesting yields the same ID |
| `INSERT OR REPLACE` | `store.py` | Re-loading replaces, never duplicates |
| Deterministic mention/relation IDs | `make_mention_id`, `make_relation_id` | Re-extraction doesn't duplicate facts |
| Deterministic ER | sorted iteration, stable tiebreaks | Same input → same entity IDs |

**Because every stage is idempotent, the retry policy is simply "run it again."**
That is a very large simplification, and it's worth naming in an interview: most
distributed-systems pain comes from operations that aren't safe to repeat.

**Retry policy:** exponential backoff with jitter; 3 attempts; then a dead-letter
queue. Distinguish **transient** failures (network, OOM, rate limit — retry)
from **permanent** ones (malformed record, unsupported language — dead-letter
immediately). Retrying a permanent failure three times just delays the alert.

---

## 5. Candidate generation at scale

Our blocking gives **94.8% reduction** on 32 records. That is not enough at
1M records — some blocks become huge (every "Kumar", every "Singh").

| Corpus size | Approach |
|---|---|
| < 100k | Exact-key blocking (what we built) |
| 100k–10M | Add ANN over embeddings: FAISS/HNSW, top-k nearest as candidates |
| > 10M | Sharded ANN + MinHash LSH over character n-grams |

**Vector database trigger:** you need ANN over embeddings *and* filtered search
("nearest neighbours **where** entity_type = PERSON **and** country = India").
Below ~1M vectors, FAISS in-process with a metadata filter is simpler and faster
than operating a vector DB. pgvector is a good middle ground because you keep
one database instead of two and avoid a sync problem.

**Guard against block skew explicitly.** `MAX_BLOCK_WARN` in `blocking.py`
already logs oversized blocks; in production, cap block size and fall back to a
tighter key, or you'll have one worker comparing 50,000 Kumars while the rest idle.

---

## 6. Model versioning

Every stored row already carries `model_version`, `extractor` and
`pipeline_version`. That matters because **model upgrades are silent quality
changes**:

- Pin exact versions (`spacy==3.8.7`, an exact HF revision hash — not `main`).
- Re-run the gold evaluation on every model change. Block the upgrade on a
  regression in any per-label F1, not just the headline number.
- **Shadow-run** the new model alongside the old, compare on live data, then
  cut over.
- Keep the old version's output long enough to diff. "Why did this entity change
  last Tuesday?" is unanswerable without it.

---

## 7. Monitoring

Log-level alerts are not enough; a pipeline can succeed loudly while degrading.

**Volume:** articles in/out per stage. A silent drop from 10,000 to 4,000 looks
exactly like success — which is why every stage in this project returns a stats
object rather than only logging.

**Quality proxies** (no labels needed, so you can run them on live data):
- mentions per document, by type
- share of relations resolved via coreference
- ER match / review / veto ratios
- review-queue growth rate — **if it grows faster than it's cleared, your
  thresholds are wrong**
- mean entity `merge_confidence`

**Drift:** distribution shift in entity types or sources means the corpus
changed and your tuned thresholds may no longer hold.

**Cost:** GPU-seconds per document, and LLM spend per document if used.

---

## 8. Where an LLM earns its cost — and where it doesn't

The comparison that matters:

| | Traditional NLP | Transformers | Embeddings | LLM |
|---|---|---|---|---|
| Cost/doc | ~0 | low | low | **100–1000×** |
| Latency | µs | ms | ms | **seconds** |
| Determinism | total | high | total | **low** |
| Training data | none | fine-tuning | none | none |
| Open schema | no | no | n/a | **yes** |
| Hallucination | impossible | impossible | n/a | **real risk** |

### Per task

| Task | Use | Why |
|---|---|---|
| **NER** | Fine-tuned encoder | Encoders already do this well and 1000× cheaper. An LLM can emit spans that aren't in the text. |
| **Coreference** | Specialised model | Purpose-built models are strong; LLMs are inconsistent on long documents and offsets. |
| **Relation extraction** | Parse/supervised baseline, **LLM for hard cases** | The best genuine use here — see below. |
| **Entity resolution** | Features + scoring | Needs to be explainable and deterministic. An LLM merge decision you can't audit is a liability. **LLM as tie-breaker on the review queue** is defensible. |
| **Summarisation** | **LLM** | Genuinely the right tool; no cheap alternative is close. |
| **Open-schema extraction** | **LLM** | If you can't enumerate your relations, rules lose by definition. |

### The three defensible LLM uses in this pipeline

1. **Review-queue triage (Phase 6).** 8 pairs needed a human. An LLM given both
   contexts and asked "same person?" costs 8 calls, not 100,000. **You pay per
   hard case, not per document.**
2. **Bootstrapping training data.** Use an LLM to label 2,000 relation examples,
   have a human verify a sample, then **distil** into a small supervised model.
   You pay once and get a cheap, fast, deterministic model.
3. **Open-schema discovery.** Run an LLM over a sample to find relation types
   your closed schema is missing, then add them as rules or training labels.

### If you do use an LLM, the non-negotiables

- **Structured output.** Constrained decoding / tool-use schemas, not "please
  return JSON".
- **Verify offsets against the source.** Make it return character spans and
  check `text[start:end] == claimed_text`. **This catches hallucination
  mechanically** — the single most important control.
- **Type-check arguments** against the same `ARGUMENT_TYPES` table the parser uses.
- **Cache on content hash.** Same input, same output, no second charge.
- **Budget caps and a fallback path.** The pipeline must still finish when the
  API is down or the budget is exhausted.
- **Log the prompt version** exactly as you log `model_version`. A prompt change
  is a model change.

---

## 9. What I would build next, in order

1. **Expand the gold set** to ~50 articles with **two annotators** and a kappa
   score. Everything else is guesswork until this exists — our current numbers
   come from 3 articles and one annotator.
2. **Learn the ER weights** by logistic regression on labelled pairs; choose the
   threshold from the precision/recall curve instead of intuition.
3. **Fix NER recall on organisations.** Evaluation showed ER errors were
   *caused* by missed ORG spans (`Board of Control for Cricket in India` was
   never extracted). That's where the marginal effort pays best — and you only
   know that because the evaluation separates the error types.
4. **GPU + document filtering for coreference** — a 94% bottleneck.
5. **Incremental ER** with a merge/unmerge audit log.
6. Only then: queues, autoscaling, a vector database.

The ordering is the point. Items 1–3 are measurement and quality; items 4–6 are
infrastructure. Doing 6 before 1 is the most common and most expensive mistake
in applied ML.
