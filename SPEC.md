# Pensieve Core — Product Plan & Technical Spec

*Domain-weighted memory infrastructure for long-running AI agents.*

Last updated: 2026-09-26

---

## 1. Elevator Pitch

**Pensieve Core is a domain-weighted memory engine for long-running AI agents.**

Standard RAG treats every semantic match as equal: a $5 coffee receipt embeds just as "relevantly" as a $50,000 wire transfer when a user asks about *spending*. Similarity tells you what a memory is *about* — it says nothing about whether it *matters*.

Pensieve Core fixes this by letting developers attach **Value Dimensions** and **category tags** to every context fragment stored in MongoDB Atlas. A dimension is any numeric axis of importance the domain cares about — `financial_impact`, `emotional_significance`, `strategic_risk`, `clinical_urgency`. Retrieval then blends three signals in a single Atlas aggregation:

1. **Semantic similarity** — Atlas Vector Search (what it's about)
2. **Categorical / temporal gates** — deterministic pre-filters (is it even in scope?)
3. **Domain weight + time decay** — computed relevance scoring (does it matter, and does it still matter *now*?)

The result: agents recall context based on **what actually matters to the user**, not just what sounds similar — eliminating memory noise and temporal hallucinations. It's a normal database query and an LLM semantic search fused into one pass: the best of both worlds.

**One-liner:** *Turn a generic vector store into a customizable, domain-aware context engine.*

---

## 2. Core Concept

Every memory is a **context fragment**: raw text + its embedding + structured metadata. The metadata is what makes Pensieve different from a plain vector DB:

- **`project`** — the **isolation key**. Every read/write/config is scoped to it, so memories never intersect across projects (e.g. `finance` vs `autobiography`). Each project carries its own Dimension Schema.
- **`categories`** — discrete tags for hard filtering (`finance`, `family`, `outage`). Answers *"is this in scope?"* deterministically.
- **`domain_weights`** — a map of numeric importance scores on developer-defined axes. Answers *"how much does this matter?"* — the axes are configurable per app, so `emotional_weight` is just one option among many.
- **`timestamp`** — enables temporal gating and decay.

Developers declare a **Dimension Schema** per deployment. The engine is domain-agnostic; the schema makes it domain-*aware*.

```jsonc
// Example dimension schema for a "Personal Wealth Companion" deployment
{
  "categories": ["spending", "investment", "tax", "income", "debt"],
  "dimensions": {
    "financial_impact":  { "range": [0, 10], "decay_half_life_days": 3650, "default": 1 },
    "liquidity_risk":    { "range": [0, 10], "decay_half_life_days": 365,  "default": 1 },
    "tax_relevance":     { "range": [0, 10], "decay_half_life_days": 400,  "default": 0 }
  }
}
```

High-emotional / high-impact memories are assigned long half-lives (they stay pristine); mundane ones decay fast (they fade). Decay is per-dimension, so a memory can fade for one lens while staying sharp for another.

---

## 3. High-Level Architecture (MongoDB Atlas Integration)

Atlas is used not merely as a vector store but as a **hybrid scoring engine** — filtering, vector search, and weighted ranking all execute inside one aggregation pipeline, close to the data.

```
  [ Raw Input ]
        │
        ▼
  [ Extraction (LLM) ] ──► structured metadata: categories[] + domain_weights{}
        │
        ▼
  [ Embedding (Voyage / OpenAI) ] ──► dense vector
        │
        ▼
  [ MongoDB Atlas Document ] ──► { text, embedding[], timestamp, categories[], domain_weights{} }
        │
        ▼
  [ Atlas Aggregation Pipeline ]
        │   Step 1: $vectorSearch  (ANN + categorical/temporal PRE-filter)
        │   Step 2: $addFields     (compute decay-adjusted weighted FinalScore)
        │   Step 3: $sort / $limit (domain-ranked, not just similarity-ranked)
        ▼
  [ Ranked Context ] ──► fed to the agent LLM
```

### 3.1 Integration approach — native driver, not the Data API

Connect with the **standard MongoDB driver** over the SRV connection string. **Do not** build on the Atlas Data API or GraphQL API — both were deprecated (announced Sept 2024) and are being retired. Vector Search is not a separate API; it is an aggregation stage available on any Atlas cluster with Search enabled (M10+, or a Flex/Serverless tier).

- **Recommended stack:** Python (FastAPI + `pymongo` 4.5+, or `motor` for async).
  - First-class SDKs for the two hot-path external calls: embedding and metadata-extraction LLM.
  - Full native `$vectorSearch` support; room to add reranking/eval later.
- **Alternative:** Node/TypeScript — only if the app backend is already JS. The Atlas Node driver supports `$vectorSearch` identically; you only give up ML tooling convenience.

### 3.2 Atlas document design

```jsonc
{
  "_id": "mem_982341",
  "text": "Closed $120k seed round investment with Venture Corp",
  "embedding": [0.012, -0.043, 0.219, /* … 1024 or 1536 dims … */],
  "timestamp": "2025-11-14T10:30:00Z",
  "categories": ["finance", "fundraising", "milestone"],
  "domain_weights": {                       // BASE importance — immutable, LLM-assigned
    "financial_impact": 9.5,
    "emotional_significance": 8.0,
    "strategic_risk": 7.0
  },
  "current_weights": {                      // decayed by the decay task (§3.4 Step 3); recall ranks by this
    "financial_impact": 8.71,
    "emotional_significance": 6.20,
    "strategic_risk": 5.90
  },
  "project": "finance",                     // isolation key (§3.9)
  "access_count": 12,                       // reinforcement — times recalled (§3.8)
  "last_accessed": "2026-09-20T14:00:00Z",
  "schema_version": 1
}
```

The embedding array **and** the scoring metadata live in the same document — one round trip, one atomic write, no join. `access_count` / `last_accessed` power recall reinforcement (§3.8); `schema_version` tracks which Dimension-Schema version a record has been scored against (§3.7).

### 3.3 Vector Search index

Fields used for pre-filtering **must** be declared as `filter` type in the index, or they can't gate the ANN search. `numDimensions` must match the embedding model (Voyage `voyage-3.5` = 1024; OpenAI `text-embedding-3-small` = 1536).

```jsonc
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine" },
    { "type": "filter", "path": "project" },
    { "type": "filter", "path": "timestamp" },
    { "type": "filter", "path": "categories" }
  ]
}
```

### 3.4 Retrieval pipeline

**Step 1 — `$vectorSearch` with pre-filtering (the Memory Gate).**

The gate is the **`filter` field *inside* `$vectorSearch`**, not a later `$match`. This matters for recall: a trailing `$match` is *post*-filtering — Atlas runs ANN over the whole space, returns *k*, then discards. In-scope-but-weaker memories get evicted before you ever see them, and you can silently get zero results. The `filter` field restricts the ANN search itself to the qualifying subset.

```jsonc
{
  "$vectorSearch": {
    "index": "vector_index",
    "path": "embedding",
    "queryVector": [ /* …query embedding… */ ],
    "numCandidates": 200,
    "limit": 50,
    "filter": {
      "categories": { "$in": ["finance"] },
      "timestamp":  { "$gte": "2025-01-01T00:00:00Z" }
    }
  }
}
```

**Step 2 — compute the weighted `FinalScore`.**

```
FinalScore = (Sim · w_vector)
           + (Weight · w_domain_bias)
           + (Reinforcement · w_reinforce)
```

where `Sim` = `$vectorSearchScore`, `Weight` = the selected dimension's **`current_weights`** value (the base `domain_weights` already degraded by the decay task — Step 3), and `Reinforcement = ln(1 + access_count)` (§3.8). **Decay is applied in storage, not here** — recall just reads the pre-decayed value. Tunable per query: raise `w_domain_bias` to favor importance, `w_vector` for topical closeness, `w_reinforce` to favor memories the agent keeps using.

```jsonc
[
  { "$addFields": {
      "sim": { "$meta": "vectorSearchScore" },
      "weight": { "$ifNull": ["$current_weights.financial_impact",
                              { "$ifNull": ["$domain_weights.financial_impact", 0] }] },
      "reinforcement": { "$ln": [ { "$add": [1, { "$ifNull": ["$access_count", 0] }] } ] } } },
  { "$addFields": {
      "score_breakdown": { "similarity": "$sim", "weight": "$weight",
                           "reinforcement": "$reinforcement" },   // §3.8 explainability
      "final_score": { "$add": [
          { "$multiply": ["$sim", 1.0] },
          { "$multiply": ["$weight", 0.15] },
          { "$multiply": ["$reinforcement", 0.1] } ] } } },
  { "$sort":    { "final_score": -1 } },
  { "$limit":   10 },
  { "$project": { "embedding": 0 } }
]
```

This pipeline is built by `pensieve/scoring.py::build_recall_pipeline`; `use_gates=false` drops the category/time gates. The demo's naive baseline is the **separate `POST /raw_recall`** endpoint — plain Voyage similarity with none of the weighting (§3.6). So the two recall-family endpoints are `/recall` (weighted) and `/raw_recall` (naive).

**Step 3 — decay as a repeated task (writes `current_weights`).**

Decay is a **scheduled task**, not a query-time value (`pensieve/decay.py`, exposed as **`GET /decay`** for Vercel Cron). Each run rewrites every memory's **`current_weights` = base `domain_weights` × exp(−age / half_life)** per dimension, using each project's schema half-lives. Recall (Step 2) ranks by `current_weights`; the base `domain_weights` is **never modified**, so decay is **lossless and idempotent** — every run recomputes from the base, no drift, and you can reset or re-derive at any time. Short half-lives (e.g. `liquidity_risk`, 365 d) fade fast; long ones (`financial_impact`, 3650 d) barely move — the "mundane fades, milestone stays" behavior, now persisted in storage rather than recomputed per query. The task also writes a single high-level **`relevance`** scalar = `max(current_weights)` — a coarse "does this still matter" score that goes low **only when a memory is old *and* unimportant on every axis** (a recent or enduring milestone stays high). It's returned on every result, so the frontend can sort or prune by it. (This reverses the earlier query-time approach per the requirement that decay mutate stored values on a schedule.)

### 3.5 Two models on the write path

Each memory is built by **two independent AI calls with different jobs** — don't conflate them:

- **Embedding model** — text → a fixed-length float vector (the `embedding` field). Sets `numDimensions` (Voyage `voyage-3.5` = 1024, OpenAI `text-embedding-3-small` = 1536). Must be the *same model* for ingest and query, or vectors are incomparable.
- **Extraction model** — an LLM that reads the text and returns the structured metadata (`categories[]`, `domain_weights{}`, `timestamp`), scored against the domain's Dimension Schema. Implemented in `pensieve/extraction.py` as a **provider-agnostic seam**: currently **`google/gemini-2.5-flash-lite` via OpenRouter** (OpenAI-compatible API), with a **strict JSON-schema** response so `categories` is an enum array and every weight is typed. Numeric range (0–10) can't be enforced by strict schema, so it's clamped client-side. Swap `settings.extract_model` / the base URL to change provider without touching anything else.

```python
def record(raw_text: str):
    meta = extract_metadata(raw_text)    # LLM (Claude) → {categories[], domain_weights{}, timestamp}
    doc = {**meta.model_dump(), "text": raw_text,
           "embedding": embed(raw_text),  # embedding model (Voyage / OpenAI)
           "created_at": datetime.utcnow()}
    col.insert_one(doc)
```

Extraction is cheap at any realistic volume: ~300–600 tokens/record, so even 10k mock records cost well under $1 on `gemini-2.5-flash-lite`. Optimize the model choice for JSON-schema reliability + scoring judgment, not price. For a large seed, parallelize the calls (currently sequential in `/record`).

### 3.6 Backend API surface

Unify writes; split reads by machinery. All JSON in / JSON out. **Strip `embedding` from every read response** (`$project: {embedding: 0}`) — the float arrays bloat payloads.

| Endpoint | Purpose | Vector search? | Returns |
|---|---|---|---|
| `POST /record` | Write — accepts **one object or an array** | — | inserted IDs |
| `POST /recall` | Semantic weighted retrieval (§3.4, the core feature) | ✅ | ranked top-N |
| `POST /raw_recall` | Plain Voyage similarity — no weights/decay/reinforcement (demo baseline, §3.4) | ✅ | ranked by similarity |
| `POST /report` | Project-scoped browse / timeline — filter by category/time, optional bucketing | ❌ | rows or time buckets |
| `POST /schema` · `GET /schema` | Configure/read a project's versioned Dimension Schema (§3.7, §3.9) | — | schema + version |
| `POST /reindex` · `GET /reindex/{job}` | Backfill new dimensions across a project's records (§3.7) | — | job id / progress |
| `GET /decay` | Repeated decay task — rewrites `current_weights` from base × age (Vercel Cron, §3.4) | — | `{updated, per_project}` |

**Every endpoint takes a required `project`** (§3.9) — it always filters, so nothing crosses project boundaries. `/recall` and `/report` are **separate** on purpose: one needs a query embedding and returns a small ranked set; the other is a plain filtered `find()` returning many rows (optionally time-bucketed). A single `mode` flag over both makes the contract ambiguous.

**Bulk ingest** — batch the embedding calls (many texts per API call) and do one `insert_many`, so onboarding mock data is fast:

```python
@app.post("/record")
def record(payload: Memory | list[Memory]):
    items = payload if isinstance(payload, list) else [payload]
    vectors = embed_batch([m.text for m in items])       # ONE embedding call
    docs = [{**m.model_dump(), "embedding": v} for m, v in zip(items, vectors)]
    return {"inserted": len(col.insert_many(docs).inserted_ids)}  # ONE round trip
```

**Time-bucketed browse** — `$group` by period, no vector search:

```jsonc
[ { "$match": { "timestamp": {"$gte": since, "$lt": until}, "categories": {"$in": cats} } },
  { "$project": { "embedding": 0 } },
  { "$group": { "_id": {"$dateTrunc": {"date": "$timestamp", "unit": "month"}},
                "memories": {"$push": "$$ROOT"} } },
  { "$sort": { "_id": 1 } } ]
```

### 3.7 Value Dimensions: schema definition + backfill

Value Dimensions are declared **ahead of time** via a versioned schema, then scored onto records. The key insight: **adding a dimension is a metadata backfill, not a vector re-index.** Embeddings are unchanged — you re-run the *extraction model* on each record's stored `text`, scoring only the new axis, and `$set` one field.

```python
class Dimension(BaseModel):
    range: tuple[float, float] = (0, 10)
    decay_half_life_days: int = 365
    default: float = 1

class DimensionSchema(BaseModel):
    categories: list[str]
    dimensions: dict[str, Dimension]   # e.g. {"financial_impact": Dimension(...), ...}
```

**Endpoints** (see the table in §3.6):
- `POST /schema` — upsert the schema (versioned); auto-enqueues a backfill for any newly added dimensions.
- `GET /schema` — read the active schema.
- `POST /reindex {dimensions:[...]}` — trigger/re-run a backfill; returns a job id immediately.
- `GET /reindex/{job_id}` — progress (`{done, total, state}`).

**Backfill worker** — four properties keep it correct and cheap:
- **Async, not in-request** — thousands of records = thousands of LLM calls; the endpoint returns a job id and the work runs in a background worker (**Batch API at 50% cost** for scale).
- **Idempotent + resumable** — the worker only touches records that *lack* the new dimension (`{f"domain_weights.{d}": {"$exists": False}}`); re-running after a crash finishes the rest. A `schema_version` stamped per doc records what's been backfilled.
- **Score-only, write-one-field** — one extraction call per text scoring *only the new axes*, then `bulk_write` of `UpdateOne` with `$set` on the nested `domain_weights.<dim>` key. **No re-embedding.**
- **Cheap interim** — seed the schema `default` via `update_many` instantly so `/recall` never errors on a missing key, then let real per-record scoring land in the background (until it does, the dimension can't discriminate — every record scores the default).

**Index caveat:** only a *gate* dimension (filtered inside `$vectorSearch`) needs a `filter` field added to the Atlas index (an online rebuild); a **scoring-only** dimension needs no index change and is live the moment the backfill writes it.

### 3.8 Ergonomic recall — reinforcement + explainability

Two features that make recall self-tuning and debuggable, implemented in `pensieve/scoring.py`:

- **Recall reinforcement (self-tuning importance).** Each memory carries `access_count` + `last_accessed`. Every *gated* `/recall` follows the result read with a `bulk_write` (`$inc access_count`, `$set last_accessed`) on the returned docs, and `FinalScore` adds `w_reinforce · ln(1 + access_count)`. Memories the agent keeps needing rise; unused ones fade faster. It's **incremental** — no batch job, no LLM — and **neutral at `access_count = 0`** (`ln 1 = 0`), so fresh records aren't penalized. The `/raw_recall` baseline never touches `access_count`, so running the two columns side-by-side doesn't skew the counts. Reinforcement is **event-driven** (on recall); time decay is its **scheduled** counterpart (§3.4 Step 3) — together they keep `current_weights` current from both usage and age.

- **Explainable results.** Every hit returns a `score_breakdown { similarity, weight, reinforcement }` next to `final_score` (`weight` is the decayed `current_weights` value). It answers "why did this surface?" for free and drives the demo's per-card score bars (§5). The `/raw_recall` baseline returns only `similarity` — no breakdown.

Both are already in the retrieval pipeline; a caller sees them on every `/recall` response with no extra request.

### 3.9 Project isolation

A **`project`** is the tenant / isolation unit. The three demo datasets are three projects — `finance`, `engineering`, `autobiography` — each with its **own Dimension Schema** and its own memories.

- **Always filters.** `project` is a `filter` field in the index and is applied inside `$vectorSearch` (and every `/report` `$match`) *regardless of `use_gates`* — recall, report, record, and schema config are all scoped to one project, so results never intersect across projects. Verified: a "migrated the database to MongoDB" query returns the MongoDB memory in `engineering` but never leaks it into `finance`.
- **Schema per project.** `POST /schema {project, categories, dimensions}` configures the value axes for one project and bumps its version; `GET /schema?project=…` reads it. Adding a dimension defines it going forward — existing memories need a `/reindex` backfill (§3.7) to be scored on it.
- **Model note.** `project` is the isolation key; `domain_weights` keeps its name (the per-dimension importance scores). Distinct concepts: `project` says *whose/which* memory store; `domain_weights` says *how much each axis matters*.

---

## 4. Domain Use Cases

| # | Use Case | Core Value Dimensions | Category Gates | High-value vs. low-value example |
|---|----------|----------------------|----------------|----------------------------------|
| 1 | **Personal Wealth Companion** | `financial_impact`, `liquidity_risk`, `tax_relevance` | `spending`, `investment`, `tax`, `income` | **High:** "Bought home for $600k in 2022" (9.8) · **Low:** "Spent $4.50 on a latte" (0.2) |
| 2 | **Corporate Project Historian** | `strategic_risk`, `architectural_impact`, `team_dependency` | `outage`, `refactor`, `decision`, `roadmap` | **High:** "Switched primary DB to MongoDB in Q2" (9.5) · **Low:** "Updated CSS button padding" (0.5) |
| 3 | **Life Autobiographer** *(original concept)* | `emotional_significance`, `relationship_impact`, `life_milestone_tier` | `career`, `family`, `travel`, `health` | **High:** "Got married in Tuscany" (9.9) · **Low:** "Watched TV after work" (0.3) |

Each use case is the *same engine* with a different Dimension Schema — the selling point for developers.

### Why this beats plain RAG (the money query)

> *"What were my most important financial decisions last year?"*

- **Plain RAG:** returns whatever text is most similar to "financial decision" — likely a mix of trivia and one or two real events, ordered by wording.
- **Pensieve Core:** temporal gate → last year only; category gate → `finance`; then ranks by `financial_impact · decay`. The $600k home and the $120k seed round surface at the top; the lattes never appear.

---

## 5. Demo App — "Pensieve Explorer"

A single-page app over **mock data** that makes the difference between plain RAG and domain-weighted retrieval *visible* side by side.

**Goal of the demo:** show that the *same query* returns *different, better* results once weighting and gates are on.

### Layout
- **Left panel — query bar + dimension controls:** a text query, a category multi-select, a time-window slider, and sliders for `w_vector` vs `w_domain_bias`.
- **Right panel — two result columns, live:**
  - **"Naive RAG"** — top-k by cosine similarity only.
  - **"Pensieve Core"** — same query through gates + weighted `FinalScore`.
  Each result card shows the text, its category chips, per-dimension weight bars, age, and the computed score breakdown (`sim`, `weight`, `decay`) so viewers *see why* it ranked where it did.

### Mock dataset
- ~150–300 synthetic memories across the three use cases above, hand-seeded so the contrast is dramatic (lots of low-weight noise sharing keywords with a few high-weight milestones).
- Pre-computed embeddings shipped with the seed so the demo runs without live embedding calls (fall back to live calls if a key is present).
- A **domain switcher** (Wealth / Project / Life) that swaps the Dimension Schema and dataset — proving the engine is config-driven, not hardcoded.

### Demo script (the 90-second wow)
1. Query *"how's my spending?"* with weighting **off** → results are dominated by coffee/lunch noise.
2. Flip weighting **on** → the home purchase and seed round jump to the top; noise sinks.
3. Drag the time window to "last 3 months" → watch anachronistic hits vanish (temporal gate).
4. Switch domain to **Project Historian**, same mechanics → proves it generalizes.

### Suggested build
- **Backend:** FastAPI + `pymongo`, two endpoints — `POST /record`, `POST /recall` (with a `use_gates: bool` flag so one endpoint serves both columns).
- **Frontend:** whatever's fastest to ship (Next.js / plain React + Vite).
- **Data:** MongoDB Atlas (Flex tier is enough for the mock volume) + a `seed.py` that builds the index and loads the dataset.

---

## 6. Open Questions / Next Steps

- [x] **Embedding model: Voyage `voyage-3.5` @ 1024 dims**, via `ai.mongodb.com/v1/embeddings`. Sets `numDimensions: 1024`.
- [x] **Default Dimension Schemas** for the three demo domains — in `seed.py`.
- [x] **Shared package** (`pensieve/`) — `config` (creds), `embeddings` (Voyage), `db` (client + index), `scoring` (recall pipeline). `seed.py` reuses these.
- [x] **Seed script** (`seed.py`) — index creation + curated mock data + real embeddings; initializes `access_count`.
- [x] **`/recall`** — semantic + gates + weighted `FinalScore`, with **reinforcement (§3.8)**, **explainable `score_breakdown` (§3.8)**, and a `use_gates` toggle for the demo's side-by-side view.
- [x] **`/record`** (accepts pre-scored metadata + embeds), **`GET /schema`**, `/health` — in `pensieve/app.py`.
- [x] **Live end-to-end** — M10 cluster seeded (26 memories, 3 schemas, `memory_index`), recall verified: weighting surfaces the seed round / home over lattes on "how's my spending?". Creds auto-load from `*-creds.txt`; `check_connection.py` (smoke test) + `demo_recall.py` (naive-vs-weighted proof) added.
- [ ] Start the API server (`uvicorn pensieve.app:app`) and exercise `/recall` over HTTP.
- [x] **Extraction on `/record`** — `google/gemini-2.5-flash-lite` via OpenRouter (`pensieve/extraction.py`), strict JSON-schema, client-side clamp. `/record` auto-extracts raw `{text, domain}` and passes pre-scored items straight through. Verified live (lake house → high impact + parsed date; gum → ~0).
- [ ] Extraction: add a *score-just-these-axes* mode for backfill (§3.7) + parallelize the per-record calls for bulk ingest.
- [x] **Project isolation** — `project` is a required, always-filtered key on `/recall`, `/report`, `/record`, `/schema`; datasets re-seeded as `finance`/`engineering`/`autobiography`; verified no cross-project leakage (§3.9).
- [x] **`POST /schema`** (per-project value/weight config, versioned) and **`/report`** (project-scoped browse + time-bucketing) implemented and tested.
- [x] **Vercel deploy** scaffolding — `index.py` entrypoint, `vercel.json` (maxDuration 60 + `/decay` cron), `DEPLOY.md` (env vars + Atlas `0.0.0.0/0` + gotchas).
- [x] **Decay as a scheduled task** — `pensieve/decay.py` rewrites `current_weights` from base `domain_weights` × age; `GET /decay` (Vercel Cron) + `decay.py` CLI. Recall ranks by `current_weights`; base is immutable. Verified.
- [x] **`/raw_recall`** — plain Voyage similarity baseline (no weights/decay/reinforcement/gates), optional filters; for the demo's "less accurate" column.
- [ ] `/reindex` backfill worker (§3.7) — must be Vercel Cron / a queue, not an in-process job.
- [ ] Wire the side-by-side demo UI (partner).
