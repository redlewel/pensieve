"""Pensieve Core — FastAPI backend (basic scaffold).

Every memory belongs to a `project` — the isolation key. Recall, report, record,
and schema config are all scoped to a project, so data never intersects across
projects.

Endpoints:
  POST /recall   — semantic + gates + weighted recall (reinforcement + breakdown)
  POST /report   — project-scoped browse / timeline (no vector search)
  POST /record   — raw text (auto-extracted) or pre-scored; partial-success
  POST /schema   — configure a project's categories + value dimensions
  GET  /schema   — read project schema(s)
  GET  /health

Run:  uvicorn pensieve.app:app --reload
(requires the cluster URI + embedding key; extraction needs the OpenRouter key —
see pensieve/config.py)

TODO (see SPEC §6): /reindex backfill worker after adding a dimension to a schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import load_settings
from .db import get_client, memories, schemas
from .decay import run_decay
from .embeddings import embed_batch, embed_one
from .extraction import extract_many
from .scoring import build_raw_pipeline, build_recall_pipeline, bump_access

settings = load_settings()
# Fail fast on a bad/unreachable cluster instead of hanging a serverless function.
_client = get_client(settings, serverSelectionTimeoutMS=8000)
_mem = memories(_client)
_sch = schemas(_client)

app = FastAPI(title="Pensieve Core", version="0.1.0")

# Demo-only: open CORS so the frontend can call the API from a browser.
# Lock this down (specific origins) before anything but the hackathon.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)


# ------------------------------------------------------------------- models ---
class Memory(BaseModel):
    text: str
    project: str                                  # isolation key
    categories: list[str] = []
    domain_weights: dict[str, float] = {}
    timestamp: Optional[datetime] = None


class RecallRequest(BaseModel):
    query: str
    project: str                                  # isolation key — always scopes
    dimension: Optional[str] = None               # which weight axis to bias by
    categories: Optional[list[str]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    w_vector: float = 1.0
    w_domain_bias: float = 0.15
    w_reinforce: float = 0.1
    limit: int = 10
    use_gates: bool = True                        # False = naive similarity-only column
    reinforce: Optional[bool] = None              # default: only when gated


class RawRecallRequest(BaseModel):
    query: str
    project: Optional[str] = None                 # optional filter (omit = all projects)
    categories: Optional[list[str]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    limit: int = 10


class ReportRequest(BaseModel):
    project: str                                  # isolation key
    categories: Optional[list[str]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    limit: int = 50
    bucket: Optional[str] = None                  # day|week|month|quarter|year to group


class Dimension(BaseModel):
    range: tuple[float, float] = (0.0, 10.0)
    decay_half_life_days: int = 365
    default: float = 1.0


class SchemaUpsert(BaseModel):
    project: str
    categories: list[str]
    dimensions: dict[str, Dimension]


# ---------------------------------------------------------------- endpoints ---
@app.get("/health")
def health():
    return {"ok": True, "model": settings.model, "dim": settings.dim}


@app.post("/record")
def record(payload: Memory | list[Memory]):
    items = payload if isinstance(payload, list) else [payload]

    # Raw-text items (no categories AND no weights) get metadata auto-extracted
    # concurrently against their project's schema; pre-scored items pass through.
    # Per-item failures (missing schema, extraction error) are collected — they
    # never abort the batch, so the good records still land.
    failed_ids: set[int] = set()
    failures: list[dict] = []
    need = [m for m in items if not m.categories and not m.domain_weights]
    if need:
        schema_cache: dict[str, Optional[dict]] = {}
        jobs, job_items = [], []
        for m in need:
            if m.project not in schema_cache:
                schema_cache[m.project] = _sch.find_one({"_id": m.project})
            ds = schema_cache[m.project]
            if not ds:
                failed_ids.add(id(m))
                failures.append({"text": m.text[:80], "error": f"no schema for project '{m.project}'"})
            else:
                jobs.append((m.text, m.project, ds))
                job_items.append(m)
        for m, (meta, err) in zip(job_items, extract_many(jobs, settings)):
            if err:
                failed_ids.add(id(m))
                failures.append({"text": m.text[:80], "error": err})
            else:
                m.categories = meta["categories"]
                m.domain_weights = meta["domain_weights"]
                if meta["timestamp"] and not m.timestamp:
                    m.timestamp = meta["timestamp"]

    to_insert = [m for m in items if id(m) not in failed_ids]
    if not to_insert:
        return {"inserted": 0, "ids": [], "failed": failures}

    vectors = embed_batch([m.text for m in to_insert], settings, input_type="document")
    now = datetime.now(timezone.utc)
    docs = []
    for m, vec in zip(to_insert, vectors):
        doc = m.model_dump()
        doc["timestamp"] = doc["timestamp"] or now
        doc.update(embedding=vec, access_count=0, last_accessed=None,
                   schema_version=1, created_at=now)
        docs.append(doc)
    res = _mem.insert_many(docs)
    return {"inserted": len(res.inserted_ids),
            "ids": [str(i) for i in res.inserted_ids],
            "failed": failures}


@app.post("/recall")
def recall(req: RecallRequest):
    qvec = embed_one(req.query, settings, input_type="query")
    pipeline = build_recall_pipeline(
        query_vector=qvec, use_gates=req.use_gates,
        project=req.project, dimension=req.dimension,
        categories=req.categories, since=req.since, until=req.until,
        w_vector=req.w_vector, w_domain_bias=req.w_domain_bias,
        w_reinforce=req.w_reinforce, limit=req.limit,
    )
    results = list(_mem.aggregate(pipeline))

    # Reinforcement: bump access on the returned docs (default: only gated recalls).
    reinforce = req.use_gates if req.reinforce is None else req.reinforce
    if reinforce:
        bump_access(_mem, [r["_id"] for r in results])

    for r in results:
        r["_id"] = str(r["_id"])
    return {"results": results}


@app.post("/raw_recall")
def raw_recall(req: RawRecallRequest):
    """Plain Voyage vector search — no weighting, decay, reinforcement, or gates.
    The intentionally 'less accurate' baseline for the demo comparison. Each result
    carries only `similarity` (no score_breakdown)."""
    qvec = embed_one(req.query, settings, input_type="query")
    results = list(_mem.aggregate(build_raw_pipeline(
        query_vector=qvec, project=req.project, categories=req.categories,
        since=req.since, until=req.until, limit=req.limit)))
    for r in results:
        r["_id"] = str(r["_id"])
    return {"results": results}


@app.post("/report")
def report(req: ReportRequest):
    match: dict = {"project": req.project}            # isolation — always applied
    if req.categories:
        match["categories"] = {"$in": req.categories}
    if req.since or req.until:
        rng: dict = {}
        if req.since:
            rng["$gte"] = req.since
        if req.until:
            rng["$lt"] = req.until
        match["timestamp"] = rng

    if req.bucket:
        buckets = list(_mem.aggregate([
            {"$match": match},
            {"$project": {"embedding": 0}},
            {"$sort": {"timestamp": -1}},
            {"$group": {"_id": {"$dateTrunc": {"date": "$timestamp", "unit": req.bucket}},
                        "memories": {"$push": "$$ROOT"}}},
            {"$sort": {"_id": 1}},
        ]))
        for b in buckets:
            for m in b["memories"]:
                m["_id"] = str(m["_id"])
        return {"project": req.project, "buckets": buckets}

    results = list(_mem.aggregate([
        {"$match": match},
        {"$project": {"embedding": 0}},
        {"$sort": {"timestamp": -1}},
        {"$limit": req.limit},
    ]))
    for r in results:
        r["_id"] = str(r["_id"])
    return {"project": req.project, "results": results}


@app.post("/schema")
def upsert_schema(body: SchemaUpsert):
    """Create or update a project's schema (categories + value dimensions).

    Bumps the version. Adding a new dimension only defines it going forward;
    existing memories need a backfill (/reindex, SPEC §3.7) to be scored on it.
    """
    prev = _sch.find_one({"_id": body.project})
    version = (prev.get("version", 0) + 1) if prev else 1
    _sch.replace_one(
        {"_id": body.project},
        {"_id": body.project, "version": version,
         "categories": body.categories,
         "dimensions": {k: v.model_dump() for k, v in body.dimensions.items()}},
        upsert=True,
    )
    return {"project": body.project, "version": version,
            "dimensions": list(body.dimensions.keys())}


@app.get("/schema")
def get_schema(project: Optional[str] = None):
    if project:
        doc = _sch.find_one({"_id": project})
        if not doc:
            raise HTTPException(404, f"no schema for project '{project}'")
        return doc
    return list(_sch.find())


@app.get("/decay")
def decay():
    """Repeated decay task: rewrite every memory's current_weights from its base
    domain_weights and age. Idempotent — wire to Vercel Cron (see vercel.json).
    Protect with a CRON_SECRET check before exposing publicly."""
    return run_decay(_mem, _sch)
