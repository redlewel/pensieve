"""Pensieve Core — FastAPI backend (basic scaffold).

Implements the core recall path with reinforcement (self-tuning importance) and
explainable score breakdowns, plus /record and schema read.

Run:  uvicorn pensieve.app:app --reload
(requires ATLAS_URI + the embedding key — see pensieve/config.py)

TODO (see SPEC §6): LLM extraction on /record (currently takes pre-scored
metadata), /schema upsert + /reindex backfill worker, /memories/query browse.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_settings
from .db import get_client, memories, schemas
from .embeddings import embed_batch, embed_one
from .scoring import build_recall_pipeline, bump_access

settings = load_settings()
_client = get_client(settings)
_mem = memories(_client)
_sch = schemas(_client)

app = FastAPI(title="Pensieve Core", version="0.1.0")


class Memory(BaseModel):
    text: str
    domain: str
    categories: list[str] = []
    domain_weights: dict[str, float] = {}
    timestamp: Optional[datetime] = None


class RecallRequest(BaseModel):
    query: str
    domain: str
    dimension: Optional[str] = None          # which weight axis to bias by
    categories: Optional[list[str]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    w_vector: float = 1.0
    w_domain_bias: float = 0.15
    w_reinforce: float = 0.1
    limit: int = 10
    use_gates: bool = True                   # False = naive similarity-only column
    reinforce: Optional[bool] = None         # default: only when gated


@app.get("/health")
def health():
    return {"ok": True, "model": settings.model, "dim": settings.dim}


@app.post("/record")
def record(payload: Memory | list[Memory]):
    items = payload if isinstance(payload, list) else [payload]
    vectors = embed_batch([m.text for m in items], settings, input_type="document")
    now = datetime.now(timezone.utc)
    docs = []
    for m, vec in zip(items, vectors):
        doc = m.model_dump()
        doc["timestamp"] = doc["timestamp"] or now
        doc.update(embedding=vec, access_count=0, last_accessed=None,
                   schema_version=1, created_at=now)
        docs.append(doc)
    res = _mem.insert_many(docs)
    return {"inserted": len(res.inserted_ids),
            "ids": [str(i) for i in res.inserted_ids]}


@app.get("/schema")
def get_schema(domain: Optional[str] = None):
    if domain:
        doc = _sch.find_one({"_id": domain})
        if not doc:
            raise HTTPException(404, f"no schema for domain '{domain}'")
        return doc
    return list(_sch.find())


@app.post("/recall")
def recall(req: RecallRequest):
    # Pull the decay half-life for the biasing dimension from the domain schema.
    half_life = 365.0
    if req.use_gates and req.dimension:
        schema = _sch.find_one({"_id": req.domain}) or {}
        dim = (schema.get("dimensions") or {}).get(req.dimension)
        if dim:
            half_life = dim.get("decay_half_life_days", 365)

    qvec = embed_one(req.query, settings, input_type="query")
    pipeline = build_recall_pipeline(
        query_vector=qvec, use_gates=req.use_gates,
        domain=req.domain, dimension=req.dimension,
        categories=req.categories, since=req.since, until=req.until,
        half_life_days=half_life,
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
