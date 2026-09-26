"""Recall pipeline construction + reinforcement write-back.

FinalScore blends three signals:

    FinalScore = Sim·w_vector
               + Weight·w_domain_bias
               + Reinforcement·w_reinforce

Weight is `current_weights` — the base `domain_weights` after the periodic decay
task (pensieve/decay.py) has degraded it by age. Decay is applied in storage, not
here, so recall just reads the already-decayed value. Reinforcement =
ln(1 + access_count) (neutral at 0 accesses: ln 1 = 0).

Every hit carries a `score_breakdown` so callers can see *why* it ranked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pymongo import UpdateOne
from pymongo.collection import Collection

from .config import INDEX_NAME


def build_recall_pipeline(
    *,
    query_vector: list[float],
    use_gates: bool,
    project: Optional[str] = None,
    dimension: Optional[str] = None,
    categories: Optional[list[str]] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    w_vector: float = 1.0,
    w_domain_bias: float = 0.15,
    w_reinforce: float = 0.1,
    pool: int = 200,
    limit: int = 10,
) -> list[dict[str, Any]]:
    vs: dict[str, Any] = {
        "index": INDEX_NAME, "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": pool * 2,   # explore wider than the re-rank pool
        "limit": pool,               # candidates handed to the weighted re-rank
    }

    # Project is the isolation scope — always applied, even for the naive column,
    # so queries never intersect across projects.
    # Category / temporal gates apply only when use_gates is on.
    flt: dict[str, Any] = {}
    if project:
        flt["project"] = project
    if use_gates:
        if categories:
            flt["categories"] = {"$in": categories}
        if since or until:
            rng: dict[str, Any] = {}
            if since:
                rng["$gte"] = since
            if until:
                rng["$lt"] = until
            flt["timestamp"] = rng
    if flt:
        vs["filter"] = flt

    pipeline: list[dict[str, Any]] = [
        {"$vectorSearch": vs},
        {"$addFields": {"sim": {"$meta": "vectorSearchScore"}}},
    ]

    if not use_gates:
        # Naive similarity-only column (the "plain RAG" comparison).
        pipeline += [
            {"$addFields": {"score_breakdown": {"similarity": "$sim"},
                            "final_score": "$sim"}},
            {"$sort": {"final_score": -1}},
            {"$limit": limit},
            {"$project": {"embedding": 0}},
        ]
        return pipeline

    # Rank by current_weights (decayed in storage by the decay task); fall back to
    # the base domain_weights for any memory the decay task hasn't touched yet.
    weight_expr: Any = ({"$ifNull": [f"$current_weights.{dimension}",
                                     {"$ifNull": [f"$domain_weights.{dimension}", 0]}]}
                        if dimension else 0)

    pipeline += [
        {"$addFields": {
            "weight": weight_expr,
            "reinforcement": {"$ln": [{"$add": [1, {"$ifNull": ["$access_count", 0]}]}]},
        }},
        {"$addFields": {
            "score_breakdown": {"similarity": "$sim", "weight": "$weight",
                                "reinforcement": "$reinforcement"},
            "final_score": {"$add": [
                {"$multiply": ["$sim", w_vector]},
                {"$multiply": ["$weight", w_domain_bias]},
                {"$multiply": ["$reinforcement", w_reinforce]},
            ]},
        }},
        {"$sort": {"final_score": -1}},
        {"$limit": limit},
        {"$project": {"embedding": 0}},
    ]
    return pipeline


def build_raw_pipeline(
    *,
    query_vector: list[float],
    project: Optional[str] = None,
    categories: Optional[list[str]] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    pool: int = 200,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Plain Voyage vector search — similarity only, no weights / decay /
    reinforcement. Optional filters (project / category / time) scope it without
    changing the ranking. The demo's intentionally "less accurate" baseline;
    omit `project` to search across everything (no isolation), which shows off
    what the full engine adds.
    """
    flt: dict[str, Any] = {}
    if project:
        flt["project"] = project
    if categories:
        flt["categories"] = {"$in": categories}
    if since or until:
        rng: dict[str, Any] = {}
        if since:
            rng["$gte"] = since
        if until:
            rng["$lt"] = until
        flt["timestamp"] = rng

    vs: dict[str, Any] = {
        "index": INDEX_NAME, "path": "embedding", "queryVector": query_vector,
        "numCandidates": pool * 2, "limit": limit,
    }
    if flt:
        vs["filter"] = flt

    # $vectorSearch already returns results sorted by similarity — no re-rank.
    return [
        {"$vectorSearch": vs},
        {"$addFields": {"similarity": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0}},
    ]


def bump_access(col: Collection, ids: list) -> None:
    """Reinforcement: record that these memories were just recalled."""
    if not ids:
        return
    now = datetime.now(timezone.utc)
    col.bulk_write(
        [UpdateOne({"_id": _id},
                   {"$inc": {"access_count": 1}, "$set": {"last_accessed": now}})
         for _id in ids],
        ordered=False,
    )
