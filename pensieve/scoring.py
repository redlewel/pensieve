"""Recall pipeline construction + reinforcement write-back.

FinalScore blends three signals:

    FinalScore = Sim·w_vector
               + Weight·w_domain_bias·Decay(age)
               + Reinforcement·w_reinforce

where Decay(age) = exp(-age_days / half_life) and
Reinforcement = ln(1 + access_count)  (neutral at 0 accesses: ln 1 = 0).

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
    domain: Optional[str] = None,
    dimension: Optional[str] = None,
    categories: Optional[list[str]] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    half_life_days: float = 365,
    w_vector: float = 1.0,
    w_domain_bias: float = 0.15,
    w_reinforce: float = 0.1,
    pool: int = 200,
    limit: int = 10,
) -> list[dict[str, Any]]:
    vs: dict[str, Any] = {
        "index": INDEX_NAME, "path": "embedding",
        "queryVector": query_vector, "numCandidates": pool, "limit": pool,
    }

    if use_gates:
        flt: dict[str, Any] = {}
        if domain:
            flt["domain"] = domain
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

    weight_expr: Any = ({"$ifNull": [f"$domain_weights.{dimension}", 0]}
                        if dimension else 0)

    pipeline += [
        {"$addFields": {
            "age_days": {"$dateDiff": {"startDate": "$timestamp",
                                       "endDate": "$$NOW", "unit": "day"}},
            "weight": weight_expr,
            "reinforcement": {"$ln": [{"$add": [1, {"$ifNull": ["$access_count", 0]}]}]},
        }},
        {"$addFields": {
            "decay": {"$exp": {"$multiply": [-1, {"$divide": ["$age_days", half_life_days]}]}},
        }},
        {"$addFields": {
            "score_breakdown": {
                "similarity": "$sim", "weight": "$weight",
                "decay": "$decay", "reinforcement": "$reinforcement",
            },
            "final_score": {"$add": [
                {"$multiply": ["$sim", w_vector]},
                {"$multiply": ["$weight", w_domain_bias, "$decay"]},
                {"$multiply": ["$reinforcement", w_reinforce]},
            ]},
        }},
        {"$sort": {"final_score": -1}},
        {"$limit": limit},
        {"$project": {"embedding": 0}},
    ]
    return pipeline


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
