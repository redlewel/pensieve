"""Time decay as a repeated task (not a query-time value).

Rewrites each memory's `current_weights` = base `domain_weights` × exp(-age/half_life),
per dimension, using each project's schema half-lives. Recall ranks by
`current_weights`; the base `domain_weights` is never modified — so decay is
**lossless and idempotent** (every run recomputes from the base, no drift).

Run it on a schedule (Vercel Cron → GET /decay, or `python decay.py`).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from pymongo import UpdateOne
from pymongo.collection import Collection

_CHUNK = 1000


def _age_days(ts: datetime, now: datetime) -> float:
    if ts is None:
        return 0.0
    if ts.tzinfo is not None:       # normalize to match naive `now`
        ts = ts.replace(tzinfo=None)
    return max(0.0, (now - ts).total_seconds() / 86400)


def run_decay(mem: Collection, sch: Collection) -> dict:
    """Recompute current_weights for every memory. Returns {updated, per_project}."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)   # pymongo returns naive UTC
    schemas = {s["_id"]: s for s in sch.find()}

    ops: list[UpdateOne] = []
    per_project: dict[str, int] = {}
    for doc in mem.find({}, {"project": 1, "timestamp": 1, "domain_weights": 1}):
        schema = schemas.get(doc.get("project"))
        if not schema:
            continue
        dims = schema.get("dimensions") or {}
        base = doc.get("domain_weights") or {}
        age = _age_days(doc.get("timestamp"), now)
        current = {}
        for name, cfg in dims.items():
            half_life = cfg.get("decay_half_life_days", 365) or 365
            factor = math.exp(-age / half_life)
            current[name] = round(base.get(name, cfg.get("default", 0)) * factor, 4)
        ops.append(UpdateOne({"_id": doc["_id"]},
                             {"$set": {"current_weights": current, "decayed_at": now}}))
        per_project[doc["project"]] = per_project.get(doc["project"], 0) + 1

    for i in range(0, len(ops), _CHUNK):
        mem.bulk_write(ops[i:i + _CHUNK], ordered=False)

    return {"updated": len(ops), "per_project": per_project}
