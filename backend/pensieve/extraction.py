"""LLM metadata extraction via OpenRouter (OpenAI-compatible).

Reads raw text and returns categories + per-dimension weights + timestamp,
scored against a domain's Dimension Schema. Uses strict JSON-schema structured
output so fields are always typed; numeric ranges are clamped client-side
(strict schema can't enforce min/max).

Provider-agnostic seam: swap `settings.extract_model` / the base URL to move off
OpenRouter without touching the rest of Pensieve.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional

from openai import OpenAI

from .config import Settings

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_MAX_WORKERS = 8   # concurrent extraction calls (I/O-bound; bounded for rate limits)


def _client(settings: Settings) -> OpenAI:
    if not settings.openrouter_key:
        raise RuntimeError(
            "OpenRouter key not set — extraction needs it (see openrouter-api-key.txt)."
        )
    return OpenAI(base_url=_OPENROUTER_BASE, api_key=settings.openrouter_key)


def _json_schema(domain_schema: dict) -> dict:
    cats = domain_schema.get("categories", [])
    dims = list((domain_schema.get("dimensions") or {}).keys())
    return {
        "name": "memory_metadata",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["categories", "domain_weights", "timestamp"],
            "properties": {
                "categories": {"type": "array",
                               "items": {"type": "string", "enum": cats}},
                "domain_weights": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": dims,
                    "properties": {d: {"type": "number"} for d in dims},
                },
                "timestamp": {"type": ["string", "null"],
                              "description": "ISO 8601 datetime if the text implies "
                                             "a date/time, else null"},
            },
        },
    }


def _system_prompt(project: str, domain_schema: dict) -> str:
    dims = domain_schema.get("dimensions") or {}
    cats = domain_schema.get("categories", [])
    lines = [
        f"You extract structured memory metadata for the '{project}' project.",
        "For the given text, assign:",
        "  1. categories — every tag that applies, from the allowed list.",
        "  2. domain_weights — a 0-10 score on each importance axis "
        "(10 = maximally significant on that axis, 0 = trivial noise).",
        "  3. timestamp — ISO 8601 if the text implies a date/time, else null.",
        "",
        "Importance axes: " + ", ".join(dims.keys()),
        "Allowed categories: " + ", ".join(cats),
        "",
        "Score honestly: a $5 coffee is ~0 on financial_impact; a $600k home is ~10. "
        "Return only the structured fields.",
    ]
    return "\n".join(lines)


def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo


def extract_metadata(text: str, project: str, domain_schema: dict,
                     settings: Settings, client: Optional[OpenAI] = None) -> dict:
    """text -> {categories: [...], domain_weights: {...}, timestamp: datetime|None}."""
    client = client or _client(settings)
    resp = client.chat.completions.create(
        model=settings.extract_model,
        messages=[{"role": "system", "content": _system_prompt(project, domain_schema)},
                  {"role": "user", "content": text}],
        response_format={"type": "json_schema", "json_schema": _json_schema(domain_schema)},
        temperature=0,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError(f"extraction returned empty content for: {text[:60]!r}")
    data = json.loads(content)

    dims = domain_schema.get("dimensions") or {}
    raw_weights = data.get("domain_weights") or {}
    weights = {}
    for name, cfg in dims.items():
        lo, hi = cfg.get("range", [0, 10])
        weights[name] = _clamp(raw_weights.get(name, cfg.get("default", 0)), lo, hi)

    parsed_ts: Optional[datetime] = None
    ts = data.get("timestamp")
    if ts:
        try:
            parsed_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            parsed_ts = None

    return {"categories": data.get("categories", []),
            "domain_weights": weights, "timestamp": parsed_ts}


def _safe_extract(job: tuple[str, str, dict], settings: Settings,
                  client: OpenAI) -> tuple[Optional[dict], Optional[str]]:
    text, project, schema = job
    try:
        return extract_metadata(text, project, schema, settings, client), None
    except Exception as e:   # isolate per-item failure — never abort the batch
        return None, f"{type(e).__name__}: {e}"


def extract_many(jobs: list[tuple[str, str, dict]], settings: Settings,
                 max_workers: int = _MAX_WORKERS,
                 ) -> list[tuple[Optional[dict], Optional[str]]]:
    """Extract metadata for many (text, domain, domain_schema) jobs concurrently.

    Order-preserving; shares one OpenAI client (connection pool) across threads.
    Each result is (metadata, None) on success or (None, error_str) on failure,
    so a single failed call never takes down the whole batch.
    """
    if not jobs:
        return []
    client = _client(settings)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
        return list(pool.map(lambda j: _safe_extract(j, settings, client), jobs))
