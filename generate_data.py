"""Generate a large, realistic mock dataset across the three demo projects.

Uses OpenRouter (Gemini) to author + score memories in strict-JSON batches (one
call yields many fully-formed memories — no separate extraction pass), Voyage for
embeddings, into the Atlas cluster. Each project gets ~N memories spread over the
last 3 years with a realistic mix of mundane low-weight entries and occasional
high-weight milestones.

Generated docs are tagged source="llm"; each run clears prior generated docs first
(the curated seed docs, which have no source tag, are preserved).

Run:  python generate_data.py              # ~200 / project
      python generate_data.py -n 300
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests
from openai import OpenAI

from pensieve.config import load_settings
from pensieve.db import get_client, memories, schemas
from pensieve.embeddings import embed_batch

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
BATCH = 20          # memories generated per LLM call
MAX_WORKERS = 8

PERSONAS = {
    "finance": "a person's personal finance journal",
    "engineering": "an engineering team's project history log (decisions, incidents, refactors, roadmap, migrations)",
    "autobiography": "a person's life autobiography (career, family, travel, health)",
}
THEMES = {
    "finance": [
        "everyday spending — coffee, groceries, gas, small subscriptions",
        "a major purchase or investment",
        "income, a bonus, or a raise",
        "taxes and financial admin",
        "debt, loans, or paying something off",
        "a significant financial milestone or decision",
    ],
    "engineering": [
        "a routine small code change or refactor",
        "a production incident or outage",
        "an architecture or tech-stack decision",
        "roadmap and planning",
        "a dependency bump or tooling change",
        "a major migration, launch, or rearchitecture",
    ],
    "autobiography": [
        "an ordinary day or small routine",
        "a family moment",
        "travel or a trip",
        "health, fitness, or a doctor visit",
        "a career event",
        "a major life milestone",
    ],
}


def _client(settings) -> OpenAI:
    return OpenAI(base_url=_OPENROUTER_BASE, api_key=settings.openrouter_key)


def credits_remaining(settings) -> float | None:
    try:
        r = requests.get(f"{_OPENROUTER_BASE}/credits",
                         headers={"Authorization": f"Bearer {settings.openrouter_key}"}, timeout=30)
        r.raise_for_status()
        d = r.json()["data"]
        return round(d["total_credits"] - d["total_usage"], 4)
    except Exception:
        return None


def _schema(project_schema: dict) -> dict:
    cats = project_schema.get("categories", [])
    dims = list((project_schema.get("dimensions") or {}).keys())
    return {
        "name": "mock_memories", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False, "required": ["memories"],
            "properties": {"memories": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["text", "categories", "domain_weights"],
                "properties": {
                    "text": {"type": "string"},
                    "categories": {"type": "array", "items": {"type": "string", "enum": cats}},
                    "domain_weights": {
                        "type": "object", "additionalProperties": False, "required": dims,
                        "properties": {d: {"type": "number"} for d in dims}},
                }}}},
        },
    }


def _prompt(project: str, project_schema: dict) -> str:
    dims = list((project_schema.get("dimensions") or {}).keys())
    cats = project_schema.get("categories", [])
    return (
        f"You author realistic first-person entries for {PERSONAS[project]}.\n"
        f"Importance axes (score each 0-10; 0 = trivial, 10 = life-defining): {', '.join(dims)}.\n"
        f"Allowed categories (choose all that apply): {', '.join(cats)}.\n"
        "Keep each entry to 1-2 sentences — a quick journal note, not an essay. "
        "Write natural, specific, varied entries — avoid generic repetition and "
        "boilerplate. Include a realistic spread of significance: most everyday and "
        "minor, a few genuinely major. Score honestly."
    )


def _clamp(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo


def gen_batch(spec, settings, client):
    project, project_schema, count, theme, nonce = spec
    try:
        resp = client.chat.completions.create(
            model=settings.extract_model, temperature=0.9, max_tokens=8000,
            messages=[
                {"role": "system", "content": _prompt(project, project_schema)},
                {"role": "user", "content":
                    f"Generate {count} distinct entries centered on: {theme}. "
                    f"(batch {nonce} — make these different from typical entries.)"},
            ],
            response_format={"type": "json_schema", "json_schema": _schema(project_schema)},
        )
        items = json.loads(resp.choices[0].message.content)["memories"]
    except Exception as e:
        print(f"  batch failed ({project}/{theme[:24]}): {type(e).__name__}: {e}")
        return []

    dims = project_schema.get("dimensions") or {}
    cats = set(project_schema.get("categories", []))
    out = []
    for it in items:
        weights = {}
        for name, cfg in dims.items():
            lo, hi = cfg.get("range", [0, 10])
            weights[name] = _clamp((it.get("domain_weights") or {}).get(name, cfg.get("default", 0)), lo, hi)
        out.append({"project": project, "text": it["text"],
                    "categories": [c for c in it.get("categories", []) if c in cats],
                    "domain_weights": weights})
    return out


def rand_ts(now):
    start = now - timedelta(days=3 * 365)
    return start + timedelta(seconds=random.random() * (now - start).total_seconds())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200, help="memories per project")
    args = ap.parse_args()

    settings = load_settings()
    if not settings.openrouter_key:
        raise SystemExit("OpenRouter key missing (see openrouter-api-key.txt)")
    client = _client(settings)
    mem, sch = memories(get_client(settings)), schemas(get_client(settings))

    before = credits_remaining(settings)
    print(f"OpenRouter credits remaining: ${before}\n" if before is not None
          else "OpenRouter credit balance unavailable\n")

    # Build batch specs across projects, cycling themes.
    specs = []
    for project in PERSONAS:
        ps = sch.find_one({"_id": project})
        if not ps:
            print(f"  no schema for '{project}' — run seed.py first; skipping")
            continue
        themes = THEMES[project]
        remaining, i = args.n, 0
        while remaining > 0:
            count = min(BATCH, remaining)
            specs.append((project, ps, count, themes[i % len(themes)], i + 1))
            remaining -= count
            i += 1

    print(f"Generating via {settings.extract_model} — {len(specs)} batches...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda s: gen_batch(s, settings, client), specs))
    docs_meta = [m for batch in results for m in batch]
    print(f"  generated {len(docs_meta)} memories")

    if not docs_meta:
        raise SystemExit("nothing generated — aborting")

    print(f"Embedding {len(docs_meta)} texts via {settings.model}...")
    vectors = embed_batch([m["text"] for m in docs_meta], settings, input_type="document")

    now = datetime.now(timezone.utc)
    docs = [{**m, "embedding": v, "timestamp": rand_ts(now),
             "access_count": 0, "last_accessed": None, "schema_version": 1,
             "created_at": now, "source": "llm"}
            for m, v in zip(docs_meta, vectors)]

    print("Writing to Atlas (clearing prior generated docs first)...")
    removed = mem.delete_many({"source": "llm"}).deleted_count
    mem.insert_many(docs)
    total = mem.count_documents({})
    print(f"  removed {removed} old generated, inserted {len(docs)}; collection now {total} docs")

    after = credits_remaining(settings)
    if before is not None and after is not None:
        print(f"\nOpenRouter: ${after} remaining (spent ${round(before - after, 4)})")


if __name__ == "__main__":
    main()
