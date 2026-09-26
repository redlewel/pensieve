"""Pensieve Core — seed script.

Creates the Atlas Vector Search index, embeds curated mock memories via Voyage
(ai.mongodb.com), and loads them across the three demo domains along with their
dimension schemas.

The mock metadata (categories, domain_weights) is hand-authored so the contrast
between high-value milestones and low-value noise is dramatic — that curation is
the point of the demo. Real records go through the LLM extraction path instead.

Credentials: ATLAS_URI (env, required) + MONGODB_MODEL_API_KEY (env or
mongodb-api-key.txt). See pensieve/config.py.

Run:  python seed.py
"""

from __future__ import annotations

import sys
from datetime import datetime

from pensieve.config import load_settings
from pensieve.db import ensure_index, get_client, memories, schemas
from pensieve.embeddings import embed_batch

# ------------------------------------------------------- dimension schemas ---
# One schema per demo domain. decay_half_life_days: high-value axes decay slowly.
SCHEMAS = {
    "wealth": {
        "categories": ["spending", "investment", "tax", "income", "debt"],
        "dimensions": {
            "financial_impact": {"range": [0, 10], "decay_half_life_days": 3650, "default": 1},
            "liquidity_risk":   {"range": [0, 10], "decay_half_life_days": 365,  "default": 1},
            "tax_relevance":    {"range": [0, 10], "decay_half_life_days": 400,  "default": 0},
        },
    },
    "project": {
        "categories": ["outage", "refactor", "decision", "roadmap"],
        "dimensions": {
            "strategic_risk":       {"range": [0, 10], "decay_half_life_days": 1825, "default": 1},
            "architectural_impact": {"range": [0, 10], "decay_half_life_days": 2555, "default": 1},
            "team_dependency":      {"range": [0, 10], "decay_half_life_days": 365,  "default": 1},
        },
    },
    "life": {
        "categories": ["career", "family", "travel", "health"],
        "dimensions": {
            "emotional_significance": {"range": [0, 10], "decay_half_life_days": 7300, "default": 1},
            "relationship_impact":    {"range": [0, 10], "decay_half_life_days": 3650, "default": 1},
            "life_milestone_tier":    {"range": [0, 10], "decay_half_life_days": 7300, "default": 0},
        },
    },
}


def _m(domain, text, ts, categories, weights):
    return {"domain": domain, "text": text,
            "timestamp": datetime.fromisoformat(ts),
            "categories": categories, "domain_weights": weights,
            "access_count": 0, "last_accessed": None, "schema_version": 1}


# ------------------------------------------------------------- mock memories ---
# Deliberately mixes high-weight milestones with low-weight noise that shares
# keywords, so weighting visibly beats plain similarity.
RAW_MEMORIES = [
    # --- Personal Wealth Companion -------------------------------------------
    _m("wealth", "Bought our first home for $600k in the suburbs.", "2022-06-14T09:00:00",
       ["spending", "debt"], {"financial_impact": 9.8, "liquidity_risk": 7.0, "tax_relevance": 6.0}),
    _m("wealth", "Closed a $120k seed round with Venture Corp.", "2025-11-14T10:30:00",
       ["income", "investment"], {"financial_impact": 9.5, "liquidity_risk": 4.0, "tax_relevance": 7.0}),
    _m("wealth", "Maxed out the 401k and opened a Roth IRA.", "2025-01-08T18:00:00",
       ["investment", "tax"], {"financial_impact": 6.5, "liquidity_risk": 2.0, "tax_relevance": 8.5}),
    _m("wealth", "Paid off the last of the student loans.", "2024-09-02T12:00:00",
       ["debt"], {"financial_impact": 7.0, "liquidity_risk": 3.0, "tax_relevance": 1.0}),
    _m("wealth", "Sold the old car for $8,500.", "2025-03-20T14:00:00",
       ["income"], {"financial_impact": 3.5, "liquidity_risk": 1.0, "tax_relevance": 1.5}),
    _m("wealth", "Spent $4.50 on a latte this morning.", "2026-09-01T08:15:00",
       ["spending"], {"financial_impact": 0.2, "liquidity_risk": 0.0, "tax_relevance": 0.0}),
    _m("wealth", "Grabbed lunch downtown, about $18.", "2026-08-28T13:00:00",
       ["spending"], {"financial_impact": 0.3, "liquidity_risk": 0.0, "tax_relevance": 0.0}),
    _m("wealth", "Monthly grocery run came to $240.", "2026-08-15T17:30:00",
       ["spending"], {"financial_impact": 1.2, "liquidity_risk": 0.5, "tax_relevance": 0.0}),
    _m("wealth", "Renewed the streaming subscription, $15.99.", "2026-07-10T20:00:00",
       ["spending"], {"financial_impact": 0.2, "liquidity_risk": 0.0, "tax_relevance": 0.0}),
    _m("wealth", "Filed quarterly estimated taxes.", "2026-06-15T11:00:00",
       ["tax"], {"financial_impact": 4.0, "liquidity_risk": 2.5, "tax_relevance": 9.0}),

    # --- Corporate Project Historian ----------------------------------------
    _m("project", "Switched the primary datastore to MongoDB in Q2.", "2025-04-18T15:00:00",
       ["decision", "refactor"], {"strategic_risk": 9.5, "architectural_impact": 9.0, "team_dependency": 7.0}),
    _m("project", "Prod outage: auth service down for 47 minutes.", "2025-08-03T02:12:00",
       ["outage"], {"strategic_risk": 8.5, "architectural_impact": 5.0, "team_dependency": 8.0}),
    _m("project", "Committed to the multi-region rollout on the H2 roadmap.", "2026-01-20T10:00:00",
       ["roadmap", "decision"], {"strategic_risk": 8.0, "architectural_impact": 7.5, "team_dependency": 6.0}),
    _m("project", "Extracted the billing module into its own service.", "2025-10-05T16:00:00",
       ["refactor", "decision"], {"strategic_risk": 6.0, "architectural_impact": 8.0, "team_dependency": 5.0}),
    _m("project", "Bumped the CSS button padding by 2px.", "2026-08-22T14:30:00",
       ["refactor"], {"strategic_risk": 0.5, "architectural_impact": 0.3, "team_dependency": 0.2}),
    _m("project", "Renamed a helper function for clarity.", "2026-08-19T11:15:00",
       ["refactor"], {"strategic_risk": 0.4, "architectural_impact": 0.4, "team_dependency": 0.3}),
    _m("project", "Fixed a typo in the onboarding tooltip.", "2026-09-02T09:45:00",
       ["refactor"], {"strategic_risk": 0.2, "architectural_impact": 0.1, "team_dependency": 0.1}),
    _m("project", "Upgraded a dev dependency, patch version.", "2026-07-30T13:20:00",
       ["refactor"], {"strategic_risk": 1.0, "architectural_impact": 0.5, "team_dependency": 0.5}),

    # --- Life Autobiographer -------------------------------------------------
    _m("life", "Got married in Tuscany surrounded by family.", "2024-06-22T16:00:00",
       ["family"], {"emotional_significance": 9.9, "relationship_impact": 9.8, "life_milestone_tier": 10.0}),
    _m("life", "Started a new job as an engineering lead.", "2025-02-03T09:00:00",
       ["career"], {"emotional_significance": 7.5, "relationship_impact": 3.0, "life_milestone_tier": 7.0}),
    _m("life", "Dad recovered well after his heart surgery.", "2025-11-30T19:00:00",
       ["family", "health"], {"emotional_significance": 9.0, "relationship_impact": 8.5, "life_milestone_tier": 6.0}),
    _m("life", "Two-week trip through Japan with old friends.", "2023-10-11T08:00:00",
       ["travel"], {"emotional_significance": 6.5, "relationship_impact": 5.0, "life_milestone_tier": 4.0}),
    _m("life", "Watched TV on the couch after work.", "2026-09-03T21:00:00",
       ["health"], {"emotional_significance": 0.3, "relationship_impact": 0.1, "life_milestone_tier": 0.0}),
    _m("life", "Went for a routine morning jog.", "2026-08-27T07:00:00",
       ["health"], {"emotional_significance": 0.8, "relationship_impact": 0.1, "life_milestone_tier": 0.0}),
    _m("life", "Grabbed coffee with a coworker.", "2026-08-25T10:30:00",
       ["career"], {"emotional_significance": 0.6, "relationship_impact": 0.5, "life_milestone_tier": 0.0}),
    _m("life", "Booked a weekend cabin getaway.", "2026-07-18T12:00:00",
       ["travel"], {"emotional_significance": 2.5, "relationship_impact": 2.0, "life_milestone_tier": 0.5}),
]


def main() -> None:
    try:
        settings = load_settings()
    except RuntimeError as e:
        sys.exit(str(e))

    client = get_client(settings)
    mem, sch = memories(client), schemas(client)

    print("Seeding dimension schemas...")
    sch.delete_many({})
    sch.insert_many([{"_id": d, "version": 1, **s} for d, s in SCHEMAS.items()])
    print(f"  wrote {len(SCHEMAS)} schemas")

    print(f"Embedding {len(RAW_MEMORIES)} memories via {settings.model} ({settings.embed_endpoint})...")
    vectors = embed_batch([m["text"] for m in RAW_MEMORIES], settings, input_type="document")
    docs = [{**m, "embedding": v} for m, v in zip(RAW_MEMORIES, vectors)]

    print("Inserting memories...")
    mem.delete_many({})
    mem.insert_many(docs)
    print(f"  inserted {len(docs)} memories")

    print("Ensuring vector search index...")
    if ensure_index(mem):
        print("  created 'memory_index' (Atlas builds it asynchronously)")
    else:
        print("  index 'memory_index' already exists")

    print("Done.")


if __name__ == "__main__":
    main()
