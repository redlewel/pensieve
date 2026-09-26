"""End-to-end proof: similarity-only vs. Pensieve weighted recall.

Waits for the vector index to be queryable, then runs the *same* query two ways
against the *same* domain-scoped candidates — differing only in whether domain
weights are applied — so you can see importance change the ranking.

Run:  python demo_recall.py
"""

import sys
import time

from pensieve.config import load_settings
from pensieve.db import get_client, memories
from pensieve.embeddings import embed_one
from pensieve.scoring import build_recall_pipeline

QUERY = "how's my spending?"
DOMAIN = "wealth"
DIMENSION = "financial_impact"
HALF_LIFE = 3650


def wait_ready(mem) -> bool:
    for _ in range(40):
        ix = list(mem.list_search_indexes())
        if ix and ix[0].get("queryable"):
            return True
        time.sleep(5)
    return False


def run(mem, qvec, w_domain_bias):
    # Both columns are domain-scoped (use_gates=True); only the weight bias
    # differs, so the comparison isolates the effect of importance weighting.
    pipeline = build_recall_pipeline(
        query_vector=qvec, use_gates=True, domain=DOMAIN, dimension=DIMENSION,
        half_life_days=HALF_LIFE, w_domain_bias=w_domain_bias, w_reinforce=0.0, limit=5,
    )
    return list(mem.aggregate(pipeline))


def main() -> None:
    settings = load_settings()
    mem = memories(get_client(settings))

    print("waiting for the vector index to build...", flush=True)
    if not wait_ready(mem):
        sys.exit("index not queryable yet — give it another minute and re-run")

    qvec = embed_one(QUERY, settings)
    print(f"\nQuery: {QUERY!r}   (domain={DOMAIN}, bias={DIMENSION})\n")

    for label, bias in [("NAIVE  — similarity only", 0.0),
                        ("PENSIEVE — + importance weight", 0.15)]:
        print(f"--- {label} ---")
        for r in run(mem, qvec, bias):
            fi = r["domain_weights"].get(DIMENSION, 0)
            print(f"  {r['final_score']:6.3f}  [{DIMENSION}={fi:>4}]  {r['text']}")
        print()


if __name__ == "__main__":
    main()
