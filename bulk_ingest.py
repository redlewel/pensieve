"""Generate N fake finance-project memories and POST them to /record.

Payloads are raw text with no metadata, so the server auto-extracts each via the
LLM — a real stress test of the concurrent, partial-success ingest path.

Start the server first:  uvicorn pensieve.app:app
Then:
  python bulk_ingest.py -n 100            # post 100 fake memories
  python bulk_ingest.py -n 20 --dry       # print payloads, don't post
  python bulk_ingest.py -n 100 --batch 25 # post in chunks of 25
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import requests

URL_DEFAULT = "http://localhost:8000/record"

# (template, min_amount, max_amount) mixing milestones with everyday noise.
_TEMPLATES = [
    ("Closed on a ${amt:,} property in {place}.", 220_000, 900_000),
    ("Raised a ${amt:,} funding round for the startup.", 50_000, 2_000_000),
    ("Paid off ${amt:,} in {loan} debt.", 4_000, 60_000),
    ("Rolled ${amt:,} from an old 401k into an IRA.", 10_000, 200_000),
    ("Invested ${amt:,} into a broad index fund.", 1_000, 40_000),
    ("Sold some stock for a ${amt:,} gain.", 500, 30_000),
    ("Spent ${amt} on {small} this week.", 3, 80),
    ("Bought a ${amt} {gadget}.", 15, 400),
    ("Auto-renewed a ${amt}.99 {sub} subscription.", 4, 30),
    ("Grabbed ${amt} of coffee and snacks.", 4, 25),
]
_PLACE = ["Austin", "Denver", "the suburbs", "Portland", "Miami"]
_LOAN = ["student", "credit card", "auto", "medical"]
_SMALL = ["lunch", "gas", "parking", "groceries", "a movie"]
_GADGET = ["keyboard", "pair of headphones", "phone case", "desk lamp"]
_SUB = ["streaming", "music", "news", "cloud storage"]


def _fill(t: str, amt: int) -> str:
    return t.format(amt=amt, place=random.choice(_PLACE), loan=random.choice(_LOAN),
                    small=random.choice(_SMALL), gadget=random.choice(_GADGET),
                    sub=random.choice(_SUB))


def generate(n: int, project: str) -> list[dict]:
    out = []
    for _ in range(n):
        tmpl, lo, hi = random.choice(_TEMPLATES)
        out.append({"text": _fill(tmpl, random.randint(lo, hi)), "project": project})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=50, help="number of fake memories")
    ap.add_argument("--project", default="finance")
    ap.add_argument("--url", default=URL_DEFAULT)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="print payloads, don't post")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    memories = generate(args.n, args.project)

    if args.dry:
        for m in memories:
            print(m)
        return

    inserted, failed, t0 = 0, [], time.time()
    for i in range(0, len(memories), args.batch):
        chunk = memories[i:i + args.batch]
        try:
            resp = requests.post(args.url, json=chunk, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            sys.exit(f"POST failed at batch {i // args.batch}: {e}")
        body = resp.json()
        inserted += body.get("inserted", 0)
        failed += body.get("failed", [])
        print(f"  batch {i // args.batch + 1}: +{body.get('inserted', 0)} "
              f"({len(body.get('failed', []))} failed)")

    dt = time.time() - t0
    print(f"\ninserted {inserted}/{args.n} in {dt:.1f}s "
          f"({args.n / dt:.1f}/s); {len(failed)} failed")
    for f in failed[:5]:
        print("  fail:", f)


if __name__ == "__main__":
    main()
