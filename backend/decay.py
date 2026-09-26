"""Run the decay task once against the cluster (local/manual trigger).

Rewrites every memory's `current_weights` from its base `domain_weights` and age.
The same logic runs on a schedule via GET /decay (Vercel Cron).

Run:  python decay.py
"""

from pensieve.config import load_settings
from pensieve.db import get_client, memories, schemas
from pensieve.decay import run_decay


def main() -> None:
    s = load_settings()
    client = get_client(s)
    result = run_decay(memories(client), schemas(client))
    print(f"decayed {result['updated']} memories: {result['per_project']}")


if __name__ == "__main__":
    main()
