"""One-command smoke test: verify the cluster + embedding creds work.

Read-only — pings the cluster and embeds a single string. Writes nothing.

Run:  python check_connection.py
"""

import sys

from pensieve.config import load_settings
from pensieve.db import get_client
from pensieve.embeddings import embed_one


def main() -> None:
    try:
        settings = load_settings()
    except RuntimeError as e:
        sys.exit(str(e))

    print(f"embedding: {settings.model} @ {settings.embed_endpoint}")

    print("pinging cluster...", end=" ", flush=True)
    try:
        get_client(settings, serverSelectionTimeoutMS=5000).admin.command("ping")
        print("OK")
    except Exception as e:
        sys.exit(f"FAILED\n  {e}\n"
                 "  (check the connection string, DB user/password, and IP allowlist)")

    print("embedding one string...", end=" ", flush=True)
    try:
        dims = len(embed_one("hello world", settings))
    except Exception as e:
        sys.exit(f"FAILED\n  {e}\n  (check the Model API key / endpoint)")
    if dims != settings.dim:
        sys.exit(f"FAILED\n  got {dims} dims, expected {settings.dim}")
    print(f"OK ({dims} dims)")

    print("\nAll good. Next: python seed.py")


if __name__ == "__main__":
    main()
