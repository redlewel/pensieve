"""Shared configuration and credential loading.

Credentials resolve from the environment first, then fall back to the
`mongodb-api-key.txt` file at the repo root. That file carries only the Voyage
embedding key + endpoint; the Atlas cluster connection string (`ATLAS_URI`) is
separate and must be provided via the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Fixed collection / index / model constants — shared by seed.py and the app so
# there is one source of truth.
DB_NAME = "pensieve"
MEMORIES_COL = "memories"
SCHEMAS_COL = "schemas"
INDEX_NAME = "memory_index"
EMBED_MODEL = "voyage-3.5"
EMBED_DIM = 1024              # must equal numDimensions in the vector index


@dataclass(frozen=True)
class Settings:
    atlas_uri: str
    embed_key: str
    embed_endpoint: str
    model: str = EMBED_MODEL
    dim: int = EMBED_DIM


# KEY=VALUE credential files at the repo root (gitignored).
_KEYFILES = ("mongodb-api-key.txt", "mongodb-cluster-creds.txt")


def _read_keyfiles() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    out: dict[str, str] = {}
    for name in _KEYFILES:
        path = root / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = (s.strip() for s in line.split("=", 1))
                out[k] = v
    return out


def load_settings() -> Settings:
    kf = _read_keyfiles()
    key = os.environ.get("MONGODB_MODEL_API_KEY") or kf.get("MONGODB_MODEL_API_KEY")
    endpoint = (os.environ.get("MONGODB_ENDPOINT")
                or kf.get("MONGODB_ENDPOINT", "ai.mongodb.com"))
    # cluster URI: env (ATLAS_URI or MONGO_DB_URI) then the creds file
    atlas = (os.environ.get("ATLAS_URI") or os.environ.get("MONGO_DB_URI")
             or kf.get("MONGO_DB_URI") or kf.get("ATLAS_URI"))

    missing = [name for name, val in
               (("MONGODB_MODEL_API_KEY", key),
                ("cluster URI (ATLAS_URI / MONGO_DB_URI)", atlas)) if not val]
    if missing:
        raise RuntimeError(
            "Missing credentials: " + ", ".join(missing)
            + ". Provide via environment or the *-creds.txt files at the repo root."
        )
    return Settings(atlas_uri=atlas, embed_key=key, embed_endpoint=endpoint)
