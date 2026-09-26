"""Voyage embeddings via ai.mongodb.com.

Same model for ingest (`input_type="document"`) and query (`input_type="query"`)
— vectors from different models are not comparable.
"""

from __future__ import annotations

import requests

from .config import Settings

_BATCH = 128   # Voyage accepts up to 1000 inputs/request; keep it modest


def embed_batch(texts: list[str], settings: Settings,
                input_type: str = "document") -> list[list[float]]:
    """Embed texts, preserving input order."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        chunk = texts[i:i + _BATCH]
        resp = requests.post(
            f"https://{settings.embed_endpoint}/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.embed_key}",
                     "Content-Type": "application/json"},
            json={"model": settings.model, "input": chunk,
                  "input_type": input_type, "output_dimension": settings.dim},
            timeout=60,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        vectors.extend(d["embedding"] for d in data)
    return vectors


def embed_one(text: str, settings: Settings, input_type: str = "query") -> list[float]:
    return embed_batch([text], settings, input_type)[0]
