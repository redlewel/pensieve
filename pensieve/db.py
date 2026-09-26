"""MongoDB client + collection handles + vector index definition."""

from __future__ import annotations

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from .config import (DB_NAME, EMBED_DIM, INDEX_NAME, MEMORIES_COL, SCHEMAS_COL,
                     Settings)


def get_client(settings: Settings, **kwargs) -> MongoClient:
    return MongoClient(settings.atlas_uri, **kwargs)


def memories(client: MongoClient) -> Collection:
    return client[DB_NAME][MEMORIES_COL]


def schemas(client: MongoClient) -> Collection:
    return client[DB_NAME][SCHEMAS_COL]


def index_model() -> SearchIndexModel:
    return SearchIndexModel(
        name=INDEX_NAME,
        type="vectorSearch",
        definition={"fields": [
            {"type": "vector", "path": "embedding",
             "numDimensions": EMBED_DIM, "similarity": "cosine"},
            # gate fields — declared here so they can pre-filter inside $vectorSearch
            {"type": "filter", "path": "domain"},
            {"type": "filter", "path": "categories"},
            {"type": "filter", "path": "timestamp"},
        ]},
    )


def ensure_index(col: Collection) -> bool:
    """Create the vector index if absent. Returns True if created."""
    existing = {ix["name"] for ix in col.list_search_indexes()}
    if INDEX_NAME in existing:
        return False
    col.create_search_index(model=index_model())
    return True
