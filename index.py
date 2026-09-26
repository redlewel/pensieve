"""Vercel entrypoint.

Vercel's FastAPI preset scans app.py / index.py / server.py (root, src/, app/)
for a module-level `app`. Our real app lives in pensieve/app.py, so re-export it
from a filename Vercel auto-detects.
"""

from pensieve.app import app  # noqa: F401  (re-exported for Vercel)
