# Deploying to Vercel

The FastAPI app deploys as a Vercel Function. It's serverless-friendly by design:
all state lives in MongoDB Atlas (reinforcement counts included), there are no
background jobs in the request path, and nothing writes to the local filesystem.

## 0. Monorepo — set the Root Directory
All backend code lives in **`backend/`**. In **Vercel → Project → Settings → General →
Root Directory**, set it to **`backend`**. Vercel then treats `backend/` as the project
root and finds `index.py`, `requirements.txt`, and `vercel.json` there. Everything below
is relative to `backend/`.

```
repo/
  backend/        ← Vercel Root Directory = this
    index.py  vercel.json  requirements.txt  pensieve/  seed.py  …
  frontend/       ← (your partner's app; its own Vercel project or root dir)
  SPEC.md  DEPLOY.md
```

## 1. Entrypoint
`backend/index.py` re-exports the app so Vercel auto-detects it:

```python
from pensieve.app import app
```

Vercel's FastAPI preset scans `app.py` / `index.py` / `server.py` (relative to the Root
Directory); our real app lives in `pensieve/app.py`, hence the re-export.

## 2. Environment variables
Secrets are **not** deployed — the `*-key.txt` / `*-creds.txt` files are gitignored.
`config.py` reads env first, so set these under **Project → Settings → Environment
Variables**:

| Var | Value | Required |
|-----|-------|----------|
| `MONGO_DB_URI` | `mongodb+srv://…` cluster string | ✅ |
| `MONGODB_MODEL_API_KEY` | Voyage embedding key | ✅ |
| `OPENROUTER_API_KEY` | extraction key (for raw-text `/record`) | for `/record` |
| `MONGODB_ENDPOINT` | `ai.mongodb.com` (default) | — |
| `EXTRACT_MODEL` | `google/gemini-2.5-flash-lite` (default) | — |

> If `MONGO_DB_URI` or `MONGODB_MODEL_API_KEY` is missing, `load_settings()` raises
> at import → the function fails to boot and every route 500s. Set them before deploy.

## 3. Dependencies
`backend/requirements.txt` (pymongo, requests, fastapi, openai). Python 3.12+.

## 4. Deploy
Push to the production branch → prod deploy; PRs get preview URLs. CORS is open
(demo-only — restrict `allow_origins` before any real use).

## Frontend (`Demo-dashboard/`) — a second Vercel project
The Vite/React dashboard deploys as its **own Vercel project from the same repo**:

- **Root Directory → `Demo-dashboard`** (Framework Preset auto-detects Vite; build
  `npm run build`, output `dist/`). No `vercel.json` needed — `vite.config.js` registers
  the React plugin (required for the JSX build).
- **Set env var `VITE_PENSIEVE_API_URL`** = the backend's deployed URL
  (`https://<your-backend>.vercel.app`) in the frontend project's Environment Variables.
  Without it, `src/api.js` uses its built-in default.
- The frontend calls the backend cross-origin; the backend's open CORS permits it. If you
  lock CORS down later, add the frontend's domain to `allow_origins`.
- Deps are **pinned** (Vite 5 / React 18) and `package-lock.json` is committed, so Vercel
  builds reproducibly. (Avoid `"latest"` — it pulled a broken rolldown build.)

Net: **two Vercel projects, one repo** — backend (Root Directory `backend`) and frontend
(Root Directory `Demo-dashboard`), pointed at each other via `VITE_PENSIEVE_API_URL`.

## Serverless gotchas
- **Atlas network access (critical).** Vercel functions have dynamic egress IPs, so
  your local-machine allowlist entry won't cover them. In Atlas → Network Access, add
  `0.0.0.0/0` (allow from anywhere) for the hackathon, or Vercel's documented ranges /
  PrivateLink for production. Without this the deployed function can't reach the
  cluster and every request fails (fast, thanks to the 8s server-selection timeout).
- **Duration limit.** A large bulk `/record` POST fires many LLM extraction calls;
  it can exceed the function timeout. `vercel.json` sets `maxDuration: 60`; keep
  bulk batches small (`bulk_ingest.py --batch 15`) or raise the limit on a paid plan.
- **Local scripts stay local.** `seed.py`, `demo_recall.py`, `bulk_ingest.py`,
  `generate_data.py`, `decay.py`, and `check_connection.py` are tooling — run them from
  `backend/` (`cd backend && python seed.py`) against Atlas, not on Vercel. (Seeding is a
  one-time step; it's already done.)
- **Future `/reindex` backfill must not be an in-process background job here.** Use
  Vercel Cron (or a queue) to invoke a backfill endpoint on a schedule instead.
- **Cold starts** rebuild the Mongo client + settings each time — fine for a demo.
