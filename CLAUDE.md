# CLAUDE.md

Personal job-search assistant: aggregates job postings, scores them for interest/candidate fit via
Claude, tailors resumes/cover letters, and tracks applications. Single user, private repo — holds
real resume content and application history, so nothing here is published anywhere public. See
`/Users/mattroth/.claude/plans/i-want-to-create-typed-cake.md` for the full design plan and milestone
breakdown this was built from.

## Commands

```bash
make serve             # uvicorn on :8000; serves frontend/dist if built
make dev                # Vite dev server (proxies /api to :8000); run `make serve` alongside
make frontend           # npm install + build frontend/dist
make test                # pytest
make ingest-resumes DIR=~/path KIND=resume   # one-time local bulk import (see jobhunt.cli.ingest_resumes)
make deploy              # Cloud Build + Cloud Run (GCP project from git-ignored .deploy.local.mk)

PYTHONPATH=src uv run pytest tests/test_resume_ingestion.py -q
PYTHONPATH=src uv run uvicorn jobhunt.api.main:app --reload --port 8000
```

## Architecture

- `src/jobhunt/db/schema.sql` — full SQLite schema, applied idempotently at startup (`db/connection.py`,
  no migration framework). "Current" profile rows (`candidate_profile`, `preference_profile`) are just
  the latest row by id — every rebuild inserts a new row rather than updating in place, so history is
  free.
- `src/jobhunt/core/llm.py` — shared Claude client + `log_call()`, which writes every LLM call to
  `llm_calls` with an estimated cost (`PRICING_PER_MTOK`, update when Anthropic pricing changes). Every
  module that calls Claude should log through this, not call `anthropic` directly — it's what backs the
  Settings-page usage log.
  - Model default is `claude-sonnet-5` (`JOBHUNT_LLM_MODEL` env override), not Opus. This is a
    deliberate cost/quality tradeoff for a cost-capped personal tool — mirrors the Haiku/Sonnet tiering
    Shelf Life's `chat.py` uses for its own Assistant. Revisit if scoring quality ever disappoints.
- `src/jobhunt/core/resume_parse.py` — pypdf/python-docx/plain-text extraction.
- `src/jobhunt/core/profile.py` — synthesizes `candidate_profile` from every uploaded resume/cover
  letter in one Claude call. Called from `api/resumes.py`'s rebuild endpoint, never automatically.
- `src/jobhunt/api/` — FastAPI route modules, one per resource (`resumes.py` so far; `jobs.py`,
  `digest.py`, `applications.py`, `sources.py`, `generate.py`, `settings.py`, `internal.py` land in
  later milestones per the plan). `api/main.py` wires routers and serves `frontend/dist` if built.
- `frontend/` — React 19 + Vite + TS, react-router-dom for navigation, no UI library. Unlike Shelf
  Life's stateless localStorage-only app, this one's source of truth is server-persisted SQLite —
  pages call `api.ts` directly rather than reading/writing a client-side store.
- Resume/cover-letter storage: `JOBHUNT_RESUME_DIR` (default `data/resumes/`) holds the actual files;
  the DB only stores the path + extracted text. In production this directory is a Cloud Run GCS-volume
  mount, so the code never needs to know it's talking to cloud storage.

## Working conventions

- Small commits, one concern each.
- This repo is **not** on the public-portfolio changelog/prompts-log pattern that `projects/01 - Shelf
  Life` uses — that machinery exists there because Shelf Life's dev history is a public product
  feature. Here it would just be noise (and a privacy problem), so there's no `DECISIONS.md`,
  `prompts/`, or Changelog page. Keep commit messages and this file as the record instead.
- Resumes/cover letters and the job/application DB are real personal data — never relax `.gitignore`
  for `data/` or commit a `.db` file, even for debugging.

## Deploy (Milestone 6, not yet done)

Cloud Run + Cloud Scheduler, same `gcloud builds submit` → `gcloud run deploy` pattern as Shelf Life,
with two differences: `--no-allow-unauthenticated` (this app must never be publicly reachable — lock it
down with Identity-Aware Proxy, which needs a one-time manual setup in the GCP console beyond what the
Makefile can script: enable the IAP API, configure the OAuth consent screen, enable IAP on the Cloud Run
service, then grant "IAP-secured Web App User" to just your own Google account) and a GCS-bucket volume
mount for `data/` (SQLite file + resume blobs) so it survives restarts, since Cloud Run's local disk is
ephemeral.
