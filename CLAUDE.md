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
make docker              # docker build -t jobhunt . (needs Docker; not verified locally -- see Deploy)
make ingest-resumes DIR=~/path KIND=resume   # one-time local bulk import (see jobhunt.cli.ingest_resumes)
make deploy              # Cloud Build + Cloud Run (GCP project from git-ignored .deploy.local.mk)
make anthropic-secret / scheduler-secret     # once each: push the two secrets to Secret Manager
make scheduler-setup     # create/update the daily Cloud Scheduler job

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
- `src/jobhunt/core/preference.py` — synthesizes `preference_profile` from feedback marks and
  application-status history, with application outcomes weighted as the stronger signal (a job that
  reached `interview`/`offer` says more than a bare `interested` mark). Rebuilt manually from
  Settings, or automatically by the daily pipeline once enough new signal has accumulated.
- `src/jobhunt/core/scoring.py` — the per-job LLM call (interest-fit + candidate-fit, High/Med/Low)
  against the candidate profile, preference profile, and recent raw feedback.
- `src/jobhunt/core/generate.py` — tailored resume/cover-letter generation, called only from a
  user action on a specific job (`api/generate.py`), never automatically.
- `src/jobhunt/core/prefilter.py` + `core/queue.py` — the cheap rule-based pass that runs before any
  LLM scoring call, and the priority queue/daily-cap mechanism it feeds (unscored jobs carry over
  with an age bonus so a capped day doesn't starve the backlog).
- `src/jobhunt/core/page_fetch.py` + `core/browser_fetch.py` — fetch a page's readable text, trying
  a plain static GET first and falling back to headless-browser rendering (Playwright) only when
  the static fetch yields too little text. The Playwright fallback is real, not speculative: at
  least one source the user wants (techjobsforgov.com, a WP Job Manager board) only renders its
  listings via client-side JS, confirmed by fetching it directly during development.
- `src/jobhunt/core/dedupe.py` — `compute_dedupe_key` prefers `(source_id, external_id)` when an
  adapter provides one, falling back to a normalized URL, then normalized company+title+location.
  The external_id preference matters: Greenhouse-shaped URLs carry the real job id in the query
  string, which URL normalization strips, so without it every posting on a board would collapse
  onto the same key (this actually happened once, caught via a live test against Stripe's board).
- `src/jobhunt/ingest/adapters/` — one module per source kind (`greenhouse`, `lever`, `ashby`,
  `adzuna`, `remoteok`, `custom_board`). Each `fetch(source: dict) -> list[dict]` returns raw
  posting dicts; the pipeline does dedup/upsert/normalization uniformly across all of them. Every
  adapter's shape was verified against its real live API during development (see each module's
  docstring), not guessed from documentation.
- `src/jobhunt/pipeline/daily_run.py` — orchestrates the full daily cycle (poll → upsert → prefilter
  → capped scoring → digest → conditional preference-profile refresh → run summary). Triggered by
  `POST /internal/run-daily` (Cloud Scheduler in production, a button on the Sources page locally).
- `src/jobhunt/api/` — one FastAPI route module per resource: `resumes`, `jobs`, `digest`,
  `applications`, `sources`, `generate`, `settings`, `internal`. `api/main.py` wires all routers and
  serves `frontend/dist` if built.
- `frontend/` — React 19 + Vite + TS, react-router-dom for navigation, no UI library. Unlike Shelf
  Life's stateless localStorage-only app, this one's source of truth is server-persisted SQLite —
  pages call `api.ts` directly rather than reading/writing a client-side store.
- Resume/cover-letter storage: `JOBHUNT_RESUME_DIR` (default `data/resumes/`) holds the actual files;
  the DB only stores the path + extracted text. In production this directory is a Cloud Run GCS-volume
  mount, so the code never needs to know it's talking to cloud storage.
- Every Claude call (scoring/parsing/generation/profile rebuilds) goes through `core/llm.py`'s
  `log_call()`, which writes to `llm_calls` with an estimated cost — this backs the Settings-page
  usage log. Claude-call failures (including missing credentials) are wrapped by
  `llm.llm_error_boundary()` into a catchable `LLMError` so they surface as a clean 502, not a raw
  500 — every call site uses it.

## Working conventions

- Small commits, one concern each.
- This repo is **not** on the public-portfolio changelog/prompts-log pattern that `projects/01 - Shelf
  Life` uses — that machinery exists there because Shelf Life's dev history is a public product
  feature. Here it would just be noise (and a privacy problem), so there's no `DECISIONS.md`,
  `prompts/`, or Changelog page. Keep commit messages and this file as the record instead.
- Resumes/cover letters and the job/application DB are real personal data — never relax `.gitignore`
  for `data/` or commit a `.db` file, even for debugging.

## Deploy (Milestone 6 — Dockerfile/Makefile ready, not yet actually deployed)

Cloud Run + Cloud Scheduler, same `gcloud builds submit` → `gcloud run deploy` pattern as Shelf Life,
with two differences from that project:

- **`--no-allow-unauthenticated`** — this app must never be publicly reachable. Lock it down with
  Identity-Aware Proxy: enable the IAP API, configure the OAuth consent screen, enable IAP on the
  Cloud Run service, then grant "IAP-secured Web App User" to just your own Google account. This is
  a one-time setup via the GCP console (Security → Identity-Aware Proxy) or `gcloud iap` — the exact
  flags shift as Google iterates on Cloud Run's IAP integration, so follow GCP's current
  documentation at deploy time rather than a command memorized here. **Before relying on it**,
  confirm from a signed-out/incognito browser that the URL actually refuses you.
- **A GCS-bucket volume mount for `data/`** (the SQLite file + resume blobs) via Cloud Run's native
  volume-mount support, so it survives restarts — Cloud Run's local disk is ephemeral otherwise.
  `--max-instances 1` avoids SQLite multi-writer issues entirely (single personal user anyway).

The image also bundles Playwright + Chromium (for `custom_board`'s JS-rendering fallback), which
makes it noticeably bigger than Shelf Life's — expect a slower Cloud Build and a slower cold start.

**Not yet done, and deliberately not run from here without you present**: actually running
`make deploy` creates a real GCP Artifact Registry image and a real Cloud Run service, and `gcloud
builds submit` consumes Cloud Build minutes — all real spend. Needs a GCP project id (`GCP_PROJECT`
in a git-ignored `.deploy.local.mk`, same pattern as Shelf Life), `make anthropic-secret` and `make
scheduler-secret` run once first, then `make deploy`, then the IAP console setup above, then `make
scheduler-setup`.
