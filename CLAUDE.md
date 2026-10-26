# CLAUDE.md

Personal job-search assistant: aggregates job postings, scores them for interest/candidate fit via
Claude, tailors resumes/cover letters, and tracks applications. Single user, private repo — holds
real resume content and application history, so nothing here is published anywhere public. See
`/Users/mattroth/.claude/plans/i-want-to-create-typed-cake.md` for the full design plan and milestone
breakdown this was built from.

## Commands

All Python runs through uv (Python 3.12). The Makefile exports `PYTHONPATH=src`. On macOS, uv's
editable-install `.pth` file sometimes gets the "hidden" flag, which makes Python skip it (same
issue Shelf Life's CLAUDE.md notes) — `import jobhunt` then fails outside `make` with
`ModuleNotFoundError: No module named 'jobhunt'`. Outside `make`, prefix commands with
`PYTHONPATH=src`.

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

- `src/jobhunt/db/schema.sql` — full SQLite schema, applied idempotently at startup
  (`db/connection.py`). `CREATE TABLE IF NOT EXISTS` never alters an existing table, so a new column
  on one needs an entry in `connection.py`'s `MIGRATIONS` list (plain `ALTER TABLE ... ADD COLUMN`,
  duplicate-column errors swallowed — safe to run on every startup). "Current" profile rows
  (`candidate_profile`, `preference_profile`) are just the latest row by id — every rebuild inserts a
  new row rather than updating in place, so history is free.
- `src/jobhunt/core/llm.py` — shared Claude client + `log_call()`, which writes every LLM call to
  `llm_calls` with an estimated cost (`PRICING_PER_MTOK`, update when Anthropic pricing changes). Every
  module that calls Claude should log through this, not call `anthropic` directly — it's what backs the
  Settings-page usage log.
  - Model default is `claude-sonnet-5` (`JOBHUNT_LLM_MODEL` env override), not Opus. This is a
    deliberate cost/quality tradeoff for a cost-capped personal tool — mirrors the Haiku/Sonnet tiering
    Shelf Life's `chat.py` uses for its own Assistant. Revisit if scoring quality ever disappoints.
- `src/jobhunt/core/resume_parse.py` — pypdf/python-docx/plain-text extraction. The docx path walks
  paragraphs *and* tables in document order (`_iter_block_items`) -- several real resumes use a
  table-based two-column layout (dates/company in one cell, bullets in another), and python-docx's
  `.paragraphs` alone silently skips table content entirely (confirmed live: several real files
  extracted ~100 characters instead of ~5000 before this was fixed).
- `src/jobhunt/cli/ingest_resumes.py` — re-runnable recursive bulk import from a real local folder
  (`make ingest-resumes DIR=~/Documents/Resumes`). Handles the mess real folders actually have:
  Word lock files (`~$*`) and `.DS_Store` skipped; prefers `.docx` over `.pdf` for the same stem
  (cleaner text, and it's what the docx-template generator can actually use); dedupes by content
  hash across reruns, not filename (catches "... copy.pdf" and renamed duplicates); classifies
  `kind` by filename ("cover" -> cover_letter), `is_archived` by a whole-word "old" in any path
  segment (so "Cold Outreach" doesn't false-positive), and `is_template` by "template" in the
  filename. Persists the root folder as the `resume_library_path` setting.
- `src/jobhunt/core/profile.py` — synthesizes `candidate_profile` from every uploaded resume/cover
  letter in one Claude call, ordered oldest-to-newest by `source_modified_at` (the *source file's*
  mtime, not when it was uploaded into the app -- those are unrelated) with `is_archived` docs
  tagged `[archived]` inline so Claude weighs current documents far more heavily, per the user's
  explicit ask, rather than treating every resume as equally representative of their current
  skills/preferences. Called from the Profile page's rebuild action, never automatically.
  `set_candidate_profile()` inserts a user-edited profile directly (no LLM call, `model` recorded
  as `manual_edit`), backing the Profile page's inline edit so the user can fix or add to what the
  LLM produced without a full re-synthesis from resumes.
- `src/jobhunt/core/docx_writer.py` — generates a tailored resume/cover-letter `.docx` by cloning
  the user's own template document's paragraph styles and exact per-run formatting (bold/size),
  not by describing formatting to Claude in prose (which can't produce real visual fidelity) and
  not by building a style sheet from scratch (which would drift from what the user actually uses).
  Validated live against a real resume: deleting all body paragraphs and re-adding clones of a
  handful of "exemplar" paragraphs survives a save/reload round trip with margins and named styles
  intact -- but font *size* is often set per-run, not just inherited from the style, so it has to
  be captured from an exemplar run and reapplied explicitly. Clearing a paragraph's old text also
  has to strip every child element, not just `.runs` -- real documents wrap text in `<w:hyperlink>`
  (tracked-change and bookmark wrappers are the same issue), which python-docx's `.runs` doesn't
  enumerate, so a naive clear leaves that text behind to leak into the new content (caught live: a
  hyperlinked word, "thesis", survived into an unrelated generated cover letter before this fix).
- `src/jobhunt/core/generate.py` — resume/cover-letter generation picks a template via
  `is_template` first (an explicit "template.docx" filename overrides the recency default), then
  most-recent non-archived, and only falls back to plain markdown text (no saved file) when the
  picked document has no usable `.docx` on disk (e.g. only ever uploaded as a PDF). The saved
  file's directory is the `resume_library_path` setting's `AI Generated` subfolder, or
  `data/resumes/generated/` if that setting isn't configured. Output is `.docx` only, deliberately
  -- `docx2pdf` (Word automation via AppleScript) was tried live and does not work on this machine's
  Word install (opens Word, hangs ~30s, fails with "Message not understood"); don't re-attempt
  automated PDF export without LibreOffice or a different conversion path.
- `src/jobhunt/core/preference.py` — synthesizes `preference_profile` from feedback marks and
  application-status history, with application outcomes weighted as the stronger signal (a job that
  reached `interview`/`offer` says more than a bare `interested` mark). Rebuilt manually from
  Settings, or automatically by the daily pipeline once enough new signal has accumulated.
- `src/jobhunt/core/scoring.py` — the per-job LLM call (interest-fit + candidate-fit, High/Med/Low)
  against the candidate profile, preference profile, recent raw feedback, desired sectors, desired
  titles, and minimum salary. Titles and salary are passed as explicit guidance, not a filter: the
  prompt tells the model a differently-worded but equivalent/adjacent title (e.g. "VP of Analytics"
  for a stated "Analytics Director") should be judged on substance, and a posting slightly below
  the salary floor isn't disqualifying on its own. `core/prefilter.py`'s cheap rule-based pass only
  *rewards* a literal title match and never penalizes a miss, for the same reason -- the literal
  string check can't tell "equivalent" from "different," so it shouldn't be allowed to sink a good
  job's scoring priority before the LLM (which can) ever sees it.
- Desired sectors (`settings.prefilter_sectors`, edited on the Profile page's "What are you looking
  for?" card) feed three places: `core/prefilter.py`'s rule-based scorer (boost on a text match,
  penalty on a clear miss -- unlike titles, sector is treated as a harder constraint per the user's
  explicit ask, not pure guidance), `core/discovery.py`'s company-suggestion prompt (treated as a
  real constraint, not a tiebreaker — a strong brand outside the sector shouldn't get suggested
  anyway), and `core/scoring.py` above.
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
- `src/jobhunt/core/company_resolver.py` — turns a company name into an actual monitorable source:
  cheap direct probes against the three ATS APIs first (confirmed live that "anthropic"/"figma"
  resolve immediately this way, no LLM needed), falling back to a Claude + web-search call only
  when none of the slug guesses pan out. Shared by favorites and by approving a discovery suggestion.
- `src/jobhunt/core/favorites.py` — favorite companies (`favorite_companies` table, intent-level:
  a favorite can exist with no resolved source yet, or a failed one with a retry option).
  `is_favorite_company_name()` drives two things: `pipeline/daily_run.py` bypasses pre-filter
  excludes and the daily cap entirely for a favorite's postings (`core/queue.py`'s `select_batch`
  always includes all pending favorite jobs, on top of the capped batch for everyone else), and
  `core/digest.py` flags + force-ranks them first regardless of score.
- `src/jobhunt/core/discovery.py` — `suggest_companies()` suggests single companies to monitor, and
  `suggest_job_boards()` suggests multi-employer boards (niche/sector boards, regional boards,
  aggregators) -- the two are deliberately separate prompts: a board suggestion needs to describe a
  *listing* worth polling repeatedly, not a single employer, and the model is told not to suggest
  a generic board (LinkedIn/Indeed) that adds no reach over what the user can already search. Both
  return suggestions only; nothing is persisted until the user approves one from the Profile page.
  Approving a company calls `company_resolver` via `POST /api/sources/resolve` (or `/api/favorites`
  to also favorite it); approving a board goes straight to `POST /api/sources` with
  `kind=custom_board`, since the board's URL is already known from the suggestion.
- `src/jobhunt/api/` — one FastAPI route module per resource: `resumes`, `jobs`, `digest`,
  `applications`, `sources`, `generate`, `settings`, `internal`, `favorites`, `discovery`.
  `api/main.py` wires all routers and serves `frontend/dist` if built.
- `frontend/` — React 19 + Vite + TS, react-router-dom for navigation, no UI library. Unlike Shelf
  Life's stateless localStorage-only app, this one's source of truth is server-persisted SQLite —
  pages call `api.ts` directly rather than reading/writing a client-side store.
- `frontend/src/components/Spinner.tsx` — small CSS-animated spinner (`@keyframes spin` in
  `index.css`), used in place of plain "...ing" button text for any action that calls Claude
  (profile rebuild, company discovery, resume/cover-letter generation, job scoring, favorite
  resolution) or otherwise takes noticeable time (test-fetch, run-daily-now).
- Digest flow: marking a job "Interested" on `DigestPage` records the feedback *and* auto-creates
  an `applications` row (status `saved`, swallowing a 409 if one already exists) -- the user
  shouldn't need a separate "track" step for a job they just said they want. `JobDetailPage` keeps
  a manual "Track application" button as a fallback for reaching the tracker without going through
  digest feedback first (e.g. a job pasted in directly). The digest is also filterable client-side
  by interest fit / candidate fit / favorite-company, reusing the scores already on each entry --
  no new endpoint needed.
- `api.ts`'s `APPLICATION_STATUS_LABELS` is a display-only relabeling (`saved` → "Drafting" is the
  only real change) -- the underlying `ApplicationStatus` enum and DB `CHECK` constraint are
  unchanged (saved/applied/phone_screen/interview/offer/rejected/withdrawn), so this is purely a
  frontend label fix, not a schema migration.
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
