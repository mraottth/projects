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
make serve-sandbox     # same on :8001 against a throwaway data/sandbox/ (empty -> onboarding); real data untouched
make reset-sandbox     # wipe data/sandbox/ to start onboarding over
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
- `src/jobhunt/core/docx_writer.py` — resumes and cover letters use two deliberately different
  strategies. Cover letters (`write_cover_letter_docx`) do positional in-place text replacement on
  the template's own existing paragraphs -- nothing is deleted/rebuilt, every paragraph keeps its
  own style/run formatting exactly. Resumes (`write_resume_edits`) are edited in place too, via a
  small set of context-anchored operations (`ResumeEditOp`: `replace`/`insert_after`/`delete`,
  each anchored to one paragraph's *exact* existing text) applied as native Word tracked changes
  (`<w:ins>`/`<w:del>`, including marking the paragraph *mark* itself inserted/deleted so Accept/
  Reject in Word cleanly adds/removes whole lines). Every paragraph not targeted by an op is left
  byte-for-byte untouched. An earlier version rebuilt the entire resume from scratch out of ~6
  cloned "exemplar" paragraphs (one per role, captured once from the template) -- that's what
  caused consistently bad formatting: every bullet in the whole document ended up sharing one
  global list-numbering definition (`numPr`/`numId`) regardless of how many distinct lists the
  real resume had, and tables were silently orphaned (not deleted, not populated). Replaced
  entirely in favor of the anchored-edit model. `write_resume_edits` resolves every op's anchor
  against the template *before* mutating anything (`AnchorNotFoundError`/`AmbiguousAnchorError`
  on zero/multiple matches, `DisallowedAnchorError` for a `title`/`contact` anchor) -- a plan with
  one bad op writes nothing to disk. New/replaced resume text inherits its font name+size from the
  *specific* neighboring run it's replacing/following (`<w:rPr>` deep-copy), not a hardcoded font
  -- this generalizes correctly to any inconsistency in the user's own template, unlike the old
  rebuild-from-scratch path's hardcoded `GENERATED_FONT = "Calibri"` (kept only for
  `write_cover_letter_docx`, which still clones/rebuilds small pieces and needs a fallback since
  it isn't editing arbitrary existing runs in place the way the resume path now does). Clearing a
  cover-letter paragraph's old text has to strip every child element, not just `.runs` -- real
  documents wrap text in `<w:hyperlink>` (tracked-change/bookmark wrappers are the same issue),
  which python-docx's `.runs` doesn't enumerate, so a naive clear leaves that text behind (caught
  live: a hyperlinked word, "thesis", survived into an unrelated generated cover letter before
  this fix).
- `src/jobhunt/core/generate.py` — `extract_addressable_lines()` tags every non-blank paragraph in
  the resume template with a structural role (title/contact/heading/job_header/bullet/description/
  normal) plus which section/job it falls under; this is what gets shown to Claude (instead of the
  old approach of pasting the reference resume's raw flattened text) so it can anchor edits to
  exact existing lines rather than generating a resume from scratch. `generate_resume()` picks a
  template via `is_template` first (an explicit "template.docx" filename overrides the recency
  default), then most-recent non-archived, and only falls back to plain markdown text (no saved
  file) when the picked document has no usable `.docx` on disk (e.g. only ever uploaded as a PDF)
  -- `python-docx` can only read `.docx`, not PDF, so a web-uploaded current resume that's a PDF
  can never use the anchored-edit path at all, making the user's current resume being a `.docx` a
  hard requirement here, not just a convenience. `AnchorNotFoundError`/`AmbiguousAnchorError`/
  `DisallowedAnchorError` from `write_resume_edits` are deliberately *not* caught alongside
  `TemplateError`/generic `Exception` in `generate_resume()` -- they mean "Claude's proposed edits
  don't line up with this template" (a failed generation attempt that must surface as a real
  error, `api/generate.py` turns it into a 502), not "template unusable, degrade to plain text
  silently" (the actual meaning of those other two cases). `content_text` (the API response's
  always-present text preview) is computed by applying the validated edit plan to the extracted
  lines in pure Python and rendering that, rather than re-reading the output `.docx`'s
  tracked-changes XML back into text -- python-docx's `.text` only sees direct `<w:r>` children
  and would conflate `<w:delText>` with surviving text, the same blind spot as the hyperlink issue
  above. The saved file's directory is the `resume_library_path` setting's `AI Generated`
  subfolder, or `data/resumes/generated/` if that setting isn't configured. Output is `.docx` only,
  deliberately -- `docx2pdf` (Word automation via AppleScript) was tried live and does not work on
  this machine's Word install (opens Word, hangs ~30s, fails with "Message not understood"); don't
  re-attempt automated PDF export without LibreOffice or a different conversion path. A `.docx`
  with tracked changes opens fine in Google Docs too, which renders Word's tracked changes as its
  own Suggesting-mode review UI on import (a long-standing, unrelated Google Docs capability) --
  considered building real Google Docs API suggested-edits support instead (Google added a native
  `writeControl.writeMode: "SUGGEST"` to `batchUpdate`, GA since 2026-09-30, works with personal
  accounts), but chose this local-`.docx` approach to avoid a new OAuth/Drive integration for the
  same end-user review experience.
- `src/jobhunt/core/preference.py` — synthesizes `preference_profile` from feedback marks and
  application-status history, with application outcomes weighted as the stronger signal (a job that
  reached `interview`/`offer` says more than a bare `interested` mark). Rebuilt manually from
  Settings, or automatically by the daily pipeline once enough new signal has accumulated.
- `src/jobhunt/core/scoring.py` — the per-job LLM call (interest-fit + candidate-fit, High/Med/Low,
  at `SCORING_EFFORT = "medium"` for speed -- raise it if ratings start missing nuance)
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
  `POST /internal/run-daily` (Cloud Scheduler, which sends `X-Scheduler-Secret`) or
  `POST /api/pipeline/run-daily` (`api/pipeline.py`, the UI's "Run now" on Manage Sources and
  onboarding's last step). The UI can't use `/internal/run-daily`: it requires the secret whenever
  `SCHEDULER_SECRET` is set -- always, in production -- so the browser got 403 "invalid or missing
  scheduler secret" there (caught live on onboarding's last step). `/api/pipeline/run-daily` relies
  on IAP for access control like every other `/api/*` route.
  A run used to do everything one at a time, and onboarding's last step (which waits on a whole
  run) took many minutes live. Now sources are fetched `POLL_CONCURRENCY` (6) at a time and jobs
  scored `SCORING_CONCURRENCY` (4) at a time. Scoring is kept low because a new API account's
  per-minute limits are low; the SDK retries a stray 429. `_upsert_job` and `add_to_digest` (which
  re-ranks the whole day per call) stay sequential in the main thread: a concurrent
  select-then-insert or re-rank could race. `run_daily(cap=)` / `?cap=N` overrides
  `daily_llm_cap` for one run; onboarding passes 20 (`FIRST_RUN_CAP`), and the rest stay queued.
  `GET /api/pipeline/progress` exposes an in-process progress dict (fine with one Cloud Run
  instance) that the finish step polls every 2 s to show "Scored 12 of 20 jobs...".
- `src/jobhunt/core/company_resolver.py` — turns a company name into an actual monitorable source:
  cheap direct probes against the three ATS APIs first (confirmed live that "anthropic"/"figma"
  resolve immediately this way, no LLM needed), falling back to a Claude + web-search call only
  when none of the slug guesses pan out. Now only a fallback for a discovery suggestion that came
  without a `careers_url` (`POST /api/sources/resolve`): live, organizations Monitored during
  onboarding never saved while boards did. Many real organizations aren't on the three ATSes, and
  this path was fragile: a probe failing with anything but an HTTP error crashed the request, and
  the web-search fallback ran at default effort with no `pause_turn` handling and raised on an
  unparseable answer. Now a probe failure of any kind just means "next guess", the fallback uses
  `discovery.run_web_search()` (low effort, resumes `pause_turn`), and an unparseable answer counts
  as not found.
- `src/jobhunt/core/discovery.py` — `suggest_companies()` suggests single companies to monitor, and
  `suggest_job_boards()` suggests multi-employer boards (niche/sector boards, regional boards,
  aggregators) -- the two are deliberately separate prompts: a board suggestion needs to describe a
  *listing* worth polling repeatedly, not a single employer, and the model is told not to suggest
  a generic board (LinkedIn/Indeed) that adds no reach over what the user can already search. Both
  return suggestions only; nothing is persisted until the user clicks Monitor on one (Manage Sources
  or onboarding). Approving a company calls `company_resolver` via `POST /api/sources/resolve`;
  approving a board goes straight to `POST /api/sources` with `kind=custom_board`, since the
  board's URL is already known from the suggestion. `api/discovery.py` passes already-monitored company names and board
  URLs (by hostname, so a path/protocol difference doesn't defeat the match) as an exclude list --
  both in the prompt (so Claude doesn't waste a suggestion slot on them) and as a belt-and-suspenders
  post-filter on the response, since prompt instructions alone aren't guaranteed to be followed.
  Company suggestions include `careers_url` (the jobs page Claude found while searching, or null),
  so Monitor adds it directly via `sourceUrl.ts`'s `detectSourceFromUrl()` (a Greenhouse/Lever/
  Ashby URL becomes that ATS, anything else a `custom_board`), the way a board suggestion already
  works. Only a suggestion without one goes through `company_resolver`, and a failure there tells
  the user to add it by URL. Both share `run_web_search()` (formerly `_run_discovery`), tuned for speed after a live run took several minutes and
  "Discover job boards" came back with nothing: `output_config.effort` is `low` (the default, high,
  reasoned at length between up to five filtered web searches -- "find a handful of real, relevant
  names" doesn't need that) and `max_uses` is 3. It resumes a `pause_turn` (a long server-tool
  turn can pause before the final JSON is written; parsing the paused turn finds no JSON), up to
  `MAX_CONTINUATIONS`, and `_extract_json()` parses the *last* `{"suggestions": ...}` object rather
  than greedily matching first-`{`-to-last-`}`, so a stray `{...}` quoted in the prose can't break
  the parse. The frontend (`components/Discovery.tsx`) says a search usually takes 1-2 minutes and
  shows a "no new suggestions" message on an empty result rather than rendering nothing.
  Each search can be steered: an optional `guidance` ("Steer this search": free text like
  "mission-driven tech startups", made the main focus of that one search) and/or `like` ("More
  like this" on a suggestion, or "Find more like this" on a tracked source -- name plus URL when
  there is one). Both are an optional JSON body on `/api/discovery/suggest[-boards]`, appended by
  `_search_direction()` after the profile; neither is saved. Manage Sources hands a tracked
  source's seed to the right Discover card through a `seed` prop (`DiscoverySeed`, with a `nonce`
  so asking twice re-runs) and scrolls to it.
- **Favorite was removed** (favorite companies and favorite sources, which bypassed the pre-filter
  and daily cap and always ranked first) at the user's request -- every source is now just a
  monitored source. `schema.sql` no longer has `sources.is_favorite`, `scoring_queue.is_favorite`,
  or `favorite_companies`, but a database created before that (the live one) still has all three as
  unused leftovers; nothing reads them, and a `SELECT *` from `sources` there still returns an
  `is_favorite` field the frontend ignores. Don't reintroduce a cap/rank bypass without being asked.
- `src/jobhunt/api/` — one FastAPI route module per resource: `resumes`, `jobs`, `digest`,
  `applications`, `sources`, `generate`, `settings`, `internal`, `pipeline`, `discovery`, `onboarding`.
  `api/main.py` wires all routers and serves `frontend/dist` if built.
- `frontend/` — React 19 + Vite + TS, react-router-dom for navigation, no UI library. Unlike Shelf
  Life's stateless localStorage-only app, this one's source of truth is server-persisted SQLite —
  pages call `api.ts` directly rather than reading/writing a client-side store.
- `frontend/src/components/Spinner.tsx` — small CSS-animated spinner (`@keyframes spin` in
  `index.css`), used in place of plain "...ing" button text for any action that calls Claude
  (profile rebuild, company discovery, resume/cover-letter generation, job scoring, company
  resolution) or otherwise takes noticeable time (test-fetch, run-daily-now).
- **Page split**, six pages each with one job (plus the first-run `/onboarding` wizard, below), nav order New & Noteworthy → All Jobs → My Jobs →
  Manage Sources → Profile → Settings: `DigestPage` (route `/`, nav label "New & Noteworthy") is
  today's capped, LLM-scored digest only. `AllJobsPage` (route `/all-jobs`) is
  `GET /api/jobs/ranked` (`api/jobs.py`, registered *before* `/{job_id}` since Starlette matches
  path templates in order -- a later `/ranked` would 404 as an invalid `job_id`) -- every active
  job from a tracked source (`source_id NOT NULL`), scored or not, ranked best-first, the uncapped
  view the daily cap's 50-job limit would otherwise hide. `ManageSourcesPage` (route `/sources`) is
  purely source management, in two sections -- **Organizations** (tracked single-employer sources,
  then Discover organizations) and **Job boards** (multi-employer boards and the keyword
  aggregators, then Discover job boards) -- then "Add a source" and run-daily-now. A source is an
  organization if it's an `ats_*` kind or has a `company_name` (`isOrganization()`); there's no
  stored type, so this needs no migration. Monitoring a suggestion refreshes the lists (`onAdded`). No job
  listing on this page at all; it used to be a single page with two tabs, split into separate
  pages/routes on request so the management page isn't also a long job feed one click away.
  `MyJobsPage` (route `/tracker`, file `MyJobsPage.tsx`) is the application tracker plus "Add a
  job" (manual paste-in, moved here from Digest) -- pasting and confirming a job scores it and
  immediately creates its `applications` row (`api/jobs.py`'s `confirm_job` calls
  `create_application` directly, swallowing a 409 so re-confirming is a no-op) rather than adding
  it to the digest, since pasting a job you already found *is* the interest signal, not a "new and
  noteworthy" discovery.
- **Onboarding** (`pages/OnboardingPage.tsx`, route `/onboarding`) is a 7-step wizard: current
  resume (required) → "What are you looking for?" → recent cover letter (upload, or "Write one for
  me") → older docs (multi-file, uploaded as archived, kind guessed by the same "cover"-in-filename
  rule as `cli/ingest_resumes.py`) → candidate profile (auto-rebuilt on arrival, then revisable) →
  company/board discovery (auto-run) → first daily run, then the digest. `App.tsx` redirects every
  route there, and hides the nav, only while `GET /api/onboarding/status` says `needs_onboarding`
  = no rows in `resumes` at all *and* the `onboarding_completed` setting is false -- so an existing
  user with documents (including bulk-CLI-imported ones) is never forced through it. Progress is
  the `onboarding_step` setting, written on every step change, so a refresh resumes in place.
  Completed steps in the stepper are buttons back to that step (forward still goes only through
  Next, which enforces the required resume). Returning to the profile step must not re-run the
  build: that's a paid call and would discard the user's revisions. So `goTo("profile")`
  auto-rebuilds only when there's no profile yet or `docsChanged` (set by any upload/generation,
  cleared once a profile is reported).
  "Skip setup" just sets `onboarding_completed`; `DigestPage` shows a "Finish setting up" card
  whenever there's no resume, which covers anyone who skipped. The three reused pieces are shared
  components rather than copies: `components/DesiredRoleForm.tsx` (also the Profile page's card),
  `components/Discovery.tsx` (also Manage Sources; `autoRun` fires on mount), and
  `components/CandidateProfileCard.tsx` (also the Profile page). Every on-mount Claude call there
  (`autoRun`, `autoRebuild`, the finish step's run-daily) is guarded by a `useRef` flag, since
  `main.tsx` uses `StrictMode`, which double-mounts in dev and would otherwise make two paid calls.
  "Write one for me" is `core/generate.py`'s `generate_starter_cover_letter()`: built from the
  current resume's text plus titles/sectors/notes (no candidate profile exists yet at that step,
  and no job), saved as a plain one-paragraph-per-element `.docx` in `RESUME_STORAGE_DIR` and
  inserted as the *current* cover letter, so later per-job generation uses it as its positional
  template. A regenerated draft deletes the previous draft rather than letting it pile up in Old,
  since a discarded AI draft isn't a real past cover letter and shouldn't feed the profile.
- `CandidateProfileCard`'s "Ask Claude to change it" is `POST /api/resumes/profile/revise`
  (`core/profile.py`'s `revise_candidate_profile()`): one call applying the user's plain-language
  change request to the current profile, inserted as a new row like every rebuild/edit.
- `api/applications.py`'s `list_applications()` used to omit `history` entirely (only the
  single-item `_get_application()` joined it), while the frontend's `MyJobsPage` called
  `app.history.map(...)` unconditionally on every row -- crashed the whole page the moment a real
  tracked application existed (caught live: worked fine with zero applications, crashed the first
  time the user actually tracked one). Fixed by having the list endpoint fetch all
  `application_status_history` rows in one extra query and group them by `application_id` in
  Python, rather than reproducing `_get_application()`'s N+1 per-row subquery pattern.
- `components/JobFilters.tsx` is the one shared filter implementation (state, pure `filterJobs()`,
  and the `<JobFilterBar>` UI) used by both the Digest and All Jobs lists -- interest fit,
  candidate fit, source (derived from the jobs actually loaded, not a separate
  API call), minimum salary, posted-within-N-days, a location substring match, and a free-text
  search across title/company/location/summary/full description. All client-side over the already
  -loaded job list, same approach as the earlier fit filters -- no new query-param surface
  on the backend.
- Empty states: `DigestPage` with no entries shows "Browse All Jobs" / "Add more sources"
  buttons; when filters hide every job, both job lists show `FilteredEmptyState` with "Clear all
  filters", which applies `NO_JOB_FILTERS` (truly empty), not `DEFAULT_JOB_FILTERS`. The defaults
  already hide Low fits *and every unscored job* (no fit to match), so after onboarding's 20-job
  first run, All Jobs can look empty under the defaults while holding many jobs.
- `components/ScoredJobCard.tsx` always shows salary and posted date now, "N/A" when either is
  missing, rather than omitting the line entirely -- and "posted" means the job description's own
  `posted_at`, never `first_seen_at` (when our pipeline happened to discover it, which isn't what
  a "date posted" label should mean to the user). Location/salary/posted/source render as small
  color-coded pill chips (`.meta-chip-*` in `index.css`, reusing the existing badge/pill visual
  language) instead of one plain dot-separated text line -- salary in the `--good` green, posted
  in the accent blue, location/source neutral, so the three scan independently instead of blurring
  into one string. The rationale blurb is clamped to 2 lines via a `.clamp-2` CSS class with a
  measured-overflow "Show more" toggle (only rendered when the text actually overflows, via a
  `scrollHeight`/`clientHeight` comparison in `useLayoutEffect`).
- `core/scoring.py`'s prompt now asks the model to wrap the 1-3 most decision-critical phrases in
  its rationale in `==double equals==` (sparingly -- short phrases, not whole sentences);
  `components/Markdown.tsx` renders `==text==` as `<mark>`, styled with `var(--warn-bg)` (the same
  soft amber already used for `badge-med`/`pill-interview`) for a subtle, on-brand highlight rather
  than a jarring bright yellow.
- On both `DigestPage` and `AllJobsPage`, marking a job "Interested" records the feedback *and*
  auto-creates an `applications` row the same way -- `components/ScoredJobCard.tsx` (`JobActions`)
  is the one shared implementation both pages use, so this only exists in one place. `JobDetailPage`
  keeps a manual "Track application" button as a fallback for reaching the tracker without going
  through feedback first. Both job-listing pages are filterable client-side by interest fit /
  candidate fit, reusing the scores already on each entry. Marking "Not interested" on
  `DigestPage` additionally removes the card from view immediately (`JobActions` takes an optional
  `onDismissed` callback, wired up only on `DigestPage`) -- `AllJobsPage` doesn't pass it, so the
  same job stays visible there, since "dismissed from New & Noteworthy" and "gone entirely" are
  different things (see next bullet).
- "New & Noteworthy" (`core/digest.py`) is deliberately narrower than "every job we found":
  `add_to_digest()` now refuses to insert a posting older than `MAX_AGE_DAYS` (90) unless its
  `posted_at` is simply unknown (missing data shouldn't read as "not new"), and `get_digest()`
  additionally excludes any job with a `not_interested` feedback row (`NOT_DISMISSED_SQL`) --
  once dismissed, a job is done surfacing here on any future read of any date, not just removed
  from the current view. Neither exclusion touches `AllJobsPage`'s `GET /api/jobs/ranked`, which
  is explicitly the uncapped, undismissable "everything" view. Default rank order (both
  `add_to_digest()`'s Python sort and `list_ranked_jobs()`'s SQL `ORDER BY`, kept in sync by hand
  since one sorts in Python and the other in SQL): interest+candidate score, then recency (`posted_at` falling back to `first_seen_at`), then
  salary (`salary_max` falling back to `salary_min`).
- `api.ts`'s `APPLICATION_STATUS_LABELS` is a display-only relabeling (`saved` → "Drafting" is the
  only real change) -- the underlying `ApplicationStatus` enum and DB `CHECK` constraint are
  unchanged (saved/applied/phone_screen/interview/offer/rejected/withdrawn), so this is purely a
  frontend label fix, not a schema migration.
- `src/jobhunt/ingest/adapters/custom_board.py` -- `max_tokens` was 4000, too small once a board
  listing page has 20-30 postings' worth of structured fields; Claude's output got cut off
  mid-JSON-string and `messages.parse()` raised a JSON-decode error (caught live against a real
  30-posting board). Raised to 12000 and `MAX_POSTINGS_PER_FETCH` lowered 50→30 for margin.
- A classic React footgun bit three places: `{x.is_favorite && <span>...</span>}` (a flag since
  removed; `is_template`/`is_archived`/`enabled` are the same shape) where the flag comes back from SQLite as the integer `0`, not `false` -- `0 && <JSX>` evaluates to
  `0`, and unlike `false`/`null`/`undefined`, React renders a bare `0` as literal text. Caught live
  as a stray "0" prefixed onto source/resume/job-card names. Fixed by coercing with `!!` everywhere
  a DB-sourced `0 | 1` flag gates JSX with `&&` -- watch for the same pattern on any other
  SQLite-backed boolean field rendered this way.
- Resume/cover-letter storage: `JOBHUNT_RESUME_DIR` (default `data/resumes/`) holds the actual files;
  the DB only stores the path + extracted text. In production this directory is a Cloud Run GCS-volume
  mount, so the code never needs to know it's talking to cloud storage.
- Every Claude call (scoring/parsing/generation/profile rebuilds) goes through `core/llm.py`'s
  `log_call()`, which writes to `llm_calls` with an estimated cost — this backs the Settings-page
  usage log. Claude-call failures (including missing credentials) are wrapped by
  `llm.llm_error_boundary()` into a catchable `LLMError` so they surface as a clean 502, not a raw
  500 — every call site uses it. An out-of-credits 400 ("credit balance is too low") gets its own plain message
  (`OUT_OF_CREDITS_MESSAGE`) pointing at platform.claude.com billing: caught live, when the user had
  paid for a claude.ai plan, which doesn't fund the API, and the raw error read like an app bug.
- `MyJobsPage` is a sortable/filterable/searchable table (Company, Title, Location, Salary, Applied
  Date, Status, Documents), not a card list -- status transitions still go through `StatusSelect`
  (same `api/applications.py` endpoint as before), and Applied Date is directly editable via a
  `type="date"` cell (`UpdateApplicationRequest.applied_at`, new). The status-change auto-set-date
  logic (`update_application()`) now checks `"applied_at" not in updates` before stamping
  `datetime('now')`, so an explicit `applied_at` sent in the same request as a status change isn't
  clobbered. "Applied Date" means what it already meant server-side (first time status left
  `saved`) -- editing it doesn't touch status or history, it's a plain field edit. Resume/cover
  letter generation moved inline (`DocActions`, calling the same `/api/generate/*` endpoints
  `JobDetailPage` uses) with a ✓ prefix once `resume_doc_id`/`cover_letter_doc_id` is set --
  `list_applications()` now selects `j.salary_min/max/currency/period` alongside the existing job
  join so the table has salary without a second request per row.
- `components/ScoredJobCard.tsx`'s `salaryText()`/`salaryValue()` take a narrow structural
  `{salary_min, salary_max, salary_period}` shape rather than the full `ScoredJob` type, so
  `MyJobsPage`'s `Application` rows (which have the same three fields via the join above) can
  reuse them directly instead of a second salary-formatting implementation.
- `CompanyLogo` (in `ScoredJobCard.tsx`) is a best-effort visual: guesses a `.com` domain from the
  company name and requests it from Clearbit's free, keyless logo API, hiding itself via `onError`
  if the guess is wrong (true for most non-trivial company names) or the request fails. No backend
  involvement, no stored state, pure progressive enhancement -- correct for well-known single-word
  company names, silently absent otherwise.
- Job-card meta chips now label the salary and posted-date ones explicitly ("Salary: $120k-150k",
  "Date Posted: 3 days ago") rather than showing bare values -- location and source stay unlabeled
  since their content alone is already self-explanatory, but a bare "N/A" chip needs the label to
  say which field is unknown. Global `input[type="number"]` rules strip the native spinner
  (`-webkit-appearance: none` on the spin-button pseudo-elements, `-moz-appearance: textfield`) --
  without that, Min Salary rendered taller than its sibling selects/inputs in some browsers.
- `JobFilterBar` (`components/JobFilters.tsx`) moved from a horizontal bar above the list to a
  vertical right-rail sidebar, modeled directly on Shelf Life's `FilterBar`/`.recs-layout` pattern:
  `.list-layout` is a CSS grid (`minmax(0,1fr) 300px`, 24px gap), `.list-main` holds the job cards,
  `.list-side` is a `position: sticky` `<aside>` (collapses to a single column with the sidebar
  reordered above the list below 980px) holding the filter `.card`, each control in its own
  `.fsection`/`.flabel` stacked vertically -- same classes the rest of the app already uses for
  form sections, so no new vocabulary. `DigestPage` and `AllJobsPage` both use it; `App.tsx` gives
  those two routes plus `/tracker` a wider `.app-main.wide` (1240px vs. the default 880px) via a
  small `useLocation()`-based check, since a sidebar and a wide table both need more room than the
  prose-oriented default.
- `MyJobsPage`'s table: `.jobs-table tbody tr:nth-child(even)` zebra-stripes rows against
  `var(--page)`, and `StatusSelect` wears the existing `pill-{status}` color classes
  (`.status-select` just adds pill shape/padding on top) instead of a plain unstyled `<select>`, so
  status is scannable by color the same way the status pill already was elsewhere in the app.
  `RemoveButton` calls the new `DELETE /api/applications/{id}` (`api/applications.py`) -- it detaches
  (doesn't delete) any `generated_documents` row still pointing at the application via
  `application_id`, so a resume/cover letter you already generated survives removing the job from
  the tracker; only the tracking row and its status history actually go. No confirmation dialog,
  matching every other delete/remove action already in this app (SourceCard,
  ResumesPage all behave the same way).
- Resume/cover-letter generation had two bugs with the same root cause: `max_tokens` on the
  structured-output calls (`core/generate.py`) was too low (4000 for `ResumeContent`, 2000 for
  `CoverLetterContent`) for realistic content length, so Claude's JSON response sometimes got cut
  off mid-string -- the same failure class as the earlier `custom_board.py` truncation bug. Raised
  to 8000 / 4000 respectively (plain-text fallback paths bumped too, for margin; the resume value
  was later reduced to 4000 after the rewrite to an edit-plan model, whose output is much smaller
  than a whole regenerated resume). But raising
  `max_tokens` only reduces the odds, so the real fix is in `api/generate.py`: a truncated/malformed
  response raises `pydantic.ValidationError`, which **subclasses `ValueError`** -- the resume route's
  `except ValueError` (meant for "no base resume uploaded") was silently swallowing it and returning
  a misleading 400 ("...— upload one on the Profile page first" appended to what was actually a
  parsing failure), while the cover-letter route had no matching `except` clause at all and let it
  fall through as a raw, undiagnosable 500. Both routes now catch `ValidationError` explicitly
  *before* any `except ValueError`, returning a clean 502 that says what actually happened. Diagnosed
  by reproducing locally with the existing `FakeAnthropicClient` test pattern against this user's
  real uploaded `.docx` templates (no API credentials needed) rather than guessing from the error
  text alone -- confirmed the docx-writing code itself was fine for the normal/overflow/underflow
  paragraph-count cases; the bug was entirely in exception handling upstream.
- Two more generation hardening passes, found by continuing to chase "Internal Server Error" past
  the ValidationError fix above: (1) the docx-writing calls in `core/generate.py` (now
  `write_resume_edits()`/`write_cover_letter_docx()`) catch `Exception` broadly (logged via
  `logging`, not silently swallowed), not just `TemplateError` -- a real docx's XML can hit an edge
  case the synthetic test fixtures don't, and that shouldn't throw away content that was already
  expensively generated; falls back to markdown-only, same as a recognized `TemplateError` (the
  resume path additionally re-raises `AnchorNotFoundError`/`AmbiguousAnchorError`/
  `DisallowedAnchorError` instead of swallowing them into that fallback -- see `core/generate.py`
  above). (2) `api/generate.py`'s post-generation
  `INSERT INTO generated_documents` had **no** exception handling at all -- a stale
  `application_id` (e.g. the job was removed from My Jobs between the frontend loading its state
  and the user clicking Generate -- very plausible right after the DELETE-application feature
  shipped) hits the FK constraint and raised an unhandled `sqlite3.IntegrityError`. Now a shared
  `_save_generated_document()` helper drops a stale `application_id` before the insert
  (`_resolve_application_id`) rather than letting it fail, and catches `sqlite3.Error` generally
  around the write, returning an actual error instead of a bare 500.
- `core/digest.py`'s `rank` column only gets recomputed when a *new* job is added to that date's
  digest (`add_to_digest()` re-sorts and rewrites rank for every row currently in the table) -- a
  feedback mark or anything else that should change what's excluded/reordered doesn't trigger that,
  so the stored rank can silently go stale (caught live: real digest entries ranked 1/3/7/10 by
  insertion history, completely out of score order). Fixed by extracting the sort key into
  `_rank_key()` and having `get_digest()` sort its results with it live, rather than trusting
  `ORDER BY de.rank` -- the stored rank is now informational/historical only, never the source of
  truth for display order.
- `core/generate.py`'s cover-letter prompt explicitly says to use the candidate's real name (e.g.
  "Matt Roth") when a signature line needs one, not a surname-only form like "Roth" -- the
  candidate profile used to call the user by surname (third-person journalistic style), and the
  generation prompt picked that up. The profile itself now refers to the user by (preferred) first
  name: `core/profile.py`'s build and revise prompts say so explicitly, since the user objected to
  "Roth" live. It stays third person, a briefing note for the scoring model. By contrast, the
  scoring *rationale* the user reads on every job card is second person ("your background..."),
  per `core/scoring.py`'s prompt, which also warns against carrying over the profile's third-person
  voice. The resume path no longer has this problem at all, for a stronger reason than a
  prompt instruction: `write_resume_edits` hard-denylists any edit anchored on the `title` line, so
  the candidate's name is never regenerated in the first place (see `core/docx_writer.py` above).
- `settings.looking_for_notes` (Profile page, "What are you looking for?" card) is a free-text
  field -- a few sentences in the user's own words, fed into both `core/discovery.py`'s
  `_profile_text()` (company/board discovery) and `core/scoring.py`'s prompt (job scoring), told to
  weigh it heavily and let it override a generic reading of the structured fields when the two
  conflict, since it's the most direct signal available, not a forced-choice structured field.
- `api/resumes.py` now treats `is_template` as the single source of truth for "current" (one per
  kind: resume, cover letter) -- `ProfilePage`'s `ResumesSection` is three pieces instead of one
  flat upload-plus-list: `CurrentDocCard` (one per kind), and `OldDocsSection` (collapsed by
  default, showing just a count, holding everything that isn't the current doc for its kind).
  Uploading a new file always becomes the new current for its kind, auto-demoting whatever was
  current before (`is_template = 0, is_archived = 1`) -- both "drag the old one to Old, then upload
  the new one" and "just upload a replacement" land in the same state. `PATCH /api/resumes/{id}`
  promotes (`is_template: true`, demoting any other current of the same kind) or archives
  (`is_archived: true`, which also clears `is_template`, since a doc can't be both). Native HTML5
  drag-and-drop (`draggable` on `CurrentDocCard`'s doc, `onDragOver`/`onDrop` on `OldDocsSection`'s
  card) moves a current file to Old by dragging; promoting the other direction is a button
  ("make current") in the expanded Old list rather than a second drop zone, since that's simpler
  and just as discoverable once the user has expanded it. This user's existing real data had a
  `is_template` cover letter already but no `is_template` resume (the old fallback logic picked
  "most recent" implicitly, never persisted it) -- fixed with a one-time direct DB update promoting
  the same resume the old fallback would have picked, so "Current resume" wasn't empty on first
  load of the new UI.
- `GET /api/generate/documents` (`api/generate.py`) lists every generated resume/cover letter
  across all jobs, joined with the job's title/company -- backs the Profile page's "Generated
  documents" section, which links each one back to the posting it was written for. Distinct from
  the existing `GET /api/generate/history?job_id=` (one job's docs, used by `JobDetailPage`).
- Job-list filters (`components/JobFilters.tsx`) are multi-select for Interest fit, Fit (née
  "Candidate fit" -- renamed in the UI only; the underlying `candidate_fit`/`candidate_fit_score`
  fields are unchanged), Source, and Location -- each a `MultiSelectGroup` (checkboxes, not a native
  `<select multiple>`, which needs ctrl/cmd-click and isn't discoverable). `filterJobs()` changed
  from equality checks to `.includes()` against the selected array; an empty array means "no
  filter," matching the previous "all" sentinel's behavior. Location switched from a free-text
  substring match to an exact multi-select over the distinct values present in the loaded jobs,
  same derivation pattern already used for Source.
- `MyJobsPage`'s table now wraps (`white-space: normal; word-break: break-word`) instead of forcing
  `nowrap` + horizontal scroll -- preferred explicitly over scrolling. The filter sidebar's `.card`
  padding and `.fsection` spacing are tightened specifically within `.filters` (not the global
  `.fsection`/`.card` rules, which are shared by every other form in the app) to cut the vertical
  padding that made the sidebar feel sparse. Between-section separation there is a `::before`
  pseudo-element on every `.fsection` but the first (an actual `border-top` would span the full
  width; this one's inset via `left: 6%; right: 6%` so it doesn't touch either edge), and `.flabel`
  is darker/bolder (`color: var(--ink); font-weight: 700`) specifically inside `.filters`, not
  changing the shared global `.flabel` used elsewhere.
- `GET /api/resumes/{id}/download` (`api/resumes.py`) returns an uploaded resume/cover letter under its
  original file name -- the Profile page's "Download" links (current docs and the Old list).
- `GET /api/generate/documents/{id}/download` (`api/generate.py`) streams the actual `.docx` via
  `FileResponse` when `file_path` exists and the file's still on disk, falling back to the
  markdown `content_text` as a downloadable `.md` (`Content-Disposition: attachment`) for the
  plain-text-only generation path or a file that's been moved/deleted since. Used from both
  `MyJobsPage`'s `DocActions` (a plain `<a download>`, not a fetch -- lets the browser handle the
  download natively) and the Profile page's Generated Documents table. In `MyJobsPage`, each
  document control reads "Tailor my resume"/"Tailor my cover letter" (same wording as
  `JobDetailPage`'s buttons) until one exists, then "Download resume"/"Download cover letter",
  which downloads instead of regenerating -- it's rendered as an `<a>` when a doc id already
  exists, a `<button>` (the generate action) otherwise;
  `a.ghost` needed `display: inline-block; text-decoration: none` added since `.ghost`'s existing
  rules were written assuming a `<button>`, which doesn't need either reset.
- `MyJobsPage`'s row-remove control is now a `×` (`.remove-x`, `aria-label`/`title` for
  accessibility since the visible glyph alone doesn't say what it does) gated by `window.confirm()`
  before calling `DELETE /api/applications/{id}` -- the one `window.confirm` in the app, used
  deliberately here per explicit request even though every other delete/remove action in the app
  (SourceCard, ResumesPage, the rest of My Jobs) stays confirmation-free; don't
  assume removing it elsewhere needs the same treatment, and don't add confirmation to those other
  actions without being asked.
- `core/generate.py`'s `_usable_template_path()` used to check only `resume.source_path`, which is
  populated solely for bulk-ingested resumes (`core/ingest_resumes.py`, the CLI import from a local
  folder) -- `stored_path` is the column that's actually `NOT NULL` for every resume regardless of
  how it got here, including anything uploaded through the web UI's "Current resume"/"Current cover
  letter" cards. A document uploaded that way silently fell back to plain-text generation (no
  `.docx` on disk from the API's point of view) even with a perfectly good Word file sitting right
  there in `stored_path` -- which is also why downloads of those came back as `.md`: the download
  endpoint was behaving correctly, `file_path` on the `generated_documents` row was just never set
  in the first place. Now checks `stored_path` first, falling back to `source_path`.
- Job-list filters: Interest fit and Fit render as horizontal clickable chip toggles
  (`ChipMultiSelect`, `.chip-toggle-group`/`.chip-toggle`, active state styled like the existing
  `.seg button.on` segmented control but as independent pills rather than a joined strip, since
  multiple can be on at once) rather than a vertical checkbox list. Source and Location render as
  a closed-by-default dropdown (`DropdownMultiSelect`) using a native `<details>`/`<summary>` --
  deliberately not a `position: absolute` floating panel, which would get clipped by `.list-side`'s
  own `overflow-y: auto` (an absolutely-positioned descendant is clipped by any ancestor that sets
  overflow to something other than visible); `<details>` instead expands inline, pushing later
  filter sections down, with zero positioning math and no click-outside-to-close logic needed. Same
  pattern Shelf Life's own filter sidebar already uses for its mobile collapse, so it's consistent
  with the house style this app follows, not a one-off.

## Working conventions

- Small commits, one concern each.
- This repo is **not** on the public-portfolio changelog/prompts-log pattern that `projects/01 - Shelf
  Life` uses — that machinery exists there because Shelf Life's dev history is a public product
  feature. Here it would just be noise (and a privacy problem), so there's no `DECISIONS.md`,
  `prompts/`, or Changelog page. Keep commit messages and this file as the record instead.
- Resumes/cover letters and the job/application DB are real personal data — never relax `.gitignore`
  for `data/` or commit a `.db` file, even for debugging.

## Deploy (Milestone 6 — live on Cloud Run)

Cloud Run + Cloud Scheduler, same `gcloud builds submit` → `gcloud run deploy` pattern as Shelf Life,
with two differences from that project:

- **`--no-allow-unauthenticated` + direct IAP** (`--iap` on `gcloud run deploy`, Cloud Run's
  native IAP integration -- no load balancer). `make iap-setup` grants just `IAP_USER`'s Google
  account access. This app must never be publicly reachable: after any change to the IAP setup,
  confirm from a signed-out/incognito browser that the URL redirects to Google sign-in.
- **A GCS-bucket volume mount for `data/`** (the SQLite file + resume blobs) via Cloud Run's native
  volume-mount support, so it survives restarts — Cloud Run's local disk is ephemeral otherwise.
  `--max-instances 1` avoids SQLite multi-writer issues entirely (single personal user anyway).
- **`--timeout 3600`** (Cloud Run's max; the default is 300 s). A full daily run -- polling plus up
  to `daily_llm_cap` serial scoring calls -- and a web-search discovery call can both outlast five
  minutes, and the UI's "Run now" waits on the whole run in one request.

The image also bundles Playwright + Chromium (for `custom_board`'s JS-rendering fallback), which
makes it noticeably bigger than Shelf Life's — expect a slower Cloud Build and a slower cold start.

Deployed to the `job-searcher-512814` GCP project, service `jobhunt`, region `us-central1`.
The URL isn't recorded in this repo; get it with `gcloud run services describe jobhunt --project
job-searcher-512814 --region us-central1 --format='value(status.url)'` or from the Cloud Run
console. One-time setup order (see the Makefile): `apis-setup` → `bucket-setup` → `anthropic-secret`
/ `scheduler-secret` → `deploy` → `iap-setup` → `scheduler-setup`; after that, `make deploy` alone
ships every update. Every one of these is real spend/real infrastructure, so don't run them from
here without the user present.
