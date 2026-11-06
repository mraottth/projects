# CLAUDE.md

Personal job-search assistant: aggregates job postings, scores them for interest/candidate fit via
Claude, tailors resumes/cover letters, and tracks applications. Built single-user; now multi-user
for up to ~20 people the admin approves (D-021, see "Multi-user" below).

**Where it lives (D-028, 2026-11-06).** This folder is `01 - Claude Code Projects/02 - Job Search
Assistant/` in the **public** `mraottth/projects` repo, next to Shelf Life (`01 - Shelf Life
(Goodreads Recommender v2)/`). It used to be the private `mraottth/job-search-assistant` repo, now
archived; its history came in by `git subtree add` after a rewrite that replaced a personal email
address, so commit hashes differ from the old repo's. The deployed app stays private (IAP plus
approval); the code, prompt log and decisions are public. So:
- Real resumes, cover letters, the database and anything under `data/` never go in git (they're
  git-ignored; never relax that), and neither do secrets, `.deploy.local.mk` or `.redact.local.json`.
- The prompt log is scrubbed on export (`scripts/redact.py`: local paths, session links, and any
  email address that isn't a placeholder). Put other literals you don't want public in the
  git-ignored `.redact.local.json`. Write deploy commands with a placeholder
  (`ADMIN_EMAIL=you@example.com`) in docs.
- Run every command from this folder (`make`, `uv`, `npm`); git commands work from anywhere in the
  repo.

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

- **Multi-user (D-021).** Up to ~20 occasional users, each approved by the admin (the owner).
  - **Identity.** `api/auth.py`'s `AuthMiddleware` (pure ASGI) runs on every `/api/*` request. In
    production it verifies IAP's signed `x-goog-iap-jwt-assertion` (ES256 against Google's
    published keys, issuer, expiry, `IAP_AUDIENCE`) and trusts only that token's email. Locally
    and in tests `JOBHUNT_DEV_EMAIL` stands in (`X-Dev-User` switches person); both are ignored
    when `IAP_AUDIENCE` is set and refused on Cloud Run (`K_SERVICE`). With neither, every
    `/api` request is a 401. A wrong audience says which audience the token carried.
  - **Approval.** The email maps to `users` (`pending`/`approved`/`declined`/`blocked`). Anyone not
    approved reaches only `/api/me` and `/api/access/request` (`api/access.py`), which back
    `pages/AccessPage.tsx` (just a name, plus a privacy note, then "Waiting for approval"; the
    optional note field was dropped at the user's ask, though the API still accepts one).
    A request emails the admin; approving or declining on `/admin` (`api/admin.py`,
    `pages/AdminPage.tsx`) emails the person (`email_digest.send_plain_email`). The admin can't
    change their own status. User 1 is the admin and owns everything from before multi-user; their
    address is `JOBHUNT_ADMIN_EMAIL` (`_ensure_admin()`; `admin@localhost` until set, which is also
    `make serve`'s default `DEV_EMAIL`).
  - **Whose data.** `core/users.py` holds the current user in a context variable, set by the
    middleware and by the daily run per person (`acting_as`). `current_user_id()` raises
    `NoUserError` when nobody is set, so a missed spot fails closed. Background threads and pools
    don't inherit it: `carry()` (discovery tasks, retag) or `daily_run._map_in_context()`. Every
    per-person table (`connection.PER_USER_TABLES`) has `user_id NOT NULL` (nullable only on
    `llm_calls`, where NULL is shared work like the scheduled run reading a board, and `runs`), and
    every query on one filters by it; reading another person's row by id is a 404. The UNIQUEs on
    `applications`, `scoring_queue`, `digest_entries` and `jobby_threads` are per person, and
    `settings`' key is `(user_id, key)`.
  - **Shared sources and jobs.** A board is one `sources` row, read once per run whoever follows it;
    `source_follows (user_id, source_id, enabled)` is who follows it, and `enabled` is per follow.
    `api/sources.py`'s `_add_or_follow()` matches an existing source by kind + config (URLs compared
    without case, protocol or trailing slash) and just follows it. Deleting unfollows (the row stays
    for others). Name/employer/config are editable only by a sole follower or the admin. Jobs from
    sources are shared; `users.VISIBLE_JOB_SQL` lets each person see those from sources they
    follow, plus any they've scored or tracked. A pasted-in job has `jobs.user_id` (its owner, the
    only one who sees or edits it; dedupe key prefixed `u<id>:`).
  - **Files** go to `RESUME_STORAGE_DIR/<user id>/` (`users.storage_dir()`). Uploads are capped at
    10 MB.
  - **Daily run** (`pipeline/daily_run.py`). It reads every source followed by an approved person,
    once (`_poll_sources(user_ids=)`, with nobody set, so board extraction is shared spend). Then
    `_run_for_person()` for each: queue that person's new jobs, score up to what's left of their
    daily cap (`users.daily_score_cap`, default 20, set on `/admin`; `remaining_cap()` counts
    today's scores, so "check now" and refreshes can't exceed it), digest, preference refresh,
    email to their own address, and a `runs` row. Someone with no candidate profile isn't scored.
    Each person's jobs are one batch, which keeps their scoring context in the prompt cache.
    "Check every source now" is the admin's (everyone); anyone else gets "Check my sources now"
    (`run_daily(only_user=)`). Progress is kept per person who started the run.
  - **Admin only:** `/admin` and the database backup download/list.
  - **The Product Timeline** is for everyone approved (the user's call). Its `changelog.json` is no
    longer bundled into the frontend, whose files anyone who gets through IAP's sign-in can fetch,
    approved or not, once `make iap-open` runs. The page loads it from `GET /api/changelog`
    (`api/changelog.py`), which needs an approved person like every `/api` route. The Dockerfile
    copies the file next to the app.
  - **"Start over"** clears only the signed-in person's rows, follows, pasted jobs and files.
  - **Migration.** On a single-user database, `init_db()` snapshots it, rebuilds each
    `PER_USER_TABLES` table from its current `schema.sql` definition with every row given to user
    1, follows every source for user 1, and gives pasted jobs to user 1. The admin keeps their old
    `daily_llm_cap` as their `daily_score_cap`; `/api/settings` still reports it as `daily_llm_cap`
    (only the admin can change it there). Tested against `tests/fixtures/schema_single_user.sql`.
  - Tests: `conftest.py` signs every request in as the admin (`JOBHUNT_DEV_EMAIL`) and runs direct
    core calls as user 1. `tests/test_multiuser.py` covers isolation (a second person on every
    endpoint), shared sources, approval, per-person runs and the migration; `tests/test_auth.py`
    covers token checks with a locally generated ES256 key.
  - **Cost controls (D-022).**
    - *Activity.* The everyone run scores only `users.active_user_ids()`: approved, and
      `COALESCE(last_seen_at, decided_at, created_at)` within `ACTIVE_DAYS` (7), so a new person
      counts until their first visit. Sources followed only by inactive people aren't read.
      `_send_paused_emails()` tells each person who just lapsed, once (`paused_email_sent_at`); a
      failed or unconfigured send stays unmarked and retries. `touch_last_seen()` clears the flag on
      the next visit. `only_user` runs ignore the rule, and the admin gets no exemption.
    - *Budget.* `core/budget.py`: spend is this calendar month's `llm_calls` for that person. It's
      enforced in `llm.get_client()`, which every paid call site calls first, by raising
      `BudgetExceeded`, an `LLMError` that routes already turn into a message. Shared work (nobody
      set) and the admin are never blocked. `_run_for_person()`/`refresh_sources()` skip scoring when
      over. `log_call()` → `budget.after_call()` emails the admin at 80% and at 100%, once each per
      person per month (`users.budget_alert_month`/`budget_block_month`, MIGRATIONS ALTERs).
      `/api/me` returns `usage` (spend, budget, over, exempt, scored today, cap), shown as Settings'
      usage line.
  - **New & Noteworthy freshness (D-023).** `GET /api/digest/status` gives `last_updated` (the
    person's latest `runs.finished_at`, daily or refresh), today's scoring against the cap,
    `over_budget`, and the Refresh cooldown. `POST /api/digest/refresh/stream` is their own check
    (`start_run_daily(only_user=, fresh_hours=1)`), streamed by the shared `pipeline.stream_run()`,
    at most every 30 minutes (`last_refresh_at` setting; a 429 says how long is left). With
    `fresh_hours`, `_poll_sources()` doesn't re-read a source anyone read successfully within the
    hour; it takes that source's jobs seen in the last day from the database
    (`_recent_jobs_for_sources`), so they're still queued and scored. The run summary's
    `cap_reached` drives "Scoring resumes tomorrow (N of N used today)". `DigestPage`'s `Freshness`
    shows "Last updated 2 hours ago · Refresh", or, a day or more since the last update, a card with
    "Refresh now". "Check my sources now" on Manage Sources has no cooldown.
  - **Deleting an account (D-024).** `core/reset.py`'s `delete_account(user_id)` shares `_wipe()`
    with Start over (one snapshot, `before-account-delete`). It removes everything Start over does,
    plus the person's settings, their `RESUME_STORAGE_DIR/<id>/` folder and the `users` row. Their
    `llm_calls` stay as spend history, unlinked (`user_id` NULL, note "deleted account: ...").
    Shared sources stay, since their jobs may be on someone else's My Jobs. The admin's account
    can't be deleted (`AdminAccountError`). Routes:
    - `POST /api/settings/delete-account` for yourself (Settings' Start over card, hidden for the
      admin);
    - `DELETE /api/admin/users/{id}` for the admin, which also emails the person.

    Both need `{"confirm": "DELETE"}` typed in the UI (`TypedDelete` in `SettingsPage.tsx`,
    `DeleteUser` in `AdminPage.tsx`; no `window.confirm`). Signing in again afterwards shows the
    request-access page. Settings' "Your data" card also shows the `PrivacyNote` from
    `AccessPage.tsx`. `_inside_storage()` now checks against `users.RESUME_STORAGE_DIR` (the same
    folder as config's).
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
  - **Prompt caching** (D-018, after Anthropic emailed that the cache hit rate was low).
    - `core/scoring.py` sends the candidate context (profiles, titles, sectors, salary floor,
      notes, recent feedback) as a second `system` block with `cache_control`, and only the posting
      as the user message. That context is identical for every job in a run, so the first call
      writes it and the rest read it. Keep it free of anything that changes per call.
    - `core/discovery.py`'s `run_web_search()` (discovery and company lookup) and Jobby's stream use
      top-level automatic caching (`cache_control={"type": "ephemeral"}`). Web search steps and
      `pause_turn` continuations re-read the whole conversation, and the API caches after each
      search result once caching is on. Jobby also keeps its explicit breakpoint on its context.
    - `messages.parse()` has no top-level `cache_control` parameter in this SDK version (2.3), so
      scoring uses a block-level marker.
    - `log_call()` takes `cache_read_tokens`/`cache_write_tokens` (use `**cache_tokens(usage)`),
      priced at 0.1x / 1.25x input (`CACHE_READ_MULTIPLIER`/`CACHE_WRITE_MULTIPLIER`, the 5-minute
      TTL), and stored in two `llm_calls` columns (a migration). Before this, cached tokens weren't
      counted at all, so Jobby read low.
    - Settings' Usage log shows the last 7 days' cache hit rate and the net saving
      (`_cache_summary()` in `api/settings.py`).
    - Not cached on purpose: board-page extraction (every page differs, read once a day), tailoring
      (calls too far apart for a 5-minute cache), and one-off or sub-1,024-token calls.
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
- **Scoring cost experiment** (asked 2026-11-03; `core/scoring_experiment.py`, the Admin page's
  last card, `GET`/`POST /api/admin/scoring-experiment`). A score costs about 2-3 cents, mostly
  output (reasoning at effort "medium" plus a free-length rationale), then the whole posting.
  `scoring.ScoringSettings` has two presets: `CURRENT` (live scoring, unchanged) and `LEAN` (effort
  "low", a 2-3 sentence rationale, the posting cut at 6,000 characters). `score_job()` keeps its
  signature and uses `CURRENT`; `score_job_with(settings=, note=)` returns the score, its estimated
  cost and output tokens. The experiment rescores up to 20 jobs the person judged (feedback marks,
  or a tracked application = wanted), balanced wanted/not wanted, under both presets, and reports
  cost per job, agreement with their marks and with each other, plus both rationales per job. It
  writes nothing to `job_scores`/queue/digest; calls log as `scoring` with note "scoring
  experiment: <arm>", and the result is kept in the `scoring_experiment` setting. Switching live
  scoring to `LEAN` was measured and declined (D-025: 0.9¢ vs 0.8¢ a job).
- **Rescore** (asked 2026-11-04; `core/rescore.py`, `api/rescore.py`, `components/RescoreButton.tsx`,
  one per list in Settings' "Rescore your jobs" card, moved there from under each page's intro at
  the user's ask). A score is made once, when a job is new, so a changed
  profile never reached older jobs. `GET /api/rescore/{my_jobs|all_jobs}` gives the job count, an
  estimate (count x the person's average of their last 50 scoring calls, 1 cent until they have
  5), when it's next available, and the budget left (null for the admin). `POST` starts it in a
  background thread (one per person; a second click on the same list follows it) and streams
  progress like the retag pass. It stores a new `job_scores` row per job (the latest is what every
  page shows) and marks a queued job scored. It ignores the daily cap (an explicit request whose
  cost is shown first) but not the budget: `BudgetExceeded` stops the run and the result says why.
  Anyone but the admin can rescore each list once a day (`last_rescore_<scope>` setting, set at
  start; 429 with the hours left). My Jobs is every tracked job (pasted ones included); All Jobs
  is the same set as `GET /api/jobs/ranked`. Generic postings are skipped. The button asks for
  confirmation, naming the estimate and, when it's more than the budget left, that it will stop.
- Desired sectors (`settings.prefilter_sectors`, edited on the Profile page's "What are you looking
  for?" card) feed three places: `core/prefilter.py`'s rule-based scorer (boost on a text match,
  penalty on a clear miss -- unlike titles, sector is treated as a harder constraint per the user's
  explicit ask, not pure guidance), `core/discovery.py`'s company-suggestion prompt (treated as a
  real constraint, not a tiebreaker — a strong brand outside the sector shouldn't get suggested
  anyway), and `core/scoring.py` above.
- **Location dealbreaker** (D-026, asked 2026-11-04; `core/locations.py`). "What are you looking
  for?" has "Dealbreaker: only show jobs in these locations" (`location_dealbreaker` setting, off by
  default, only effective with `prefilter_locations` set). `locations.verdict()` returns
  match/remote/unknown/multiple/elsewhere, and only `elsewhere` is hidden. It errs towards keeping:
  - remote jobs always stay (`remote_type`, or "remote"/"anywhere"/"WFH" in the text); missing or
    unreadable locations stay, as do "United States" alone, bare "Washington", and any place we
    don't know.
  - Several locations (amended at the user's ask): `split_places()` splits on ";", "|", "/",
    newline, " & " and lowercase " or "/" and ", then on commas, keeping a part that's only a state,
    zip, country or "USA" with the place before it ("Austin, TX, Seattle, WA" is two places,
    "Washington, DC, United States" one). Each place is judged alone. Any match matches. The job is
    hidden only when every place is clearly elsewhere and nothing in `VAGUE` ("+2 more",
    "3 locations", "multiple", "various", "and more", "nationwide"...) says the list is incomplete;
    one unreadable place keeps it.
  - "Elsewhere" needs evidence: a state (full name, ", VA" after a comma, Workday "US-VA-..." or a
    bare "DC"; a bare uppercase pair elsewhere isn't trusted, so "IN OFFICE" isn't Indiana and
    "Portland, OR" isn't "or"), a metro's core name (implies its home state), or a non-US country or
    big foreign city. A written state code wins over state names inside a city's name ("Kansas
    City, MO" is MO). A suburb name alone never proves "elsewhere".
  - Entries are read generously: "DC"/"D.C."/"District of Columbia"/"DMV"/"Washington, DC" are the
    DC metro (its cities: Arlington, Bethesda, Tysons...; the user chose metro areas); "Washington"
    or "NY" match every reading; "Arlington, VA" is its metro; an unknown "City, ST" is that city in
    that state; anything else is matched as written. ~25 US metros (`METROS`), states, countries.
  - Applied through `core/dealbreakers.py` (with the salary one, below) in `queue.select_batch()` (a
    hidden job stays pending, takes no slot, and comes back if the setting changes), `get_digest()`
    (so New & Noteworthy and the email), All Jobs (each job gets `dealbreakers`: `["location"]`,
    `["salary"]`, both or `[]`; a "Show N jobs your dealbreakers hide" filter toggle, off, and
    "Clear all" keeps it off) and the All Jobs rescore. My Jobs is never filtered.
  - `GET /api/settings/location-check` shows how each entry is read and what it hides (or would
    hide) among All Jobs, by location text, shown under the checkbox, so a wrong reading is visible
    before it's switched on. Baltimore isn't in the DC metro; the preview shows it if hidden.
- **Salary dealbreaker** (asked 2026-11-04; `core/dealbreakers.py`, `salary_dealbreaker` setting,
  shown under Minimum salary once one is set). Hides a job whose posted pay, annualized at the low
  end (`salary_min`, else the one figure given; hour x2080, day x260, week x52, month x12), is below
  `salary_floor`. No pay, a non-USD currency, an unknown period, or a figure under 1,000 with no
  period all stay. `GET /api/settings/salary-check` previews it (how many list pay, how many it
  hides, a few examples). Same places as the location dealbreaker; `dealbreakers.reasons()` says which.
- `src/jobhunt/core/prefilter.py` + `core/queue.py` — the cheap rule-based pass that runs before any
  LLM scoring call, and the priority queue/daily-cap mechanism it feeds (unscored jobs carry over
  with an age bonus so a capped day doesn't starve the backlog). `select_batch()` takes jobs
  round-robin across sources (`ROW_NUMBER() OVER (PARTITION BY source_id)`): every source's best
  job, then every source's second-best, and so on. A plain top-N by priority let one big board
  whose postings all matched the titles/sectors take every slot of onboarding's 20-job first run
  (caught live: only Civis jobs got scored, so All Jobs showed only Civis).
- `src/jobhunt/core/page_fetch.py` + `core/browser_fetch.py` — fetch a page's readable text, trying
  a plain static GET first and falling back to headless-browser rendering (Playwright) only when
  the static fetch yields too little text. The Playwright fallback is real, not speculative: at
  least one source the user wants (techjobsforgov.com, a WP Job Manager board) only renders its
  listings via client-side JS, confirmed by fetching it directly during development.
  `MIN_CHARS` (200) alone missed the common case, though: an organization's /careers page has
  plenty of nav/"about us"/footer text, so it never reached the browser, while its listings load
  via JS or sit in an embedded ATS iframe (caught live: "Test fetch" found nothing on several
  careers pages, e.g. blueroseresearch.org/jobs, and All Jobs had 2 listings). So `custom_board.fetch()`
  now (1) checks the raw HTML for an embedded Greenhouse/Lever/Ashby board
  (`core/ats_detect.py`; an embed script/iframe/API URL counts on any page, a plain link only on a
  single-organization source, and two or more distinct boards count as none) and reads it through
  that ATS's adapter, no Claude call; (2) if Claude finds zero postings in a static page, re-renders
  it in the browser (`fetch_page(force_render=True)`) and tries again, skipping the second paid call
  when the rendered text is identical. `render_page_html()` appends every child frame's HTML (an
  ATS board is often an iframe, which `page.content()` omits) and keeps whatever rendered when
  `networkidle` times out (analytics/chat widgets can keep the network busy forever).
  That still left pages with *some* static text: Test fetch found 1 posting on bluestate.co/careers
  and 0 on codeforamerica.org/jobs, both with many openings (live; neither site is reachable from
  the dev sandbox, so the exact cause wasn't confirmed). Now `fetch_with_report()`:
  - re-renders when the static read found fewer than `FEW_POSTINGS` (3), not just zero. It pays
    for a second read only if the rendered text has more links (or, with zero found, any change),
    and keeps the bigger result;
  - `render_page_html()` also records requests the page made to job-board hosts
    (`ATS_REQUEST_HINTS`) in a trailing HTML comment, so a board only the page's scripts call
    (e.g. `boards-api.greenhouse.io/v1/boards/<token>`) is still detected and read via its API;
  - with still too few, reads the one hosted board the page links to on an ATS with no adapter
    here (`ats_detect.detect_hosted_board()`: Workable, BambooHR, JazzHR, Breezy, Recruitee,
    Rippling, Paylocity, Jobvite, SmartRecruiters, Dover, Pinpoint, Workday).
  **WP Job Manager boards** (asked 2026-11-06: "job boards rarely seem to work"; e.g. Civic Work
  Studio, which techjobsforgov.com now serves) are read through the plugin's own listings endpoint
  (`core/wp_job_manager.py`), right after the embedded-ATS check and before any Claude call. Their
  page arrives with an empty `ul.job_listings` and the plugin's script fetches the jobs from
  `/jm-ajax/get_listings/` a few at a time, so reading the page, even in the browser, found at most
  the first handful of the board's 362. Detected by the script's `"ajax_url":".../jm-ajax/%%endpoint%%/"`
  setting (or its script id). It posts 100 per page, up to `MAX_PAGES` (5), with the source URL's
  query string as `form_data` (how the plugin's script sends the board's own filters, e.g. remote
  only and job types; verified live: the owner's filtered URL gave its 30 remote jobs) plus the
  standard fields directly. Cards give title, organization, location, posted date, pay
  (`extract_salary` on the card) and `remote_type` "remote" when the card says so. No `external_id`,
  so jobs keep the URL-based dedupe keys of those stored before. An unreadable endpoint, or cards
  it can't make out, falls back to the page reader; zero matches is an answer. Free: no Claude call.
  Live, the deployed app's Test fetch then failed on that board before the reader ran: the page
  itself "found too little text", while Cloud Shell (also Google Cloud) loaded both the page and
  the endpoint fine, so the cause is still unknown. Two changes followed: when the page won't load,
  `custom_board` tries `wp_job_manager.fetch(url, None, ...)`, which posts to the site's standard
  `/jm-ajax/get_listings/` and counts only a WP Job Manager-shaped answer (`found_jobs`); and
  `page_fetch`'s "too little text" error now names what each way got (the plain request's status,
  timeout or text length, and the browser's text length), which it used to hide.
  The next Test fetch didn't answer at all ("Lost the connection to the app"): the plain request
  (15 s), the browser (20 s plus start-up) and the endpoint try (20 s) together ran past the point
  where a silent request gets dropped, which suggests the site doesn't answer the deployed app.
  So a plain request that times out or can't connect now fails at once, skipping the browser (it
  would only wait out its own timeout), and the endpoint try after a failed page waits 10 s
  (`PROBE_TIMEOUT`): about 25 s at worst, with the reason in the message.
  **Unchanged pages aren't paid for twice** (D-025): `_extract_postings()` fingerprints the page
  text (sha256, with relative dates like "3 days ago"/"today" blanked by `RELATIVE_DATE`) and, when
  the shared `page_extractions` table has that URL + fingerprint, rebuilds the postings from the
  stored listings with no Claude call (the step reads "...: 12 (unchanged, no Claude call)").
  Listings are stored, not finished postings, so a changed `company_name` still applies. Only reads
  that found postings are stored, so an empty or failed read retries; rows older than 30 days are
  pruned per URL. Not per person, so Start over and account deletion leave it alone.
  `POST /api/sources/{id}/test` returns these `steps` ("read the page as plain HTML: 1; read the
  page in the browser: 12"), and Test fetch shows them after the count.
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
  instance).
  The UI never waits on one silent request for a run: "Check every source now" sat silent for the
  whole multi-minute run and got cut ("Lost the connection to the app", caught live).
  - `POST /api/pipeline/run-daily/stream` starts the run in its own thread (`start_run_daily()`:
    a second click while one is going follows that run rather than paying for another). It then
    streams SSE `started`/`progress` (every 2 s)/`done`|`error` events, so data keeps flowing and
    Cloud Run keeps CPU allocated while the request is open. A run started purely in the background
    and only polled would be CPU-throttled between polls under request-based billing.
  - The run's outcome lands in the progress dict (`run_id`, `finished_run_id`, `last_summary`,
    `last_error`). `api.ts`'s `runDaily()`, behind `runDailyNow(cap, onProgress)`, falls back to
    polling `/progress` until that run's id finishes if the stream drops.
  - Onboarding's finish step and Manage Sources both show `runProgressText()` ("Scored 12 of 20
    jobs..."), which while polling also names what's still being read (`sources_pending`,
    maintained by `_fetch_source`): "Still reading Tech Jobs for Good, Climatebase +2 more". The
    sources counter only moves when one finishes, six at a time, so a run looked stuck on "1 of N"
    (caught live). The plain `POST /api/pipeline/run-daily` remains for scripts and tests.
  - `custom_board` extraction (and manual job paste-in) run at `output_config.effort` "low"
    (`EXTRACTION_EFFORT`). At the default (high), each web-page board spent a minute or more
    reasoning before copying postings off the page, which is mechanical work, and paid for those
    reasoning tokens.
  A newly added (or re-enabled) source is checked right away rather than at the next daily run
  (asked for live): `api/sources.py`'s create/resolve/re-enable call
  `schedule_source_refresh()`, which gathers adds made within `REFRESH_GATHER_SECONDS` (3) into
  one background `refresh_sources(ids)`: poll just those sources, prefilter, score up to
  `NEW_SOURCE_SCORE_CAP` (10, ~30 cents) of *their* jobs (`queue.select_batch(source_ids=)`), and
  add them to today's digest. Skipped while onboarding isn't completed, since its last step runs
  the whole pipeline anyway. `_run_lock` keeps a refresh and a daily run from overlapping.
  `conftest.py` stubs `schedule_source_refresh` for every test so creating a source never reaches
  a real job board.
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
  The UI doesn't wait on one long request for a search: `POST /api/discovery/suggest[-boards]/start`
  runs the same route handler in a background thread and returns a `task_id`, and `api.ts`'s
  `runDiscovery()` polls `GET /api/discovery/tasks/{id}` every 2 s (tolerating a few dropped polls).
  A single request that's silent for 1-2 minutes got cut before the answer arrived (caught live in
  Safari as "Load failed" on "Find more like this"). Tasks are in-process memory (one Cloud Run
  instance), pruned after an hour. The synchronous `/suggest[-boards]` routes still exist. `api.ts`'s
  `request()` also turns a bare fetch network failure ("Load failed"/"Failed to fetch") into a
  readable message.
  Each search can be steered: an optional `guidance` ("Steer this search": free text like
  "mission-driven tech startups", made the main focus of that one search) and/or `like` ("More
  like this" on a suggestion, or "Find more like this" on a tracked source -- name plus URL when
  there is one). Both are an optional JSON body on `/api/discovery/suggest[-boards]`, appended by
  `_search_direction()` after the profile; neither is saved. Manage Sources hands a tracked
  source's seed to the right Discover card through a `seed` prop (`DiscoverySeed`, with a `nonce`
  so asking twice re-runs), opening that section's "Add more" dialog on its Discover tab. In
  `DiscoveryControls`, the big primary "Suggest ..." button (`.discover-go`) comes first and the
  steering box sits under it, per the user's ask; `bare` drops the card and heading for use
  inside the dialog.
- **Favorite was removed** (favorite companies and favorite sources, which bypassed the pre-filter
  and daily cap and always ranked first) at the user's request -- every source is now just a
  monitored source. `schema.sql` no longer has `sources.is_favorite`, `scoring_queue.is_favorite`,
  or `favorite_companies`, but a database created before that (the live one) still has all three as
  unused leftovers; nothing reads them, and a `SELECT *` from `sources` there still returns an
  `is_favorite` field the frontend ignores. Don't reintroduce a cap/rank bypass without being asked.
- `src/jobhunt/api/` — one FastAPI route module per resource: `resumes`, `jobs`, `digest`,
  `applications`, `sources`, `generate`, `settings`, `internal`, `pipeline`, `discovery`, `onboarding`,
  `jobby`.
  `api/main.py` wires all routers and serves `frontend/dist` if built.
- `frontend/` — React 19 + Vite + TS, react-router-dom for navigation, no UI library. Unlike Shelf
  Life's stateless localStorage-only app, this one's source of truth is server-persisted SQLite —
  pages call `api.ts` directly rather than reading/writing a client-side store.
- `frontend/src/components/Spinner.tsx` — small CSS-animated spinner (`@keyframes spin` in
  `index.css`), used in place of plain "...ing" button text for any action that calls Claude
  (profile rebuild, company discovery, resume/cover-letter generation, job scoring, company
  resolution) or otherwise takes noticeable time (test-fetch, run-daily-now).
- **Backups** (D-019; `core/backup.py`). The live database is a single SQLite file on the Cloud
  Storage volume, and before this nothing copied it. `snapshot()` uses SQLite's online backup API
  (`copy_database()`, consistent even mid-write) into `<DATA_DIR>/backups/`:
  - `jobhunt-YYYY-MM-DD.db` at the start of every daily run (`_run_daily()`, before anything
    writes), one per day, kept `KEEP_DAYS` (30);
  - `jobhunt-…-HHMMSS-before-reset.db` from "Start over", and `…-before-migration.db` from
    `init_db()` when `_needs_status_migration()` (the newest `KEEP_EVENTS` (10) are kept).
  - `backups/` is outside `RESUME_STORAGE_DIR`, so "Start over" never deletes it. A failed snapshot
    is logged and never blocks the thing it guards.
  - Settings' "Your data" card lists them (`GET /api/settings/backups`) and offers "Download a
    backup" (`GET /api/settings/backup`, a fresh copy, for one kept off Google Cloud). My Jobs'
    toolbar has "Export CSV" (`GET /api/applications/export.csv`, registered before `/{app_id}`,
    status labels as shown on the page).
  - Restoring a snapshot (production runs on Litestream, D-020, so the bucket's `jobhunt.db` is only
    the seed): with the service idle (no requests for 15+ minutes, so the server has shut down and
    flushed; not near the 2pm run), move `gs://<bucket>/litestream/` aside, copy the snapshot over
    `gs://<bucket>/jobhunt.db`, delete `jobhunt.db-wal`/`-shm` and `litestream-seeded` next to it,
    then open the app: the new server finds no Litestream copy and no marker, and seeds from it.
    For a point in time instead, `litestream restore -timestamp ...` from the replica.
  - `copy_sqlite()` builds every copy in a local temp file, then copies the finished file into
    `backups/` in one sequential write (`.part`, then rename). Writing it in place on the bucket mount
    failed live with `BufferedWriteHandler.OutOfOrderError` on the backup and its journal.
  - **Why the daily run failed Oct 31 – Nov 2 (diagnosed 2026-11-02).** The app returned 500 after
    6-25 minutes, and the mount logged `OutOfOrderError`, "stale file handle ... clobbered" on
    `jobhunt.db-shm`, and 429 throttling. That's SQLite's lock file on the FUSE mount, the D-020
    risk. Litestream (deployed after the last of these runs) moved the live database to local disk.
    The scheduler's own log showed `DEADLINE_EXCEEDED` every day, even on successful runs, because
    its default attempt deadline is 3 minutes. `scheduler-setup` now sets `--attempt-deadline
    1800s`. To check runs: `gcloud scheduler jobs describe run-daily ...`, and the Cloud Run log
    filtered on `httpRequest.requestUrl:"/internal/run-daily"` for status and latency.
  - **...and the cause still there after Litestream (2026-11-03): `StopIteration`.** Every run (the
    2pm one, Refresh, "Check every source now") failed with a bare "StopIteration:". Once enough new
    feedback had built up, the run rebuilt the preference profile, and Claude's reply had no text
    block: its thinking used up the 1500 `max_tokens`. Six call sites took the first text block with
    a bare `next(...)`, which raised StopIteration. That isn't an `LLMError`, so it took down the
    whole run, and the rebuild was retried, and failed, every time.
    - Now every reply goes through `llm.response_text(response, action)`, which raises `LLMError`
      on an empty reply.
    - The preference rebuild has 4000 tokens at effort "low", and the profile build/revise 4000 at
      "medium".
    - A failed preference rebuild is just noted in the run's errors.
    - `_run_for_person_safely()` keeps one person's failure from stopping the others. It writes no
      `runs` row for them, so "Last updated" doesn't claim a check that didn't happen.
    - `describe_error()` adds the file, line and function to the Daily check card's error.
  - **The Admin page's "Daily check" card.** `run_daily()` gives the everyone run (scheduled, or the
    admin's "Check every source now", `started_by`) its own `runs` row with `user_id` NULL. The row is
    written as it starts (`status` 'running'), then set to 'ok' with counts or 'failed' with the
    `error` (`_system_run_started`/`_finished`; MIGRATIONS add `status`, `started_by`, `error`). A row
    still 'running' after 70 minutes shows as "Didn't finish": the server was shut down or crashed
    mid-run. `GET /api/admin/users` returns it as `daily_run` (`_daily_run_status`), with the last
    clean finish. One person's own check doesn't count. Per-person `runs` rows (`user_id` set) still
    back "Last updated".
  - `make bucket-protect` makes sure the bucket's soft delete (7 days, covering overwrites too) is
    on. Object versioning was left off on purpose: the database is rewritten on every save.
- **Settings' "Start over"** (`core/reset.py`'s `delete_all_data()`, `POST
  /api/settings/delete-all-data` with `{"confirm": "DELETE"}`, which the UI makes the user type)
  deletes every row of every table except `llm_calls` (real spend history; its `job_id`s are
  nulled), re-inserts `DEFAULT_SETTINGS` (so `onboarding_completed` is false again), and deletes
  uploaded/generated files, but only ones inside `RESUME_STORAGE_DIR`, never the user's own
  `resume_library_path` folder. With no resumes left, the next load goes through onboarding; the
  UI does a full page load to `/onboarding` so `App.tsx` re-checks status. Tables are listed from
  `sqlite_master` with foreign keys off, so the live DB's leftover tables are cleared too.
- **Daily email** (`core/email_digest.py`): at the end of every `run_daily` (Cloud Scheduler
  now runs it at 2:00pm America/New_York, `0 14 * * *` with `--time-zone`), `send_new_jobs_email()`
  emails New & Noteworthy jobs from today's and yesterday's digest with High or Med interest and a
  High or Med fit (`GOOD_INTEREST`/`GOOD_FITS`; widened from High interest only on 2026-11-05 at
  the user's ask, after a Med/Med match never arrived), best first, each one once (`digest_entries.emailed_at`, a
  migration). No new matches means no email. It's sent over Gmail SMTP (`smtp.gmail.com:465`) with an app
  password, from `EMAIL_ADDRESS` to itself (or `EMAIL_TO`), with posting links and, given
  `APP_URL`, "Open in the app" links. Without `EMAIL_ADDRESS` + `GMAIL_APP_PASSWORD` nothing is
  sent, so dev and tests never email. A send failure is logged and lands in the run summary's
  `email`, never failing the run. `refresh_sources()` doesn't email; its jobs go out with the next
  daily email. Settings' "Daily email" card shows who it goes to and has "Send a test email"
  (`POST /api/settings/test-email`: today's top 5 matches, without marking them emailed).
  Deploy wiring: `make email-secret` stores `GMAIL_APP_PASSWORD` as `gmail-app-password` (and
  grants the compute service account access). After that every `make deploy` turns email on: the
  sender is `EMAIL_ADDRESS` if given (command line or `.deploy.local.mk`), else `ADMIN_EMAIL`, when
  the `gmail-app-password` secret exists (`HAS_EMAIL_SECRET`, a lazy `gcloud secrets describe`), and
  deploy prints "Daily email: on, sent from ..." or "OFF -- ...". It used to need `EMAIL_ADDRESS` on
  every deploy, and `--set-env-vars`/`--set-secrets` replace the whole set, so the usual
  `make deploy ADMIN_EMAIL=...` quietly switched email off (no emails arrived, found 2026-11-05).
  Each check records what its email step did as the person's `last_daily_email` setting
  (`email_digest.record_outcome()`: sent and to whom, not configured, nothing new with how many
  were on today's list and the best interest, or the error), returned as `last` by
  `GET /api/settings/email-status` and shown on Settings' Daily email card ("Last check (...)").
- **Settings is a gear icon** (`.settings-gear`, right end of the top bar), not a nav tab.
- **About** (`pages/AboutPage.tsx`, route `/about`, an "About" link at the right of the top bar
  before Admin and the gear; asked 2026-11-04): what the project is, how the daily pipeline works,
  a "Getting started" guide linking each page, costs and limits (the person's own daily cap and
  budget from `/api/me`'s `usage`), the `PrivacyNote`, what it's built with, and the Product
  Timeline's milestones (`MilestoneTimeline`, loaded from `GET /api/changelog`) with "Open the full
  Timeline". The Timeline card moved here from Settings. A milestone's prompt or decision opens
  `/changelog?prompt=<id>` / `?decision=<id>`, which `ChangelogView` scrolls to on load.
- **Page split**, seven pages each with one job (plus the first-run `/onboarding` wizard, below), nav order New & Noteworthy → All Jobs → My Jobs →
  Jobby → Manage Sources → Profile → Settings: `DigestPage` (route `/`, nav label "New & Noteworthy") is
  today's capped, LLM-scored digest only. `AllJobsPage` (route `/all-jobs`) is
  `GET /api/jobs/ranked` (`api/jobs.py`, registered *before* `/{job_id}` since Starlette matches
  path templates in order -- a later `/ranked` would 404 as an invalid `job_id`) -- every active
  job from a tracked source (`source_id NOT NULL`), scored or not, ranked best-first, the uncapped
  view the daily cap's 50-job limit would otherwise hide. `ManageSourcesPage` (route `/sources`, a
  wide route) is purely source management, in two side-by-side `SourcePanel`s (`.sources-grid`;
  one column under 860px), each a fixed-height card whose list scrolls. A source is a compact
  `SourceRow`: name, status ("N jobs · checked 3h ago" from `GET /api/sources`' `job_count`,
  "Checking for jobs..." for five minutes after it's added and before its first check, while the
  page polls every 4 s, or "disabled"), kind and host/board, the last error clamped to two lines,
  and text-link actions. The name links (new tab) to the page it reads (`sourceLink()`): a web
  page's own URL, or the public Greenhouse/Lever/Ashby board built from its token/slug. The
  Adzuna/RemoteOK keyword searches have no single page, so their names stay plain text. The two sections are **Organizations** (single-employer sources) and
  **Job boards** (multi-employer boards and the keyword aggregators), each with a "+ Add more"
  button opening `AddMoreDialog` (`components/Modal.tsx`, a native `<dialog>`). It has two tabs:
  "Add by link" (name + URL through `detectSourceFromUrl()`; an organization gets `company_name`
  = its name so it files under Organizations) and "Discover suggestions" (the Discover cards,
  which no longer sit inline on the page). The dialog stays mounted while closed, so discovery
  results survive closing and reopening. "Add a source manually" stays at the bottom, as the only
  way to add the Adzuna/RemoteOK keyword aggregators, then run-daily-now. Its fields carry hints
  (the user asked what Name vs. the organization field meant): "Name" is the source's label, and
  "Employer name (optional)", formerly "Organization display name", is `company_name`. That's the
  company shown on postings that don't name one, and it files the source under Organizations;
  leave it blank for multi-employer boards. A source is an
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
- **Jobby** (`core/jobby.py`, `api/jobby.py`, `pages/JobbyPage.tsx`, route `/jobby?job=<id>`) is the
  interview-prep chat. There's one saved conversation per job (`jobby_threads.job_id` UNIQUE), plus
  one general conversation (`job_id` NULL; SQLite's UNIQUE allows many NULLs, so `get_thread()`
  enforces "only one"). "Start over" deletes the thread. The picker lists My Jobs (open
  applications first) plus "No specific job"; "Prep with Jobby" on My Jobs rows and `JobDetailPage`
  links straight to a job's conversation. `build_context()` puts the posting, tracking status/notes
  and fit rationale, the candidate profile, "What are you looking for?", the current and tailored
  documents (whole, up to generous limits), and older documents (trimmed to a shared budget) into
  the system prompt behind a `cache_control` breakpoint, so every later turn re-reads it from cache.
  Replies stream as server-sent events over the POST (`stream_reply()` yields `delta`/`status`/
  `done`/`error`; `api.ts`'s `streamJobbyMessage()` reads the body, since `EventSource` can't POST).
  Web search is on (`max_uses` 3, `pause_turn` resumed), and a search shows "Searching the web...".
  A failed turn keeps nothing, not even the user's message, so the UI puts their text back to
  retry. Turns log as `call_type='generation'` with a `jobby chat` note, because `llm_calls`'
  CHECK constraint can't be widened on the live table without a rebuild. The four starter prompts
  (mock interview, likely questions, frame my background, prep & research plan; "questions to ask
  them" was dropped at the user's request)
  are frontend constants (`STARTERS`) sent as ordinary user messages; the mock-interview behavior
  (one question at a time, feedback, wrap-up) lives in `PERSONA`. `JobbyLogo.tsx` renders the
  user-supplied Jobby art: `src/assets/jobby-avatar.png` (a tight head crop, for the nav tab,
  header and message avatars) and `jobby-full.png` (Jobby at a laptop, the empty-conversation
  welcome). The off-white backgrounds were cut out to transparent PNGs with a one-off Pillow
  flood fill, so Pillow isn't a project dependency. The art was replaced once (2026-10-31) the same
  way: flood fill from the edges, a slightly blurred alpha edge, cropped to the figure and padded
  square (128px avatar, 480px full). They're imported rather than put in
  `public/`: `api/main.py`'s SPA fallback answers every non-`/assets/` path with `index.html`.
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
- Both job lists have a "Sort by" select above the list (`JobListHeader`, which also shows "N of
  M jobs"; moved out of the filter sidebar at the user's request), stored as `JobFilterState.sort`
  and applied by `sortJobs()` in `components/JobFilters.tsx`: Best match (the server's order,
  unchanged), Interest, Fit (each on its 0-100 score, the other score breaking ties), Date posted,
  Salary (`salary_max` falling back to `salary_min`; hourly pay x2080 so it compares with annual
  pay, and the Minimum salary filter does the same). A job missing the value (unscored, no salary, no
  posted date) sorts last, and the stable sort keeps best-match order among ties. Client-side, like
  the filters. "Clear all" keeps the chosen sort.
- General-application / talent-pool / resume-drop "postings" ("Don't see your role?", "Join our
  Talent Community") are excluded at the user's request: `core/generic_postings.py`'s
  `is_generic_posting()` matches the *title* only (a real job's description often mentions a
  talent community). It's applied when jobs are queued for scoring (`_prefilter_and_enqueue`
  skips them, and `_score_and_digest` marks any already-queued ones `skipped`, so none is paid
  for), in `add_to_digest()`/`get_digest()`, and in `GET /api/jobs/ranked`. Filtering at read
  time covers rows stored before this existed, with no migration. Manually pasted My Jobs entries
  aren't affected.
- Empty states: `DigestPage` with no entries shows "Browse All Jobs" / "Add more sources"
  buttons; when filters hide every job, both job lists show `FilteredEmptyState` with "Clear all
  filters", which applies `NO_JOB_FILTERS` (truly empty), not `DEFAULT_JOB_FILTERS`. The defaults
  hide Low fits. Unscored jobs (no fit at all) skip the fit filters and are governed only by
  `includeUnscored` ("Include jobs not yet scored", on by default) -- they used to fail the
  High/Med fit filters, so every job the daily cap hadn't reached yet was invisible on All Jobs.
- **Salary** used to read "N/A" on nearly every card: the ATS adapters hard-coded it empty. Now
  Lever's `salaryRange`, Ashby's `compensation` (`includeCompensation=true`) and Greenhouse's
  `pay_input_ranges` (`pay_transparency=true`) fill it. Those shapes come from each API's docs and
  haven't been seen on a live board from this sandbox, which can't reach them. Otherwise
  `core/salary_extract.py` reads it from the description: ranges like "$120K-$150K", "$45-60/hr",
  "GBP 55,000-65,000", "between $160,000 and $210,000", a pay-labeled range with no currency
  sign ("Salary: 160,000 - 210,000 USD"), or a single amount only next to a pay word, ignoring
  figures outside a plausible pay range (funding rounds, signing bonuses). The pipeline applies it
  when a job is first saved (`_with_salary`), when a job saved without pay is seen again (never
  overwriting a stored salary), and when a custom-board job's description is fetched before
  scoring. Scoring also returns the posting's stated pay (`ScoredPosting`, a `JobScore` subclass
  with nullable salary fields, the same pattern as manual_import's), and `store_score()` fills an
  empty salary from it: still "N/A" live while Claude's own summary quoted "$160-210k".
  `backfill_salaries()` (regex only, free) runs at app startup (`api/main.py`'s lifespan, in a
  background thread so a cold start doesn't wait on it; a deploy still shows on existing cards
  within moments) and in every daily run, over active jobs still missing
  pay, reading the description and then the latest score summary.
  `salaryText()` prints hourly pay as whole dollars ("$45-60/hr"; rounding to thousands showed
  "$0k") and uses the currency's symbol. Yearly is the assumed default and gets no suffix
  ("$120k-150k", at the user's ask); any other period keeps one ("/hr", or "/<period>"). The daily
  email's `_salary_text()` matches.
- **Job description formatting.** `core/html_extract.py`'s `html_to_markdown()` turns a
  description's HTML into the Markdown subset `components/Markdown.tsx` renders: `##`/`###`
  headings, `- `/`1. ` lists, `**bold**`, `*italic*`, links, and blank-line-separated paragraphs.
  `extract_readable_text()` flattened it to one line per text node instead, losing headings and
  bullets and splitting every bolded phrase onto its own line (live descriptions read as
  run-together blocks). No Claude call, and about a millisecond per job.
  - Greenhouse and RemoteOK convert their HTML. Lever builds `description` + `lists` (each a
    heading + items) + `additional`, falling back to `descriptionPlain`. Ashby uses
    `descriptionHtml`, falling back to `descriptionPlain`. The Lever/Ashby fields come from their
    docs and haven't been seen live from the sandbox.
  - A custom-board job's description, fetched before scoring, converts the job page's `<main>`/
    `<article>` (`main_content=True`), falling back to the page text when that comes out too
    short.
  - `_upsert_job` replaces a re-seen job's description with the adapter's current one (never with
    None), which is how jobs stored before this get reformatted on their next daily check.
  - For text that's still plain (Adzuna snippets, plain-text fallbacks), `frontend/src/
    descriptionText.ts`'s `tidyDescription()` (used by `JobDetailPage`, inside `.job-description`)
    turns bullet-character lines into lists, short colon/ALL-CAPS lines into headings, and rejoins
    only clear mid-sentence breaks (a lowercase continuation, or a line ending "a"/"the"/...).
    Anything with blank lines or a `##` heading passes through untouched.
- **Profile page layout** (2026-11-04, at the user's ask; `pages/ProfilePage.tsx`, a wide route):
  "What are you looking for?" (`DesiredRoleForm`) on the left and a narrower, sticky "My documents"
  card on the right (`.profile-top`, 2fr/1fr, stacked under 900px): current resume, current cover
  letter (each with Download and a Replace/Upload button that opens the file picker and uploads at
  once, `UploadButton`, so there's no wide native file input) and "Older documents", collapsed,
  still a drop target for retiring a current one. The candidate profile and generated documents
  follow at full width.
- **Job card chips** (2026-11-04): no meta chip uses an Interest/Fit color (green/amber/red), since a
  green salary chip read as "strong fit". Organization is slate (`.meta-chip-org`, new; the
  title line still reads "Title — Organization", which the user wanted kept), location grey, salary violet, posted blue, source outlined.
  The source chip shows only for a job board (asked 2026-11-06): `source_is_board` on each job
  (`digest.SOURCE_IS_BOARD_SQL`, used by `get_digest()` and `GET /api/jobs/ranked`) is Manage
  Sources' `isOrganization()` rule reversed: not an `ats_*` kind and no `company_name`. An
  organization's own page would only repeat the organization chip. It's also hidden when it's the
  same name as the organization, ignoring case and punctuation (a board's own postings). The
  organization and location chips are buttons (`FilterChip`) on New & Noteworthy and All Jobs: a
  click adds that value to the Organization (new) or Location filter, or removes it, and the chip
  shows a check while its filter is on (`toggleFilterValue()` in `JobFilters.tsx`).
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
- `core/scoring.py`'s prompt asks the model to mark the 2-4 most decision-critical phrases in its
  rationale (short phrases, not whole sentences), tagged with the rating each explains:
  `==i:...==` for interest, `==f:...==` for fit. `components/Markdown.tsx` renders them as
  `<mark class="mark-i|mark-f">`, and the wrapper's `rationaleClass(interest, fit)` (in
  `ScoredJobCard.tsx`, used there and on `JobDetailPage`) colors each with exactly its rating's
  chip colors, background and text (`.badge-high/-med/-low`): green High, amber Med, red Low. The
  user asked for highlights in the color of the assessment they support, matching the chips. An
  untagged `==text==` (rationales scored before this) looks like the Med chip.
  Settings' "Color older highlights" (`core/retag.py`, `GET`/`POST /api/settings/retag-highlights`)
  fixes those older ones. A refresh only scores new jobs, so live every card stayed amber. It shows
  how many latest-score summaries have untagged marks and an estimate. Then, in a background thread
  streamed as SSE progress like the pipeline run, it asks Claude (low effort, structured
  `HighlightTags`) only for an `i`/`f` per numbered highlight and splices the tags in, so wording
  and ratings never change. An answer whose length doesn't match the highlights leaves that
  summary as it was. It logs as `parsing` with note "retag highlights", and the card hides once
  nothing's left.
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
  from the current view. It also excludes any job on the person's My Jobs (`NOT_TRACKED_SQL`, an
  `applications` row for them; asked 2026-11-04), so a tracked job never shows here or in the daily
  email, which reads `get_digest()`; removing it from My Jobs brings it back. The card you mark
  Interested stays, saying "Added to My Jobs", until the page reloads. None of these exclusions touches `AllJobsPage`'s `GET /api/jobs/ranked`, which
  is explicitly the uncapped, undismissable "everything" view. Default rank order (both
  `add_to_digest()`'s Python sort and `list_ranked_jobs()`'s SQL `ORDER BY`, kept in sync by hand
  since one sorts in Python and the other in SQL): interest+candidate score, then recency (`posted_at` falling back to `first_seen_at`), then
  salary (`salary_max` falling back to `salary_min`).
- **Application statuses** (D-017) are `saved` ("Drafting"), `waiting`, `interviewing`, `offer`,
  `rejected_no_interview`, `rejected_interviewed`, `withdrawn`, renamed at the user's ask from
  applied/phone_screen/interview/rejected. Labels live in `api.ts`'s `APPLICATION_STATUS_LABELS`,
  order in `APPLICATION_STATUSES`, and the list in `api/applications.py`'s `VALID_STATUSES` and
  `schema.sql`'s CHECK. Since SQLite can't alter a CHECK, `db/connection.py`'s
  `_migrate_application_statuses()` rebuilds `applications` once on a database that still has the
  old names (foreign keys off, `legacy_alter_table` on for the rename): applied → waiting,
  phone_screen/interview → interviewing, and rejected → `rejected_interviewed` if its history ever
  reached a screen or interview, else `rejected_no_interview`. Status history rows are renamed the
  same way.
- **My Jobs** (`pages/MyJobsPage.tsx`), changes the user asked for on 2026-11-02:
  - Rows keep the order they had when the list loaded or a column was last clicked (`order`, a list
    of ids). Editing a row (date, status) no longer re-sorts it. Re-sorting on every edit moved a
    row out from under the cursor, and its native date picker could reopen on whichever row took
    its place. A "Re-sort with your changes" link appears when the order is out of date.
  - The default order is best match: interest + fit score (the job's latest `job_scores` row, which
    `GET /api/applications` now joins in), then most recently applied. Every column sort breaks ties
    the same way, and "Sort by best match" returns to it.
  - The Interest and Fit chips (`FitBadge`, exported from `ScoredJobCard.tsx`) sit under the title,
    12px below it. Columns run Title, Organization (the `company` field; renamed and moved after
    Title at the user's ask), Location, Salary, Applied, Status, Prep. Title has a 300px minimum (200px
    under 1240px) and Organization/Location are capped (170/200px), so most titles fit on one line;
    `/tracker` alone gets `.app-main.extra-wide` (1440px) for the room.
  - Layout (second pass, same day): "How your applications are going" (left, the taller one) and
    "Add a job" (right; swapped at the user's ask so the short card's white space sits on the right)
    sit side by side (`.myjobs-top`, 3fr/2fr, stacked under 900px) so the table starts above the
    fold. The search box,
    an Active / Closed / All toggle (`.seg`; Closed = both rejections and withdrawn, default
    Active) and the sort links form one toolbar inside the table card. That toolbar and the header
    row are `position: sticky` under the app's sticky top bar. `useStickyOffsets()` measures both
    with a ResizeObserver into `--sticky-top`/`--thead-top`. The card has no `overflow-x`, which
    would break sticky.
  - Column fit: cells use `overflow-wrap: break-word` (`word-break: break-word` split "$183k/yr"
    mid-word). Salary, Applied and Status don't wrap. The status menu uses `field-sizing: content`,
    or a native select is as wide as "Rejected - no interview". The Documents column (three
    buttons, the widest column) became **Prep**: one small round "+"/"−" toggle per row
    (`PrepCell`, `.prep-toggle`; green when a tailored document exists, and its tooltip says how
    many) that opens `DocActions` stacked vertically inside the cell. It expands inline rather than as a floating menu, so nothing gets
    clipped by the sticky header.
  - "How your applications are going" counts applications, waiting, interviews (and rate), offers
    and rejections from the loaded list, plus the interview rate by fit rating, with no API call.
    "Ask Jobby what's working" (`core/app_insights.py`, `GET`/`POST /api/applications/insights`) is
    one Claude call in Jobby's voice, on request, about which kinds of jobs are getting interviews
    and which aren't. The answer is kept in the `application_insights` setting so a revisit costs
    nothing.
    - The card shows Jobby's avatar and the opening one-sentence takeaway (the prompt asks for one
      before the first heading; `takeaway()`), plus "Read Jobby's take". The full read-out opens
      in a `Modal`, whose `title` now takes any node so it can carry the avatar.
    - The pop-up says when it was last updated, at the top above the read-out. Only when it's out
      of date (any application's
      `updated_at` is later than `generated_at`, or the job count differs) does it add a Refresh
      button. An always-there "Ask Jobby again" on the card read as confusing.
    - "Talk it over with Jobby" opens the general Jobby chat. `jobby.build_context(None)` includes
      the latest read-out, so Jobby knows what it said.
    - Logs as `profile_rebuild` with note "application insights" (the `llm_calls` CHECK, as with
      Jobby).
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
  exists, a `<button>` (the generate action) otherwise. The download link wears `.doc-download`
  (green `--good` tint, "↓" prefix) so a finished document stands apart from the outlined action
  buttons beside it, at the user's request. `JobDetailPage`'s "Pursuing this one?" card puts
  Track application, both Tailor buttons and Prep with Jobby in one wrapping row
  (`.pursue-actions`), all plain `ghost` buttons (the Tailor ones used to be `primary`, and Track
  sat on its own line); once tracked, the status pill shows above the row;
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
- **Show the deploy command** (the owner's ask, 2026-11-06) whenever a reply hands over a change
  that needs a deploy to go live: merged or about to be, or the owner asks to ship. Give it in full,
  ready to paste into Cloud Shell, in its own `bash` block, as one line in the shape the owner is
  used to (git works from the project folder, and `git pull` fetches the branch):
  `cd ~/projects/"01 - Claude Code Projects/02 - Job Search Assistant" && git checkout main && git pull && git merge origin/<branch> && git push && make deploy ADMIN_EMAIL=<the admin's sign-in address>`
  Leave out the merge when the branch is already on `main`. In the reply use the real address; in
  files, a placeholder. Cloud Shell's `~/projects` must use the SSH remote
  (`git@github.com:mraottth/projects.git`): an HTTPS clone asks for a GitHub password on push.
- **Product Timeline (D-008):** this project uses the same changelog/prompts-log pattern as Shelf
  Life. The in-app Timeline is visible only to approved people, but since the move into
  `mraottth/projects` (D-028) the records behind it (`prompts/`, `DECISIONS.md`, commits) are
  public in the repo, as Shelf Life's are. See `DECISIONS.md` for the full reasoning.
  - **`DECISIONS.md`:** add an entry (next `D-NNN`, same fields as the others) whenever a choice
    or tradeoff is made. Link prompt ids and commits; quote measured numbers; say so rather than
    inventing it if the reasoning wasn't stated.
  - **`prompts/`:** a `Stop` hook (`.claude/settings.json`) runs `scripts/export_prompts.py` after
    every turn, keeping `prompts/<date>_<session>.md/.json` current. Commit those files with the
    related work. **Only prompts that drive the product belong on the Timeline** (the owner's
    rule, 2026-11-06): feature requests, design feedback, bug reports, and questions that led
    straight to a change or a decision. Everything else goes in `prompts/omit.json` ({id: reason},
    "operational: ..." or "general question, no product change: ..."), which also removes it from
    the exported files on the next `--refresh`:
    - housekeeping: commit/push/deploy requests, repo moves, setup and sign-in troubleshooting,
      pasted command output, "OK to run it?" answers, subagent hand-backs, resent duplicates;
    - general questions that changed nothing ("what are the criteria?", "show me the prompts",
      cost breakdowns, ideas that weren't built).

    Label each new prompt either way when exporting (`build_changelog.py` warns about unlabeled
    ones), and drop an omitted prompt from any milestone or decision that cites it.
    `prompts/trim.json` ({id: text}) cuts an operational aside out of an otherwise-relevant prompt. `prompts/categories.json` /
    `commit_categories.json` are hand labels (categories: `ui` · `ai` · `pipeline` · `infra` ·
    `docs` · `discussion`) — commits are categorized first by `commit_categories.json`, then by
    which files they touch (`FILE_CATEGORIES` in `scripts/build_changelog.py`), since this repo's
    commits use conventional-commit types (`feat:`/`fix:`), not Shelf Life's bare `category:`
    prefix. **This hook only fires for a Claude Code session whose project root is this folder** —
    one started at the `projects` repo root triggers the root hook, which exports into *Shelf
    Life's* `prompts/` folder (this happened once, caught and cleaned up before anything was
    committed or deployed). Start Job Search Assistant sessions rooted in this folder.
  - **Importing a Claude Code on the web session:** its commits carry a `Claude-Session:` trailer
    (shown as a "Claude Code on the web" badge on the Timeline) but its prompt-by-prompt transcript
    isn't local. The system handles this fine as-is -- those commits still show up, just without
    linked prompts, same as any commit whose transcript was never captured. To import one anyway:
    `claude --teleport <session id>` from this repo root with a clean working tree, send one
    message so the transcript saves locally, exit, then run `python3 scripts/export_prompts.py
    ~/.claude/projects/<project>/<new session>.jsonl`. Confirmed live that `--teleport` is blocked
    outright ("Cloud sessions are disabled by your organization's policy") on this account's plan
    -- not something to keep retrying. The other way in: run the export *inside* the web session,
    where the transcript does exist (`python3 scripts/export_prompts.py
    ~/.claude/projects/<project>/<session>.jsonl`, then commit `prompts/` and add its omit/category
    entries). That's how `prompts/2026-10-28_ab602548.*` (onboarding v2, Jobby, the daily email,
    D-009 to D-014) got here. A web session's commits stay prompt-less only if nobody does that.
  - **Milestones:** `prompts/milestones.json` feeds the visual timeline on the About page and the
    Timeline page. Every decision must belong to a milestone: when you add a `D-NNN`, add a
    milestone (or extend the latest one) naming it, plus its prompts and commits. A process-only
    decision goes in `NO_MILESTONE` in `scripts/build_changelog.py` (D-007 is). `tests/test_milestones.py`
    fails on a decision with no milestone, an unknown prompt/decision/commit, or ids/dates out of
    order, and the build warns. They had fallen behind once (stopped at M10 while decisions reached
    D-027), caught 2026-11-04.
  - **Build:** `scripts/build_changelog.py` writes `frontend/src/changelog.json` (git-ignored, a
    build artifact) from git, `prompts/`, and `DECISIONS.md`. `make frontend` and `make deploy` run
    it — `make deploy` runs it locally, before `gcloud builds submit`, since Cloud Build's context
    has no `.git` to read commit history from.
- Resumes/cover letters and the job/application DB are real personal data — never relax `.gitignore`
  for `data/` or commit a `.db` file, even for debugging.

## Deploy (Milestone 6 — live on Cloud Run)

Cloud Run + Cloud Scheduler, same `gcloud builds submit` → `gcloud run deploy` pattern as Shelf Life,
with two differences from that project:

- **`--no-allow-unauthenticated` + direct IAP** (`--iap` on `gcloud run deploy`, Cloud Run's
  native IAP integration -- no load balancer). `make iap-setup` grants just `IAP_USER`'s Google
  account access. This app must never be publicly reachable: after any change to the IAP setup,
  confirm from a signed-out/incognito browser that the URL redirects to Google sign-in.
  Multi-user (D-021): `make iap-open` lets any Google account through IAP's sign-in (and
  `make iap-close` undoes it). The app still admits only people the admin approves (api/auth.py),
  so a signed-out browser must still hit Google sign-in, and an unapproved account must see only
  the request-access page. `make deploy` sets `JOBHUNT_ADMIN_EMAIL` (`ADMIN_EMAIL`, defaulting to
  `EMAIL_ADDRESS`; deploy stops if it's empty), `IAP_AUDIENCE`
  (`/projects/<number>/locations/<region>/services/jobhunt`, overridable) and `APP_URL`.
- **A GCS-bucket volume mount for `data/`** (resume blobs, `backups/`, and the pre-Litestream
  database) via Cloud Run's native volume-mount support, so it survives restarts.
  `--max-instances 1` avoids SQLite multi-writer issues entirely.
- **The live database is on local disk, streamed by Litestream** (D-020; `litestream.yml`,
  `scripts/start.sh`, the Dockerfile's pinned v0.3.13 with its SHA-256). SQLite on a Cloud Storage
  FUSE mount doesn't get real locking or durable writes, so `JOBHUNT_DB_PATH=/tmp/db/jobhunt.db`
  and Litestream copies every change to `gs://<bucket>/litestream/jobhunt` within about a second.
  `start.sh` restores that copy at startup, runs the app under `litestream replicate -exec`
  (which flushes on SIGTERM), and the very first time seeds from the old `/data/jobhunt.db`
  (`JOBHUNT_SEED_DB`, via `python -m jobhunt.cli.seed_db`), writing a `/data/litestream-seeded`
  marker. If the copy is ever missing after that, it refuses to start rather than come up empty
  (onboarding) or resurrect the stale seed. Without `LITESTREAM_REPLICA_URL` (local Docker) it's
  plain uvicorn. Deploy flags: `--no-cpu-throttling` (streaming never stalls between requests),
  `--min-instances 0` (still shuts down when idle), 2 GiB (the database now lives in memory-backed
  local disk, plus Chromium).
  **Cold starts (D-029).** After ~15 idle minutes the next visit starts a new server, so the first
  page waits for it. The warm instance that would avoid this (`--min-instances 1`) was priced at
  about $52-57/month with the CPU always on, so it was declined in favor of cheaper fixes:
  `--cpu-boost` (extra CPU during startup); `litestream.yml`'s `snapshot-interval: 1h` (was 24h,
  so a restore replays at most an hour of changes); the salary backfill off the startup path; and
  the Anthropic SDK imported on first use in `core/llm.py` (it was half of the app's import time:
  1.26 s down to 0.55 s locally; `tests/test_startup.py` keeps it that way). Caveat: a deploy briefly overlaps old and new servers, and a write
  to the old one in those seconds can be lost, so don't use the app mid-deploy. Tested locally
  with the real binary and a `file://` replica: seed, a write right before SIGTERM, restore onto
  an empty disk (all rows back), and the refusal; not yet against GCS from here.
- **`--timeout 3600`** (Cloud Run's max; the default is 300 s). A full daily run -- polling plus up
  to `daily_llm_cap` serial scoring calls -- and a web-search discovery call can both outlast five
  minutes, and the UI's "Run now" waits on the whole run in one request.

The image also bundles Playwright + Chromium (for `custom_board`'s JS-rendering fallback), which
makes it noticeably bigger than Shelf Life's — expect a slower Cloud Build and a slower cold start.

Deployed to the `job-searcher-512814` GCP project, service `jobhunt`, region `us-central1`.
The URL isn't recorded in this repo; get it with `gcloud run services describe jobhunt --project
job-searcher-512814 --region us-central1 --format='value(status.url)'` or from the Cloud Run
console. Deploy from this folder in Cloud Shell's checkout of `mraottth/projects` (`cd
~/projects/"01 - Claude Code Projects/02 - Job Search Assistant"`), with `.deploy.local.mk` copied
there from the old checkout. One-time setup order (see the Makefile): `apis-setup` → `bucket-setup` → `anthropic-secret`
/ `scheduler-secret` (/ `email-secret` for the daily email) → `deploy` → `iap-setup` →
`scheduler-setup` (the daily run at 2:00pm ET), then `bucket-protect` once; after that, `make
deploy` alone ships every update. Every one of these is real spend/real infrastructure, so don't run them from
here without the user present.
