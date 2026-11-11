# Decisions

The choices that shaped this app, oldest first. Each entry links the prompts where it was discussed
(ids from [`prompts/`](prompts/)) and the commits that carried it out. Where the reasoning wasn't
recorded, the entry says so rather than inventing it.

This log is a backfill, started 2026-10-30, not a complete retroactive audit. It covers the
decisions from two Claude Code sessions that were clear enough to write up accurately:
- this machine's own session (prompts/2026-10-26_e2f9082b.*), D-001 to D-008;
- the Claude Code on the web session that built the onboarding wizard, Jobby, the daily email,
  salary extraction and discovery steering (prompts/2026-10-28_ab602548.*), D-009 onward.

`claude --teleport` couldn't pull the web session's transcript here, because cloud sessions are
disabled by this account's org policy. So that session exported its own prompt log from inside
the cloud container with `scripts/export_prompts.py`, and its decisions were added afterwards.

Categories: `ui` · `ai` · `pipeline` · `infra` · `docs`

## D-001 · LinkedIn/Indeed/Glassdoor via manual paste-in, not scraping
- **Date:** 2026-10-26
- **Category:** pipeline
- **Prompts:** e2f9082b-002, e2f9082b-003

**Decision.** The big three job sites are handled by the user pasting in a URL or raw text, which Claude parses into structured fields — not automated scraping.

**Context.** All three actively restrict automated scraping via ToS and anti-bot measures; none has a public job-search API (official APIs are employer-only or don't exist for this use case).

**Alternatives considered.** Automated scraping anyway; skipping these sources entirely.

**Why.** Scraping these specific sites risks account/IP bans for a personal tool with no real upside over a 10-second paste. Automated aggregation instead comes from legitimate sources: aggregator APIs, public ATS job-board endpoints (Greenhouse/Lever/Ashby), and a generic custom-board adapter for anything else.

## D-002 · Lock the deployed app down with Identity-Aware Proxy, not public access
- **Date:** 2026-10-26
- **Category:** infra
- **Prompts:** e2f9082b-005

**Decision.** Once deployed to Cloud Run, the service is gated by IAP to the user's own Google account — not `--allow-unauthenticated` (Shelf Life's pattern).

**Context.** This app holds real personal data: resume contents, target companies, application statuses. Shelf Life is a public portfolio piece with no such data, so its pattern doesn't transfer.

**Why.** The whole point of hosting this anywhere is so the daily pipeline runs without the laptop on — nobody else should ever be able to reach it.

## D-003 · Headless-browser fallback for JS-rendered job boards
- **Date:** 2026-10-26
- **Category:** pipeline
- **Prompts:** e2f9082b-007

**Decision.** `core/page_fetch.py` tries a plain static GET first and only falls back to Playwright (headless Chromium) when that yields too little text.

**Context.** Tested techjobsforgov.com directly: it's a WordPress/WP Job Manager site that only renders its listings via client-side JS — a static fetch returns an almost-empty shell.

**Alternatives considered.** Skipping sources that need JS rendering; always using a headless browser (slower, heavier image).

**Why.** Most boards don't need it, so paying the Playwright cost only when the static fetch actually comes back thin keeps the common case fast and the Docker image's extra weight justified only where it's needed.

## D-004 · Tailor resumes by editing the real file in place, not rebuilding from scratch
- **Date:** 2026-10-27
- **Category:** ai
- **Prompts:** e2f9082b-047, e2f9082b-048
- **Commits:** 899346e

**Decision.** Resume tailoring proposes a short list of context-anchored edits (replace/insert/delete, each anchored to one line's exact existing text) applied as native Word tracked changes to a copy of the user's real template — not a full regeneration from a handful of cloned "exemplar" paragraphs.

**Context.** The previous approach (delete the whole document body, rebuild it from ~6 cloned paragraph shapes) produced consistently bad formatting: every bullet ended up sharing one global list-numbering definition regardless of how many distinct lists the real resume had, and tables were silently orphaned.

**Alternatives considered.** Convert the resume to a Google Doc and submit edits via the Docs API's native suggested-edit mode (`writeControl.writeMode: "SUGGEST"`, GA since 2026-09-30, no load balancer needed) — genuinely viable and close to what the user first proposed, but would need a new OAuth/Drive integration and a stored credential for a project with no Google API dependency at all today.

**Why.** Both options give a real accept/reject review experience (a `.docx` with tracked changes opens in Google Docs as Suggesting-mode on import too). The local-file approach gets there with zero new infrastructure, reusing the storage/download plumbing that already existed.

## D-005 · A separate, dedicated GCP project for this app
- **Date:** 2026-10-27
- **Category:** infra
- **Prompts:** e2f9082b-055

**Decision.** Deploy to `job-searcher-512814`, not Shelf Life's `book-recommender-406619`.

**Context.** An unused, billing-enabled project with that name already existed, apparently set up in advance for this.

**Why.** Keeps this app's sensitive personal data isolated from a public portfolio project's GCP project, rather than mixing them.

## D-006 · Cloud Scheduler authenticates through IAP with a real identity, not a bare header
- **Date:** 2026-10-28
- **Category:** infra
- **Commits:** b069971

**Decision.** The daily pipeline's Cloud Scheduler job authenticates with an OIDC identity token (audience = the IAP OAuth client id) from the default compute service account, which is granted `run.invoker` + `iap.httpsResourceAccessor` for this. The original plan's custom `X-Scheduler-Secret` header is kept as a second check, but is no longer the actual security boundary.

**Context.** Confirmed live: once `--iap` protects the whole service, IAP itself rejects a request with no Google-signed identity (`401 Invalid IAP credentials: empty token`) before it ever reaches the app's own header check. A plain header only ever worked because the service was merely `--no-allow-unauthenticated`, pre-IAP.

**Why.** This is the documented pattern for letting a non-interactive caller through IAP (Cloud Scheduler can't do an interactive Google sign-in) — not a workaround.

## D-007 · Merge the cloud-session branch straight into main
- **Date:** 2026-10-30
- **Category:** infra
- **Prompts:** e2f9082b-072, e2f9082b-073

**Decision.** Fast-forward `main` onto a 50-commit Claude Code on the web branch (onboarding wizard, Jobby, email digests, salary extraction, discovery steering, settings overhaul) without a separate review pass.

**Context.** The branch's merge-base was exactly `main`'s current tip — a clean linear continuation, zero conflicts. It's the user's own work from a session they ran.

**Alternatives considered.** Reviewing a structured summary of the diff before merging.

**Why.** No daylight between "review it first" and "it's already mine" for a conflict-free fast-forward; verified with the full test suite and a frontend build immediately after instead.

## D-008 · Bring the Timeline/decision-log system here too, now that the app is private
- **Date:** 2026-10-30
- **Category:** docs
- **Prompts:** e2f9082b-072

**Decision.** Port Shelf Life's Timeline page, `DECISIONS.md`, and the `prompts/` export pipeline (Stop hook + `scripts/export_prompts.py`) to this repo too — superseding the explicit decision recorded earlier in `CLAUDE.md` *not* to.

**Context.** The earlier note's reasoning was that this machinery exists on Shelf Life because its dev history is a *public* product feature, and here it would "just be noise (and a privacy problem)." That reasoning predates this app being locked down behind IAP (D-002) — with the whole app private to one Google account, a Timeline page carries the same exposure as any other page in it, i.e. none.

**Why.** Revisited now that the privacy objection no longer holds, at the user's request, for the same reason it's valuable on Shelf Life: a browsable record of what was built, why, and in what order.

## D-009 · Onboarding: only for users with no documents, skippable, ending in a real first run
- **Date:** 2026-10-28
- **Category:** ui
- **Prompts:** ab602548-003, ab602548-004, ab602548-013, ab602548-014
- **Commits:** 8027d3d, 906a15c, dce7234, ffc96b4

**Decision.** The wizard only appears when there are no uploaded documents and onboarding hasn't been completed, and every step can be skipped. The candidate-profile step allows both a direct edit and "Ask Claude to change it". The last step runs the pipeline instead of dropping the user on an empty digest, but scores only the top 20 jobs (the rest stay queued). Scoring itself moved from high to medium effort.

**Context.** A new user landed on a blank New & Noteworthy page. The first version of the last step took many minutes, since it waited on a full daily run at high effort.

**Alternatives considered.** Required steps; just going to the (empty) digest at the end; scoring the full daily limit on the first run; low or unchanged scoring effort.

**Why.** The user picked the recommended option each time. An existing user with documents, including bulk-imported ones, is never forced through the wizard. Twenty scored jobs gives the digest something to show straight away. Medium effort trades a little care for faster, cheaper scores, and `SCORING_EFFORT` can go back up if ratings start missing nuance.

## D-010 · Jobby: one saved conversation per job, web search on, My Jobs plus a general chat
- **Date:** 2026-10-28
- **Category:** ai
- **Prompts:** ab602548-022, ab602548-023, ab602548-024, ab602548-027
- **Commits:** f719a86, 3944384, f73b19e, ad24c2b

**Decision.** Jobby keeps one saved conversation per job plus one general one. It can search the web (up to three searches a turn), and its picker lists the jobs on My Jobs plus "No specific job". The posting, profile and documents sit in a cached system prompt, and replies stream.

**Alternatives considered.** Conversations that last one visit; offline only; any job in the database.

**Why.** The user chose the recommended options. A mock interview can then be picked up the next day, and the research checklist can use the company's recent news. Caching the large context keeps later turns cheap. The "Questions to ask them" starter was dropped later at the user's request.

## D-011 · Score a mix of sources each run, and read careers pages like a browser would
- **Date:** 2026-10-28
- **Category:** pipeline
- **Prompts:** ab602548-028, ab602548-031, ab602548-033
- **Commits:** 3300b0e, a570d76, fed313f

**Decision.** Each run takes jobs for scoring round-robin across sources rather than a plain top-N. A careers page is checked first for an embedded Greenhouse/Lever/Ashby board, which is then read through that ATS's API with no Claude call. A page where Claude finds nothing is re-rendered in a headless browser and read again. New sources are checked as soon as they're added.

**Context.** All Jobs showed only Civis: its postings all matched the titles, so they took every slot of the 20-job first run, and unscored jobs were hidden by the default filters. "Test fetch" also found nothing on careers pages such as blueroseresearch.org/jobs, whose listings load through JS or an ATS iframe.

**Why.** One big board shouldn't crowd out every other source. And a careers page has enough nav and footer text that the old "too little text, so use the browser" rule never fired.

## D-012 · Daily email over Gmail SMTP: High interest with High or Med fit, nothing on empty days
- **Date:** 2026-10-29
- **Category:** pipeline
- **Prompts:** ab602548-040, ab602548-041
- **Commits:** 60f4d6c

**Decision.** The daily run moved to 2:00pm ET and ends by emailing new New & Noteworthy jobs with High interest and High or Med fit, each one once. The email is sent through Gmail SMTP with an app password stored in Secret Manager, and a day with no matches sends nothing.

**Alternatives considered.** Resend (an email API); High on both ratings; anything High or Med; a short "nothing new" email.

**Why.** The user picked the recommended options. Both senders are free at this volume, and a Gmail app password needs no new account.

## D-013 · Keep run progress flowing over one streamed request instead of polling a background job
- **Date:** 2026-10-29
- **Category:** pipeline
- **Prompts:** ab602548-047, ab602548-053
- **Commits:** 24cb942, 647ac27

**Decision.** "Check every source now" starts the run in a thread and streams progress events every 2 seconds over the same request. If the stream drops, the page falls back to polling. Web-page boards are read at low effort, and the progress line names the sources still being read.

**Context.** The button waited silently on one request for the whole run and got "Lost the connection to the app". Discovery's "Load failed" had the same cause.

**Alternatives considered.** A purely background run that the page polls (what discovery uses).

**Why.** On Cloud Run, CPU is throttled outside an open request, so a run nobody is streaming from would crawl between polls. Keeping the request open keeps the CPU allocated. Low effort removed a minute or more of reasoning per board spent on what is mechanical copying.

## D-014 · Re-tag old rationale highlights with a small Claude pass, not a re-score or a heuristic
- **Date:** 2026-10-29
- **Category:** ai
- **Prompts:** ab602548-051, ab602548-052, ab602548-054, ab602548-055
- **Commits:** b3e920d, eaf8c33, 6df34b9

**Decision.** New scores tag each highlight as explaining interest or fit, drawn in exactly that rating's chip color. Older rationales are fixed by a Settings button that asks Claude only for an interest/fit tag per highlight and splices the tags in, leaving the wording and ratings unchanged.

**Context.** After deploying, every highlight was still amber, because a refresh only scores new jobs.

**Alternatives considered.** Waiting for new jobs to replace the old ones; guessing the tag from nearby words with no Claude call.

**Why.** The user chose the re-tag ("#2"). It's roughly $1 for a few hundred jobs, runs only when the user clicks, and the guess would sometimes pick the wrong color.

## D-015 · Read a careers page harder when it shows only a posting or two
- **Date:** 2026-10-31
- **Category:** pipeline
- **Prompts:** ab602548-060

**Decision.** A web-page source that yields fewer than three postings is re-rendered in the browser, which also checks for job-board API calls the page's scripts make. If it still yields fewer than three, the one hosted ATS board the page links to (Workable, BambooHR, Workday and others) is read instead. The biggest result wins, and Test fetch lists each step it took.

**Context.** Test fetch found 1 posting on bluestate.co/careers and 0 on codeforamerica.org/jobs, both with many openings. Neither site was reachable from the dev sandbox, so the exact cause couldn't be confirmed. These are the three ways the old flow could miss them: it only re-rendered when zero postings were found, it couldn't see a board named only in a script's network request, and it never followed "See all openings" to an ATS it has no adapter for.

**Alternatives considered.** Writing a dedicated adapter for each extra ATS.

**Why.** The browser step costs time rather than money. A second Claude read happens only when the rendered page has more links than before, so an organization that really has one opening isn't billed twice every day. The step list lets the next live Test fetch show which path worked, or that none did.

## D-017 · Application statuses that say how far it got, and a frozen row order on My Jobs
- **Date:** 2026-11-02
- **Category:** ui
- **Prompts:** ab602548-066

**Decision.** The statuses become Drafting, Waiting, Interviewing, Offer, Rejected - no interview, Rejected - interviewed, and Withdrawn. They are renamed in the database too, not just relabeled, through a one-time rebuild of the `applications` table that maps existing rows, including splitting old rejections by whether their history reached an interview. My Jobs rows keep their order until the page reloads or a column is clicked. The default order is best match (interest + fit score), then most recent application. The page gets status filter chips, counts of how applications are going, and an on-demand Claude read-out of what's working.

**Context.** The user asked for all of this together. Splitting "Rejected" needs a value the old CHECK constraint didn't allow, so a label-only change (how "Drafting" was done) couldn't work. Re-sorting on every edit also caused a bug: changing a date moved the row, and the date picker could reopen on another row.

**Why.** "Rejected without an interview" and "rejected after interviewing" are different signals: one says the match looked weak on paper, the other that it got through. That difference is what the insights, and the preference profile built from application history, need. The counts cost nothing, and the Claude read-out runs only on request and is cached, so visiting the page stays free.

## D-018 · Cache the per-run candidate context for scoring, and web-search turns
- **Date:** 2026-11-02
- **Category:** ai
- **Prompts:** ab602548-067

**Decision.** Scoring moves the candidate context into a cached system block, with only the posting left in the user message. Discovery, company lookup and Jobby use automatic caching. Cached tokens are logged and priced, and Settings shows the hit rate and the saving.

**Context.** Anthropic emailed that the account's cache hit rate was low, estimating "up to 56%" savings. Only Jobby's system prompt was cached. Every scoring call repeated about 2,000-2,600 tokens of identical context (instructions, profiles and preferences), with only the 800-1,500-token posting varying.

**Alternatives considered.** The 1-hour cache lifetime, and caching tailoring and board-page reading.

**Why.** Estimated from the code, not measured: about 15-25% off scoring and 20-40% off each web-search call, roughly 15-30% of the bill overall. 56% is Anthropic's upper bound; output tokens and the varying postings and pages can't be cached. Scoring calls in a run start well under 5 minutes apart, so the default lifetime stays warm on its own, and the 2x write of the 1-hour lifetime would buy nothing. Tailoring and board reading never repeat a prefix within 5 minutes.

## D-019 · Daily database snapshots in the app, plus a downloadable copy
- **Date:** 2026-11-02
- **Category:** infra
- **Prompts:** ab602548-073

**Decision.** The app snapshots its SQLite database (SQLite's online backup API) into `backups/` in the data bucket: daily before each run (30 kept), and before "Start over" or a data migration. Settings gets "Download a backup" and My Jobs gets "Export CSV" for copies kept off Google Cloud. `make bucket-protect` makes sure the bucket's 7-day soft delete is on.

**Context.** Asked where the data was backed up, the answer was that it wasn't. The one live SQLite file on a Cloud Storage FUSE mount was the only copy. Google warns that such mounts don't give databases normal locking or durable writes. "Start over" and the status migration rewrite data with no copy first.

**Alternatives considered.** Bucket object versioning; a second bucket in another region.

**Why.** Soft delete already keeps overwritten objects for 7 days. The database is rewritten on every save, so versioning would mostly add copies of the same churn. Snapshots taken by the app are consistent copies at known points (before a run, a reset, a migration) and easy to find and restore. The download covers losing the bucket or project, without a second paid bucket.

## D-020 · Litestream: the live database on local disk, streamed to Cloud Storage
- **Date:** 2026-11-02
- **Category:** infra
- **Prompts:** ab602548-076, ab602548-077, ab602548-079, ab602548-081

**Decision.** The live SQLite database moves off the Cloud Storage FUSE mount onto the server's own disk, and Litestream (v0.3.13) streams every change to `gs://<bucket>/litestream/jobhunt` within about a second. It restores from there at startup and flushes on shutdown. The first start seeds from the old `/data/jobhunt.db` and leaves a marker; if the copy is later missing, the server refuses to start rather than come up empty. Cloud Run gets `--no-cpu-throttling`, `--min-instances 0` and 2 GiB. The D-019 snapshots continue, as a second layer.

**Context.** Step 1 of the multi-user plan (up to about 20 occasional users, each approved by the owner), and worth doing even for one user. Google warns that FUSE mounts don't give databases normal locking or durable writes, and the database was the one file everything depends on.

**Alternatives considered.** Cloud SQL Postgres (the cheapest tier costs about $10 a month or more, plus a rewrite of every query, for capacity 20 sparse users don't need); keeping the FUSE mount and relying on the daily snapshots alone (up to a day of changes lost); running more than one server copy (SQLite needs one writer).

**Why.** Litestream is free apart from a little storage, needs no code changes beyond the startup script, and makes every write durable within a second while SQLite keeps local-disk speed. One server copy stays the limit, which suits the expected load. CPU always-on while the server is up costs a little more per active minute, but with min instances 0 it still scales to zero when idle.

## D-021 · Multi-user: per-person data, shared sources, IAP sign-in with admin approval
- **Date:** 2026-11-02
- **Category:** infra
- **Prompts:** ab602548-075, ab602548-076, ab602548-078, ab602548-079, ab602548-081

**Decision.** The app serves up to about 20 people. Nobody gets in until the admin approves their request.
- Every per-person table gets a `user_id`, and the existing data migrates to the admin as user 1.
- Sources are shared rows that people follow, and each is read once per run.
- Who's signed in comes only from IAP's verified JWT. A context variable carries the person through each request and the daily run, and it fails closed when nobody is set.
- Unapproved people see only a request-access page. The admin approves, declines or blocks people and sets each person's daily scoring cap on `/admin`.
- The Timeline's data moves out of the frontend bundle into an API that only approved people can reach. It was admin-only at first; the owner opened it to everyone approved.

**Context.** The owner asked what it would take to let others use the app, with join requests they approve to keep costs in check. They expect at most 20 users with sparse use. The settled plan: SQLite plus Litestream (D-020), one server copy, shared sources, scoring only for people active in the last 7 days with a one-time "paused" email, and New & Noteworthy freshness controls. This is steps 2 and 3 of that plan; the activity rule, budgets and refresh come next.

**Alternatives considered.**
- Postgres, for capacity a 20-person sparse load doesn't need.
- A database or deployment per person: 20 copies of everything, and no shared board reads.
- Passing a user id through every function signature. The context variable keeps signatures stable, and its fail-closed check stood in for auditing every call path.
- Trusting IAP's plain email header, which is unsigned.
- Keeping the Timeline bundled into the frontend: once any Google account can pass IAP, the static files are readable before approval.

**Why.**
- Shared sources make each board's daily read cost nothing extra per follower, and the round-robin, per-person scoring keeps each person's prompt cache warm.
- Fail-closed identity (no user, no data; NOT NULL `user_id`) means a missed spot errors in tests rather than leaking. It caught every raw insert in the old tests.
- IAP still does the Google sign-in, so there are no passwords to handle; the app verifies the signed token itself.
- The daily cap moved onto the user, so only the admin can raise it.

## D-022 · Cost controls: score only active people, enforce each person's monthly budget
- **Date:** 2026-11-02
- **Category:** infra
- **Prompts:** ab602548-079, ab602548-080, ab602548-081

**Decision.**
- The daily run scores only people who opened the app in the last 7 days. A person who lapses gets one "your matches are paused" email, and their next visit resumes scoring.
- Each person has a monthly Claude budget ($10 by default, set on /admin). It's enforced where every paid call gets its client. When someone reaches it, their paid features stop with a plain message until next month or until the admin raises it.
- The admin gets an email at 80% and at 100%. The admin's own account has no limit.

**Context.** Step 4 of the multi-user plan (D-021), which the owner wanted in place before inviting anyone, so costs can't run away. They first chose a 3-day activity window, then went back to 7, keeping the paused email ("scratch the 3day idea. Let's keep it at 7. The paused email still makes sense").

**Alternatives considered.**
- A 3-day window: the owner's first choice, reversed.
- Checking the budget in each route: 17 call sites to keep in sync.
- Hard-stopping on the daily cap alone: it doesn't bound Jobby, tailoring or discovery.
- Charging shared board reads to followers: unpredictable per-person costs for work done once for everyone.

**Why.**
- Nobody pays for matches no one reads.
- One choke point means a new paid feature is covered automatically.
- Raising an `LLMError` reuses every route's existing error path.
- A first visit after a lapse fixes everything without the person doing anything.

## D-023 · New & Noteworthy: "Last updated" and a 30-minute Refresh
- **Date:** 2026-11-02
- **Category:** ui
- **Prompts:** ab602548-080, ab602548-081

**Decision.**
- New & Noteworthy shows when the person's matches were last updated, with a Refresh that runs their own check. Refresh has a 30-minute cooldown per person, and scoring stays within today's cap and the month's budget.
- Sources anyone read in the last hour aren't read again; their stored jobs are used instead.
- After a day or more without an update, the line becomes a prominent "Refresh now" card. Once the cap or budget is used, the page says when scoring resumes.

**Context.** Step 5 of the multi-user plan. The owner asked for a last-updated timestamp and a refresh button, alongside the decision to score only people active in the last week (D-022). Someone coming back after a pause otherwise sees stale matches until the next 2pm run.

**Alternatives considered.**
- Refreshing automatically when the page opens: it costs money on every visit.
- No cooldown: the daily cap bounds scoring, but every refresh would still re-read every board.
- Re-reading every source on each refresh: wasteful when someone else just read the same board.

**Why.**
- The cooldown and the one-hour freshness window keep a refresh cheap, mostly shared reads and only new scoring.
- Taking a skipped source's jobs from the database means a second person refreshing right after the first still gets their matches scored.

## D-024 · Deleting an account: everything goes except anonymized spend
- **Date:** 2026-11-02
- **Category:** infra
- **Prompts:** ab602548-079, ab602548-081

**Decision.**
- People can delete their own account from Settings, and the admin can delete anyone else's from the Admin page. Both require typing DELETE.
- It removes everything "Start over" does, plus their settings, files folder and account. A snapshot is taken first.
- Their Claude usage rows stay, unlinked from them, so the month's spend still adds up.
- The admin's own account can't be deleted.
- Coming back means requesting access again.

**Context.** Step 6 of the multi-user plan ("Delete my account" and a privacy note), which the owner approved as part of the full plan. The privacy note on the request page says people can delete what they've added.

**Alternatives considered.**
- Deleting the usage rows too: the admin's spend totals would quietly shrink.
- Also deleting sources nobody follows any more: their jobs can still be on other people's My Jobs, and an unfollowed source is already never read.
- A browser confirm dialog: CLAUDE.md keeps `window.confirm` to the one place the owner asked for it.

**Why.**
- Delete as much as can be deleted without breaking shared rows or the spend record.
- The snapshot keeps a mistaken deletion recoverable for 30 days of backups.

## D-025 · Scoring cost: keep the current settings; stop paying to re-read unchanged board pages
- **Date:** 2026-11-04
- **Category:** ai
- **Prompts:** ab602548-095, ab602548-096a, ab602548-098, ab602548-100

**Decision.**
- Live scoring stays at effort "medium" with the full posting and rationale.
- A web-page board whose text hasn't changed since an earlier read reuses that read's postings, with no Claude call (`page_extractions`, `custom_board._extract_postings`). Relative dates ("3 days ago") are ignored when comparing.

**Context.**
- The owner asked what rescoring would cost, then how to cut the scoring cost. I estimated 2–3¢ a job.
- The Admin page's scoring experiment measured 20 of the owner's judged jobs under both settings:
  - current settings: 0.9¢ a job, 463 output tokens;
  - lean settings (effort "low", a 2–3 sentence rationale, the posting cut at 6,000 characters): 0.8¢ a job, 309 output tokens;
  - the two gave the same interest rating on 95% of jobs and the same fit rating on 85%.
- The owner then asked for savings with no quality risk.

**Alternatives considered.**
- Switch to lean: 6% cheaper, with some fit ratings changing. Not worth the risk at 0.9¢.
- The Batch API (50% off): results come back late, and it needs a second step in the daily run. That's real complexity for about $2.50 per person a month.
- A cheaper model: the most likely to hurt rating quality.

**Why.**
- At under 1¢ a job, scoring is already cheap.
- Board pages are read every day whether or not they've changed, and each read pays for up to 40k characters in and up to 30 postings written out. The same text always yields the same postings, so reusing them can't change what anyone sees.
- How much this saves depends on how often the boards actually change; not yet measured.

**Also seen.** Both settings rated most jobs the owner had added to My Jobs as Low interest (32% High/Med). That points to the interest rating, not the cost, and is still to be looked into.

## D-026 · A location dealbreaker that only hides what it's sure about
- **Date:** 2026-11-04
- **Category:** pipeline
- **Prompts:** ab602548-107, ab602548-108

**Decision.**
- "What are you looking for?" gets an opt-in "Dealbreaker: only show jobs in these locations".
- When on, a job clearly located elsewhere isn't scored, isn't on New & Noteworthy or in the daily email, and is hidden on All Jobs behind a toggle. My Jobs is never filtered.
- A job is hidden only when its location is determined and clearly elsewhere.
  - Remote, unknown and multi-location jobs always stay.
  - A city entry covers its metro area: "DC" includes Arlington, Bethesda, Tysons and so on.

**Context.** The owner wanted the option only if we're confident it won't wrongly filter out good jobs. For example, "DC" must match "Washington DC", "Washington D.C." and "District of Columbia". Jobs whose location can't be determined, or that list several, must stay.
- The owner chose: metro areas count, remote always passes, and failing jobs are hidden everywhere except My Jobs.

**Alternatives considered.**
- **Substring matching** (what the prefilter's location boost does): "DC" misses "District of Columbia", "D.C." and every suburb, and "Washington" would match Seattle.
- **Asking Claude per job:** it costs money, and the result could vary from one read to the next.
- **A geocoding service:** a new dependency, and it can't tell an ambiguous "Arlington" or "Springfield" apart any better.
- **Throwing hidden jobs out of the queue:** switching the setting off couldn't bring them back.

**Why.**
- A local gazetteer of states, about 25 metros with their suburbs, and countries, plus rules that need evidence before hiding, keeps the failure mode on the safe side. A wrong reading shows a job you didn't want rather than hiding one you did.
- The Settings readout shows how each entry was read and lists every location it hides, so mistakes are visible before the switch is turned on. 71 test cases cover the DC spellings, the suburbs, and the keep rules.

**Amended (2026-11-04, prompt ab602548-109).**
- At the owner's ask, a job listing several places is hidden when every place is clearly elsewhere and nothing marks the list as incomplete. "New York, NY; San Francisco, CA" goes for someone who wants DC.
- It's still kept if any place matches, if one can't be read, or if the wording is vague ("+2 more", "3 locations", "multiple", "nationwide").
- A written state code now outranks state names inside a city's name, so "Kansas City, MO" reads as Missouri, not Kansas plus Missouri.

## D-027 · A salary dealbreaker on the low end of the posted range
- **Date:** 2026-11-04
- **Category:** pipeline
- **Prompts:** ab602548-110

**Decision.**
- "Minimum salary" gets an opt-in dealbreaker. A job is hidden when its posted pay, annualized at the low end of the range, is below the minimum.
- It applies everywhere the location dealbreaker does: not scored, not on New & Noteworthy or in the email, behind a toggle on All Jobs, and never on My Jobs.

**Context.** The owner asked to match "an annualized total at the lowest end of a posted range". It came with the Profile redesign and two chip changes (no chip in the Interest/Fit colors; clickable organization and location chips).

**Alternatives considered.**
- **Compare the top of the range:** more lenient, but not what was asked.
- **Treat a missing salary as failing:** most postings list no pay, so it would hide most jobs.

**Why.**
- As with location, it hides only what it's sure about. No pay listed, a non-USD currency, an unclear period, or a bare number under 1,000 all stay.
- The Settings readout shows how many jobs list pay and which would be hidden before it's switched on.

## D-028 · Move into the public projects repo, with a personal email scrubbed from the history
- **Date:** 2026-11-06
- **Category:** infra
- **Prompts:** ab602548-122, ab602548-123
- **Commits:** 6e5cc83, d208597

**Decision.**
- The code moves from the private `mraottth/job-search-assistant` repo into the public `mraottth/projects` portfolio, as `01 - Claude Code Projects/02 - Job Search Assistant/`. Shelf Life moves into the same folder as `01 - Shelf Life (Goodreads Recommender v2)/`. The old repo is archived.
- The full history comes along (`git subtree add`), after a rewrite (`git filter-repo`) that replaced the owner's email address in the prompt log with a placeholder and set the 33 commits authored under that address to the owner's GitHub no-reply address. Every commit got a new hash, so DECISIONS, milestones and commit labels were remapped to them.
- `scripts/redact.py` now replaces any real email address on every prompt-log export and in commit text on the Timeline, so a deploy command can't publish it again.
- The deployed app is unchanged and stays private (IAP plus approval). The admin's address reaches production only at deploy time and in the live database, neither of which is in git.

**Context.** A move like this was done once on Oct 31 and undone at the owner's request. Picking it up again, the history had grown to 170 commits and 135 prompts. The prompt log held the owner's Gmail address 15 times, in the deploy commands. The owner chose to scrub it "as long as that doesn't create any issues in production", and asked that the move not put the live Shelf Life site at risk.

**Alternatives considered.**
- **Move the history as it was:** the earlier choice. It would have published the email address.
- **Copy the current code with one fresh commit:** nothing old published, but the Timeline would lose every commit link.
- **Leave Shelf Life where it is and add this as a new top-level folder:** no change to Shelf Life at all. The owner preferred the shared `01 - Claude Code Projects/` folder.

**Why.**
- The scrub keeps the whole Timeline: every commit, prompt and decision is still linked, and the build's output was compared before and after (identical apart from the commit links).
- Shelf Life's move was checked the same way: its Changelog output is identical, its tests give the same result from the new folder as from the old (58 passed; the same 10 need model files that aren't in git), and its frontend builds. Its live site changes only when it's next deployed.
- This project's tests (473) and frontend build pass from the new folder, and `make deploy` expands the same commands there.

## D-029 · Faster cold starts instead of a warm instance
- **Date:** 2026-11-06
- **Category:** infra
- **Prompts:** ab602548-125, ab602548-126, ab602548-127
- **Commits:** cd32c20

**Decision.**
- Keep `--min-instances 0`, so the server still shuts down after about 15 idle minutes, and make the start that follows faster:
  - `--cpu-boost` on deploy: extra CPU while the server starts;
  - Litestream snapshots every hour instead of every day, so a restore replays at most an hour of changes;
  - the startup salary backfill runs in the background instead of before the first request;
  - the Anthropic SDK is imported on first use instead of at startup.

**Context.** The owner asked what a warm instance would cost. With 1 vCPU, 2 GiB and the CPU always on (which Litestream needs), Cloud Run bills a minimum instance at the full rate around the clock: about $57 a month at list prices, about $52 if the free allowance isn't already used by Shelf Life's warm instance. Turning CPU throttling back on to get the cheaper idle rate (about $19) would stall Litestream between requests.

**Alternatives considered.**
- **A warm instance all day:** no cold starts, about $52-57 a month.
- **Warm only during the day** (a scheduled switch of the minimum between 1 and 0): about $33 a month, and more moving parts.
- **A smaller image:** Cloud Run streams images, so size matters less than it seems, and the one big piece (Chromium) is needed for some job boards.

**Why.**
- Together these cost close to nothing: the boost bills only for startup seconds, and hourly snapshots add some storage in proportion to the database's size.
- The app's import time dropped from 1.26 s to 0.55 s locally; the SDK was about half of it. The other gains (boost, restore) can only be measured on Cloud Run, from the startup time in the logs after an idle spell.

## D-030 · Keep the server warm 9:30am-8:30pm Eastern, with scheduled pings
- **Date:** 2026-11-10
- **Category:** infra
- **Prompts:** ab602548-148, ab602548-150, ab602548-151
- **Commits:** 9a24eb3

**Decision.**
- Cloud Scheduler calls a no-op `GET /internal/ping` every 10 minutes from 9:30am to 8:30pm Eastern, every day (`make keep-warm`; `make keep-warm-off` removes it). A request keeps a Cloud Run instance up for about 15 minutes, so the server stays up through the window, and outside it the app still starts on demand.
- Three Scheduler jobs, since one cron line can't start and stop at half past: 9:30-9:50, 10:00-19:50, 20:00-20:30. They reuse the 2pm job's service account and sign-in audience.

**Context.** After D-029's cheaper cold-start fixes, the owner asked what a warm instance costs, then for a narrower daytime window to save money, and chose 9:30am-8:30pm.

**Alternatives considered.**
- **A warm instance around the clock** (`--min-instances 1`): about $52-57 a month with the CPU always on, which Litestream needs.
- **Switching `--min-instances` between 1 and 0 on a schedule:** every switch deploys a new revision, with the brief old/new overlap in which a database write can be lost (D-020). Pings change nothing about the service.
- **Throttled CPU for a cheaper idle rate (about $19 a month at 2 GiB):** stalls Litestream between requests.

**Why.**
- Cost scales with hours up: 11 hours a day at about $0.08 an hour (1 vCPU, 2 GiB, CPU always on, list prices) is about $26 a month, against $52-57 around the clock.
- The pings don't count as anyone's visit (they aren't `/api` requests), so they don't affect who counts as active for scoring.
- The fourth Scheduler job is past the three free per billing account, about $0.10 a month.

## D-031 · Jobby leans on resumes, the profile and best-fit jobs; cover letters are a writing sample
- **Date:** 2026-11-10
- **Category:** ai
- **Prompts:** ab602548-157
- **Commits:** bbe8b2b

**Decision.**
- Jobby's context puts resumes first and gives the older-document allowance to resumes only.
- Only the current cover letter goes in, cut to 3,000 characters and labeled a sample of how they write. Older cover letters are left out; a line says how many.
- Added: the preference profile, and the 8 jobs that score highest on interest + fit (both High or Med, not dismissed, not the job being prepared for), each with the reason it scored well and its My Jobs status.
- The persona says what each part is for, and not to read what a cover letter emphasizes as what matters most.

**Context.** The owner found Jobby "indexes a little too heavily" on cover letters, "a kind of arbitrary bias, since not all jobs require a cover letter and I haven't uploaded all of them", and asked it to focus on resumes, the profile, and the jobs that score high on interest and fit and the qualities that make them score well.

**Alternatives considered.**
- **Leave cover letters out entirely:** loses the one sample of how they write, which helps when drafting answers in their voice.
- **Keep them but say they're partial:** a long letter still takes a big share of the context and pulls attention whatever the label says.

**Why.**
- Resumes and the profile are the record of what they've done; top-scoring jobs and their reasons show what they're suited to and want. All of those exist for every person, while cover letters depend on which applications needed one.
- About 5,000 characters for the top jobs, against up to 15,000 for the current letter plus 20,000 shared with older documents before.

## D-032 · A public demo recorded from the real app, static and read-only
- **Date:** 2026-11-11
- **Category:** infra
- **Prompts:** ab602548-169, ab602548-170, ab602548-171
- **Commits:** 1b02608

**Decision.**
- A static, read-only demo on GitHub Pages, with a fictional job seeker (Jordan Rivera) whose search looks like the owner's: an analytics leader in civic tech and public health. Every organization, posting and link is made up.
- No server and no Claude calls. `demo/seed.py` builds a throwaway database with the app's own code. `demo/record.py` runs the real app on it and saves its answer to every GET the pages make. A demo build of the frontend (`src/demo.ts`) answers from that recording.
- Visitors can look at everything but change nothing: writes are refused with a toast, and Jobby's message box, adding sources or jobs, and uploads are hidden. Dates move forward to the day of the visit.
- Published by `.github/workflows/jsa-demo.yml` on every push to main that touches the project.

**Context.** The owner wanted people to be able to try the app before requesting an account. Their choices: a static demo, no Claude calls, nothing changeable ("they should just be able to see and navigate all the dummy content"), a persona like their own search, and GitHub Pages.

**Alternatives considered.**
- **A page inside the real app:** it would need part of the app outside Google sign-in, which the app must never allow.
- **A second Cloud Run service with its own database:** real interactivity, but a public server, a key to protect, and a cost.
- **Hand-written fixtures in the frontend:** about 45 endpoints' worth that would drift from the real responses.

**Why.**
- Recording the real backend keeps the demo exactly in step: re-running the build picks up any change in what the app returns, and `tests/test_demo.py` fails if the frontend calls a GET the recording doesn't cover.
- Free to host, no keys, and nothing near the real app's infrastructure or data.
- Building it found a real bug: a rationale with two tagged highlights counted as untagged (3ddb3da).

