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
- **Commits:** 1d10205

**Decision.** Resume tailoring proposes a short list of context-anchored edits (replace/insert/delete, each anchored to one line's exact existing text) applied as native Word tracked changes to a copy of the user's real template — not a full regeneration from a handful of cloned "exemplar" paragraphs.

**Context.** The previous approach (delete the whole document body, rebuild it from ~6 cloned paragraph shapes) produced consistently bad formatting: every bullet ended up sharing one global list-numbering definition regardless of how many distinct lists the real resume had, and tables were silently orphaned.

**Alternatives considered.** Convert the resume to a Google Doc and submit edits via the Docs API's native suggested-edit mode (`writeControl.writeMode: "SUGGEST"`, GA since 2026-09-30, no load balancer needed) — genuinely viable and close to what the user first proposed, but would need a new OAuth/Drive integration and a stored credential for a project with no Google API dependency at all today.

**Why.** Both options give a real accept/reject review experience (a `.docx` with tracked changes opens in Google Docs as Suggesting-mode on import too). The local-file approach gets there with zero new infrastructure, reusing the storage/download plumbing that already existed.

## D-005 · A separate, dedicated GCP project for this app
- **Date:** 2026-10-27
- **Category:** infra
- **Prompts:** e2f9082b-055, e2f9082b-056

**Decision.** Deploy to `job-searcher-512814`, not Shelf Life's `book-recommender-406619`.

**Context.** An unused, billing-enabled project with that name already existed, apparently set up in advance for this.

**Why.** Keeps this app's sensitive personal data isolated from a public portfolio project's GCP project, rather than mixing them.

## D-006 · Cloud Scheduler authenticates through IAP with a real identity, not a bare header
- **Date:** 2026-10-28
- **Category:** infra
- **Commits:** 0fbe098

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
- **Commits:** 379f309, bb23a91, f37c72a, 42ded0d

**Decision.** The wizard only appears when there are no uploaded documents and onboarding hasn't been completed, and every step can be skipped. The candidate-profile step allows both a direct edit and "Ask Claude to change it". The last step runs the pipeline instead of dropping the user on an empty digest, but scores only the top 20 jobs (the rest stay queued). Scoring itself moved from high to medium effort.

**Context.** A new user landed on a blank New & Noteworthy page. The first version of the last step took many minutes, since it waited on a full daily run at high effort.

**Alternatives considered.** Required steps; just going to the (empty) digest at the end; scoring the full daily limit on the first run; low or unchanged scoring effort.

**Why.** The user picked the recommended option each time. An existing user with documents, including bulk-imported ones, is never forced through the wizard. Twenty scored jobs gives the digest something to show straight away. Medium effort trades a little care for faster, cheaper scores, and `SCORING_EFFORT` can go back up if ratings start missing nuance.

## D-010 · Jobby: one saved conversation per job, web search on, My Jobs plus a general chat
- **Date:** 2026-10-28
- **Category:** ai
- **Prompts:** ab602548-022, ab602548-023, ab602548-024, ab602548-027
- **Commits:** f66caea, 22d3315, 699f323, 90edb75

**Decision.** Jobby keeps one saved conversation per job plus one general one. It can search the web (up to three searches a turn), and its picker lists the jobs on My Jobs plus "No specific job". The posting, profile and documents sit in a cached system prompt, and replies stream.

**Alternatives considered.** Conversations that last one visit; offline only; any job in the database.

**Why.** The user chose the recommended options. A mock interview can then be picked up the next day, and the research checklist can use the company's recent news. Caching the large context keeps later turns cheap. The "Questions to ask them" starter was dropped later at the user's request.

## D-011 · Score a mix of sources each run, and read careers pages like a browser would
- **Date:** 2026-10-28
- **Category:** pipeline
- **Prompts:** ab602548-028, ab602548-031, ab602548-033
- **Commits:** 18cb51d, 18273fe, ba604fc

**Decision.** Each run takes jobs for scoring round-robin across sources rather than a plain top-N. A careers page is checked first for an embedded Greenhouse/Lever/Ashby board, which is then read through that ATS's API with no Claude call. A page where Claude finds nothing is re-rendered in a headless browser and read again. New sources are checked as soon as they're added.

**Context.** All Jobs showed only Civis: its postings all matched the titles, so they took every slot of the 20-job first run, and unscored jobs were hidden by the default filters. "Test fetch" also found nothing on careers pages such as blueroseresearch.org/jobs, whose listings load through JS or an ATS iframe.

**Why.** One big board shouldn't crowd out every other source. And a careers page has enough nav and footer text that the old "too little text, so use the browser" rule never fired.

## D-012 · Daily email over Gmail SMTP: High interest with High or Med fit, nothing on empty days
- **Date:** 2026-10-29
- **Category:** pipeline
- **Prompts:** ab602548-040, ab602548-041
- **Commits:** a473298

**Decision.** The daily run moved to 2:00pm ET and ends by emailing new New & Noteworthy jobs with High interest and High or Med fit, each one once. The email is sent through Gmail SMTP with an app password stored in Secret Manager, and a day with no matches sends nothing.

**Alternatives considered.** Resend (an email API); High on both ratings; anything High or Med; a short "nothing new" email.

**Why.** The user picked the recommended options. Both senders are free at this volume, and a Gmail app password needs no new account.

## D-013 · Keep run progress flowing over one streamed request instead of polling a background job
- **Date:** 2026-10-29
- **Category:** pipeline
- **Prompts:** ab602548-047, ab602548-053
- **Commits:** 28f9cd9, 603caa8

**Decision.** "Check every source now" starts the run in a thread and streams progress events every 2 seconds over the same request. If the stream drops, the page falls back to polling. Web-page boards are read at low effort, and the progress line names the sources still being read.

**Context.** The button waited silently on one request for the whole run and got "Lost the connection to the app". Discovery's "Load failed" had the same cause.

**Alternatives considered.** A purely background run that the page polls (what discovery uses).

**Why.** On Cloud Run, CPU is throttled outside an open request, so a run nobody is streaming from would crawl between polls. Keeping the request open keeps the CPU allocated. Low effort removed a minute or more of reasoning per board spent on what is mechanical copying.

## D-014 · Re-tag old rationale highlights with a small Claude pass, not a re-score or a heuristic
- **Date:** 2026-10-29
- **Category:** ai
- **Prompts:** ab602548-051, ab602548-052, ab602548-054, ab602548-055
- **Commits:** 10af09b, fa6f205, 9d9ec21

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
