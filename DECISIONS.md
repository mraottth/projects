# Decisions

The choices that shaped this app, oldest first. Each entry links the prompts where it was discussed
(ids from [`prompts/`](prompts/)) and the commits that carried it out. Where the reasoning wasn't
recorded, the entry says so rather than inventing it.

This log is a backfill, started 2026-10-30, not a complete retroactive audit — it covers the
decisions from this machine's own Claude Code session (prompts/2026-10-26_e2f9082b.*) that were
clear enough to write up accurately. A separate block of work (onboarding wizard, the "Jobby"
interview-prep chat, email digests, salary extraction, discovery steering) was built in a Claude
Code on the web session (commits tagged `web_session` on the Timeline) merged straight into main
on 2026-10-30; its own decisions aren't reflected here yet — importing that session's transcript
turned out not to be possible (cloud sessions are disabled by this account's org policy; see
CLAUDE.md → Working conventions), so those commits will stay prompt-less on the Timeline.

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
