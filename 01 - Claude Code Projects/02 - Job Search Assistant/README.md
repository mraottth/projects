# Job Search Assistant

A web app that finds job postings at the organizations and job boards you choose, has Claude rate
each one for how interested you'd be and how strong a candidate you are, tailors your resume and
cover letter to the ones you pursue (as Word files with tracked changes), and tracks your
applications. It's a personal project, shared with a few people the owner approves.

* **Daily pipeline:** reads every followed source once a day (ATS APIs for Greenhouse, Lever and
  Ashby; Claude extraction for any other careers page or board, with a headless-browser fallback),
  runs a free rule-based pre-filter, then scores the best postings with Claude within each person's
  daily cap and monthly budget.
* **New & Noteworthy and a daily email:** each day's best new matches, with highlighted reasons,
  opt-in location and salary dealbreakers, and feedback that teaches it what you like.
* **Tailoring:** edits your own resume in place as Word tracked changes, so every change is yours
  to accept or reject; cover letters keep your template's formatting.
* **Jobby:** an interview-prep chat that knows the posting, your documents and your profile.
* **Multi-user:** Google sign-in through Identity-Aware Proxy, admin approval, and per-person data
  isolation on one SQLite database (streamed to Cloud Storage by Litestream).
* **Built in conversation with Claude Code:** the in-app Timeline shows every prompt, commit and
  decision (`prompts/`, `DECISIONS.md`).

The app itself is private (behind IAP and an approval step); this folder is its code. **[Try the demo](https://mraottth.github.io/projects/jsa-demo/)**: a
read-only copy with a fictional job seeker's data, no sign-in needed (`demo/`, `make demo`). See
[`CLAUDE.md`](CLAUDE.md) for architecture, commands and deployment.

**Filetree:**
```
├── src/jobhunt/
│   ├── ingest/      source adapters (Greenhouse, Lever, Ashby, Adzuna, RemoteOK, any web page)
│   ├── pipeline/    the daily run
│   ├── core/        scoring, tailoring, Jobby, digest, email, dealbreakers, users and budgets
│   ├── api/         FastAPI routes, IAP sign-in, admin
│   └── db/          SQLite schema and migrations
├── frontend/        React + Vite + TypeScript
├── scripts/         Timeline build, prompt-log export, container start (Litestream)
├── prompts/         the exported Claude Code prompt log and milestones
└── tests/
```
