# Job Search Assistant

Personal tool: aggregates job postings, scores interest/candidate fit with Claude, tailors
resumes/cover letters, and tracks applications. Private, single-user — see `CLAUDE.md` for
architecture and commands.

**Status:** Milestones 1–5 complete (scaffold, resume ingestion, manual job scoring + digest,
feedback loop + application tracker, automated source adapters + pre-filter + daily cap, tailored
resume/cover-letter generation) — runs fully locally, plus a cold-start pass (Profile page for
desired titles/locations/salary, Claude+web-search company discovery, favorite companies that
bypass the daily cap and get flagged in the digest). Milestone 6 (Cloud Run deploy) has a Dockerfile
and Makefile targets ready but hasn't actually been deployed; see `CLAUDE.md` → Deploy.
