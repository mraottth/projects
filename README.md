# Job Search Assistant

Personal tool: aggregates job postings, scores interest/candidate fit with Claude, tailors
resumes/cover letters, and tracks applications. Private, single-user — see `CLAUDE.md` for
architecture and commands.

**Status:** Milestones 1–5 complete (scaffold, resume ingestion, manual job scoring + digest,
feedback loop + application tracker, automated source adapters + pre-filter + daily cap, tailored
resume/cover-letter generation) — runs fully locally. Milestone 6 (Cloud Run deploy) has a
Dockerfile and Makefile targets ready but hasn't actually been deployed; see `CLAUDE.md` → Deploy.
