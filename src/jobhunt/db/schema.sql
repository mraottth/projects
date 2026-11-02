-- Full schema, applied once at startup if tables are missing (no migration framework; see db/connection.py).
-- "Current" profile rows are just the latest by id/generated_at — no separate history tables.
--
-- Multi-user (D-021): every per-person table has a `user_id` (indexes on it are in
-- db/connection.py's USER_INDEXES, since on an older database the column only exists once the
-- multi-user migration has run). Sources and the jobs read from them are shared; who follows which
-- source is `source_follows`. A pasted-in job has an owner (`jobs.user_id`) and stays private.

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name TEXT,
    note TEXT,                      -- what they wrote when asking for access
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'declined', 'blocked')),
    is_admin INTEGER NOT NULL DEFAULT 0,
    daily_score_cap INTEGER NOT NULL DEFAULT 20,
    monthly_budget_usd REAL NOT NULL DEFAULT 10,
    last_seen_at TEXT,
    paused_email_sent_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at TEXT,
    budget_alert_month TEXT,   -- 'YYYY-MM' the admin was told this person passed 80% of their budget
    budget_block_month TEXT    -- 'YYYY-MM' the admin was told they reached it (core/budget.py)
);

CREATE TABLE IF NOT EXISTS source_follows (
    user_id INTEGER NOT NULL REFERENCES users(id),
    source_id INTEGER NOT NULL REFERENCES sources(id),
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, source_id)
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('aggregator_adzuna', 'aggregator_remoteok', 'ats_greenhouse', 'ats_lever', 'ats_ashby', 'custom_board')),
    name TEXT NOT NULL,
    company_name TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    poll_frequency_hours INTEGER NOT NULL DEFAULT 24,
    last_polled_at TEXT,
    last_success_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    source_id INTEGER REFERENCES sources(id),
    external_id TEXT,
    dedupe_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    remote_type TEXT,
    salary_min REAL,
    salary_max REAL,
    salary_currency TEXT,
    salary_period TEXT,
    description_text TEXT,
    raw_json TEXT,
    url TEXT,
    posted_at TEXT,
    first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'expired', 'removed')),
    needs_review INTEGER NOT NULL DEFAULT 0,
    user_id INTEGER REFERENCES users(id)  -- set only on a pasted-in job: its owner, who alone sees it
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source_id);

CREATE TABLE IF NOT EXISTS scoring_queue (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    prefilter_score REAL NOT NULL DEFAULT 0,
    priority REAL NOT NULL DEFAULT 0,
    matched_rules_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'in_progress', 'scored', 'skipped')),
    queued_at TEXT NOT NULL DEFAULT (datetime('now')),
    scored_at TEXT,
    UNIQUE(user_id, job_id)
);
CREATE INDEX IF NOT EXISTS idx_scoring_queue_status ON scoring_queue(status, priority DESC);

CREATE TABLE IF NOT EXISTS job_scores (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    scored_at TEXT NOT NULL DEFAULT (datetime('now')),
    model TEXT,
    interest_fit TEXT CHECK(interest_fit IN ('High', 'Med', 'Low')),
    interest_score REAL,
    candidate_fit TEXT CHECK(candidate_fit IN ('High', 'Med', 'Low')),
    candidate_fit_score REAL,
    summary_text TEXT,
    rationale_text TEXT,
    raw_response_json TEXT,
    prompt_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_scores_job ON job_scores(job_id);

CREATE TABLE IF NOT EXISTS digest_entries (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    digest_date TEXT NOT NULL,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    rank INTEGER NOT NULL,
    emailed_at TEXT,
    UNIQUE(user_id, digest_date, job_id)
);
CREATE INDEX IF NOT EXISTS idx_digest_entries_date ON digest_entries(digest_date);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    interested TEXT NOT NULL CHECK(interested IN ('interested', 'not_interested', 'maybe')),
    note TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_feedback_job ON feedback(job_id);

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    status TEXT NOT NULL DEFAULT 'saved' CHECK(status IN ('saved', 'waiting', 'interviewing', 'offer', 'rejected_no_interview', 'rejected_interviewed', 'withdrawn')),
    applied_at TEXT,
    resume_doc_id INTEGER,       -- generated_documents.id, no hard FK (avoids circular FK ordering)
    cover_letter_doc_id INTEGER, -- generated_documents.id
    notes TEXT,
    next_action TEXT,
    next_action_date TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, job_id)
);

CREATE TABLE IF NOT EXISTS application_status_history (
    id INTEGER PRIMARY KEY,
    application_id INTEGER NOT NULL REFERENCES applications(id),
    status TEXT NOT NULL,
    changed_at TEXT NOT NULL DEFAULT (datetime('now')),
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_app_status_history_app ON application_status_history(application_id);

CREATE TABLE IF NOT EXISTS resumes (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    kind TEXT NOT NULL CHECK(kind IN ('resume', 'cover_letter')),
    label TEXT,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    raw_text TEXT,
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    source_path TEXT,              -- absolute path on disk, if ingested from a local folder
    source_modified_at TEXT,       -- the source file's own mtime -- recency weighting uses this,
                                    -- not uploaded_at, since uploading is unrelated to how current
                                    -- the document's content actually is
    is_archived INTEGER NOT NULL DEFAULT 0,  -- was found under a folder named/containing "old"
    is_template INTEGER NOT NULL DEFAULT 0,  -- filename contains "template" -- the strongest
                                              -- signal for which doc to clone when generating new
                                              -- ones, overriding the archived/recency ordering
    content_hash TEXT              -- sha256 of normalized extracted text, for cross-run dedup
);
-- idx_resumes_content_hash is created in db/connection.py's MIGRATIONS, not here: on an existing
-- database this column doesn't exist until that migration runs, which happens *after* this script.

CREATE TABLE IF NOT EXISTS candidate_profile (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    text TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT (datetime('now')),
    model TEXT,
    source_resume_ids_json TEXT
);

CREATE TABLE IF NOT EXISTS preference_profile (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    text TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    based_on_feedback_count INTEGER,
    based_on_application_events_count INTEGER,
    model TEXT
);

CREATE TABLE IF NOT EXISTS generated_documents (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    application_id INTEGER REFERENCES applications(id),
    kind TEXT NOT NULL CHECK(kind IN ('resume', 'cover_letter')),
    based_on_resume_id INTEGER REFERENCES resumes(id),
    content_text TEXT NOT NULL,
    file_path TEXT,  -- where the formatted .docx was saved on disk, if the template approach applied
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    model TEXT,
    prompt_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_generated_documents_job ON generated_documents(job_id);

CREATE TABLE IF NOT EXISTS settings (
    user_id INTEGER NOT NULL REFERENCES users(id),
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),  -- whose check this was
    run_date TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    sources_polled INTEGER,
    jobs_fetched INTEGER,
    jobs_deduped INTEGER,
    prefiltered_count INTEGER,
    llm_scored_count INTEGER,
    cap_used INTEGER,
    carried_over_count INTEGER,
    errors_json TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),  -- who it was for; NULL for shared work (reading a board)
    call_type TEXT NOT NULL CHECK(call_type IN ('scoring', 'parsing', 'generation', 'profile_rebuild', 'career_profile')),
    job_id INTEGER REFERENCES jobs(id),
    model TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    note TEXT,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at);

-- Jobby (interview-prep chat): one saved conversation per tracked job, plus one general
-- conversation per person (job_id NULL -- SQLite's UNIQUE allows many NULLs, so "only one general thread"
-- is enforced in core/jobby.py rather than here).
CREATE TABLE IF NOT EXISTS jobby_threads (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    job_id INTEGER REFERENCES jobs(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, job_id)
);

CREATE TABLE IF NOT EXISTS jobby_messages (
    id INTEGER PRIMARY KEY,
    thread_id INTEGER NOT NULL REFERENCES jobby_threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_jobby_messages_thread ON jobby_messages(thread_id, id);
