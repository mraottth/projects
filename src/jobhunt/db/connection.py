import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from jobhunt.config import DATA_DIR, DB_PATH, DEFAULT_SETTINGS

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# schema.sql's CREATE TABLE IF NOT EXISTS never alters an already-existing table, so a new column
# on an existing table needs an explicit migration here. Each statement is applied once; "duplicate
# column" errors (already applied) are swallowed so this is safe to run on every startup.
# (The Favorite feature's `sources.is_favorite` / `scoring_queue.is_favorite` columns and
# `favorite_companies` table were removed from schema.sql; databases created before that still have
# them, unused and harmless.)
MIGRATIONS = [
    "ALTER TABLE resumes ADD COLUMN source_path TEXT",
    "ALTER TABLE resumes ADD COLUMN source_modified_at TEXT",
    "ALTER TABLE resumes ADD COLUMN is_archived INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE resumes ADD COLUMN is_template INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE resumes ADD COLUMN content_hash TEXT",
    "ALTER TABLE generated_documents ADD COLUMN file_path TEXT",
    # Must run after the content_hash column migration above (order matters: on an existing DB,
    # the column doesn't exist until that ALTER TABLE runs).
    "CREATE INDEX IF NOT EXISTS idx_resumes_content_hash ON resumes(content_hash)",
    # When a digest entry went out in the daily email (core/email_digest.py), so it's sent once.
    "ALTER TABLE digest_entries ADD COLUMN emailed_at TEXT",
    # Prompt-cache reads and writes per Claude call (core/llm.py), priced separately from input.
    "ALTER TABLE llm_calls ADD COLUMN cache_read_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE llm_calls ADD COLUMN cache_write_tokens INTEGER NOT NULL DEFAULT 0",
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for statement in MIGRATIONS:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise


# Application statuses were renamed on 2026-11-02 (user's ask): applied -> waiting, phone_screen and
# interview -> interviewing, and rejected split by whether it got as far as an interview. The old
# names are in the applications table's CHECK constraint, which SQLite can't alter, so a database
# from before that gets the table rebuilt once (detected by the old name in its definition).
_NEW_APPLICATIONS_TABLE = """CREATE TABLE applications_new (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
    status TEXT NOT NULL DEFAULT 'saved' CHECK(status IN ('saved', 'waiting', 'interviewing', 'offer',
        'rejected_no_interview', 'rejected_interviewed', 'withdrawn')),
    applied_at TEXT,
    resume_doc_id INTEGER,
    cover_letter_doc_id INTEGER,
    notes TEXT,
    next_action TEXT,
    next_action_date TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)"""
# A rejection counts as "interviewed" if the application ever reached a screen or interview.
_INTERVIEWED = """EXISTS (SELECT 1 FROM application_status_history h WHERE h.application_id = {app_id}
    AND h.status IN ('phone_screen', 'interview', 'interviewing'))"""
_RENAMED = """CASE {col}
    WHEN 'applied' THEN 'waiting'
    WHEN 'phone_screen' THEN 'interviewing'
    WHEN 'interview' THEN 'interviewing'
    WHEN 'rejected' THEN CASE WHEN {interviewed} THEN 'rejected_interviewed' ELSE 'rejected_no_interview' END
    ELSE {col} END"""


def _needs_status_migration(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'applications'"
    ).fetchone()
    return row is not None and "phone_screen" in row["sql"]


def _migrate_application_statuses(conn: sqlite3.Connection) -> None:
    if not _needs_status_migration(conn):
        return
    conn.execute("PRAGMA foreign_keys = OFF")
    # Keep other tables' REFERENCES applications(id) pointing at the renamed table as written.
    conn.execute("PRAGMA legacy_alter_table = ON")
    try:
        conn.execute("BEGIN")
        conn.execute(_NEW_APPLICATIONS_TABLE)
        status = _RENAMED.format(col="a.status", interviewed=_INTERVIEWED.format(app_id="a.id"))
        conn.execute(
            f"""INSERT INTO applications_new SELECT a.id, a.job_id, {status}, a.applied_at,
                   a.resume_doc_id, a.cover_letter_doc_id, a.notes, a.next_action, a.next_action_date,
                   a.created_at, a.updated_at FROM applications a"""
        )
        history = _RENAMED.format(col="status", interviewed=_INTERVIEWED.format(app_id="hist.application_id"))
        conn.execute(f"UPDATE application_status_history AS hist SET status = {history}")
        conn.execute("DROP TABLE applications")
        conn.execute("ALTER TABLE applications_new RENAME TO applications")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA legacy_alter_table = OFF")
        conn.execute("PRAGMA foreign_keys = ON")


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.executescript(SCHEMA_PATH.read_text())
        _apply_migrations(conn)
        if _needs_status_migration(conn):
            # It rewrites every application, so keep a copy of the database as it was first.
            from jobhunt.core import backup  # here, not at the top: backup imports this module

            backup.snapshot("before-migration")
            _migrate_application_statuses(conn)
        for key, value in DEFAULT_SETTINGS.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value_json) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
    finally:
        conn.close()


@contextmanager
def get_db():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()
