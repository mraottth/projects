import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from jobhunt.config import DATA_DIR, DB_PATH, DEFAULT_SETTINGS

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# schema.sql's CREATE TABLE IF NOT EXISTS never alters an already-existing table, so a new column
# on an existing table needs an explicit migration here. Each statement is applied once; "duplicate
# column" errors (already applied) are swallowed so this is safe to run on every startup.
MIGRATIONS = [
    "ALTER TABLE sources ADD COLUMN is_favorite INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE scoring_queue ADD COLUMN is_favorite INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE resumes ADD COLUMN source_path TEXT",
    "ALTER TABLE resumes ADD COLUMN source_modified_at TEXT",
    "ALTER TABLE resumes ADD COLUMN is_archived INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE resumes ADD COLUMN is_template INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE resumes ADD COLUMN content_hash TEXT",
    "ALTER TABLE generated_documents ADD COLUMN file_path TEXT",
    # Must run after the content_hash column migration above (order matters: on an existing DB,
    # the column doesn't exist until that ALTER TABLE runs).
    "CREATE INDEX IF NOT EXISTS idx_resumes_content_hash ON resumes(content_hash)",
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for statement in MIGRATIONS:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise


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
