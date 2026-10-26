import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from jobhunt.config import DATA_DIR, DB_PATH, DEFAULT_SETTINGS

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


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
