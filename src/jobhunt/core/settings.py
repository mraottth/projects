import json

from jobhunt.db.connection import get_db


def get_settings() -> dict:
    with get_db() as conn:
        rows = conn.execute("SELECT key, value_json FROM settings").fetchall()
    return {row["key"]: json.loads(row["value_json"]) for row in rows}


def get_setting(key: str, default=None):
    with get_db() as conn:
        row = conn.execute("SELECT value_json FROM settings WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value_json"]) if row else default


def update_settings(partial: dict) -> dict:
    with get_db() as conn:
        for key, value in partial.items():
            conn.execute(
                "INSERT INTO settings (key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (key, json.dumps(value)),
            )
    return get_settings()
