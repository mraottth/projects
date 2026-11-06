"""Per-person settings (D-021): every row belongs to the current user."""

import json

from jobhunt.core.users import current_user_id
from jobhunt.db.connection import get_db


def get_settings() -> dict:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT key, value_json FROM settings WHERE user_id = ?", (current_user_id(),)
        ).fetchall()
    return {row["key"]: json.loads(row["value_json"]) for row in rows}


def get_setting(key: str, default=None):
    with get_db() as conn:
        row = conn.execute(
            "SELECT value_json FROM settings WHERE user_id = ? AND key = ?", (current_user_id(), key)
        ).fetchone()
    return json.loads(row["value_json"]) if row else default


def update_settings(partial: dict) -> dict:
    user_id = current_user_id()
    with get_db() as conn:
        for key, value in partial.items():
            conn.execute(
                "INSERT INTO settings (user_id, key, value_json) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value_json = excluded.value_json",
                (user_id, key, json.dumps(value)),
            )
    return get_settings()
