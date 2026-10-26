"""Preference-profile getter. The rebuild (from feedback + application outcomes) lands in M3."""

from jobhunt.db.connection import get_db


def get_current_preference_profile() -> str | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT text FROM preference_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["text"] if row else None
