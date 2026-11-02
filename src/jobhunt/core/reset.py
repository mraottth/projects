"""Settings' "Delete all my data": wipes everything back to a first-run state, so the next page
load goes through onboarding again (no resumes + onboarding_completed reset to false).

Deletes every row of every table except `llm_calls` -- the Claude usage log is a record of real
money already spent, which the user tracks on Settings, so it survives (its job_id links are
cleared, since those jobs are gone). Also deletes the uploaded and generated document files, but
only ones inside RESUME_STORAGE_DIR: a generated file can live in the user's own local resume
folder (`resume_library_path`'s "AI Generated"), which isn't the app's to delete. A snapshot of the
database is taken first (core/backup.py, `backups/` is outside RESUME_STORAGE_DIR), so a mistaken
reset can still be undone."""

import json
from pathlib import Path

from jobhunt.config import DEFAULT_SETTINGS, RESUME_STORAGE_DIR
from jobhunt.core import backup
from jobhunt.db.connection import get_db

KEPT_TABLES = {"llm_calls"}


def _inside_storage(path: str | None) -> Path | None:
    if not path:
        return None
    resolved = Path(path).resolve()
    return resolved if resolved.is_relative_to(RESUME_STORAGE_DIR.resolve()) else None


def delete_all_data() -> dict:
    kept = backup.snapshot("before-reset")
    with get_db() as conn:
        files = [
            row[0]
            for query in (
                "SELECT stored_path FROM resumes",
                "SELECT file_path FROM generated_documents",
            )
            for row in conn.execute(query).fetchall()
        ]
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if row["name"] not in KEPT_TABLES
        ]
        # Foreign keys can only be toggled outside a transaction; with them off, table order
        # doesn't matter (this also covers leftover tables, like the live DB's favorite_companies).
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")
        try:
            conn.execute("UPDATE llm_calls SET job_id = NULL")
            for table in tables:
                conn.execute(f'DELETE FROM "{table}"')
            for key, value in DEFAULT_SETTINGS.items():
                conn.execute("INSERT INTO settings (key, value_json) VALUES (?, ?)", (key, json.dumps(value)))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

    deleted_files = 0
    for path in files:
        target = _inside_storage(path)
        if target and target.is_file():
            target.unlink()
            deleted_files += 1
    return {
        "tables_cleared": len(tables),
        "files_deleted": deleted_files,
        "backup": kept.name if kept else None,
    }
