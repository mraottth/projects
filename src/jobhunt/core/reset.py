"""Settings' "Start over": wipes the current person's data back to a first-run state, so their next
page load goes through onboarding again (no resumes + onboarding_completed reset to false).

Multi-user (D-021): only the signed-in person's rows go. That's every per-person table (except
`llm_calls`: the usage log is a record of real money already spent, so it survives, with its
job_id links cleared), their pasted-in jobs, and their source follows. Shared sources and the jobs
read from them stay, for whoever else follows them. Also deletes their uploaded and generated
document files, but only ones inside RESUME_STORAGE_DIR: a generated file can live in the admin's
own local resume folder (`resume_library_path`'s "AI Generated"), which isn't the app's to delete.
A snapshot of the database is taken first (core/backup.py, `backups/` is outside
RESUME_STORAGE_DIR), so a mistaken reset can still be undone."""

from pathlib import Path

from jobhunt.config import RESUME_STORAGE_DIR
from jobhunt.core import backup
from jobhunt.core.users import current_user_id, insert_default_settings
from jobhunt.db.connection import PER_USER_TABLES, get_db

KEPT_TABLES = {"llm_calls"}


def _inside_storage(path: str | None) -> Path | None:
    if not path:
        return None
    resolved = Path(path).resolve()
    return resolved if resolved.is_relative_to(RESUME_STORAGE_DIR.resolve()) else None


def delete_all_data() -> dict:
    user_id = current_user_id()
    kept = backup.snapshot("before-reset")
    with get_db() as conn:
        files = [
            row[0]
            for query in (
                "SELECT stored_path FROM resumes WHERE user_id = ?",
                "SELECT file_path FROM generated_documents WHERE user_id = ?",
            )
            for row in conn.execute(query, (user_id,)).fetchall()
        ]
        own_jobs = "SELECT id FROM jobs WHERE user_id = :uid"
        # Foreign keys can only be toggled outside a transaction; with them off, order doesn't matter.
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")
        try:
            conn.execute("UPDATE llm_calls SET job_id = NULL WHERE user_id = :uid", {"uid": user_id})
            # Children of per-person rows first: they have no user_id of their own.
            conn.execute(
                "DELETE FROM application_status_history WHERE application_id IN "
                "(SELECT id FROM applications WHERE user_id = :uid)",
                {"uid": user_id},
            )
            conn.execute(
                "DELETE FROM jobby_messages WHERE thread_id IN "
                "(SELECT id FROM jobby_threads WHERE user_id = :uid)",
                {"uid": user_id},
            )
            tables = [t for t in PER_USER_TABLES if t not in KEPT_TABLES]
            for table in tables:
                conn.execute(f"DELETE FROM {table} WHERE user_id = :uid", {"uid": user_id})
            conn.execute("DELETE FROM source_follows WHERE user_id = :uid", {"uid": user_id})
            # Their pasted-in jobs, now unreferenced (another person's rows never point at them).
            conn.execute(f"UPDATE llm_calls SET job_id = NULL WHERE job_id IN ({own_jobs})", {"uid": user_id})
            conn.execute(f"DELETE FROM jobs WHERE id IN ({own_jobs})", {"uid": user_id})
            insert_default_settings(conn, user_id)
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
        "tables_cleared": len(tables) + 3,
        "files_deleted": deleted_files,
        "backup": kept.name if kept else None,
    }
