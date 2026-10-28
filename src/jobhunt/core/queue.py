"""The pre-filter -> LLM-scoring cap/backlog mechanism. A job's prefilter_score sets its initial
priority; anything left pending past today's cap gets an age bonus each run so it isn't starved
indefinitely by a stream of new, higher-scoring jobs -- this *is* the backlog carry-over."""

import json

from jobhunt.db.connection import get_db

AGE_BONUS_PER_DAY = 5.0


def enqueue(job_id: int, prefilter_score: float, matched_rules: dict) -> None:
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id, status FROM scoring_queue WHERE job_id = ?", (job_id,)
        ).fetchone()
        if existing and existing["status"] in ("pending", "in_progress"):
            # Refresh the score but keep the original queued_at so its age bonus keeps accruing.
            conn.execute(
                "UPDATE scoring_queue SET prefilter_score = ?, matched_rules_json = ? WHERE id = ?",
                (prefilter_score, json.dumps(matched_rules), existing["id"]),
            )
            return
        conn.execute(
            """INSERT INTO scoring_queue
               (job_id, prefilter_score, priority, matched_rules_json, status)
               VALUES (?, ?, ?, ?, 'pending')""",
            (job_id, prefilter_score, prefilter_score, json.dumps(matched_rules)),
        )


def _apply_age_bonus() -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE scoring_queue SET priority = prefilter_score + "
            "(julianday('now') - julianday(queued_at)) * ? WHERE status = 'pending'",
            (AGE_BONUS_PER_DAY,),
        )


def select_batch(cap: int) -> list[int]:
    """The `cap` highest-priority pending jobs, marked in_progress."""
    _apply_age_bonus()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, job_id FROM scoring_queue WHERE status = 'pending' "
            "ORDER BY priority DESC LIMIT ?",
            (cap,),
        ).fetchall()
        ids = [row["id"] for row in rows]
        if ids:
            conn.executemany(
                "UPDATE scoring_queue SET status = 'in_progress' WHERE id = ?", [(i,) for i in ids]
            )
    return [row["job_id"] for row in rows]


def requeue_stranded() -> int:
    """Returns jobs left in_progress by a run that never finished (the instance was restarted or
    replaced mid-run) to pending. select_batch only picks pending rows, so without this they'd
    never be scored. Called at the start of each run; runs don't overlap in practice (one daily
    scheduled run, plus the occasional manual one)."""
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE scoring_queue SET status = 'pending' WHERE status = 'in_progress'"
        )
    return cursor.rowcount


def mark_scored(job_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE scoring_queue SET status = 'scored', scored_at = datetime('now') WHERE job_id = ?",
            (job_id,),
        )


def mark_pending_again(job_id: int) -> None:
    """Revert an in_progress row to pending on a scoring failure, so it's retried, not dropped."""
    with get_db() as conn:
        conn.execute("UPDATE scoring_queue SET status = 'pending' WHERE job_id = ?", (job_id,))


def pending_count() -> int:
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM scoring_queue WHERE status = 'pending'").fetchone()
    return row["n"]
