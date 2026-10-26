"""The pre-filter -> LLM-scoring cap/backlog mechanism. A job's prefilter_score sets its initial
priority; anything left pending past today's cap gets an age bonus each run so it isn't starved
indefinitely by a stream of new, higher-scoring jobs -- this *is* the backlog carry-over."""

import json

from jobhunt.db.connection import get_db

AGE_BONUS_PER_DAY = 5.0


def enqueue(
    job_id: int, prefilter_score: float, matched_rules: dict, *, is_favorite: bool = False
) -> None:
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id, status FROM scoring_queue WHERE job_id = ?", (job_id,)
        ).fetchone()
        if existing and existing["status"] in ("pending", "in_progress"):
            # Refresh the score but keep the original queued_at so its age bonus keeps accruing.
            conn.execute(
                "UPDATE scoring_queue SET prefilter_score = ?, matched_rules_json = ?, "
                "is_favorite = ? WHERE id = ?",
                (prefilter_score, json.dumps(matched_rules), int(is_favorite), existing["id"]),
            )
            return
        conn.execute(
            """INSERT INTO scoring_queue
               (job_id, prefilter_score, priority, matched_rules_json, is_favorite, status)
               VALUES (?, ?, ?, ?, ?, 'pending')""",
            (job_id, prefilter_score, prefilter_score, json.dumps(matched_rules), int(is_favorite)),
        )


def _apply_age_bonus() -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE scoring_queue SET priority = prefilter_score + "
            "(julianday('now') - julianday(queued_at)) * ? WHERE status = 'pending'",
            (AGE_BONUS_PER_DAY,),
        )


def select_batch(cap: int) -> list[int]:
    """Favorite-company jobs bypass the cap entirely (all pending ones are always included); the
    cap applies in full to everything else, independent of how many favorites there are."""
    _apply_age_bonus()
    with get_db() as conn:
        favorite_rows = conn.execute(
            "SELECT id, job_id FROM scoring_queue WHERE status = 'pending' AND is_favorite = 1"
        ).fetchall()
        other_rows = conn.execute(
            "SELECT id, job_id FROM scoring_queue WHERE status = 'pending' AND is_favorite = 0 "
            "ORDER BY priority DESC LIMIT ?",
            (cap,),
        ).fetchall()
        rows = [*favorite_rows, *other_rows]
        ids = [row["id"] for row in rows]
        if ids:
            conn.executemany(
                "UPDATE scoring_queue SET status = 'in_progress' WHERE id = ?", [(i,) for i in ids]
            )
    return [row["job_id"] for row in rows]


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
