"""The pre-filter -> LLM-scoring cap/backlog mechanism. A job's prefilter_score sets its initial
priority; anything left pending past today's cap gets an age bonus each run so it isn't starved
indefinitely by a stream of new, higher-scoring jobs -- this *is* the backlog carry-over."""

import json

from jobhunt.db.connection import get_db
from jobhunt.core.users import current_user_id

AGE_BONUS_PER_DAY = 5.0


def enqueue(job_id: int, prefilter_score: float, matched_rules: dict) -> None:
    user_id = current_user_id()
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id, status FROM scoring_queue WHERE user_id = ? AND job_id = ?", (user_id, job_id)
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
               (user_id, job_id, prefilter_score, priority, matched_rules_json, status)
               VALUES (?, ?, ?, ?, ?, 'pending')
               ON CONFLICT(user_id, job_id) DO UPDATE SET status = 'pending',
                   prefilter_score = excluded.prefilter_score, priority = excluded.priority,
                   matched_rules_json = excluded.matched_rules_json, queued_at = datetime('now')""",
            (user_id, job_id, prefilter_score, prefilter_score, json.dumps(matched_rules)),
        )


def _apply_age_bonus() -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE scoring_queue SET priority = prefilter_score + "
            "(julianday('now') - julianday(queued_at)) * ? WHERE user_id = ? AND status = 'pending'",
            (AGE_BONUS_PER_DAY, current_user_id()),
        )


def select_batch(cap: int, source_ids: list[int] | None = None) -> list[int]:
    """Up to `cap` pending jobs, marked in_progress, taken round-robin across sources: every
    source's best job first, then every source's second-best, and so on (highest priority first
    within each round). A plain top-N by priority let one big board whose postings all match the
    titles/sectors (Civis, live) take every slot of onboarding's 20-job first run, leaving every
    other source unscored."""
    _apply_age_bonus()
    only_sources = ""
    if source_ids:
        # Ints from our own DB, so inlining them is safe; it keeps the LIMIT parameter simple.
        only_sources = f"AND j.source_id IN ({','.join(str(int(i)) for i in source_ids)})"
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT id, job_id FROM (
                   SELECT q.id, q.job_id, q.priority,
                          ROW_NUMBER() OVER (
                              PARTITION BY j.source_id ORDER BY q.priority DESC, q.id
                          ) AS source_rank
                     FROM scoring_queue q JOIN jobs j ON j.id = q.job_id
                    WHERE q.user_id = ? AND q.status = 'pending' {only_sources}
               )
               ORDER BY source_rank, priority DESC, id
               LIMIT ?""",
            (current_user_id(), cap),
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
        # Everyone's: it runs at the start of a run, before any of them is in progress.
        cursor = conn.execute("UPDATE scoring_queue SET status = 'pending' WHERE status = 'in_progress'")
    return cursor.rowcount


def mark_scored(job_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE scoring_queue SET status = 'scored', scored_at = datetime('now') "
            "WHERE user_id = ? AND job_id = ?",
            (current_user_id(), job_id),
        )


def mark_skipped(job_id: int) -> None:
    """Takes a job out of the queue for good (a general-application posting: nothing to score)."""
    with get_db() as conn:
        conn.execute(
            "UPDATE scoring_queue SET status = 'skipped' WHERE user_id = ? AND job_id = ?",
            (current_user_id(), job_id),
        )


def mark_pending_again(job_id: int) -> None:
    """Revert an in_progress row to pending on a scoring failure, so it's retried, not dropped."""
    with get_db() as conn:
        conn.execute(
            "UPDATE scoring_queue SET status = 'pending' WHERE user_id = ? AND job_id = ?",
            (current_user_id(), job_id),
        )


def pending_count() -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM scoring_queue WHERE user_id = ? AND status = 'pending'",
            (current_user_id(),),
        ).fetchone()
    return row["n"]
