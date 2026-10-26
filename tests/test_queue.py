from jobhunt.core import queue
from jobhunt.db.connection import get_db


def _insert_job(title="Engineer") -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company) VALUES (?, ?, 'Acme')", (title, title)
        )
        return cursor.lastrowid


def test_enqueue_and_select_batch_respects_priority():
    low_id = _insert_job("Low")
    high_id = _insert_job("High")
    queue.enqueue(low_id, 20.0, {})
    queue.enqueue(high_id, 90.0, {})

    batch = queue.select_batch(cap=1)
    assert batch == [high_id]

    with get_db() as conn:
        row = conn.execute("SELECT status FROM scoring_queue WHERE job_id = ?", (high_id,)).fetchone()
        low_row = conn.execute("SELECT status FROM scoring_queue WHERE job_id = ?", (low_id,)).fetchone()
    assert row["status"] == "in_progress"
    assert low_row["status"] == "pending"


def test_mark_scored_and_pending_again():
    job_id = _insert_job()
    queue.enqueue(job_id, 50.0, {})
    queue.select_batch(cap=1)

    queue.mark_scored(job_id)
    with get_db() as conn:
        row = conn.execute("SELECT status FROM scoring_queue WHERE job_id = ?", (job_id,)).fetchone()
    assert row["status"] == "scored"


def test_mark_pending_again_allows_retry():
    job_id = _insert_job()
    queue.enqueue(job_id, 50.0, {})
    queue.select_batch(cap=1)
    queue.mark_pending_again(job_id)

    assert queue.pending_count() == 1
    assert queue.select_batch(cap=1) == [job_id]


def test_enqueue_is_idempotent_for_pending_jobs():
    job_id = _insert_job()
    queue.enqueue(job_id, 20.0, {"a": 1})
    queue.enqueue(job_id, 80.0, {"a": 2})  # re-prefiltered with a different score

    with get_db() as conn:
        rows = conn.execute("SELECT * FROM scoring_queue WHERE job_id = ?", (job_id,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["prefilter_score"] == 80.0


def test_pending_count():
    queue.enqueue(_insert_job("A"), 50.0, {})
    queue.enqueue(_insert_job("B"), 50.0, {})
    assert queue.pending_count() == 2


def test_favorite_jobs_bypass_the_cap():
    favorite_ids = [_insert_job(f"Fav{i}") for i in range(3)]
    other_ids = [_insert_job(f"Other{i}") for i in range(5)]
    for job_id in favorite_ids:
        queue.enqueue(job_id, 10.0, {"favorite_company": True}, is_favorite=True)
    for job_id in other_ids:
        queue.enqueue(job_id, 50.0, {})

    batch = queue.select_batch(cap=2)

    # all 3 favorites included despite their low raw score, PLUS the full cap of 2 others --
    # total exceeds the cap because favorites don't count against it.
    assert set(favorite_ids).issubset(set(batch))
    assert len(batch) == 5


def test_favorite_jobs_do_not_shrink_the_cap_for_others():
    favorite_ids = [_insert_job(f"Fav{i}") for i in range(10)]
    other_ids = [_insert_job(f"Other{i}") for i in range(4)]
    for job_id in favorite_ids:
        queue.enqueue(job_id, 10.0, {}, is_favorite=True)
    for job_id in other_ids:
        queue.enqueue(job_id, 50.0, {})

    batch = queue.select_batch(cap=4)
    assert len(batch) == 14  # 10 favorites + all 4 others, cap untouched for the non-favorites
