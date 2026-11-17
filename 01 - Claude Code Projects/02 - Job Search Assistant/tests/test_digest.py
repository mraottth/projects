from jobhunt.core import digest
from jobhunt.db.connection import get_db


def _insert_job(
    title: str,
    company: str = "Acme",
    posted_at: str | None = None,
    salary_max: float | None = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO jobs (user_id, dedupe_key, title, company, posted_at, salary_max)
               VALUES (1, ?, ?, ?, ?, ?)""",
            (title, title, company, posted_at, salary_max),
        )
        return cursor.lastrowid


def _score_job(
    job_id: int,
    interest_score: float = 80,
    candidate_fit_score: float = 80,
    interest: str = "High",
    fit: str = "High",
) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO job_scores
               (user_id, job_id, model, interest_fit, interest_score, candidate_fit, candidate_fit_score,
                summary_text, rationale_text)
               VALUES (1, ?, 'x', ?, ?, ?, ?, '', '')""",
            (job_id, interest, interest_score, fit, candidate_fit_score),
        )


def _mark_not_interested(job_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, job_id, interested) VALUES (1, ?, 'not_interested')", (job_id,)
        )


def _days_ago(n: int) -> str:
    with get_db() as conn:
        return conn.execute("SELECT datetime('now', ?) AS d", (f"-{n} days",)).fetchone()["d"]


def test_add_to_digest_skips_postings_older_than_90_days():
    job_id = _insert_job("Old Role", posted_at=_days_ago(120))
    _score_job(job_id)
    digest.add_to_digest(job_id, "2026-01-01")
    assert digest.get_digest("2026-01-01") == []


def test_add_to_digest_includes_unknown_posted_date():
    job_id = _insert_job("Mystery Role", posted_at=None)
    _score_job(job_id)
    digest.add_to_digest(job_id, "2026-01-01")
    entries = digest.get_digest("2026-01-01")
    assert len(entries) == 1
    assert entries[0]["title"] == "Mystery Role"


def test_add_to_digest_includes_recent_posting():
    job_id = _insert_job("Fresh Role", posted_at=_days_ago(10))
    _score_job(job_id)
    digest.add_to_digest(job_id, "2026-01-01")
    entries = digest.get_digest("2026-01-01")
    assert [e["title"] for e in entries] == ["Fresh Role"]


def test_get_digest_excludes_jobs_marked_not_interested():
    job_id = _insert_job("Dismissed Role", posted_at=_days_ago(5))
    _score_job(job_id)
    digest.add_to_digest(job_id, "2026-01-01")
    assert len(digest.get_digest("2026-01-01")) == 1

    _mark_not_interested(job_id)
    assert digest.get_digest("2026-01-01") == []


def test_rank_order_is_score_then_recency_then_salary():
    # Same (high) score for all three -- recency should decide the tie, then salary would, but
    # recency alone already separates job_a from job_b here.
    job_a = _insert_job("Older Higher Salary", posted_at=_days_ago(20), salary_max=200000)
    job_b = _insert_job("Newer Lower Salary", posted_at=_days_ago(1), salary_max=100000)
    _score_job(job_a, 80, 80)
    _score_job(job_b, 80, 80)

    digest.add_to_digest(job_a, "2026-01-01")
    digest.add_to_digest(job_b, "2026-01-01")

    titles = [e["title"] for e in digest.get_digest("2026-01-01")]
    assert titles == ["Newer Lower Salary", "Older Higher Salary"]


def test_rank_order_salary_breaks_tie_when_score_and_recency_match():
    same_day = _days_ago(5)
    job_a = _insert_job("Lower Salary", posted_at=same_day, salary_max=90000)
    job_b = _insert_job("Higher Salary", posted_at=same_day, salary_max=150000)
    _score_job(job_a, 70, 70)
    _score_job(job_b, 70, 70)

    digest.add_to_digest(job_a, "2026-01-01")
    digest.add_to_digest(job_b, "2026-01-01")

    titles = [e["title"] for e in digest.get_digest("2026-01-01")]
    assert titles == ["Higher Salary", "Lower Salary"]


def test_rank_order_score_beats_recency_and_salary():
    job_a = _insert_job("Best Score, Old, Low Salary", posted_at=_days_ago(80), salary_max=50000)
    job_b = _insert_job("Worse Score, New, High Salary", posted_at=_days_ago(1), salary_max=300000)
    _score_job(job_a, 95, 95)
    _score_job(job_b, 40, 40)

    digest.add_to_digest(job_a, "2026-01-01")
    digest.add_to_digest(job_b, "2026-01-01")

    titles = [e["title"] for e in digest.get_digest("2026-01-01")]
    assert titles == ["Best Score, Old, Low Salary", "Worse Score, New, High Salary"]


def test_get_digest_sorts_live_even_if_stored_rank_is_stale():
    # Regression: digest_entries.rank is only recomputed when a *new* job gets added to that
    # date's digest -- scoring/feedback changes that happen afterward don't trigger a re-rank, so
    # a stale stored rank used to silently win over the current scores. get_digest() must sort
    # live, not trust `rank`.
    job_a = _insert_job("Low Score", posted_at=_days_ago(1))
    job_b = _insert_job("High Score", posted_at=_days_ago(1))
    _score_job(job_a, 20, 20)
    _score_job(job_b, 90, 90)

    digest.add_to_digest(job_a, "2026-01-01")
    digest.add_to_digest(job_b, "2026-01-01")
    # Correct immediately after insertion.
    assert [e["title"] for e in digest.get_digest("2026-01-01")] == ["High Score", "Low Score"]

    # Simulate a stale rank -- as if these rows were written before scores were what they are now,
    # and no new job has been added since to trigger a recompute.
    with get_db() as conn:
        conn.execute(
            "UPDATE digest_entries SET rank = 1 WHERE digest_date = '2026-01-01' AND job_id = ?",
            (job_a,),
        )
        conn.execute(
            "UPDATE digest_entries SET rank = 2 WHERE digest_date = '2026-01-01' AND job_id = ?",
            (job_b,),
        )

    assert [e["title"] for e in digest.get_digest("2026-01-01")] == ["High Score", "Low Score"]


def test_only_jobs_rated_med_or_better_on_both_land_on_new_and_noteworthy():
    ratings = {
        "High/High": ("High", "High"),
        "Med/Med": ("Med", "Med"),
        "High/Med": ("High", "Med"),
        "Low/Low": ("Low", "Low"),
        "Med/Low": ("Med", "Low"),
        "Low/High": ("Low", "High"),
    }
    for title, (interest, fit) in ratings.items():
        job_id = _insert_job(title, posted_at=_days_ago(1))
        _score_job(job_id, interest=interest, fit=fit)
        digest.add_to_digest(job_id, "2026-01-01")
    unscored = _insert_job("Not rated", posted_at=_days_ago(1))
    digest.add_to_digest(unscored, "2026-01-01")
    titles = {e["title"] for e in digest.get_digest("2026-01-01")}
    assert titles == {"High/High", "Med/Med", "High/Med"}
    with get_db() as conn:
        stored = conn.execute(
            "SELECT COUNT(*) FROM digest_entries WHERE digest_date = '2026-01-01'"
        ).fetchone()[0]
    assert stored == 3  # the others never went in


def test_a_low_row_already_stored_or_rescored_low_drops_out():
    kept = _insert_job("Still good", posted_at=_days_ago(1))
    rescored = _insert_job("Rescored low", posted_at=_days_ago(1))
    stored_low = _insert_job("Stored before the rule", posted_at=_days_ago(1))
    for job_id in (kept, rescored):
        _score_job(job_id)
        digest.add_to_digest(job_id, "2026-01-01")
    _score_job(stored_low, interest="Low", fit="Low")
    with get_db() as conn:  # as an older version would have stored it
        conn.execute(
            "INSERT INTO digest_entries (user_id, digest_date, job_id, rank) VALUES (1, '2026-01-01', ?, 0)",
            (stored_low,),
        )
    _score_job(rescored, interest="Med", fit="Low")  # the latest score is what counts
    assert [e["title"] for e in digest.get_digest("2026-01-01")] == ["Still good"]
