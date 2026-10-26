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
            """INSERT INTO jobs (dedupe_key, title, company, posted_at, salary_max)
               VALUES (?, ?, ?, ?, ?)""",
            (title, title, company, posted_at, salary_max),
        )
        return cursor.lastrowid


def _score_job(job_id: int, interest_score: float, candidate_fit_score: float) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO job_scores
               (job_id, model, interest_fit, interest_score, candidate_fit, candidate_fit_score,
                summary_text, rationale_text)
               VALUES (?, 'x', 'High', ?, 'High', ?, '', '')""",
            (job_id, interest_score, candidate_fit_score),
        )


def _mark_not_interested(job_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO feedback (job_id, interested) VALUES (?, 'not_interested')", (job_id,)
        )


def _days_ago(n: int) -> str:
    with get_db() as conn:
        return conn.execute("SELECT datetime('now', ?) AS d", (f"-{n} days",)).fetchone()["d"]


def test_add_to_digest_skips_postings_older_than_90_days():
    job_id = _insert_job("Old Role", posted_at=_days_ago(120))
    digest.add_to_digest(job_id, "2026-01-01")
    assert digest.get_digest("2026-01-01") == []


def test_add_to_digest_includes_unknown_posted_date():
    job_id = _insert_job("Mystery Role", posted_at=None)
    digest.add_to_digest(job_id, "2026-01-01")
    entries = digest.get_digest("2026-01-01")
    assert len(entries) == 1
    assert entries[0]["title"] == "Mystery Role"


def test_add_to_digest_includes_recent_posting():
    job_id = _insert_job("Fresh Role", posted_at=_days_ago(10))
    digest.add_to_digest(job_id, "2026-01-01")
    entries = digest.get_digest("2026-01-01")
    assert [e["title"] for e in entries] == ["Fresh Role"]


def test_get_digest_excludes_jobs_marked_not_interested():
    job_id = _insert_job("Dismissed Role", posted_at=_days_ago(5))
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


def test_favorite_still_ranks_above_everything_else():
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sources (kind, name, config_json, is_favorite) "
            "VALUES ('custom_board', 'Fav', '{}', 1)"
        )
        source_id = conn.execute("SELECT id FROM sources WHERE name = 'Fav'").fetchone()["id"]
        cursor = conn.execute(
            "INSERT INTO jobs (source_id, dedupe_key, title, company, posted_at) VALUES (?, ?, ?, ?, ?)",
            (source_id, "fav-job", "Low Score Favorite", "Acme", _days_ago(50)),
        )
        fav_job_id = cursor.lastrowid
    _score_job(fav_job_id, 10, 10)

    strong_job = _insert_job("High Score Non-Favorite", posted_at=_days_ago(1))
    _score_job(strong_job, 99, 99)

    digest.add_to_digest(fav_job_id, "2026-01-01")
    digest.add_to_digest(strong_job, "2026-01-01")

    titles = [e["title"] for e in digest.get_digest("2026-01-01")]
    assert titles == ["Low Score Favorite", "High Score Non-Favorite"]
