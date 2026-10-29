from jobhunt.core import queue
from jobhunt.core.settings import update_settings
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run


def _source(name):
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', ?, '{}')", (name,)
        ).lastrowid


def test_refresh_polls_only_new_sources_and_scores_into_digest(monkeypatch):
    old = _source("Old board")
    new = _source("New board")
    fetched = []

    def fake_adapter(source):
        fetched.append(source["id"])
        return [
            {"title": f"Analyst {i}", "company": "Acme", "url": f"https://x.org/{source['id']}/{i}"}
            for i in range(15)
        ]

    monkeypatch.setitem(daily_run.ADAPTERS, "custom_board", fake_adapter)
    scored = []

    def fake_score_one(job_id, score_args):
        scored.append(job_id)
        return True, None

    monkeypatch.setattr(daily_run, "_score_one", fake_score_one)
    digested = []
    monkeypatch.setattr(daily_run, "add_to_digest", digested.append)

    result = daily_run.refresh_sources([new])

    assert fetched == [new]
    assert result["jobs_fetched"] == 15
    assert len(scored) == daily_run.NEW_SOURCE_SCORE_CAP
    assert digested == scored
    assert queue.pending_count() == 15 - daily_run.NEW_SOURCE_SCORE_CAP  # the rest wait their turn
    with get_db() as conn:
        polled = conn.execute("SELECT id, last_success_at FROM sources ORDER BY id").fetchall()
    assert polled[0]["id"] == old and polled[0]["last_success_at"] is None
    assert polled[1]["last_success_at"] is not None


def test_select_batch_can_be_limited_to_sources():
    a, b = _source("A"), _source("B")
    with get_db() as conn:
        ja = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company, source_id) VALUES ('a', 'x', 'y', ?)", (a,)
        ).lastrowid
        jb = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company, source_id) VALUES ('b', 'x', 'y', ?)", (b,)
        ).lastrowid
    queue.enqueue(ja, 90.0, {})
    queue.enqueue(jb, 10.0, {})
    assert queue.select_batch(5, source_ids=[b]) == [jb]


def test_refresh_is_skipped_during_onboarding(monkeypatch):
    started = []
    monkeypatch.setattr(daily_run.threading, "Thread", lambda **k: started.append(k) or None)
    daily_run.schedule_source_refresh(1)
    assert started == []

    update_settings({"onboarding_completed": True})

    class FakeThread:
        def __init__(self, **k):
            started.append(k)

        def start(self):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(daily_run.threading, "Thread", FakeThread)
    monkeypatch.setattr(daily_run, "_refresh_worker", None)
    daily_run.schedule_source_refresh(1)
    daily_run.schedule_source_refresh(2)  # gathered into the same pending refresh
    assert len(started) == 1
    assert daily_run._pending_refresh == {1, 2}
    daily_run._pending_refresh.clear()


def test_salary_is_read_from_the_description_and_filled_on_reseen_jobs():
    source = _source("Board")
    posting = {
        "title": "Data Director",
        "company": "Acme",
        "external_id": "1",
        "description_text": "The salary range for this role is $150,000 - $180,000.",
    }
    job_id = daily_run._upsert_job(source, posting)
    with get_db() as conn:
        row = conn.execute(
            "SELECT salary_min, salary_max, salary_period FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert tuple(row) == (150000, 180000, "year")

    # A job first saved without a salary gets one when it's seen again with pay information...
    with get_db() as conn:
        conn.execute("UPDATE jobs SET salary_min = NULL, salary_max = NULL WHERE id = ?", (job_id,))
    daily_run._upsert_job(source, posting)
    # ...but an existing salary is never overwritten.
    daily_run._upsert_job(source, {**posting, "salary_min": 1, "salary_max": 2})
    with get_db() as conn:
        row = conn.execute("SELECT salary_min, salary_max FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert tuple(row) == (150000, 180000)


def test_backfill_salaries_reads_existing_descriptions():
    source = _source("Board")
    with get_db() as conn:
        job_id = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company, source_id, description_text) "
            "VALUES ('k', 'Analyst', 'Acme', ?, 'Pay: $40 - $55 per hour')",
            (source,),
        ).lastrowid
    assert daily_run.backfill_salaries() == 1
    with get_db() as conn:
        row = conn.execute(
            "SELECT salary_min, salary_max, salary_period FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert tuple(row) == (40, 55, "hour")


def test_backfill_falls_back_to_the_score_summary():
    source = _source("Board")
    with get_db() as conn:
        job_id = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company, source_id, description_text) "
            "VALUES ('s', 'Data Scientist', 'Blue Rose', ?, 'About us. Benefits.')",
            (source,),
        ).lastrowid
        conn.execute(
            "INSERT INTO job_scores (job_id, model, interest_fit, interest_score, candidate_fit, "
            "candidate_fit_score, summary_text, rationale_text) VALUES (?, 'x', 'High', 80, 'Med', 50, "
            "'Building LLM tools for campaigns, remote or NYC, $160-210k.', 'r')",
            (job_id,),
        )
    assert daily_run.backfill_salaries() == 1
    with get_db() as conn:
        row = conn.execute("SELECT salary_min, salary_max FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert tuple(row) == (160000, 210000)


def test_reseen_job_gets_the_newly_formatted_description():
    source = _source("Board")
    posting = {
        "title": "Analyst",
        "company": "Acme",
        "external_id": "9",
        "description_text": "Old\nflat text",
    }
    job_id = daily_run._upsert_job(source, posting)
    daily_run._upsert_job(source, {**posting, "description_text": "## Role\n\n- Formatted"})
    # A source that doesn't provide one (a web-page board) never blanks a stored description.
    daily_run._upsert_job(source, {**posting, "description_text": None})
    with get_db() as conn:
        row = conn.execute("SELECT description_text FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["description_text"] == "## Role\n\n- Formatted"
