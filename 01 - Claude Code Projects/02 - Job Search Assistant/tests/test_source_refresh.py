from jobhunt.core import queue
from jobhunt.core.settings import update_settings
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run

import pytest as _pytest

from tests.fakes import give_profile


@_pytest.fixture(autouse=True)
def _profile():
    give_profile()


def _source(name):
    with get_db() as conn:
        source_id = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', ?, '{}')", (name,)
        ).lastrowid
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (1, ?)", (source_id,))
        return source_id


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
    assert daily_run._pending_refresh == {1: {1, 2}}  # per person (D-021)
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


def test_a_reseen_job_takes_its_current_location_and_remote_flag():
    # How an adapter fix (Ashby's workplaceType and secondaryLocations, 2026-11-12) reaches jobs saved
    # before it: on their next check.
    source = _source("Board")
    posting = {"title": "ER Partner", "company": "Harvey", "external_id": "1", "location": "New York",
               "remote_type": "remote"}  # fmt: skip
    job_id = daily_run._upsert_job(source, posting)
    daily_run._upsert_job(
        source, {**posting, "location": "New York | San Francisco", "remote_type": "hybrid"}
    )
    with get_db() as conn:
        row = conn.execute("SELECT location, remote_type FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert tuple(row) == ("New York | San Francisco", "hybrid")
    # A posting that comes back without a location never wipes the stored one.
    daily_run._upsert_job(source, {**posting, "location": None, "remote_type": None})
    with get_db() as conn:
        row = conn.execute("SELECT location, remote_type FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert tuple(row) == ("New York | San Francisco", "hybrid")


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
            "INSERT INTO job_scores (user_id, job_id, model, interest_fit, interest_score, candidate_fit, "
            "candidate_fit_score, summary_text, rationale_text) VALUES (1, ?, 'x', 'High', 80, 'Med', 50, "
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


def test_progress_lists_sources_still_being_read(monkeypatch):
    slow = _source("Slow board")
    seen_while_reading = []

    def adapter(source):
        seen_while_reading.append(list(daily_run.get_progress()["sources_pending"]))
        return []

    monkeypatch.setitem(daily_run.ADAPTERS, "custom_board", adapter)
    daily_run._poll_sources([slow])
    assert seen_while_reading == [["Slow board"]]
    assert daily_run.get_progress()["sources_pending"] == []


def _job_status(source_id):
    with get_db() as conn:
        rows = conn.execute("SELECT title, status FROM jobs WHERE source_id = ? ORDER BY title", (source_id,))
        return {r["title"]: r["status"] for r in rows}


def test_a_complete_read_closes_jobs_it_no_longer_lists(monkeypatch):
    # A Greenhouse/Lever/Ashby read lists every open posting; one that's gone is closed (2026-11-12).
    source = _source("Board")
    listed = [{"title": t, "company": "Acme", "external_id": t} for t in ("A", "B", "C")]

    def adapter(src):
        src["read_complete"] = True  # as custom_board does for an embedded ATS board
        return list(listed)

    monkeypatch.setitem(daily_run.ADAPTERS, "custom_board", adapter)
    daily_run._poll_sources(source_ids=[source])
    listed.pop(1)
    daily_run._poll_sources(source_ids=[source])
    assert _job_status(source) == {"A": "active", "B": "expired", "C": "active"}
    listed.append({"title": "B", "company": "Acme", "external_id": "B"})  # back again: reopened
    daily_run._poll_sources(source_ids=[source])
    assert _job_status(source)["B"] == "active"


def test_a_page_read_by_claude_never_closes_jobs(monkeypatch):
    source = _source("Board")
    listed = [{"title": t, "company": "Acme", "url": f"https://x.org/{t}"} for t in ("A", "B")]
    monkeypatch.setitem(daily_run.ADAPTERS, "custom_board", lambda src: list(listed))
    daily_run._poll_sources(source_ids=[source])
    listed.pop()
    daily_run._poll_sources(source_ids=[source])
    assert _job_status(source) == {"A": "active", "B": "active"}  # it may only have read some


def test_an_empty_read_of_a_source_with_jobs_is_an_error_not_a_success(monkeypatch):
    source = _source("Board")
    listed = [{"title": "A", "company": "Acme", "external_id": "A"}]
    monkeypatch.setitem(daily_run.ADAPTERS, "custom_board", lambda src: list(listed))
    daily_run._poll_sources(source_ids=[source])
    with get_db() as conn:
        conn.execute("UPDATE sources SET last_success_at = '2026-01-01 00:00:00' WHERE id = ?", (source,))
    listed.clear()
    result = daily_run._poll_sources(source_ids=[source])
    assert result["errors"] == ["Board: found no postings this time (it had 1); will try again next check"]
    with get_db() as conn:
        row = conn.execute(
            "SELECT last_success_at, last_error FROM sources WHERE id = ?", (source,)
        ).fetchone()
    assert row["last_success_at"] == "2026-01-01 00:00:00" and "found no postings" in row["last_error"]
    assert _job_status(source) == {"A": "active"}
