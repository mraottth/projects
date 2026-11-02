from jobhunt.core import queue
from jobhunt.core.scoring import JobScore
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run
from tests.fakes import FakeAnthropicClient, FakeCreateResponse, FakeParseResponse

import pytest as _pytest

from tests.fakes import give_profile


@_pytest.fixture(autouse=True)
def _profile():
    give_profile()



def _make_posting(title: str, company: str = "Acme") -> dict:
    return {
        "external_id": title,
        "title": title,
        "company": company,
        "location": "Remote",
        "remote_type": "remote",
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
        "description_text": "Great role.",
        "url": f"https://example.com/{title}",
        "posted_at": "2026-10-20",
        "raw_json": {},
    }


def _add_source(kind="ats_greenhouse", config=None) -> int:
    import json

    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES (?, ?, ?)",
            (kind, "Test Source", json.dumps(config or {})),
        )
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (1, ?)", (cursor.lastrowid,))
        return cursor.lastrowid


def _fake_score_client(interest="High", candidate="Med"):
    score = JobScore(
        interest_fit=interest,
        interest_score=80,
        candidate_fit=candidate,
        candidate_fit_score=60,
        summary_text="summary",
        rationale_text="rationale",
    )
    return FakeAnthropicClient(parse_result=FakeParseResponse(score))


def test_run_daily_full_cycle(monkeypatch):
    _add_source()
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"ats_greenhouse": lambda source: [_make_posting("Engineer A"), _make_posting("Engineer B")]},
    )
    from jobhunt.core import scoring

    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    summary = daily_run.run_daily()

    assert summary["sources_polled"] == 1
    assert summary["jobs_fetched"] == 2
    assert summary["jobs_upserted"] == 2
    assert summary["prefiltered_count"] == 2
    assert summary["llm_scored_count"] == 2
    assert summary["carried_over_count"] == 0

    with get_db() as conn:
        jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
        scores = conn.execute("SELECT COUNT(*) AS n FROM job_scores").fetchone()["n"]
        digest = conn.execute("SELECT COUNT(*) AS n FROM digest_entries").fetchone()["n"]
        runs = conn.execute("SELECT COUNT(*) AS n FROM runs WHERE user_id IS NOT NULL").fetchone()["n"]
    assert jobs == 2
    assert scores == 2
    assert digest == 2
    assert runs == 1


def test_run_daily_respects_daily_cap_and_carries_over(monkeypatch):
    from jobhunt.core import scoring

    _add_source()
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"ats_greenhouse": lambda source: [_make_posting(f"Job {i}") for i in range(5)]},
    )
    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())
    with get_db() as conn:  # the cap lives on the user (D-021)
        conn.execute("UPDATE users SET daily_score_cap = 2 WHERE id = 1")

    summary = daily_run.run_daily()

    assert summary["llm_scored_count"] == 2
    assert summary["carried_over_count"] == 3
    assert queue.pending_count() == 3


def test_run_daily_dedupes_already_seen_jobs(monkeypatch):
    _add_source()
    monkeypatch.setattr(
        daily_run, "ADAPTERS", {"ats_greenhouse": lambda source: [_make_posting("Same Job")]}
    )
    from jobhunt.core import scoring

    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    daily_run.run_daily()
    daily_run.run_daily()  # same posting seen again on a second run

    with get_db() as conn:
        jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
        scores = conn.execute("SELECT COUNT(*) AS n FROM job_scores").fetchone()["n"]
    assert jobs == 1
    assert scores == 1  # not re-scored once it already has a score


def test_run_daily_one_bad_source_does_not_abort_run(monkeypatch):
    _add_source()
    _add_source()

    def failing_adapter(source):
        raise RuntimeError("boom")

    def working_adapter(source):
        return [_make_posting("Survivor")]

    monkeypatch.setattr(
        daily_run, "ADAPTERS", {"ats_greenhouse": failing_adapter}
    )
    # Second source shares the same kind in this fixture, so patch per-call via a counter instead.
    calls = {"n": 0}

    def dispatch(source):
        calls["n"] += 1
        if calls["n"] == 1:
            return failing_adapter(source)
        return working_adapter(source)

    monkeypatch.setattr(daily_run, "ADAPTERS", {"ats_greenhouse": dispatch})
    from jobhunt.core import scoring

    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    summary = daily_run.run_daily()
    assert summary["jobs_fetched"] == 1
    assert len(summary["errors"]) == 1


def test_run_daily_triggers_preference_rebuild_when_signal_present(monkeypatch):
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company) VALUES (1, 'k1', 'Engineer', 'Acme')"
        )
        job_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO feedback (user_id, job_id, interested) VALUES (1, ?, 'interested')", (job_id,)
        )

    from jobhunt.core import preference

    monkeypatch.setattr(daily_run, "ADAPTERS", {})
    monkeypatch.setattr(
        preference,
        "get_client",
        lambda: FakeAnthropicClient(create_result=FakeCreateResponse("profile text")),
    )

    summary = daily_run.run_daily()
    assert summary["preference_profile_rebuilt"] is True


def test_run_daily_cap_override_scores_only_that_many(monkeypatch):
    # Onboarding passes a small cap so the first run is quick; the rest carry over.
    from jobhunt.core import scoring

    _add_source()
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"ats_greenhouse": lambda source: [_make_posting(f"Job {i}") for i in range(5)]},
    )
    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    summary = daily_run.run_daily(cap=2)

    assert summary["llm_scored_count"] == 2
    assert summary["cap_used"] == 2
    assert queue.pending_count() == 3


def test_run_daily_scores_jobs_concurrently(monkeypatch):
    import time

    _add_source()
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"ats_greenhouse": lambda source: [_make_posting(f"Job {i}") for i in range(8)]},
    )
    scores = {f"Job {i}": 10 * i for i in range(8)}

    def slow_score(job, *args):
        time.sleep(0.3)
        return JobScore(
            interest_fit="Med",
            interest_score=scores[job["title"]],
            candidate_fit="Med",
            candidate_fit_score=0,
            summary_text="s",
            rationale_text="r",
        )

    monkeypatch.setattr(daily_run, "score_job", slow_score)

    started = time.monotonic()
    summary = daily_run.run_daily()
    elapsed = time.monotonic() - started

    assert summary["llm_scored_count"] == 8
    # 8 jobs x 0.3s serially would be 2.4s; 4 at a time is ~0.6s.
    assert elapsed < 1.5

    from jobhunt.core.digest import get_digest, today_str

    titles = [entry["title"] for entry in get_digest(today_str())]
    assert titles == [f"Job {i}" for i in reversed(range(8))]


def test_run_daily_reports_progress(monkeypatch):
    from jobhunt.core import scoring

    _add_source()
    _add_source()
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"ats_greenhouse": lambda source: [_make_posting(f"Job {source['id']}-{i}") for i in range(2)]},
    )
    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    daily_run.run_daily(cap=3)

    from fastapi.testclient import TestClient

    from jobhunt.api.main import app

    with TestClient(app) as client:
        progress = client.get("/api/pipeline/progress").json()
    counts = {k: progress[k] for k in ("stage", "sources_done", "sources_total", "jobs_done", "jobs_total")}
    assert counts == {
        "stage": "done",
        "sources_done": 2,
        "sources_total": 2,
        "jobs_done": 3,
        "jobs_total": 3,
    }


def test_run_daily_marks_progress_failed_on_crash(monkeypatch):
    import pytest

    def boom():
        raise RuntimeError("db gone")

    monkeypatch.setattr(daily_run, "_expire_stale_jobs", boom)
    with pytest.raises(RuntimeError):
        daily_run.run_daily()
    assert daily_run.get_progress()["stage"] == "failed"
