"""Scoring one job on request (asked 2026-11-12): "Score this job" on a card."""

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import queue, scoring
from jobhunt.core.llm import BudgetExceeded
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeParseResponse, give_profile

SCORE = scoring.ScoredPosting(
    interest_fit="High",
    interest_score=85,
    candidate_fit="Med",
    candidate_fit_score=65,
    summary_text="A data role.",
    rationale_text="==i:Civic work== you want.",
    salary_min=None,
    salary_max=None,
    salary_currency=None,
    salary_period=None,
)


def _job():
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('ats_greenhouse', 'Board', '{}')"
        ).lastrowid
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (1, ?)", (source,))
        job_id = conn.execute(
            "INSERT INTO jobs (source_id, dedupe_key, title, company, description_text) "
            "VALUES (?, 'k', 'Director of Data', 'Acme', 'Lead the data team.')",
            (source,),
        ).lastrowid
    queue.enqueue(job_id, 10, {})
    return job_id


def test_a_job_is_scored_on_request_and_leaves_the_queue(monkeypatch):
    give_profile()
    job_id = _job()
    monkeypatch.setattr(
        scoring, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(SCORE))
    )
    with TestClient(app) as http:
        resp = http.post(f"/api/jobs/{job_id}/score")
        assert resp.status_code == 200
        assert resp.json()["interest_fit"] == "High" and resp.json()["candidate_fit"] == "Med"
        ranked = {j["id"]: j for j in http.get("/api/jobs/ranked").json()}
    assert ranked[job_id]["interest_fit"] == "High"
    with get_db() as conn:
        assert (
            conn.execute("SELECT status FROM scoring_queue WHERE job_id = ?", (job_id,)).fetchone()[0]
            == "scored"
        )
    assert queue.select_batch(10) == []


def test_someone_elses_job_is_a_404():
    with get_db() as conn:
        job_id = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company) VALUES ('x', 'Private', 'Co')"
        ).lastrowid
    with TestClient(app) as http:
        assert http.post(f"/api/jobs/{job_id}/score").status_code == 404


def test_an_exhausted_budget_is_a_clear_error(monkeypatch):
    job_id = _job()

    def over():
        raise BudgetExceeded("You've used your $10.00 Claude budget for this month.")

    monkeypatch.setattr(scoring, "get_client", over)
    with TestClient(app) as http:
        resp = http.post(f"/api/jobs/{job_id}/score")
    assert resp.status_code == 502 and "budget" in resp.json()["detail"]
