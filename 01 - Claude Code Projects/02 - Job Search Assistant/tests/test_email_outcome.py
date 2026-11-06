"""Each check records what its email step did, so Settings can say why no email came (asked
2026-11-05: no emails arrived and nothing said why)."""

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import email_digest
from jobhunt.core.digest import add_to_digest
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run


def _job_on_todays_list(interest: str) -> None:
    with get_db() as conn:
        job = conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company) VALUES (NULL, 'k', 'Analyst', 'Co')"
        ).lastrowid
        conn.execute(
            "INSERT INTO job_scores (user_id, job_id, model, interest_fit, interest_score, candidate_fit, "
            "candidate_fit_score, summary_text, rationale_text) "
            "VALUES (1, ?, 'm', ?, 50, 'High', 80, 's', 'r')",
            (job, interest),
        )
    add_to_digest(job)


def _status():
    with TestClient(app) as http:
        return http.get("/api/settings/email-status").json()


def test_not_configured_is_recorded_by_the_daily_run(monkeypatch):
    monkeypatch.delenv("EMAIL_ADDRESS", raising=False)
    monkeypatch.setattr(daily_run, "ADAPTERS", {})
    daily_run.run_daily()
    status = _status()
    assert status["configured"] is False
    assert status["last"]["sent"] is False and status["last"]["reason"] == "not configured"


def test_nothing_new_says_what_was_on_the_list():
    _job_on_todays_list("Med")
    last = email_digest.record_outcome({"sent": False, "reason": "nothing new"})
    assert (last["on_list_today"], last["best_interest_today"]) == (1, "Med")
    assert _status()["last"]["reason"] == "nothing new"


def test_sent_is_recorded():
    last = email_digest.record_outcome({"sent": True, "count": 2, "to": "me@example.com"})
    assert (last["sent"], last["count"], last["to"]) == (True, 2, "me@example.com")
    assert _status()["last"]["count"] == 2
