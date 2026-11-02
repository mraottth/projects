"""Step 5 of D-021: New & Noteworthy's "Last updated" and Refresh."""

import json

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import scoring, users
from jobhunt.core.scoring import JobScore
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run
from tests.fakes import FakeAnthropicClient, FakeParseResponse, give_profile

B = {"X-Dev-User": "b@example.com"}


def _score_client():
    score = JobScore(
        interest_fit="High",
        interest_score=80,
        candidate_fit="Med",
        candidate_fit_score=60,
        summary_text="s",
        rationale_text="r",
    )
    return FakeAnthropicClient(parse_result=FakeParseResponse(score))


def _source(*user_ids, read_minutes_ago=None) -> int:
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        if read_minutes_ago is not None:
            conn.execute(
                "UPDATE sources SET last_success_at = datetime('now', ?) WHERE id = ?",
                (f"-{read_minutes_ago} minutes", source),
            )
        for user_id in user_ids:
            conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (?, ?)", (user_id, source))
    return source


def _events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_status_before_and_after_a_refresh(monkeypatch):
    give_profile(1)
    _source(1)
    monkeypatch.setattr(
        daily_run, "ADAPTERS", {"custom_board": lambda s: [{"title": "Analyst", "company": "Acme"}]}
    )
    monkeypatch.setattr(scoring, "get_client", _score_client)
    with TestClient(app) as client:
        status = client.get("/api/digest/status").json()
        assert status["last_updated"] is None and status["can_refresh"] is True
        assert (status["scored_today"], status["daily_score_cap"]) == (0, 50)

        events = _events(client.post("/api/digest/refresh/stream"))
        assert events[0]["type"] == "started" and events[-1]["type"] == "done"
        assert events[-1]["summary"]["llm_scored_count"] == 1

        status = client.get("/api/digest/status").json()
        assert status["last_updated"] is not None and status["scored_today"] == 1
        assert status["can_refresh"] is False and status["refresh_available_at"]
        assert len(client.get("/api/digest/today").json()["entries"]) == 1

        again = client.post("/api/digest/refresh/stream")
        assert again.status_code == 429 and "refresh again in 30 minutes" in again.json()["detail"]


def test_cooldowns_are_per_person(monkeypatch):
    users.create_user("b@example.com", "Bea", status="approved")
    monkeypatch.setattr(daily_run, "start_run_daily", lambda **k: 0)
    monkeypatch.setattr(
        daily_run, "get_progress", lambda: {"finished_run_id": 0, "last_error": None, "last_summary": {}}
    )
    with TestClient(app) as client:
        assert client.post("/api/digest/refresh/stream").status_code == 200
        assert client.post("/api/digest/refresh/stream").status_code == 429
        assert client.post("/api/digest/refresh/stream", headers=B).status_code == 200


def test_a_just_read_source_is_not_read_again_but_its_jobs_still_get_scored(monkeypatch):
    b = users.create_user("b@example.com", "Bea", status="approved")["id"]
    give_profile(b)
    fresh = _source(1, b, read_minutes_ago=10)
    stale = _source(b, read_minutes_ago=180)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (source_id, dedupe_key, title, company) VALUES (?, 'f1', 'Data Lead', 'Acme')",
            (fresh,),
        )
    reads = []
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"custom_board": lambda s: reads.append(s["id"]) or [{"title": "PM", "company": "Co"}]},
    )
    monkeypatch.setattr(scoring, "get_client", _score_client)
    with users.acting_as(b):
        summary = daily_run.run_daily(only_user=b, fresh_hours=1)
    assert reads == [stale]
    assert summary["sources_fresh"] == 1
    with get_db() as conn:
        titles = {
            r[0]
            for r in conn.execute(
                "SELECT j.title FROM job_scores s JOIN jobs j ON j.id = s.job_id WHERE s.user_id = ?", (b,)
            )
        }
    assert titles == {"Data Lead", "PM"}


def test_the_summary_says_when_todays_cap_is_used(monkeypatch):
    give_profile(1)
    _source(1)
    with get_db() as conn:
        conn.execute("UPDATE users SET daily_score_cap = 1 WHERE id = 1")
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"custom_board": lambda s: [{"title": "A", "company": "Acme"}, {"title": "B", "company": "Acme"}]},
    )
    monkeypatch.setattr(scoring, "get_client", _score_client)
    summary = daily_run.run_daily(only_user=1)
    assert summary["llm_scored_count"] == 1 and summary["cap_reached"] is True
