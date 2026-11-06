"""Step 4 of D-021: only people active in the last week are scored (with a one-time "paused"
email), and each person's monthly Claude budget is enforced, with alerts to the admin."""

import pytest
from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import budget, email_digest, llm, scoring, users
from jobhunt.core.llm import BudgetExceeded
from jobhunt.core.scoring import JobScore
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run
from tests.fakes import FakeAnthropicClient, FakeParseResponse, give_profile

B = "b@example.com"


@pytest.fixture
def mail(monkeypatch):
    sent = []

    def record(to, subject, body):
        sent.append({"to": to, "subject": subject, "body": body})
        return {"sent": True}

    monkeypatch.setattr(daily_run, "send_plain_email", record)
    monkeypatch.setattr(email_digest, "send_plain_email", record)
    monkeypatch.setattr(email_digest, "admin_address", lambda: "admin@example.com")
    return sent


def _person(email=B, name="Bea", days_away=None) -> int:
    user_id = users.create_user(email, name, status="approved")["id"]
    with get_db() as conn:
        conn.execute("UPDATE users SET decided_at = datetime('now', '-30 days') WHERE id = ?", (user_id,))
        if days_away is not None:
            conn.execute(
                "UPDATE users SET last_seen_at = datetime('now', ?) WHERE id = ?",
                (f"-{days_away} days", user_id),
            )
    return user_id


def _spend(user_id, usd):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO llm_calls (user_id, call_type, model, estimated_cost_usd) "
            "VALUES (?, 'scoring', 'm', ?)",
            (user_id, usd),
        )


def _source_followed_by(*user_ids) -> int:
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        for user_id in user_ids:
            conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (?, ?)", (user_id, source))
    return source


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


def test_only_people_active_this_week_are_scored(monkeypatch, mail):
    away = _person("away@example.com", "Al", days_away=8)
    here = _person(B, "Bea", days_away=2)
    new = users.create_user("new@example.com", "Ned", status="approved")["id"]  # never visited yet
    with get_db() as conn:
        conn.execute("UPDATE users SET last_seen_at = datetime('now') WHERE id = 1")
    for user_id in (away, here, new):
        give_profile(user_id)
    give_profile(1)
    only_away = _source_followed_by(away)
    _source_followed_by(1, here, new)
    reads = []
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"custom_board": lambda s: reads.append(s["id"]) or [{"title": f"Job {s['id']}", "company": "Acme"}]},
    )
    monkeypatch.setattr(scoring, "get_client", _score_client)

    summary = daily_run.run_daily()

    assert only_away not in reads  # nobody active follows it, so it isn't read
    with get_db() as conn:
        scored = {r[0] for r in conn.execute("SELECT DISTINCT user_id FROM job_scores")}
    assert scored == {1, here, new}
    assert summary["paused_emails"] == 1
    assert [m["to"] for m in mail] == ["away@example.com"]
    assert "paused" in mail[0]["subject"]

    # Told once: the next run doesn't email again.
    daily_run.run_daily()
    assert len(mail) == 1


def test_a_visit_clears_the_paused_flag(mail):
    away = _person(days_away=9)
    daily_run._send_paused_emails()
    assert users.get_user(away)["paused_email_sent_at"] is not None
    with TestClient(app) as client:
        client.get("/api/settings", headers={"X-Dev-User": B})
    user = users.get_user(away)
    assert user["paused_email_sent_at"] is None and away in users.active_user_ids()


def test_a_failed_paused_email_is_retried(monkeypatch):
    away = _person(days_away=9)
    monkeypatch.setattr(daily_run, "send_plain_email", lambda *a: {"sent": False, "reason": "not configured"})
    assert daily_run._send_paused_emails() == 0
    assert users.get_user(away)["paused_email_sent_at"] is None


def test_over_budget_blocks_paid_calls_but_not_the_admin(monkeypatch):
    b = _person(days_away=0)
    _spend(b, 10.0)
    with users.acting_as(b), pytest.raises(BudgetExceeded, match=r"used your \$10.00 Claude budget"):
        llm.get_client()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _spend(1, 500.0)
    with users.acting_as(1):
        assert llm.get_client() is not None  # the admin has no limit
    with users.acting_as(b):
        with get_db() as conn:
            conn.execute("UPDATE users SET monthly_budget_usd = 25 WHERE id = ?", (b,))
        assert llm.get_client() is not None  # raised on /admin: unblocked at once

    # Shared work (nobody set) is never blocked.
    token = users.set_current_user(None)
    try:
        assert llm.get_client() is not None
    finally:
        users.reset_current_user(token)


def test_a_paid_route_says_the_budget_is_used_up():
    b = _person(days_away=0)
    _spend(b, 12.0)
    give_profile(b)
    with TestClient(app) as client:
        resp = client.post(
            "/api/resumes/profile/revise", json={"instructions": "shorter"}, headers={"X-Dev-User": B}
        )
    assert resp.status_code == 502 and "Claude budget" in resp.json()["detail"]


def test_the_daily_run_skips_scoring_once_the_budget_is_used(monkeypatch, mail):
    b = _person(days_away=0)
    give_profile(b)
    _source_followed_by(b)
    _spend(b, 10.0)
    monkeypatch.setattr(
        daily_run, "ADAPTERS", {"custom_board": lambda s: [{"title": "Job", "company": "Acme"}]}
    )
    monkeypatch.setattr(scoring, "get_client", _score_client)
    summary = daily_run.run_daily(only_user=b)
    assert summary["llm_scored_count"] == 0
    assert any("Claude budget" in e for e in summary["errors"])


def test_the_admin_hears_at_80_and_100_percent_once_each(mail):
    b = _person(days_away=0)
    with users.acting_as(b):
        llm.log_call("scoring", "claude-sonnet-5", 0, 0)  # costs nothing: no alert
        _spend(b, 8.5)
        llm.log_call("scoring", "claude-sonnet-5", 0, 0)
        llm.log_call("scoring", "claude-sonnet-5", 0, 0)
        assert (
            len(mail) == 1
            and "passed 80%" in mail[0]["subject"]
            and "$8.50 of their $10.00" in mail[0]["body"]
        )
        _spend(b, 2.0)
        llm.log_call("scoring", "claude-sonnet-5", 0, 0)
        llm.log_call("scoring", "claude-sonnet-5", 0, 0)
    assert len(mail) == 2 and "reached" in mail[1]["subject"]
    assert all(m["to"] == "admin@example.com" for m in mail)
    # The admin's own spend never alerts.
    _spend(1, 1000.0)
    with users.acting_as(1):
        llm.log_call("scoring", "claude-sonnet-5", 0, 0)
    assert len(mail) == 2


def test_me_reports_usage():
    b = _person(days_away=0)
    _spend(b, 3.2)
    with TestClient(app) as client:
        usage = client.get("/api/me", headers={"X-Dev-User": B}).json()["usage"]
        admin_usage = client.get("/api/me").json()["usage"]
    assert usage["spend_this_month"] == pytest.approx(3.2)
    assert (usage["monthly_budget_usd"], usage["over"], usage["exempt"]) == (10, False, False)
    assert (usage["scored_today"], usage["daily_score_cap"]) == (0, 20)
    assert admin_usage["exempt"] is True


def test_budget_status_counts_only_this_month():
    b = _person(days_away=0)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO llm_calls (user_id, call_type, model, estimated_cost_usd, created_at) "
            "VALUES (?, 'scoring', 'm', 9.0, datetime('now', 'start of month', '-1 day'))",
            (b,),
        )
    _spend(b, 1.0)
    assert budget.spend_this_month(b) == pytest.approx(1.0)
