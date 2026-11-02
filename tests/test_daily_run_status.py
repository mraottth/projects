"""The Admin page's "Daily check" card: the everyone run records how it went, including a crash
or a run that never finished (three days of failed scheduled runs went unseen on 2026-11-02)."""

import pytest
from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import users
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run


def _status(client, headers=None):
    return client.get("/api/admin/users", headers=headers or {})


def test_no_runs_yet():
    with TestClient(app) as client:
        assert _status(client).json()["daily_run"] == {"latest": None, "last_ok_finished_at": None}


def test_a_scheduled_run_records_its_outcome(monkeypatch):
    monkeypatch.setattr(daily_run, "ADAPTERS", {})
    token = users.set_current_user(None)  # the scheduler's request: nobody signed in
    try:
        daily_run.run_daily()
    finally:
        users.reset_current_user(token)
    with TestClient(app) as client:
        status = _status(client).json()["daily_run"]
    latest = status["latest"]
    assert (latest["status"], latest["started_by"], latest["error"]) == ("ok", "scheduled", None)
    assert latest["finished_at"] and status["last_ok_finished_at"] == latest["finished_at"]
    assert latest["source_errors"] == [] and latest["people"] == 1


def test_a_crashed_run_shows_the_error(monkeypatch):
    daily_run.run_daily()  # one good run first (as the admin: "Check every source now")

    def boom(*a, **k):
        raise OSError("disk I/O error")

    monkeypatch.setattr(daily_run, "_poll_sources", boom)
    with pytest.raises(OSError):
        daily_run.run_daily()
    with TestClient(app) as client:
        status = _status(client).json()["daily_run"]
    assert status["latest"]["status"] == "failed"
    assert status["latest"]["error"] == "OSError: disk I/O error"
    assert status["latest"]["started_by"] == "admin"
    assert status["last_ok_finished_at"] is not None  # the earlier good run


def test_a_run_that_never_finished_is_flagged():
    with get_db() as conn:
        conn.execute(
            "INSERT INTO runs (user_id, run_date, started_at, status, started_by) "
            "VALUES (NULL, date('now'), strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', '-3 hours'), "
            "'running', 'scheduled')"
        )
    with TestClient(app) as client:
        assert _status(client).json()["daily_run"]["latest"]["status"] == "unfinished"


def test_a_person_s_own_check_is_not_the_daily_run():
    daily_run.run_daily(only_user=1)
    with TestClient(app) as client:
        assert _status(client).json()["daily_run"]["latest"] is None


def test_only_the_admin_sees_it():
    users.create_user("b@example.com", "Bea", status="approved")
    with TestClient(app) as client:
        assert _status(client, {"X-Dev-User": "b@example.com"}).status_code == 403
