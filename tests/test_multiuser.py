"""Multi-user (D-021): one person never sees another's data, sources are shared and followed,
nobody gets in without the admin's approval, and the old single-user database migrates to the admin."""

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jobhunt.api import access, admin
from jobhunt.api.main import app
from jobhunt.core import scoring, users
from jobhunt.core.scoring import JobScore
from jobhunt.db import connection
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run
from tests.fakes import FakeAnthropicClient, FakeParseResponse, give_profile

B = "b@example.com"
AS_B = {"X-Dev-User": B}
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sent(monkeypatch):
    mails = []

    def record(to, subject, body):
        mails.append({"to": to, "subject": subject, "body": body})
        return {"sent": True}

    monkeypatch.setattr(access, "send_plain_email", record)
    monkeypatch.setattr(admin, "send_plain_email", record)
    monkeypatch.setattr(access, "admin_address", lambda: "admin@example.com")
    return mails


def _approved_b() -> int:
    return users.create_user(B, "Bea", status="approved")["id"]


def _admin_data(client) -> dict:
    """The admin's own things: a document, a source, a pasted job they track, feedback, a
    generated document, a Jobby conversation, a setting, and a Claude call."""
    resume_id = client.post(
        "/api/resumes/upload",
        files={"file": ("resume.txt", b"Admin resume text.", "text/plain")},
        data={"kind": "resume", "is_archived": "false"},
    ).json()["id"]
    source_id = client.post(
        "/api/sources",
        json={"kind": "custom_board", "name": "Board", "config": {"url": "https://x.org/jobs"}},
    ).json()["id"]
    with get_db() as conn:
        job_id = conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company) VALUES (1, 'u1:k', 'Secret Role', 'Acme')"
        ).lastrowid
        doc_id = conn.execute(
            "INSERT INTO generated_documents (user_id, job_id, kind, content_text) "
            "VALUES (1, ?, 'resume', 'x')",
            (job_id,),
        ).lastrowid
        thread = conn.execute(
            "INSERT INTO jobby_threads (user_id, job_id) VALUES (1, ?)", (job_id,)
        ).lastrowid
        conn.execute(
            "INSERT INTO jobby_messages (thread_id, role, content) VALUES (?, 'user', 'hi')", (thread,)
        )
        conn.execute(
            "INSERT INTO llm_calls (user_id, call_type, model, estimated_cost_usd) "
            "VALUES (1, 'scoring', 'm', 0.5)"
        )
    app_id = client.post("/api/applications", json={"job_id": job_id, "status": "waiting"}).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/feedback", json={"interested": "interested"}).status_code == 200
    client.put("/api/settings", json={"looking_for_notes": "admin's private notes"})
    with get_db() as conn:
        conn.execute(
            "INSERT INTO digest_entries (user_id, digest_date, job_id, rank) VALUES (1, date('now'), ?, 1)",
            (job_id,),
        )
    return {"resume": resume_id, "source": source_id, "job": job_id, "doc": doc_id, "app": app_id}


def test_one_person_never_sees_anothers_data():
    _approved_b()
    with TestClient(app) as client:
        a = _admin_data(client)

        def get(path):
            return client.get(path, headers=AS_B)

        assert get("/api/me").json()["email"] == B
        assert get("/api/applications").json() == []
        assert get(f"/api/applications/{a['app']}").status_code == 404
        assert (
            client.patch(f"/api/applications/{a['app']}", json={"notes": "x"}, headers=AS_B).status_code
            == 404
        )
        assert client.delete(f"/api/applications/{a['app']}", headers=AS_B).status_code == 404
        assert get("/api/applications/export.csv").text.strip().count("\n") == 0  # header row only
        assert get("/api/applications/insights").json() is None
        assert get("/api/resumes").json() == []
        assert get(f"/api/resumes/{a['resume']}/download").status_code == 404
        assert client.delete(f"/api/resumes/{a['resume']}", headers=AS_B).status_code == 404
        assert (
            client.patch(f"/api/resumes/{a['resume']}", json={"is_archived": True}, headers=AS_B).status_code
            == 404
        )
        assert get(f"/api/jobs/{a['job']}").status_code == 404
        assert all(j["id"] != a["job"] for j in get("/api/jobs").json())
        assert (
            client.post(
                f"/api/jobs/{a['job']}/feedback", json={"interested": "maybe"}, headers=AS_B
            ).status_code
            == 404
        )
        assert client.post("/api/applications", json={"job_id": a["job"]}, headers=AS_B).status_code == 404
        assert get("/api/sources").json() == []
        assert get(f"/api/sources/{a['source']}").status_code == 404
        assert client.delete(f"/api/sources/{a['source']}", headers=AS_B).status_code == 404
        assert get("/api/generate/documents").json() == []
        assert get(f"/api/generate/documents/{a['doc']}/download").status_code == 404
        assert get(f"/api/generate/history?job_id={a['job']}").json() == []
        assert client.post("/api/generate/resume", json={"job_id": a["job"]}, headers=AS_B).status_code == 404
        assert get(f"/api/jobby/thread?job_id={a['job']}").status_code == 404
        assert get("/api/jobby/thread").json()["messages"] == []
        assert get("/api/settings").json()["looking_for_notes"] == ""
        assert get("/api/settings/usage-log").json()["calls"] == []
        assert get("/api/digest/today").json()["entries"] == []
        assert get("/api/digest/dates").json() == []
        assert get("/api/onboarding/status").json()["needs_onboarding"] is True
        # The whole database is everyone's: admin only.
        assert get("/api/settings/backup").status_code == 403
        assert get("/api/settings/backups").status_code == 403
        assert get("/api/admin/users").status_code == 403
        # The Timeline is for everyone approved (404 here only because it isn't built in tests).
        assert get("/api/changelog").status_code in (200, 404)

        # And the admin still sees all of it.
        assert len(client.get("/api/applications").json()) == 1
        assert client.get(f"/api/jobs/{a['job']}").status_code == 200
        assert len(client.get("/api/digest/today").json()["entries"]) == 1


def test_start_over_only_clears_your_own_data():
    b = _approved_b()
    with TestClient(app) as client:
        a = _admin_data(client)
        assert (
            client.post("/api/settings/delete-all-data", json={"confirm": "DELETE"}, headers=AS_B).status_code
            == 200
        )
        assert len(client.get("/api/applications").json()) == 1
        assert client.get(f"/api/resumes/{a['resume']}/download").status_code == 200
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) FROM settings WHERE user_id = ?", (b,)).fetchone()[0] > 0


def test_sources_are_shared_and_followed():
    _approved_b()
    board = {"kind": "custom_board", "name": "Board", "config": {"url": "https://x.org/jobs"}}
    with TestClient(app) as client:
        mine = client.post("/api/sources", json=board).json()
        # The same board, written a little differently, is the same source: B just follows it.
        theirs = client.post(
            "/api/sources",
            json={**board, "name": "My name", "config": {"url": "http://X.org/jobs/"}},
            headers=AS_B,
        ).json()
        assert theirs["id"] == mine["id"] and theirs["followers"] == 2
        assert theirs["name"] == "Board"
        # Shared details can't be changed by one of several followers; their own switch can.
        assert (
            client.patch(f"/api/sources/{mine['id']}", json={"name": "Mine now"}, headers=AS_B).status_code
            == 403
        )
        off = client.patch(f"/api/sources/{mine['id']}", json={"enabled": False}, headers=AS_B).json()
        assert off["enabled"] == 0
        assert client.get(f"/api/sources/{mine['id']}").json()["enabled"] == 1
        # Removing it only unfollows.
        assert client.delete(f"/api/sources/{mine['id']}", headers=AS_B).status_code == 200
        assert client.get("/api/sources", headers=AS_B).json() == []
        assert [s["id"] for s in client.get("/api/sources").json()] == [mine["id"]]


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


def _posting(title):
    return {"external_id": title, "title": title, "company": "Acme", "url": f"https://x.org/{title}"}


def test_daily_run_reads_a_shared_source_once_and_scores_each_person(monkeypatch):
    b = _approved_b()
    users.create_user("c@example.com", "Cy", status="approved")  # follows nothing, no profile
    give_profile(1)
    give_profile(b)
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        conn.executemany(
            "INSERT INTO source_follows (user_id, source_id) VALUES (?, ?)", [(1, source), (b, source)]
        )
        conn.execute("UPDATE users SET daily_score_cap = 1 WHERE id = ?", (b,))
    reads = []
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"custom_board": lambda s: reads.append(s["id"]) or [_posting("A"), _posting("B")]},
    )
    monkeypatch.setattr(scoring, "get_client", _score_client)

    summary = daily_run.run_daily()

    assert reads == [source]  # once, for both followers
    assert summary["people"] == 3
    with get_db() as conn:
        scored = dict(conn.execute("SELECT user_id, COUNT(*) FROM job_scores GROUP BY user_id").fetchall())
        digests = dict(
            conn.execute("SELECT user_id, COUNT(*) FROM digest_entries GROUP BY user_id").fetchall()
        )
        runs = [
            r[0] for r in conn.execute("SELECT user_id FROM runs WHERE user_id IS NOT NULL ORDER BY user_id")
        ]
    assert scored == {1: 2, b: 1}  # B's cap is 1
    assert digests == {1: 2, b: 1}
    assert runs == [1, b, b + 1]

    # B's own "check my sources" reads only B's sources and scores only for B.
    reads.clear()
    with users.acting_as(b):
        daily_run.run_daily(only_user=b)
    assert reads == [source]
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_scores WHERE user_id = 1").fetchone()[0] == 2


def test_check_now_is_everyone_for_the_admin_and_yourself_for_others(monkeypatch):
    b = _approved_b()
    calls = []
    monkeypatch.setattr(
        daily_run, "run_daily", lambda cap=None, only_user=None: calls.append(only_user) or {}
    )
    with TestClient(app) as client:
        client.post("/api/pipeline/run-daily")
        client.post("/api/pipeline/run-daily", headers=AS_B)
    assert calls == [None, b]


def test_access_requests_need_the_admins_approval(sent):
    with TestClient(app) as client:
        assert client.get("/api/me", headers=AS_B).json()["status"] == "none"
        blocked = client.get("/api/applications", headers=AS_B)
        assert blocked.status_code == 403 and blocked.json()["code"] == "none"
        assert client.get("/api/changelog", headers=AS_B).status_code == 403  # not before approval

        assert client.post(
            "/api/access/request", json={"name": "Bea", "note": "friend of Matt"}, headers=AS_B
        ).json() == {"status": "pending"}
        assert sent[-1]["to"] == "admin@example.com" and sent[-1]["subject"] == "Access request: Bea"
        assert "friend of Matt" in sent[-1]["body"]
        assert client.get("/api/applications", headers=AS_B).json()["code"] == "pending"

        pending = [u for u in client.get("/api/admin/users").json()["users"] if u["email"] == B][0]
        assert (
            client.patch(f"/api/admin/users/{pending['id']}", json={"status": "approved"}).status_code == 200
        )
        assert sent[-1]["to"] == B and "approved" in sent[-1]["body"]
        assert client.get("/api/applications", headers=AS_B).status_code == 200
        assert client.get("/api/settings", headers=AS_B).json()["daily_llm_cap"] == 20

        # Blocking takes effect on the next request.
        client.patch(f"/api/admin/users/{pending['id']}", json={"status": "blocked"})
        assert client.get("/api/applications", headers=AS_B).json()["code"] == "blocked"
        assert client.post("/api/access/request", json={"name": "Bea"}, headers=AS_B).status_code == 403

        # The admin can't lock themselves out, and a non-admin can't use the admin API.
        assert client.patch("/api/admin/users/1", json={"status": "blocked"}).status_code == 400


def test_a_user_cannot_raise_their_own_cap():
    _approved_b()
    with TestClient(app) as client:
        assert (
            client.put("/api/settings", json={"daily_llm_cap": 500}, headers=AS_B).json()["daily_llm_cap"]
            == 20
        )
        assert client.put("/api/settings", json={"daily_llm_cap": 30}).json()["daily_llm_cap"] == 30


def test_nobody_set_means_no_data():
    token = users.set_current_user(None)
    try:
        with pytest.raises(users.NoUserError):
            users.current_user_id()
        with TestClient(app):
            pass
        from jobhunt.core.settings import get_settings

        with pytest.raises(users.NoUserError):
            get_settings()
    finally:
        users.reset_current_user(token)


def test_single_user_database_migrates_to_the_admin(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    old = sqlite3.connect(db_path, isolation_level=None)
    old.executescript((FIXTURES / "schema_single_user.sql").read_text())
    old.executescript(
        """
        INSERT INTO settings (key, value_json)
        VALUES ('looking_for_notes', '"mine"'), ('daily_llm_cap', '35');
        INSERT INTO sources (id, kind, name, config_json, enabled) VALUES (1, 'custom_board', 'B', '{}', 0);
        INSERT INTO jobs (id, source_id, dedupe_key, title, company) VALUES (1, 1, 'a', 'From board', 'X');
        INSERT INTO jobs (id, dedupe_key, title, company) VALUES (2, 'b', 'Pasted', 'Y');
        INSERT INTO applications (id, job_id, status) VALUES (1, 2, 'waiting');
        INSERT INTO application_status_history (application_id, status) VALUES (1, 'waiting');
        INSERT INTO job_scores (job_id, interest_fit) VALUES (1, 'High');
        INSERT INTO scoring_queue (job_id) VALUES (1);
        INSERT INTO digest_entries (digest_date, job_id, rank) VALUES ('2026-11-01', 1, 1);
        INSERT INTO feedback (job_id, interested) VALUES (1, 'interested');
        INSERT INTO resumes (kind, file_name, file_type, stored_path)
        VALUES ('resume', 'r.docx', 'docx', '/x');
        INSERT INTO candidate_profile (text) VALUES ('profile');
        INSERT INTO jobby_threads (job_id) VALUES (2);
        INSERT INTO llm_calls (call_type, model) VALUES ('scoring', 'm');
        """
    )
    old.close()
    monkeypatch.setattr(connection, "DB_PATH", db_path)
    monkeypatch.setenv("JOBHUNT_ADMIN_EMAIL", "me@example.com")
    connection.init_db()
    connection.init_db()  # a second start finds nothing to migrate

    (kept,) = (tmp_path / "backups").glob("jobhunt-*-before-migration.db")
    assert kept.exists()
    with get_db() as conn:
        for table in connection.PER_USER_TABLES:
            owners = {r[0] for r in conn.execute(f"SELECT user_id FROM {table}")}
            assert owners <= {1}, table
        assert conn.execute("SELECT COUNT(*) FROM job_scores").fetchone()[0] == 1
        admin_row = dict(conn.execute("SELECT * FROM users").fetchone())
        assert (admin_row["email"], admin_row["is_admin"], admin_row["status"]) == (
            "me@example.com",
            1,
            "approved",
        )
        assert admin_row["daily_score_cap"] == 35  # the cap they'd set
        assert [tuple(r) for r in conn.execute("SELECT user_id, source_id, enabled FROM source_follows")] == [
            (1, 1, 0)
        ]
        assert dict(conn.execute("SELECT id, user_id FROM jobs").fetchall()) == {1: None, 2: 1}
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        # The new per-person uniqueness holds: someone else can track the same job.
        conn.execute("INSERT INTO users (id, email, status) VALUES (2, 'b@x', 'approved')")
        conn.execute("INSERT INTO applications (user_id, job_id) VALUES (2, 1)")
        conn.execute("INSERT INTO applications (user_id, job_id) VALUES (1, 1)")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO applications (user_id, job_id) VALUES (1, 1)")

    with users.acting_as(1), TestClient(app) as client:
        assert client.get("/api/me", headers={"X-Dev-User": "me@example.com"}).json()["is_admin"] is True
        assert (
            client.get("/api/settings", headers={"X-Dev-User": "me@example.com"}).json()["looking_for_notes"]
            == "mine"
        )
