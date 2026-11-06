"""Step 6 of D-021: deleting an account, by its owner or by the admin."""

import pytest
from fastapi.testclient import TestClient

from jobhunt.api import admin
from jobhunt.api.main import app
from jobhunt.core import backup, users
from jobhunt.db.connection import PER_USER_TABLES, get_db

B_EMAIL = "b@example.com"
B = {"X-Dev-User": B_EMAIL}


@pytest.fixture
def mail(monkeypatch):
    sent = []
    monkeypatch.setattr(admin, "send_plain_email", lambda to, subject, body: sent.append((to, subject)) or {})
    return sent


def _person_with_data(client) -> int:
    b = users.create_user(B_EMAIL, "Bea", status="approved")["id"]
    client.post(
        "/api/resumes/upload",
        files={"file": ("resume.txt", b"Bea's resume.", "text/plain")},
        data={"kind": "resume", "is_archived": "false"},
        headers=B,
    )
    client.post(
        "/api/sources",
        json={"kind": "custom_board", "name": "Board", "config": {"url": "https://x.org"}},
        headers=B,
    )
    with get_db() as conn:
        job = conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company) VALUES (?, 'ub:k', 'Pasted', 'Co')", (b,)
        ).lastrowid
        conn.execute(
            "INSERT INTO llm_calls (user_id, call_type, model, estimated_cost_usd, note) "
            "VALUES (?, 'scoring', 'm', 0.4, 'x')",
            (b,),
        )
    client.post("/api/applications", json={"job_id": job, "status": "waiting"}, headers=B)
    client.put("/api/settings", json={"looking_for_notes": "Bea's notes"}, headers=B)
    return b


def _admin_data(client):
    client.put("/api/settings", json={"looking_for_notes": "admin's notes"})


def test_you_can_delete_your_own_account():
    with TestClient(app) as client:
        _admin_data(client)
        b = _person_with_data(client)
        folder = users.RESUME_STORAGE_DIR / str(b)
        assert any(folder.iterdir())

        assert (
            client.post("/api/settings/delete-account", json={"confirm": "nope"}, headers=B).status_code
            == 400
        )
        resp = client.post("/api/settings/delete-account", json={"confirm": "DELETE"}, headers=B)
        assert resp.status_code == 200 and resp.json()["backup"].endswith("-before-account-delete.db")

        assert client.get("/api/me", headers=B).json()["status"] == "none"  # would have to ask again
        assert client.get("/api/settings").json()["looking_for_notes"] == "admin's notes"

    assert not folder.exists()
    with get_db() as conn:
        for table in PER_USER_TABLES:
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (b,)).fetchone()[0] == 0, (
                table
            )
        assert conn.execute("SELECT COUNT(*) FROM source_follows WHERE user_id = ?", (b,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM jobs WHERE user_id = ?", (b,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM users WHERE id = ?", (b,)).fetchone()[0] == 0
        # Spend history stays, no longer linked to anyone.
        kept = conn.execute("SELECT user_id, note, estimated_cost_usd FROM llm_calls").fetchall()
        assert [tuple(r) for r in kept] == [(None, "deleted account: x", 0.4)]
        assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1  # shared: kept
    assert any(backup.backup_dir().glob("*-before-account-delete.db"))


def test_the_admin_account_cant_be_deleted():
    with TestClient(app) as client:
        resp = client.post("/api/settings/delete-account", json={"confirm": "DELETE"})
        assert resp.status_code == 400 and "admin account" in resp.json()["detail"]
        resp = client.request("DELETE", "/api/admin/users/1", json={"confirm": "DELETE"})
        assert resp.status_code == 400
    assert users.get_user(1) is not None


def test_the_admin_can_delete_someone(mail):
    with TestClient(app) as client:
        b = _person_with_data(client)
        assert client.request("DELETE", f"/api/admin/users/{b}", json={"confirm": "no"}).status_code == 400
        # Only the admin.
        assert (
            client.request(
                "DELETE", f"/api/admin/users/{b}", json={"confirm": "DELETE"}, headers=B
            ).status_code
            == 403
        )
        assert (
            client.request("DELETE", f"/api/admin/users/{b}", json={"confirm": "DELETE"}).status_code == 200
        )
        assert (
            client.request("DELETE", f"/api/admin/users/{b}", json={"confirm": "DELETE"}).status_code == 404
        )
    assert users.get_user(b) is None
    assert mail == [(B_EMAIL, "Your Job Search Assistant account was deleted")]
