from pathlib import Path

from fastapi.testclient import TestClient

from jobhunt.api import resumes as resumes_api
from jobhunt.api.main import app
from jobhunt.core import reset
from jobhunt.db.connection import get_db


def _count(table):
    with get_db() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_delete_all_data_returns_to_onboarding(tmp_path, monkeypatch):
    storage = tmp_path / "resumes"
    monkeypatch.setattr(resumes_api, "RESUME_STORAGE_DIR", storage)
    monkeypatch.setattr(reset, "RESUME_STORAGE_DIR", storage)
    outside = tmp_path / "my-own-folder" / "Tailored.docx"
    outside.parent.mkdir()
    outside.write_bytes(b"not the app's to delete")

    with TestClient(app) as client:
        client.post(
            "/api/resumes/upload",
            files={"file": ("resume.txt", b"Some resume text content.", "text/plain")},
            data={"kind": "resume", "is_archived": "false"},
        )
        client.post(
            "/api/sources", json={"kind": "custom_board", "name": "Board", "config": {"url": "https://x.org"}}
        )
        client.put("/api/settings", json={"onboarding_completed": True, "daily_llm_cap": 10})
        with get_db() as conn:
            job_id = conn.execute(
                "INSERT INTO jobs (dedupe_key, title, company) VALUES ('k', 'Analyst', 'Acme')"
            ).lastrowid
            conn.execute(
                "INSERT INTO generated_documents (job_id, kind, content_text, file_path) "
                "VALUES (?, 'resume', 'x', ?)",
                (job_id, str(outside)),
            )
            conn.execute(
                "INSERT INTO llm_calls (call_type, job_id, model, estimated_cost_usd) "
                "VALUES ('scoring', ?, 'claude-sonnet-5', 0.03)",
                (job_id,),
            )
        stored = list(storage.iterdir())
        assert len(stored) == 1
        assert client.get("/api/onboarding/status").json()["needs_onboarding"] is False

        assert client.post("/api/settings/delete-all-data", json={"confirm": "nope"}).status_code == 400
        assert _count("sources") == 1

        resp = client.post("/api/settings/delete-all-data", json={"confirm": "DELETE"})
        assert resp.status_code == 200
        assert resp.json()["files_deleted"] == 1

        assert client.get("/api/onboarding/status").json()["needs_onboarding"] is True
        settings = client.get("/api/settings").json()
        assert settings["daily_llm_cap"] == 50
        assert settings["onboarding_completed"] is False

    for table in ("resumes", "sources", "jobs", "generated_documents"):
        assert _count(table) == 0
    assert not Path(stored[0]).exists()
    assert outside.exists()  # outside the app's storage: left alone
    with get_db() as conn:
        row = conn.execute("SELECT job_id, estimated_cost_usd FROM llm_calls").fetchone()
    assert row["job_id"] is None and row["estimated_cost_usd"] == 0.03  # usage history kept
