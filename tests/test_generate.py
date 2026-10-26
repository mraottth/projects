from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import generate
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeCreateResponse


def _insert_job() -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company, description_text) "
            "VALUES ('k1', 'Engineer', 'Acme', 'Build things')"
        )
        return cursor.lastrowid


def _insert_resume(text="Experienced engineer. Built many things.") -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO resumes (kind, file_name, file_type, stored_path, raw_text) "
            "VALUES ('resume', 'r.md', 'md', '/tmp/r.md', ?)",
            (text,),
        )
        return cursor.lastrowid


def test_generate_resume_requires_a_base_resume():
    job_id = _insert_job()
    with TestClient(app) as client:
        resp = client.post("/api/generate/resume", json={"job_id": job_id})
        assert resp.status_code == 400
        assert "upload one" in resp.json()["detail"]


def test_generate_resume_and_cover_letter_flow(monkeypatch):
    job_id = _insert_job()
    _insert_resume()
    monkeypatch.setattr(
        generate,
        "get_client",
        lambda: FakeAnthropicClient(create_result=FakeCreateResponse("# Tailored Resume\n...")),
    )

    with TestClient(app) as client:
        app_resp = client.post("/api/applications", json={"job_id": job_id})
        application_id = app_resp.json()["id"]

        resp = client.post(
            "/api/generate/resume", json={"job_id": job_id, "application_id": application_id}
        )
        assert resp.status_code == 200
        assert "Tailored Resume" in resp.json()["content_text"]

        monkeypatch.setattr(
            generate,
            "get_client",
            lambda: FakeAnthropicClient(create_result=FakeCreateResponse("Dear Hiring Manager...")),
        )
        resp = client.post(
            "/api/generate/cover-letter", json={"job_id": job_id, "application_id": application_id}
        )
        assert resp.status_code == 200
        assert "Dear Hiring Manager" in resp.json()["content_text"]

        app_after = client.get(f"/api/applications/{application_id}").json()
        assert app_after["resume_doc_id"] is not None
        assert app_after["cover_letter_doc_id"] is not None

        history = client.get(f"/api/generate/history?job_id={job_id}").json()
        assert len(history) == 2


def test_generate_resume_with_explicit_resume_id(monkeypatch):
    job_id = _insert_job()
    _insert_resume("Old resume")
    newer_id = _insert_resume("Newer resume")
    monkeypatch.setattr(
        generate, "get_client", lambda: FakeAnthropicClient(create_result=FakeCreateResponse("tailored"))
    )

    with TestClient(app) as client:
        resp = client.post("/api/generate/resume", json={"job_id": job_id, "resume_id": newer_id})
        assert resp.status_code == 200
        with get_db() as conn:
            row = conn.execute(
                "SELECT based_on_resume_id FROM generated_documents WHERE id = ?",
                (resp.json()["id"],),
            ).fetchone()
        assert row["based_on_resume_id"] == newer_id
