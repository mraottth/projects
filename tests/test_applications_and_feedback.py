from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.db.connection import get_db


def _insert_job(title="Engineer", company="Acme") -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company) VALUES (?, ?, ?)",
            (f"{title}-{company}", title, company),
        )
        return cursor.lastrowid


def test_feedback_roundtrip():
    job_id = _insert_job()
    with TestClient(app) as client:
        resp = client.post(f"/api/jobs/{job_id}/feedback", json={"interested": "interested"})
        assert resp.status_code == 200

        resp = client.post(f"/api/jobs/{job_id}/feedback", json={"interested": "bogus"})
        assert resp.status_code == 400


def test_application_lifecycle_and_history():
    job_id = _insert_job()
    with TestClient(app) as client:
        resp = client.post("/api/applications", json={"job_id": job_id})
        assert resp.status_code == 200
        app_id = resp.json()["id"]
        assert resp.json()["status"] == "saved"
        assert resp.json()["applied_at"] is None

        # duplicate application for the same job is rejected
        dup = client.post("/api/applications", json={"job_id": job_id})
        assert dup.status_code == 409

        resp = client.patch(f"/api/applications/{app_id}", json={"status": "applied"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "applied"
        assert body["applied_at"] is not None
        assert [h["status"] for h in body["history"]] == ["saved", "applied"]

        resp = client.patch(f"/api/applications/{app_id}", json={"status": "interview"})
        assert resp.status_code == 200
        assert [h["status"] for h in resp.json()["history"]] == ["saved", "applied", "interview"]

        listing = client.get("/api/applications").json()
        assert len(listing) == 1
        assert listing[0]["title"] == "Engineer"
        # Regression: the list endpoint used to omit history entirely, while the frontend called
        # .map() on it unconditionally -- crashed the My Jobs page the moment a real application
        # with history existed.
        assert [h["status"] for h in listing[0]["history"]] == ["saved", "applied", "interview"]


def test_list_applications_includes_history_for_each_app():
    job_a = _insert_job("Engineer", "Acme")
    job_b = _insert_job("Designer", "Beta")
    with TestClient(app) as client:
        client.post("/api/applications", json={"job_id": job_a})
        client.post("/api/applications", json={"job_id": job_b})
        listing = client.get("/api/applications").json()
        assert len(listing) == 2
        for entry in listing:
            assert [h["status"] for h in entry["history"]] == ["saved"]


def test_application_rejects_invalid_status():
    job_id = _insert_job()
    with TestClient(app) as client:
        resp = client.post("/api/applications", json={"job_id": job_id, "status": "bogus"})
        assert resp.status_code == 400
