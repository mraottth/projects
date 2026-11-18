from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.db.connection import get_db


def _insert_job(title="Engineer", company="Acme", salary_min=None, salary_max=None) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company, salary_min, salary_max) "
            "VALUES (1, ?, ?, ?, ?, ?)",
            (f"{title}-{company}", title, company, salary_min, salary_max),
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

        resp = client.patch(f"/api/applications/{app_id}", json={"status": "waiting"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "waiting"
        assert body["applied_at"] is not None
        assert [h["status"] for h in body["history"]] == ["saved", "waiting"]

        resp = client.patch(f"/api/applications/{app_id}", json={"status": "interviewing"})
        assert resp.status_code == 200
        assert [h["status"] for h in resp.json()["history"]] == ["saved", "waiting", "interviewing"]

        listing = client.get("/api/applications").json()
        assert len(listing) == 1
        assert listing[0]["title"] == "Engineer"
        # Regression: the list endpoint used to omit history entirely, while the frontend called
        # .map() on it unconditionally -- crashed the My Jobs page the moment a real application
        # with history existed.
        assert [h["status"] for h in listing[0]["history"]] == ["saved", "waiting", "interviewing"]


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


def test_applications_include_salary_fields():
    job_id = _insert_job(salary_min=120000, salary_max=150000)
    with TestClient(app) as client:
        resp = client.post("/api/applications", json={"job_id": job_id})
        assert resp.json()["salary_min"] == 120000
        assert resp.json()["salary_max"] == 150000

        listing = client.get("/api/applications").json()
        assert listing[0]["salary_min"] == 120000
        assert listing[0]["salary_max"] == 150000


def test_applied_at_is_directly_editable():
    job_id = _insert_job()
    with TestClient(app) as client:
        app_id = client.post("/api/applications", json={"job_id": job_id}).json()["id"]

        resp = client.patch(f"/api/applications/{app_id}", json={"applied_at": "2026-01-15"})
        assert resp.status_code == 200
        assert resp.json()["applied_at"] == "2026-01-15"
        # Still "saved" -- editing the date directly shouldn't change status or add history.
        assert resp.json()["status"] == "saved"
        assert [h["status"] for h in resp.json()["history"]] == ["saved"]


def test_explicit_applied_at_is_not_overwritten_by_status_change_in_same_request():
    job_id = _insert_job()
    with TestClient(app) as client:
        app_id = client.post("/api/applications", json={"job_id": job_id}).json()["id"]

        resp = client.patch(
            f"/api/applications/{app_id}",
            json={"status": "waiting", "applied_at": "2026-02-01"},
        )
        assert resp.status_code == 200
        assert resp.json()["applied_at"] == "2026-02-01"


def test_delete_application_removes_it_from_listing():
    job_id = _insert_job()
    with TestClient(app) as client:
        app_id = client.post("/api/applications", json={"job_id": job_id}).json()["id"]
        assert len(client.get("/api/applications").json()) == 1

        resp = client.delete(f"/api/applications/{app_id}")
        assert resp.status_code == 200
        assert client.get("/api/applications").json() == []

        assert client.delete(f"/api/applications/{app_id}").status_code == 404


def test_delete_application_preserves_generated_documents_but_detaches_them():
    job_id = _insert_job()
    with TestClient(app) as client:
        app_id = client.post("/api/applications", json={"job_id": job_id}).json()["id"]

    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO generated_documents (user_id, job_id, application_id, kind, content_text) "
            "VALUES (1, ?, ?, 'resume', 'tailored content')",
            (job_id, app_id),
        )
        doc_id = cursor.lastrowid

    with TestClient(app) as client:
        resp = client.delete(f"/api/applications/{app_id}")
        assert resp.status_code == 200

    with get_db() as conn:
        doc = conn.execute(
            "SELECT application_id FROM generated_documents WHERE id = ?", (doc_id,)
        ).fetchone()
    assert doc is not None
    assert doc["application_id"] is None


def test_a_reason_for_not_interested_reaches_the_scoring_prompt():
    """Not Interested asks why (asked 2026-11-18); the reason is the feedback's note, and every
    score reads the reasons, newest first, without waiting for a preference rebuild."""
    from jobhunt.core import scoring
    from jobhunt.core.users import acting_as

    senior, sales, liked, skipped = (
        _insert_job("VP of Data", "BigCo"),
        _insert_job("Sales Analyst", "Shop"),
        _insert_job("Data Lead", "Civic"),
        _insert_job("Analyst", "Other"),
    )
    with TestClient(app) as client:
        for job, value, note in (
            (senior, "not_interested", "  Too senior — I want a hands-on role "),
            (sales, "not_interested", "Not my line of work"),
            (liked, "interested", "Great mission"),
            (skipped, "not_interested", "   "),
        ):
            resp = client.post(f"/api/jobs/{job}/feedback", json={"interested": value, "note": note})
            assert resp.status_code == 200
        too_long = client.post(
            f"/api/jobs/{senior}/feedback", json={"interested": "not_interested", "note": "x" * 1001}
        )
        assert too_long.status_code == 422

    with get_db() as conn:
        assert conn.execute("SELECT note FROM feedback WHERE job_id = ?", (skipped,)).fetchone()[0] is None
    with acting_as(1):
        reasons = scoring._passed_reasons_text()
        assert reasons.splitlines() == [
            "- Sales Analyst at Shop: Not my line of work",
            "- VP of Data at BigCo: Too senior — I want a hands-on role",
        ]
        system, _ = scoring._request_parts(
            {"title": "Director", "company": "X"}, "profile", None, None, None, None, None
        )
    context = system[1]["text"]
    assert "## Why they passed on earlier jobs" in context and "Not my line of work" in context
    assert "reasons they gave for passing" in system[0]["text"]


def test_no_reasons_yet_reads_as_none():
    from jobhunt.core import scoring
    from jobhunt.core.users import acting_as

    with acting_as(1):
        assert scoring._passed_reasons_text() == "(none given yet)"
