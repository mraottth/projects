from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import scoring
from jobhunt.core.scoring import JobScore
from jobhunt.ingest import manual_import
from jobhunt.ingest.manual_import import ParsedJob
from tests.fakes import FakeAnthropicClient, FakeParseResponse


def test_manual_job_confirm_scores_and_auto_tracks(monkeypatch):
    parsed = ParsedJob(
        title="Staff Engineer",
        company="Acme Corp",
        location="Remote",
        remote_type="remote",
        salary_min=180000,
        salary_max=220000,
        salary_currency="USD",
        salary_period="year",
        description_text="Lead the platform team.",
        posted_at="2026-10-20",
    )
    monkeypatch.setattr(
        manual_import, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(parsed))
    )

    score = JobScore(
        interest_fit="High",
        interest_score=88,
        candidate_fit="High",
        candidate_fit_score=91,
        summary_text="Platform leadership role.",
        rationale_text="Strong match on backend + leadership experience.",
    )
    monkeypatch.setattr(
        scoring, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(score))
    )

    with TestClient(app) as client:
        resp = client.post("/api/jobs/manual", json={"raw_text": "Staff Engineer at Acme Corp..."})
        assert resp.status_code == 200
        job = resp.json()
        assert job["needs_review"] == 1

        resp = client.post(f"/api/jobs/{job['id']}/confirm")
        assert resp.status_code == 200
        confirmed = resp.json()
        assert confirmed["needs_review"] == 0
        assert confirmed["scores"][0]["interest_fit"] == "High"

        # Confirming a manually pasted job IS the interest signal -- it goes straight to My Jobs
        # (an auto-created application), not through the Digest's "new and noteworthy" list.
        resp = client.get("/api/digest/today")
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

        resp = client.get("/api/applications")
        assert resp.status_code == 200
        apps = resp.json()
        assert len(apps) == 1
        assert apps[0]["job_id"] == job["id"]
        assert apps[0]["status"] == "saved"

        # Re-confirming (e.g. a page refresh) doesn't error or create a second application.
        resp = client.post(f"/api/jobs/{job['id']}/confirm")
        assert resp.status_code == 200
        assert len(client.get("/api/applications").json()) == 1


def test_duplicate_manual_job_rejected(monkeypatch):
    parsed = ParsedJob(
        title="Engineer",
        company="Acme",
        location="NYC",
        remote_type=None,
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        salary_period=None,
        description_text="desc",
        posted_at=None,
    )
    monkeypatch.setattr(
        manual_import, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(parsed))
    )

    with TestClient(app) as client:
        first = client.post("/api/jobs/manual", json={"raw_text": "Engineer at Acme in NYC"})
        assert first.status_code == 200

        second = client.post("/api/jobs/manual", json={"raw_text": "Engineer at Acme in NYC"})
        assert second.status_code == 409
