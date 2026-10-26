from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.pipeline import daily_run


def test_source_crud_and_test_fetch(monkeypatch):
    monkeypatch.setitem(
        daily_run.ADAPTERS,
        "ats_greenhouse",
        lambda source: [{"title": "Engineer", "company": "Acme"}],
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/sources",
            json={"kind": "ats_greenhouse", "name": "Acme board", "config": {"board_token": "acme"}},
        )
        assert resp.status_code == 200
        source = resp.json()
        assert source["config"] == {"board_token": "acme"}
        assert source["enabled"] is True or source["enabled"] == 1

        resp = client.post(f"/api/sources/{source['id']}/test")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

        resp = client.patch(f"/api/sources/{source['id']}", json={"enabled": False})
        assert resp.status_code == 200

        resp = client.get("/api/sources")
        assert len(resp.json()) == 1

        resp = client.delete(f"/api/sources/{source['id']}")
        assert resp.status_code == 200
        assert client.get("/api/sources").json() == []


def test_create_source_rejects_unknown_kind():
    with TestClient(app) as client:
        resp = client.post("/api/sources", json={"kind": "bogus", "name": "x"})
        assert resp.status_code == 400


def test_internal_run_daily_without_secret_configured(monkeypatch):
    monkeypatch.delenv("SCHEDULER_SECRET", raising=False)
    monkeypatch.setattr(daily_run, "ADAPTERS", {})

    with TestClient(app) as client:
        resp = client.post("/internal/run-daily")
        assert resp.status_code == 200
        assert "llm_scored_count" in resp.json()


def test_internal_run_daily_rejects_bad_secret(monkeypatch):
    monkeypatch.setenv("SCHEDULER_SECRET", "s3cr3t")

    with TestClient(app) as client:
        resp = client.post("/internal/run-daily")
        assert resp.status_code == 403

        resp = client.post("/internal/run-daily", headers={"X-Scheduler-Secret": "wrong"})
        assert resp.status_code == 403

        resp = client.post("/internal/run-daily", headers={"X-Scheduler-Secret": "s3cr3t"})
        assert resp.status_code == 200
