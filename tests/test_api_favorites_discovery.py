from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.api import discovery as discovery_api
from jobhunt.core import company_resolver, discovery, favorites


def test_favorite_lifecycle_via_api(monkeypatch):
    monkeypatch.setattr(
        favorites,
        "resolve_company",
        lambda name: company_resolver.ResolvedSource(
            kind="ats_greenhouse", config={"board_token": "acme"}, note="x"
        ),
    )

    with TestClient(app) as client:
        resp = client.post("/api/favorites", json={"company_name": "Acme"})
        assert resp.status_code == 200
        favorite = resp.json()
        assert favorite["resolution_status"] == "resolved"

        resp = client.get("/api/favorites")
        assert len(resp.json()) == 1

        resp = client.delete(f"/api/favorites/{favorite['id']}")
        assert resp.status_code == 200
        assert client.get("/api/favorites").json() == []


def test_favorite_retry_via_api(monkeypatch):
    monkeypatch.setattr(favorites, "resolve_company", lambda name: None)

    with TestClient(app) as client:
        resp = client.post("/api/favorites", json={"company_name": "Acme"})
        assert resp.json()["resolution_status"] == "failed"
        favorite_id = resp.json()["id"]

        monkeypatch.setattr(
            favorites,
            "resolve_company",
            lambda name: company_resolver.ResolvedSource(kind="ats_lever", config={}, note="x"),
        )
        resp = client.post(f"/api/favorites/{favorite_id}/retry")
        assert resp.json()["resolution_status"] == "resolved"


def test_discovery_suggest_via_api(monkeypatch):
    monkeypatch.setattr(
        discovery_api,
        "suggest_companies",
        lambda profile, settings: [
            discovery.CompanySuggestion(company_name="Acme", reason="Hiring now")
        ],
    )

    with TestClient(app) as client:
        resp = client.post("/api/discovery/suggest")
        assert resp.status_code == 200
        assert resp.json() == [{"company_name": "Acme", "reason": "Hiring now"}]


def test_resolve_source_via_api(monkeypatch):
    from jobhunt.api import sources

    monkeypatch.setattr(
        sources,
        "resolve_company",
        lambda name: company_resolver.ResolvedSource(
            kind="ats_ashby", config={"board_name": "acme"}, note="x"
        ),
    )

    with TestClient(app) as client:
        resp = client.post("/api/sources/resolve", json={"company_name": "Acme"})
        assert resp.status_code == 200
        source = resp.json()
        assert source["kind"] == "ats_ashby"
        assert source["is_favorite"] == 0


def test_resolve_source_404_when_unresolvable(monkeypatch):
    from jobhunt.api import sources

    monkeypatch.setattr(sources, "resolve_company", lambda name: None)

    with TestClient(app) as client:
        resp = client.post("/api/sources/resolve", json={"company_name": "Nonexistent"})
        assert resp.status_code == 404
