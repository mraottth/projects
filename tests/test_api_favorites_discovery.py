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
        lambda profile, settings, exclude_names=None: [
            discovery.CompanySuggestion(company_name="Acme", reason="Hiring now")
        ],
    )

    with TestClient(app) as client:
        resp = client.post("/api/discovery/suggest")
        assert resp.status_code == 200
        assert resp.json() == [{"company_name": "Acme", "reason": "Hiring now"}]


def test_discovery_suggest_excludes_already_monitored_and_favorited_companies(monkeypatch):
    monkeypatch.setattr(
        favorites,
        "resolve_company",
        lambda name: company_resolver.ResolvedSource(kind="ats_lever", config={}, note="x"),
    )
    captured = {}

    def fake_suggest_companies(profile, settings, exclude_names=None):
        captured["exclude_names"] = exclude_names
        return []

    monkeypatch.setattr(discovery_api, "suggest_companies", fake_suggest_companies)

    with TestClient(app) as client:
        client.post("/api/favorites", json={"company_name": "Acme"})
        client.post(
            "/api/sources",
            json={"kind": "custom_board", "name": "Niche Board", "company_name": "Niche Co", "config": {}},
        )
        resp = client.post("/api/discovery/suggest")
        assert resp.status_code == 200

    assert sorted(captured["exclude_names"]) == ["Acme", "Niche Co"]


def test_discovery_suggest_boards_excludes_already_monitored_boards(monkeypatch):
    captured = {}

    def fake_suggest_job_boards(profile, settings, exclude_urls=None):
        captured["exclude_urls"] = exclude_urls
        return []

    monkeypatch.setattr(discovery_api, "suggest_job_boards", fake_suggest_job_boards)

    with TestClient(app) as client:
        client.post(
            "/api/sources",
            json={
                "kind": "custom_board",
                "name": "Climate Jobs List",
                "config": {"url": "https://climatejobslist.com"},
            },
        )
        resp = client.post("/api/discovery/suggest-boards")
        assert resp.status_code == 200

    assert captured["exclude_urls"] == ["https://climatejobslist.com"]


def test_discovery_suggest_returns_clean_502_on_unparseable_llm_response(monkeypatch):
    # Regression: _extract_json raises a plain ValueError when Claude's response has no
    # parseable JSON (e.g. cut off mid-output after a long web-search tool-use turn). The route
    # used to catch only LLMError, so this propagated as an unhandled exception -> raw 500.
    def raise_value_error(profile, settings, exclude_names=None):
        raise ValueError("no JSON object found in response: 'some truncated text'")

    monkeypatch.setattr(discovery_api, "suggest_companies", raise_value_error)

    with TestClient(app) as client:
        resp = client.post("/api/discovery/suggest")
        assert resp.status_code == 502
        assert "couldn't be parsed" in resp.json()["detail"]


def test_discovery_suggest_boards_returns_clean_502_on_unparseable_llm_response(monkeypatch):
    def raise_value_error(profile, settings, exclude_urls=None):
        raise ValueError("no JSON object found in response: 'some truncated text'")

    monkeypatch.setattr(discovery_api, "suggest_job_boards", raise_value_error)

    with TestClient(app) as client:
        resp = client.post("/api/discovery/suggest-boards")
        assert resp.status_code == 502
        assert "couldn't be parsed" in resp.json()["detail"]


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
