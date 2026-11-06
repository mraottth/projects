import time

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.api import discovery as discovery_api
from jobhunt.core import company_resolver, discovery


def test_discovery_suggest_via_api(monkeypatch):
    monkeypatch.setattr(
        discovery_api,
        "suggest_companies",
        lambda profile, settings, exclude_names=None, **kwargs: [
            discovery.CompanySuggestion(company_name="Acme", reason="Hiring now")
        ],
    )

    with TestClient(app) as client:
        resp = client.post("/api/discovery/suggest")
        assert resp.status_code == 200
        assert resp.json() == [{"company_name": "Acme", "reason": "Hiring now", "careers_url": None}]


def test_discovery_suggest_excludes_already_monitored_companies(monkeypatch):
    captured = {}

    def fake_suggest_companies(profile, settings, exclude_names=None, **kwargs):
        captured["exclude_names"] = exclude_names
        return []

    monkeypatch.setattr(discovery_api, "suggest_companies", fake_suggest_companies)

    with TestClient(app) as client:
        client.post(
            "/api/sources",
            json={"kind": "custom_board", "name": "Niche Board", "company_name": "Niche Co", "config": {}},
        )
        resp = client.post("/api/discovery/suggest")
        assert resp.status_code == 200

    assert captured["exclude_names"] == ["Niche Co"]


def test_discovery_suggest_boards_excludes_already_monitored_boards(monkeypatch):
    captured = {}

    def fake_suggest_job_boards(profile, settings, exclude_urls=None, **kwargs):
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
    def raise_value_error(profile, settings, exclude_names=None, **kwargs):
        raise ValueError("no JSON object found in response: 'some truncated text'")

    monkeypatch.setattr(discovery_api, "suggest_companies", raise_value_error)

    with TestClient(app) as client:
        resp = client.post("/api/discovery/suggest")
        assert resp.status_code == 502
        assert "couldn't be parsed" in resp.json()["detail"]


def test_discovery_suggest_boards_returns_clean_502_on_unparseable_llm_response(monkeypatch):
    def raise_value_error(profile, settings, exclude_urls=None, **kwargs):
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


def test_resolve_source_404_when_unresolvable(monkeypatch):
    from jobhunt.api import sources

    monkeypatch.setattr(sources, "resolve_company", lambda name: None)

    with TestClient(app) as client:
        resp = client.post("/api/sources/resolve", json={"company_name": "Nonexistent"})
        assert resp.status_code == 404


def test_discovery_passes_guidance_and_like_through(monkeypatch):
    captured = {}

    def fake_suggest_job_boards(profile, settings, exclude_urls=None, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(discovery_api, "suggest_job_boards", fake_suggest_job_boards)
    with TestClient(app) as client:
        resp = client.post(
            "/api/discovery/suggest-boards",
            json={"guidance": "mission-driven tech startups", "like": "Tech Jobs for Good"},
        )
        assert resp.status_code == 200
        assert captured == {"guidance": "mission-driven tech startups", "like": "Tech Jobs for Good"}

        # No body at all still works (the original, unsteered search).
        assert client.post("/api/discovery/suggest-boards").status_code == 200
        assert captured == {"guidance": None, "like": None}


def _wait_for_task(client, task_id):
    for _ in range(100):
        body = client.get(f"/api/discovery/tasks/{task_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError("discovery task never finished")


def test_discovery_background_task_returns_suggestions(monkeypatch):
    captured = {}

    def fake_suggest_companies(profile, settings, exclude_names=None, **kwargs):
        captured.update(kwargs)
        return [discovery.CompanySuggestion(company_name="Acme", reason="Hiring now")]

    monkeypatch.setattr(discovery_api, "suggest_companies", fake_suggest_companies)
    with TestClient(app) as client:
        resp = client.post("/api/discovery/suggest/start", json={"like": "Civis Analytics"})
        assert resp.status_code == 200
        body = _wait_for_task(client, resp.json()["task_id"])

    assert body == {
        "status": "done",
        "result": [{"company_name": "Acme", "reason": "Hiring now", "careers_url": None}],
    }
    assert captured["like"] == "Civis Analytics"


def test_discovery_background_task_reports_errors(monkeypatch):
    def raise_value_error(profile, settings, exclude_urls=None, **kwargs):
        raise ValueError("no JSON object found in response")

    monkeypatch.setattr(discovery_api, "suggest_job_boards", raise_value_error)
    with TestClient(app) as client:
        task_id = client.post("/api/discovery/suggest-boards/start").json()["task_id"]
        body = _wait_for_task(client, task_id)

    assert body["status"] == "error"
    assert "couldn't be parsed" in body["error"]


def test_discovery_unknown_task_404s():
    with TestClient(app) as client:
        assert client.get("/api/discovery/tasks/nope").status_code == 404
