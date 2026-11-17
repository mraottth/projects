"""Editing a source's name or address on Manage Sources (asked 2026-11-17)."""

import json

from fastapi.testclient import TestClient

from jobhunt.api import sources as sources_api
from jobhunt.api.main import app
from jobhunt.core import users
from jobhunt.db.connection import get_db


def _add(http, **body) -> dict:
    resp = http.post("/api/sources", json={"kind": "custom_board", "name": "Acme", **body})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_renaming_keeps_the_address_and_doesnt_recheck(monkeypatch):
    checked: list[int] = []
    monkeypatch.setattr(sources_api, "schedule_source_refresh", checked.append)
    with TestClient(app) as http:
        s = _add(http, config={"url": "https://acme.example/jobs"}, company_name="Acme")
        checked.clear()
        out = http.patch(f"/api/sources/{s['id']}", json={"name": "Acme Labs"}).json()
    assert (out["name"], out["company_name"], out["config"]) == (
        "Acme Labs",
        "Acme Labs",  # an organization's employer name follows its name
        {"url": "https://acme.example/jobs"},
    )
    assert checked == []


def test_a_new_address_replaces_the_config_clears_the_error_and_rechecks(monkeypatch):
    checked: list[int] = []
    monkeypatch.setattr(sources_api, "schedule_source_refresh", checked.append)
    with TestClient(app) as http:
        s = _add(http, config={"url": "https://acme.example/jobs", "embedded_ats": {"kind": "ats_ashby"}})
        with get_db() as conn:
            conn.execute("UPDATE sources SET last_error = 'found nothing' WHERE id = ?", (s["id"],))
        checked.clear()
        # A careers page that's really a Greenhouse board: the kind changes with the address.
        out = http.patch(
            f"/api/sources/{s['id']}", json={"kind": "ats_greenhouse", "config": {"board_token": "acme"}}
        ).json()
    assert (out["kind"], out["config"], out["last_error"]) == (
        "ats_greenhouse",
        {"board_token": "acme"},
        None,
    )
    assert checked == [s["id"]]
    with get_db() as conn:
        stored = json.loads(
            conn.execute("SELECT config_json FROM sources WHERE id = ?", (s["id"],)).fetchone()[0]
        )
    assert "embedded_ats" not in stored


def test_an_unknown_kind_is_refused():
    with TestClient(app) as http:
        s = _add(http, config={"url": "https://acme.example/jobs"})
        resp = http.patch(f"/api/sources/{s['id']}", json={"kind": "ats_nope"})
    assert resp.status_code == 400


def test_a_shared_source_cant_be_edited_by_a_non_admin():
    users.create_user("bea@example.com", "Bea", status="approved")
    with TestClient(app) as http:
        s = _add(http, config={"url": "https://acme.example/jobs"})
        bea = {"X-Dev-User": "bea@example.com"}
        http.post(
            "/api/sources",
            json={"kind": "custom_board", "name": "Acme", "config": {"url": "https://acme.example/jobs"}},
            headers=bea,
        )
        resp = http.patch(f"/api/sources/{s['id']}", json={"name": "Renamed"}, headers=bea)
    assert resp.status_code == 403
