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


def _saved_job(source_id: int, title: str, company: str) -> int:
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO jobs (source_id, dedupe_key, title, company) VALUES (?, ?, ?, ?)",
            (source_id, f"{source_id}-{title}", title, company),
        ).lastrowid


def _company(job_id: int) -> str:
    with get_db() as conn:
        return conn.execute("SELECT company FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]


def test_renaming_an_organization_renames_its_saved_jobs():
    with TestClient(app) as http:
        s = _add(http, config={"url": "https://acme.example/jobs"}, company_name="Acme")
        mine = _saved_job(s["id"], "Analyst", "Acme")
        other = _saved_job(s["id"], "Partner role", "Some Partner Org")  # names another employer
        http.patch(f"/api/sources/{s['id']}", json={"name": "Acme Labs"})
        tracked = http.post("/api/applications", json={"job_id": mine})
        assert tracked.status_code in (200, 201), tracked.text
        assert http.get("/api/applications").json()[0]["company"] == "Acme Labs"  # My Jobs
    assert (_company(mine), _company(other)) == ("Acme Labs", "Some Partner Org")


def test_a_greenhouse_board_renamed_takes_the_new_name_over_its_token():
    with TestClient(app) as http:
        s = http.post(
            "/api/sources",
            json={"kind": "ats_greenhouse", "name": "acme", "config": {"board_token": "acme"}},
        ).json()
        job = _saved_job(s["id"], "Engineer", "acme")
        out = http.patch(f"/api/sources/{s['id']}", json={"name": "Acme Corp"}).json()
    assert out["company_name"] == "Acme Corp"
    assert _company(job) == "Acme Corp"


def test_renaming_a_job_board_leaves_its_employers_alone():
    with TestClient(app) as http:
        s = _add(http, config={"url": "https://board.example/jobs"}, name="Board")
        job = _saved_job(s["id"], "Analyst", "Civic Signal")
        out = http.patch(f"/api/sources/{s['id']}", json={"name": "Better Board"}).json()
    assert out["company_name"] is None and _company(job) == "Civic Signal"
