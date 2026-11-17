"""Settings' daily run audit (asked 2026-11-17): checks, sources, new and scored jobs, the queue."""

import json
from datetime import date

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import run_audit, users
from jobhunt.core.settings import update_settings
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run

DAY = date(2026, 11, 17)
NOON = "2026-11-17 17:00:00"  # noon in New York


def _source(name, company=None, user_id=1, kind="custom_board") -> int:
    with get_db() as conn:
        sid = conn.execute(
            "INSERT INTO sources (kind, name, company_name, config_json) VALUES (?, ?, ?, '{}')",
            (kind, name, company),
        ).lastrowid
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (?, ?)", (user_id, sid))
    return sid


def _job(source_id, title, seen=NOON, location=None) -> int:
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO jobs (source_id, dedupe_key, title, company, location, first_seen_at) "
            "VALUES (?, ?, ?, 'Acme', ?, ?)",
            (source_id, f"{source_id}-{title}", title, location, seen),
        ).lastrowid


def _score(job_id, at=NOON, user_id=1):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO job_scores (user_id, job_id, scored_at, interest_fit, interest_score, "
            "candidate_fit, candidate_fit_score) VALUES (?, ?, ?, 'High', 80, 'Med', 50)",
            (user_id, job_id, at),
        )


def _queue(job_id, priority, screen=None, user_id=1):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO scoring_queue (user_id, job_id, priority, screen) VALUES (?, ?, ?, ?)",
            (user_id, job_id, priority, screen),
        )


def _run(checked, failed=None, started="2026-11-17T19:00:00+00:00", user_id=1):
    fails = {str(k): v for k, v in (failed or {}).items()}
    with get_db() as conn:
        conn.execute(
            "INSERT INTO runs (user_id, run_date, started_at, finished_at, audit_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, "2026-11-17", started, started, json.dumps({"checked": checked, "failed": fails})),
        )


def test_a_check_records_the_sources_it_read_and_those_that_failed(monkeypatch):
    good = _source("Good Board")
    bad = _source("Bad Org", company="Bad Org")
    users.create_user("bea@example.com", "Bea", status="approved")
    theirs = _source("Their Board", user_id=2)

    def read(source):
        if source["id"] == bad:
            raise RuntimeError("timed out")
        return [{"title": f"Job {source['id']}", "company": "Acme"}]

    monkeypatch.setattr(daily_run, "ADAPTERS", {"custom_board": read})
    daily_run.run_daily(only_user=1)
    with get_db() as conn:
        audit = json.loads(
            conn.execute("SELECT audit_json FROM runs WHERE user_id = 1 ORDER BY id DESC").fetchone()[0]
        )
    assert audit["checked"] == [good] and theirs not in audit["checked"]
    assert audit["failed"] == {str(bad): "timed out"}


def test_the_audit_for_a_day():
    board = _source("Big Board")
    org = _source("Acme", company="Acme")
    _source("Quiet Org", company="Quiet Org")
    gh = _source("Greenhouse Org", kind="ats_greenhouse")
    _run([board, org], failed={gh: "404"}, started="2026-11-17T19:00:00+00:00")
    _run([gh], started="2026-11-17T20:00:00+00:00")  # a later read clears the earlier failure
    _run([board], started="2026-11-16T19:00:00+00:00")  # another day

    a, b = _job(board, "Analyst"), _job(board, "Lead")
    _job(org, "Director")
    _job(board, "Old", seen="2026-11-15 12:00:00")
    _job(board, "Late night", seen="2026-11-18 03:00:00")  # 10pm on the 17th in New York
    _score(a)
    _score(b, at="2026-11-10 12:00:00")
    _score(b)  # a rescore

    out = run_audit.audit(DAY, "America/New_York")
    assert len(out["checks"]) == 2
    src = out["sources"]
    assert src["recorded"] and src["failed"] == []
    assert src["organizations"] == {"checked": 2, "total": 3}  # Acme, Greenhouse Org; not Quiet Org
    assert src["boards"] == {"checked": 1, "total": 1}
    assert out["new_jobs"] == [{"source": "Big Board", "jobs": 3}, {"source": "Acme", "jobs": 1}]
    assert [(s["title"], s["rescored"]) for s in out["scored"]] == [("Analyst", False), ("Lead", True)]
    assert "2026-11-17" in out["days"] and "2026-11-16" in out["days"]


def test_the_queue_is_in_scoring_order_and_counts_what_it_hides():
    one, two = _source("One"), _source("Two")
    first = _job(one, "Top of One")
    second = _job(two, "Top of Two")
    third = _job(one, "Second of One")
    dropped = _job(two, "Set aside")
    far = _job(two, "Far away", location="London, UK")
    for job, priority in ((first, 50), (third, 40), (second, 10), (far, 90)):
        _queue(job, priority)
    _queue(dropped, 99, screen="drop")
    with users.acting_as(1):
        update_settings({"prefilter_locations": ["DC"], "location_dealbreaker": True})
    q = run_audit.audit(DAY, "America/New_York")["queue"]
    # Round-robin like queue.select_batch: each source's best, then each source's second best.
    assert [j["title"] for j in q["jobs"]] == ["Top of One", "Top of Two", "Second of One"]
    assert (q["set_aside"], q["hidden"]) == (1, 1)


def test_the_route_is_per_person_and_checks_its_date():
    bea = users.create_user("bea@example.com", "Bea", status="approved")["id"]
    mine = _source("Mine")
    _score(_job(mine, "My job"))
    with TestClient(app) as http:
        assert http.get("/api/digest/run-audit?date=2026-11-17&tz=America/New_York").json()["scored"]
        theirs = http.get(
            "/api/digest/run-audit?date=2026-11-17&tz=America/New_York",
            headers={"X-Dev-User": "bea@example.com"},
        ).json()
        assert theirs["scored"] == [] and theirs["new_jobs"] == []
        assert http.get("/api/digest/run-audit?date=nope").status_code == 400
        assert http.get("/api/digest/run-audit?tz=Not/AZone").status_code == 200
    assert bea


def test_scored_jobs_come_best_first():
    board = _source("Board")
    with get_db() as conn:
        for title, interest, fit in (("Okay", 50, 40), ("Best", 90, 85), ("Weak", 20, 10)):
            job = _job(board, title)
            conn.execute(
                "INSERT INTO job_scores (user_id, job_id, scored_at, interest_fit, interest_score, "
                "candidate_fit, candidate_fit_score) VALUES (1, ?, ?, 'Med', ?, 'Med', ?)",
                (job, NOON, interest, fit),
            )
    scored = run_audit.audit(DAY, "America/New_York")["scored"]
    assert [s["title"] for s in scored] == ["Best", "Okay", "Weak"]
