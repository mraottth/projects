from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.db.connection import get_db


def _insert_source() -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES (?, ?, '{}')",
            ("custom_board", "Some Board"),
        )
        conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (1, ?)", (cursor.lastrowid,))
        return cursor.lastrowid


def _insert_job(dedupe_key: str, title: str, source_id: int | None, score: float | None = None) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO jobs (source_id, dedupe_key, title, company)
               VALUES (?, ?, ?, 'Acme')""",
            (source_id, dedupe_key, title),
        )
        job_id = cursor.lastrowid
        if score is not None:
            conn.execute(
                """INSERT INTO job_scores
                   (user_id, job_id, model, interest_fit, interest_score, candidate_fit, candidate_fit_score,
                    summary_text, rationale_text)
                   VALUES (1, ?, 'x', 'High', ?, 'High', ?, '', '')""",
                (job_id, score, score),
            )
        return job_id


def test_ranked_jobs_excludes_manually_pasted_jobs():
    source_id = _insert_source()
    _insert_job("k1", "Sourced Job", source_id, score=50)
    _insert_job("k2", "Pasted Job", None)  # no source_id -- manually pasted

    with TestClient(app) as client:
        resp = client.get("/api/jobs/ranked")
    assert resp.status_code == 200
    titles = {row["title"] for row in resp.json()}
    assert titles == {"Sourced Job"}


def test_ranked_jobs_sorts_by_score_with_unscored_last():
    source = _insert_source()
    _insert_job("k1", "Low Score", source, score=10)
    _insert_job("k2", "High Score", source, score=90)
    _insert_job("k3", "Unscored", source)

    with TestClient(app) as client:
        resp = client.get("/api/jobs/ranked")
    assert resp.status_code == 200
    titles = [row["title"] for row in resp.json()]
    assert titles == ["High Score", "Low Score", "Unscored"]


def test_each_job_says_its_status_on_my_jobs():
    """All Jobs cards show where a job stands on My Jobs (asked 2026-11-10); another person's
    tracking never shows."""
    from jobhunt.core import users

    source_id = _insert_source()
    tracked = _insert_job("k1", "Tracked", source_id, score=60)
    _insert_job("k2", "Not tracked", source_id, score=50)
    bea = users.create_user("bea@example.com", "Bea", status="approved")["id"]
    with get_db() as conn:
        conn.execute(
            "INSERT INTO applications (user_id, job_id, status) VALUES (1, ?, 'waiting')", (tracked,)
        )
        other = _insert_job("k3", "Bea's", source_id, score=40)
        conn.execute(
            "INSERT INTO applications (user_id, job_id, status) VALUES (?, ?, 'offer')", (bea, other)
        )

    with get_db() as conn:
        conn.execute("UPDATE applications SET status = 'saved' WHERE user_id = 1")
    with TestClient(app) as client:
        statuses = {row["title"]: row["application_status"] for row in client.get("/api/jobs/ranked").json()}
    assert statuses == {"Tracked": "saved", "Not tracked": None, "Bea's": None}


def test_a_job_youve_applied_to_leaves_all_jobs_until_its_removed_from_my_jobs():
    """Asked 2026-11-12: applied-to jobs live on My Jobs only; Drafting ones stay on All Jobs."""
    source_id = _insert_source()
    drafting = _insert_job("k1", "Drafting", source_id, score=60)
    applied = _insert_job("k2", "Applied", source_id, score=50)
    with get_db() as conn:
        conn.execute("INSERT INTO applications (user_id, job_id, status) VALUES (1, ?, 'saved')", (drafting,))
        app_id = conn.execute(
            "INSERT INTO applications (user_id, job_id, status) VALUES (1, ?, 'waiting')", (applied,)
        ).lastrowid

    with TestClient(app) as client:
        assert [r["title"] for r in client.get("/api/jobs/ranked").json()] == ["Drafting"]
        assert client.delete(f"/api/applications/{app_id}").status_code in (200, 204)
        assert {r["title"] for r in client.get("/api/jobs/ranked").json()} == {"Drafting", "Applied"}


def test_the_list_leaves_out_descriptions_and_raw_data_which_load_separately():
    """All Jobs sends its whole list at once, so it carries only what cards and filters use; the
    descriptions (for the search box) come from their own route (asked 2026-11-10: slow loads)."""
    source_id = _insert_source()
    job_id = _insert_job("k1", "Analyst", source_id, score=50)
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET description_text = 'Full text', raw_json = '{\"big\": 1}' WHERE id = ?",
            (job_id,),
        )
    with TestClient(app) as client:
        [row] = client.get("/api/jobs/ranked").json()
        descriptions = client.get("/api/jobs/ranked/descriptions").json()
        gzipped = client.get("/api/jobs/ranked", headers={"Accept-Encoding": "gzip"})
    assert row["description_text"] is None and "raw_json" not in row and "dedupe_key" not in row
    assert row["title"] == "Analyst" and row["interest_fit"] == "High"
    assert descriptions == {str(job_id): "Full text"}
    assert gzipped.status_code == 200
