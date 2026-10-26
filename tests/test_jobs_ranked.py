from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.db.connection import get_db


def _insert_source(is_favorite: int = 0) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO sources (kind, name, config_json, is_favorite) VALUES (?, ?, '{}', ?)",
            ("custom_board", "Some Board", is_favorite),
        )
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
                   (job_id, model, interest_fit, interest_score, candidate_fit, candidate_fit_score,
                    summary_text, rationale_text)
                   VALUES (?, 'x', 'High', ?, 'High', ?, '', '')""",
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


def test_ranked_jobs_sorts_favorites_first_then_by_score():
    fav_source = _insert_source(is_favorite=1)
    plain_source = _insert_source(is_favorite=0)
    _insert_job("k1", "Low Score Favorite", fav_source, score=10)
    _insert_job("k2", "High Score Non-Favorite", plain_source, score=90)
    _insert_job("k3", "Unscored Non-Favorite", plain_source)

    with TestClient(app) as client:
        resp = client.get("/api/jobs/ranked")
    assert resp.status_code == 200
    titles = [row["title"] for row in resp.json()]
    assert titles == ["Low Score Favorite", "High Score Non-Favorite", "Unscored Non-Favorite"]
