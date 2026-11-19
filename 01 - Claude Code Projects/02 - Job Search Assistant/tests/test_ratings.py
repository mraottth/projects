"""The person's own Interest and Fit ratings (asked 2026-11-19): clicking a chip changes it, every
page follows, and scoring, the preference profile and rescores take it into account."""

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import digest, preference, ratings, rescore, reset, scoring, users
from jobhunt.db.connection import get_db

B = {"X-Dev-User": "b@example.com"}


def _source(*user_ids) -> int:
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        for uid in user_ids:
            conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (?, ?)", (uid, source))
    return source


def _job(title, source, interest="High", fit="Med", user_id=1, company="Civic Signal") -> int:
    with get_db() as conn:
        job = conn.execute(
            "INSERT INTO jobs (source_id, dedupe_key, title, company, description_text) "
            "VALUES (?, ?, ?, ?, 'x')",
            (source, f"k-{title}", title, company),
        ).lastrowid
        if interest:
            conn.execute(
                """INSERT INTO job_scores (user_id, job_id, model, interest_fit, interest_score,
                   candidate_fit, candidate_fit_score, summary_text, rationale_text)
                   VALUES (?, ?, 'claude', ?, 77, ?, 61, 'summary', 'why')""",
                (user_id, job, interest, fit),
            )
    return job


def _in_digest(job: int) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO digest_entries (user_id, digest_date, job_id, rank) VALUES (1, ?, ?, 1)",
            (digest.today_str(), job),
        )


def test_a_rating_is_stored_and_shown_everywhere():
    job = _job("Analytics Lead", _source(1))
    _in_digest(job)
    http = TestClient(app)
    resp = http.put(
        f"/api/jobs/{job}/rating",
        json={"interest_fit": "High", "candidate_fit": "High", "note": "  my campaign work counts  "},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["rated_by_you"] is True
    # The unchanged label keeps its number; the changed one gets its band's middle.
    assert (body["interest_score"], body["candidate_fit"], body["candidate_fit_score"]) == (77, "High", 83)
    assert body["summary_text"] == "summary"  # the reasons stay

    with get_db() as conn:
        corr = conn.execute("SELECT * FROM rating_corrections").fetchone()
    assert (corr["from_interest"], corr["from_fit"], corr["to_interest"], corr["to_fit"]) == (
        "High",
        "Med",
        "High",
        "High",
    )
    assert corr["note"] == "my campaign work counts"

    ranked = {j["id"]: j for j in http.get("/api/jobs/ranked").json()}
    assert ranked[job]["candidate_fit"] == "High" and ranked[job]["rated_by_you"] is True
    [entry] = digest.get_digest(digest.today_str())
    assert entry["candidate_fit"] == "High" and entry["rated_by_you"] is True
    assert http.get(f"/api/jobs/{job}").json()["rated_by_you"] is True


def test_rating_a_job_low_takes_it_off_new_and_noteworthy():
    job = _job("Analytics Lead", _source(1))
    _in_digest(job)
    assert digest.get_digest(digest.today_str())
    ratings.correct(job, "Low", "Med")
    assert digest.get_digest(digest.today_str()) == []


def test_the_errors():
    source = _source(1)
    unscored = _job("Unscored", source, interest=None)
    http = TestClient(app)
    assert (
        http.put(
            f"/api/jobs/{unscored}/rating", json={"interest_fit": "High", "candidate_fit": "High"}
        ).status_code
        == 400
    )
    job = _job("Scored", source)
    assert (
        http.put(
            f"/api/jobs/{job}/rating", json={"interest_fit": "Great", "candidate_fit": "High"}
        ).status_code
        == 400
    )
    # Another person can't see (or rate) a job from a source they don't follow.
    users.create_user("b@example.com", "Bea", status="approved")
    resp = http.put(
        f"/api/jobs/{job}/rating", json={"interest_fit": "High", "candidate_fit": "High"}, headers=B
    )
    assert resp.status_code == 404


def test_corrections_reach_scoring_and_the_preference_profile():
    source = _source(1)
    first = _job("Data Manager", source, company="Open Ballot Lab")
    second = _job("Analytics Lead", source)
    assert scoring._corrections_text() == "(none yet)"
    ratings.correct(first, "Low", "Med", "too much people management")
    ratings.correct(second, "High", "Med")  # agreeing with the model
    text = scoring._corrections_text()
    lines = text.splitlines()
    assert lines[0] == "- Analytics Lead at Civic Signal: rated interest High, fit Med; they agreed"
    assert lines[1] == (
        "- Data Manager at Open Ballot Lab: rated interest High, fit Med; "
        "they said interest Low, fit Med (too much people management)"
    )
    system, _ = scoring._request_parts(
        {"title": "T", "company": "C", "description_text": "d"}, None, None, [], [], None, "", scoring.CURRENT
    )
    assert "## Where they corrected our ratings" in system[1]["text"]
    assert "too much people management" in system[1]["text"]

    signal, feedback_count, _ = preference._signal_text()
    assert "too much people management" in signal
    assert feedback_count == 2


def test_rescores_and_score_this_job_leave_your_rating_alone():
    source = _source(1)
    mine = _job("Mine", source)
    other = _job("Other", source)
    ratings.correct(mine, "Low", "Low")
    assert rescore.job_ids("all_jobs") == [other]
    resp = TestClient(app).post(f"/api/jobs/{mine}/score")
    assert resp.status_code == 409


def test_start_over_clears_corrections():
    job = _job("Analytics Lead", _source(1))
    ratings.correct(job, "Low", "Low")
    reset.delete_all_data()
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM rating_corrections").fetchone()[0] == 0
