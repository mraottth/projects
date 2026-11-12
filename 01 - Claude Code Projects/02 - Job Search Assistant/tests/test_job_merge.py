"""A pasted job and the same job found by a source become one job (D-035, core/job_merge.py)."""

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import job_merge
from jobhunt.core.job_merge import posting_identity
from jobhunt.db.connection import get_db
from jobhunt.ingest import manual_import
from jobhunt.ingest.manual_import import ParsedJob
from jobhunt.pipeline import daily_run
from tests.fakes import FakeAnthropicClient, FakeParseResponse


def _source(user_ids=(1,)) -> int:
    with get_db() as conn:
        sid = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'Board', '{}')"
        ).lastrowid
        for uid in user_ids:
            conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (?, ?)", (uid, sid))
    return sid


def _pasted(title, company="Acme", url=None, location=None, description="d", user_id=1) -> int:
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company, location, url, description_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                f"u{user_id}:{title}:{url}:{location}:{hash(description)}",
                title,
                company,
                location,
                url,
                description,
            ),
        ).lastrowid


def _track(job_id, status="saved", notes=None, user_id=1) -> int:
    with get_db() as conn:
        app_id = conn.execute(
            "INSERT INTO applications (user_id, job_id, status, notes) VALUES (?, ?, ?, ?)",
            (user_id, job_id, status, notes),
        ).lastrowid
        conn.execute(
            "INSERT INTO application_status_history (application_id, status) VALUES (?, ?)", (app_id, status)
        )
    return app_id


def _found(source_id, title, company="Acme", url=None, location=None, description="d", ext=None) -> int:
    return daily_run._upsert_job(
        source_id,
        {
            "title": title,
            "company": company,
            "url": url,
            "location": location,
            "description_text": description,
            "external_id": ext or f"{title}-{url}-{location}",
        },
    )


def _jobs():
    with get_db() as conn:
        return [dict(r) for r in conn.execute("SELECT id, source_id, user_id FROM jobs ORDER BY id")]


# ---- identity


def test_one_greenhouse_job_has_one_identity_whatever_its_address():
    same = {
        posting_identity("https://boards.greenhouse.io/acme/jobs/123"),
        posting_identity("https://job-boards.greenhouse.io/acme/jobs/123?gh_src=abc"),
        posting_identity("https://acme.com/careers/?gh_jid=123&utm_source=x"),
    }
    assert same == {"greenhouse:123"}
    assert posting_identity("https://acme.com/careers?gh_jid=456") != "greenhouse:123"


def test_lever_ashby_and_other_sites():
    uuid = "0f2b8c4e-1a2b-4c3d-8e9f-0123456789ab"
    assert posting_identity(f"https://jobs.lever.co/acme/{uuid}/apply") == f"lever:{uuid}"
    assert posting_identity(f"https://jobs.ashbyhq.com/acme/{uuid.upper()}") == f"ashby:{uuid}"
    # Another site keeps its query (the job's id can live there), minus tracking parameters.
    assert posting_identity("https://acme.org/jobs?id=7&utm_medium=email") == "url:acme.org/jobs?id=7"
    assert posting_identity("https://acme.org/jobs?id=7") != posting_identity("https://acme.org/jobs?id=8")


# ---- a source finds a job you pasted


def test_a_source_finding_your_pasted_job_merges_everything_onto_it():
    sid = _source()
    pasted = _pasted("Director of Data", url="https://acme.com/careers?gh_jid=55")
    app_id = _track(pasted, "interviewing", notes="Spoke to Sam")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO job_scores (user_id, job_id, interest_fit, interest_score, candidate_fit, "
            "candidate_fit_score, summary_text, rationale_text, model) "
            "VALUES (1, ?, 'High', 80, 'High', 80, 's', 'r', 'm')",
            (pasted,),
        )
        thread = conn.execute(
            "INSERT INTO jobby_threads (user_id, job_id) VALUES (1, ?)", (pasted,)
        ).lastrowid
        conn.execute(
            "INSERT INTO jobby_messages (thread_id, role, content) VALUES (?, 'user', 'hi')", (thread,)
        )

    found = _found(sid, "Director, Data", url="https://boards.greenhouse.io/acme/jobs/55")

    assert [j["id"] for j in _jobs()] == [found]
    with get_db() as conn:
        row = dict(conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone())
        assert (row["job_id"], row["status"], row["notes"]) == (found, "interviewing", "Spoke to Sam")
        assert conn.execute("SELECT job_id FROM job_scores").fetchone()[0] == found
        assert conn.execute("SELECT job_id FROM jobby_threads WHERE id = ?", (thread,)).fetchone()[0] == found
    with TestClient(app) as http:
        # Applied to, so it's on My Jobs only (not All Jobs).
        assert found not in {j["id"] for j in http.get("/api/jobs/ranked").json()}
        assert [a["job_id"] for a in http.get("/api/applications").json()] == [found]


def test_same_title_and_organization_merges_only_when_its_the_one():
    sid = _source()
    pasted = _pasted("Sr. Data Scientist", company="Acme, Inc.", location="Washington, DC")
    _track(pasted)
    found = _found(sid, "Senior Data Scientist", company="Acme", location="Washington, DC (Hybrid)")
    assert [j["id"] for j in _jobs()] == [found]


def test_a_different_city_is_a_different_job():
    sid = _source()
    pasted = _pasted("Data Scientist", location="New York, NY")
    _found(sid, "Data Scientist", location="Austin, TX")
    assert pasted in [j["id"] for j in _jobs()]


def test_the_description_picks_among_same_titled_jobs_and_otherwise_nothing_merges():
    sid = _source()
    team_a = "Build forecasting models for our elections team using survey data and turnout history."
    team_b = "Own billing analytics: revenue dashboards, churn reporting and pricing experiments."
    _found(sid, "Data Scientist", description=team_a, ext="a")
    pasted = _pasted("Data Scientist", description=team_b)
    unsure = _pasted("Data Scientist", description="Something else entirely about kitchens and recipes.")
    b = _found(sid, "Data Scientist", description=team_b, ext="b")
    ids = [j["id"] for j in _jobs()]
    assert pasted not in ids and b in ids
    assert unsure in ids  # matches neither closely: left alone


def test_when_both_copies_are_tracked_the_further_along_one_stays_with_both_notes():
    sid = _source()
    found = _found(sid, "Analytics Lead", url="https://example.org/jobs/9")
    _track(found, "saved", notes="Looks good")
    pasted = _pasted("Analytics Lead", url="https://example.org/jobs/9/")
    kept = _track(pasted, "waiting", notes="Applied via referral")
    assert job_merge.merge_existing_duplicates() == 1
    with get_db() as conn:
        apps = [dict(r) for r in conn.execute("SELECT * FROM applications")]
        history = conn.execute(
            "SELECT COUNT(*) FROM application_status_history WHERE application_id = ?", (kept,)
        )
        assert history.fetchone()[0] == 2
    assert len(apps) == 1
    assert (apps[0]["id"], apps[0]["job_id"], apps[0]["status"]) == (kept, found, "waiting")
    assert apps[0]["notes"] == "Applied via referral\n\nLooks good"


def test_another_persons_pasted_job_is_never_merged():
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (id, email, name, status) VALUES (2, 'b@example.com', 'B', 'approved')"
        )
    sid = _source(user_ids=(1,))
    theirs = _pasted("Analytics Lead", url="https://example.org/jobs/9", user_id=2)
    _found(sid, "Analytics Lead", url="https://example.org/jobs/9")
    assert theirs in [j["id"] for j in _jobs()]


# ---- pasting a job a source already has


def test_pasting_an_address_a_source_has_tracks_that_job_without_reading_it(monkeypatch):
    sid = _source()
    found = _found(sid, "Analytics Lead", url="https://boards.greenhouse.io/acme/jobs/77")
    monkeypatch.setattr(
        manual_import, "get_client", lambda: (_ for _ in ()).throw(AssertionError("paid read"))
    )
    with TestClient(app) as http:
        job = http.post("/api/jobs/manual", json={"url": "https://acme.com/careers?gh_jid=77"}).json()
        assert (job["id"], job["already_in_sources"]) == (found, True)
        assert [a["job_id"] for a in http.get("/api/applications").json()] == [found]
    assert len(_jobs()) == 1


def test_pasting_text_of_a_job_a_source_has_tracks_that_job(monkeypatch):
    sid = _source()
    found = _found(sid, "Analytics Lead", company="Civic Signal", location="Remote")
    parsed = ParsedJob(
        title="Analytics Lead", company="Civic Signal, Inc.", location="Remote", remote_type="remote",
        salary_min=None, salary_max=None, salary_currency=None, salary_period=None,
        description_text="d", posted_at=None,
    )  # fmt: skip
    monkeypatch.setattr(
        manual_import, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(parsed))
    )
    with TestClient(app) as http:
        job = http.post("/api/jobs/manual", json={"raw_text": "Analytics Lead at Civic Signal"}).json()
    assert (job["id"], job["already_in_sources"]) == (found, True)
    assert len(_jobs()) == 1
