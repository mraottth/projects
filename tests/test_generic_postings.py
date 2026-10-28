from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core.generic_postings import is_generic_posting
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run

GENERIC = [
    "General Application",
    "General Interest - Engineering",
    "Don't see the right role? Apply here",
    "Dont See Your Role? Submit Your Resume",
    "Join our Talent Community",
    "Talent Pool: Data & Analytics",
    "Resume Drop",
    "Open Application",
    "Future Opportunities",
    "Data Scientist (Future Opportunities)",
    "Expression of Interest – Policy",
]
REAL = [
    "General Counsel",
    "General Manager, Analytics",
    "Talent Acquisition Partner",
    "Open Source Engineer",
    "Future of Work Researcher",
    "Director of Data Science",
    "Resume Parsing Engineer",
    None,
]


def test_generic_titles_are_recognized():
    assert [t for t in GENERIC if not is_generic_posting(t)] == []
    assert [t for t in REAL if is_generic_posting(t)] == []


def test_generic_postings_are_hidden_from_all_jobs_and_not_queued():
    with get_db() as conn:
        source = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES ('custom_board', 'B', '{}')"
        ).lastrowid
        insert = "INSERT INTO jobs (dedupe_key, title, company, source_id) VALUES (?, ?, 'A', ?)"
        real = conn.execute(insert, ("r", "Data Director", source)).lastrowid
        generic = conn.execute(insert, ("g", "General Application", source)).lastrowid

    with TestClient(app) as client:
        titles = [j["title"] for j in client.get("/api/jobs/ranked").json()]
    assert titles == ["Data Director"]

    assert daily_run._prefilter_and_enqueue([real, generic]) == 1
    with get_db() as conn:
        queued = [r["job_id"] for r in conn.execute("SELECT job_id FROM scoring_queue")]
    assert queued == [real]
