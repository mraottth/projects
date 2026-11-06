"""Job cards name the source only when it's a job board (asked 2026-11-06): each job on All Jobs and
New & Noteworthy says whether its source is a board, by the same rule Manage Sources files sources
by. An ATS board or a page with an employer name is one organization; anything else is a board."""

from datetime import date

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core.digest import add_to_digest, get_digest
from jobhunt.db.connection import get_db

SOURCES = {
    # name: (kind, company_name, is a board)
    "Idealist": ("custom_board", None, True),
    "Remote OK": ("aggregator_remoteok", None, True),
    "Empty employer": ("custom_board", "", True),
    "Acme careers": ("custom_board", "Acme", False),
    "Acme Greenhouse": ("ats_greenhouse", None, False),
}
EXPECTED = {f"Job from {name}": board for name, (_, _, board) in SOURCES.items()}


def _seed_scored_jobs() -> list[int]:
    jobs = []
    with get_db() as conn:
        for i, (name, (kind, company, _)) in enumerate(SOURCES.items()):
            source = conn.execute(
                "INSERT INTO sources (kind, name, company_name, config_json) VALUES (?, ?, ?, '{}')",
                (kind, name, company),
            ).lastrowid
            conn.execute("INSERT INTO source_follows (user_id, source_id) VALUES (1, ?)", (source,))
            job = conn.execute(
                "INSERT INTO jobs (source_id, dedupe_key, title, company) VALUES (?, ?, ?, 'Acme')",
                (source, f"k{i}", f"Job from {name}"),
            ).lastrowid
            conn.execute(
                "INSERT INTO job_scores (user_id, job_id, model, interest_fit, interest_score, "
                "candidate_fit, candidate_fit_score, summary_text, rationale_text) "
                "VALUES (1, ?, 'x', 'High', 80, 'High', 80, '', '')",
                (job,),
            )
            jobs.append(job)
    return jobs


def test_all_jobs_and_the_digest_say_which_sources_are_boards():
    for job in _seed_scored_jobs():
        add_to_digest(job)

    with TestClient(app) as client:
        ranked = client.get("/api/jobs/ranked").json()
    assert {j["title"]: bool(j["source_is_board"]) for j in ranked} == EXPECTED

    digest = get_digest(date.today().isoformat())
    assert {j["title"]: bool(j["source_is_board"]) for j in digest} == EXPECTED
