import sqlite3

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import app_insights
from jobhunt.db import connection
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeCreateResponse

# The applications table as created before the statuses were renamed (2026-11-02).
OLD_APPLICATIONS = """CREATE TABLE applications (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
    status TEXT NOT NULL DEFAULT 'saved' CHECK(status IN ('saved', 'applied', 'phone_screen', 'interview',
        'offer', 'rejected', 'withdrawn')),
    applied_at TEXT,
    resume_doc_id INTEGER,
    cover_letter_doc_id INTEGER,
    notes TEXT,
    next_action TEXT,
    next_action_date TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)"""


def _insert_job(title, company="Acme") -> int:
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company) VALUES (1, ?, ?, ?)",
            (f"{title}-{company}", title, company),
        ).lastrowid


def test_old_statuses_are_migrated_once(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path, isolation_level=None)
    schema = connection.SCHEMA_PATH.read_text()
    start = schema.index("CREATE TABLE IF NOT EXISTS applications (")
    end = schema.index(");", start) + 2
    conn.executescript(schema[:start] + OLD_APPLICATIONS + ";" + schema[end:])
    conn.execute(
        "INSERT INTO jobs (id, dedupe_key, title, company) VALUES (1, 'a', 'A', 'X'), (2, 'b', 'B', 'X'),"
        " (3, 'c', 'C', 'X'), (4, 'd', 'D', 'X')"
    )
    history = {
        1: ["saved", "applied"],
        2: ["saved", "applied", "rejected"],
        3: ["saved", "applied", "phone_screen", "rejected"],
        4: ["saved", "interview"],
    }
    for app_id, statuses in history.items():
        conn.execute(
            "INSERT INTO applications (id, job_id, status) VALUES (?, ?, ?)", (app_id, app_id, statuses[-1])
        )
        for status in statuses:
            conn.execute(
                "INSERT INTO application_status_history (application_id, status) VALUES (?, ?)",
                (app_id, status),
            )
    conn.close()

    monkeypatch.setattr(connection, "DB_PATH", db_path)
    connection.init_db()
    connection.init_db()  # a second startup finds nothing left to migrate
    # One copy of the database as it was, taken just before the rewrite, and only that once.
    (kept,) = (tmp_path / "backups").glob("jobhunt-*-before-migration.db")
    old = sqlite3.connect(kept)
    assert old.execute("SELECT status FROM applications WHERE id = 1").fetchone()[0] == "applied"
    old.close()

    with get_db() as conn:
        statuses = dict(conn.execute("SELECT id, status FROM applications").fetchall())
        hist = [
            tuple(r)
            for r in conn.execute("SELECT application_id, status FROM application_status_history ORDER BY id")
        ]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.execute(
            "UPDATE applications SET status = 'rejected_interviewed' WHERE id = 1"
        )  # new CHECK allows it
    assert statuses == {
        1: "waiting",
        2: "rejected_no_interview",
        3: "rejected_interviewed",
        4: "interviewing",
    }
    assert (3, "interviewing") in hist and (3, "rejected_interviewed") in hist
    assert (2, "rejected_no_interview") in hist


def test_old_status_names_are_refused():
    job_id = _insert_job("Analyst")
    with TestClient(app) as client:
        app_id = client.post("/api/applications", json={"job_id": job_id}).json()["id"]
        assert client.patch(f"/api/applications/{app_id}", json={"status": "applied"}).status_code == 400
        assert (
            client.patch(f"/api/applications/{app_id}", json={"status": "rejected_no_interview"}).status_code
            == 200
        )


def test_list_includes_the_latest_score():
    job_id = _insert_job("Analyst")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO job_scores (user_id, job_id, interest_fit, interest_score, candidate_fit,"
            " candidate_fit_score, scored_at) VALUES (1, ?, 'Low', 20, 'Low', 20, '2026-01-01')",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO job_scores (user_id, job_id, interest_fit, interest_score, candidate_fit,"
            " candidate_fit_score, scored_at) VALUES (1, ?, 'High', 90, 'Med', 60, '2026-02-01')",
            (job_id,),
        )
    with TestClient(app) as client:
        client.post("/api/applications", json={"job_id": job_id})
        row = client.get("/api/applications").json()[0]
    assert (row["interest_fit"], row["interest_score"], row["candidate_fit"]) == ("High", 90, "Med")


def test_insights_need_a_sent_application():
    _insert_job("Analyst")
    with TestClient(app) as client:
        client.post("/api/applications", json={"job_id": 1})
        assert client.post("/api/applications/insights").status_code == 400
        assert client.get("/api/applications/insights").json() is None


def test_insights_are_built_and_saved(monkeypatch):
    seen = {}
    fake = FakeAnthropicClient(create_result=FakeCreateResponse("## What's working\n- Data roles"))
    original = fake.messages.create

    def recording_create(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    fake.messages.create = recording_create
    monkeypatch.setattr(app_insights, "get_client", lambda: fake)
    a, b = _insert_job("Data Lead", "Civic"), _insert_job("Sales Rep", "Shop")
    with TestClient(app) as client:
        client.post("/api/applications", json={"job_id": a, "status": "interviewing"})
        client.post("/api/applications", json={"job_id": b, "status": "rejected_no_interview"})
        body = client.post("/api/applications/insights").json()
        assert body["text"].startswith("## What's working")
        assert (body["applications"], body["applied"]) == (2, 2)
        assert client.get("/api/applications/insights").json()["text"] == body["text"]
    prompt = seen["messages"][0]["content"]
    assert "Data Lead at Civic" in prompt and "interviewing" in prompt
    assert "Sales Rep at Shop" in prompt and "rejected without an interview" in prompt
