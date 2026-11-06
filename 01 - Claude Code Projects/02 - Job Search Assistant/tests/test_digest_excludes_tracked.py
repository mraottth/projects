"""Jobs on My Jobs stay out of New & Noteworthy and the daily email (asked 2026-11-04)."""

from jobhunt.core import email_digest, users
from jobhunt.core.digest import add_to_digest, get_digest, today_str
from jobhunt.core.users import acting_as
from jobhunt.db.connection import get_db


def _scored_job(title: str) -> int:
    with get_db() as conn:
        job = conn.execute(
            "INSERT INTO jobs (user_id, dedupe_key, title, company) VALUES (NULL, ?, ?, 'Co')",
            (f"k-{title}", title),
        ).lastrowid
        for uid in (1, 2):
            conn.execute(
                "INSERT INTO job_scores (user_id, job_id, model, interest_fit, interest_score, "
                "candidate_fit, candidate_fit_score, summary_text, rationale_text) "
                "VALUES (?, ?, 'm', 'High', 90, 'High', 90, 's', 'r')",
                (uid, job),
            )
    return job


def _titles() -> set[str]:
    return {j["title"] for j in get_digest(today_str())}


def test_tracked_jobs_leave_new_and_noteworthy_and_the_email():
    users.create_user("b@example.com", "Bea", status="approved")
    tracked, other = _scored_job("Tracked"), _scored_job("Other")
    for uid in (1, 2):
        with acting_as(uid):
            add_to_digest(tracked)
            add_to_digest(other)

    with get_db() as conn:
        app_id = conn.execute(
            "INSERT INTO applications (user_id, job_id, status) VALUES (1, ?, 'saved')", (tracked,)
        ).lastrowid

    assert _titles() == {"Other"}
    assert {j["title"] for j in email_digest.qualifying_jobs(include_emailed=True)} == {"Other"}
    with acting_as(2):  # someone else tracking it doesn't hide it for you
        assert _titles() == {"Tracked", "Other"}

    with get_db() as conn:  # taken off My Jobs: back on the list
        conn.execute("DELETE FROM applications WHERE id = ?", (app_id,))
    assert _titles() == {"Tracked", "Other"}
