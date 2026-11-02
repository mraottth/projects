from datetime import date

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import email_digest
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run

SENT = []


class FakeSMTP:
    def __init__(self, host, port, timeout=None):
        assert (host, port) == ("smtp.gmail.com", 465)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        assert (user, password) == ("me@example.com", "app-pass")

    def send_message(self, message):
        SENT.append(message)


def _configure(monkeypatch):
    SENT.clear()
    monkeypatch.setenv("EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-pass")
    monkeypatch.setenv("APP_URL", "https://jobhunt.example.run.app/")
    monkeypatch.setattr(email_digest.smtplib, "SMTP_SSL", FakeSMTP)


def _digest_job(title, interest, fit, interest_score=80, fit_score=60, **job):
    with get_db() as conn:
        job_id = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company, url, salary_min, salary_max, salary_period) "
            "VALUES (?, ?, 'Acme', ?, ?, ?, ?)",
            (
                title,
                title,
                f"https://acme.example/jobs/{title}",
                job.get("salary_min"),
                job.get("salary_max"),
                job.get("salary_period"),
            ),
        ).lastrowid
        conn.execute(
            "INSERT INTO job_scores (job_id, model, interest_fit, interest_score, candidate_fit, "
            "candidate_fit_score, summary_text, rationale_text) VALUES (?, 'x', ?, ?, ?, ?, ?, 'r')",
            (job_id, interest, interest_score, fit, fit_score, f"About {title}."),
        )
        conn.execute(
            "INSERT INTO digest_entries (digest_date, job_id, rank) VALUES (?, ?, 0)",
            (date.today().isoformat(), job_id),
        )
    return job_id


def test_not_configured_sends_nothing(monkeypatch):
    monkeypatch.delenv("EMAIL_ADDRESS", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    _digest_job("Data Director", "High", "High")
    assert email_digest.send_new_jobs_email() == {"sent": False, "reason": "not configured"}


def test_sends_new_high_interest_matches_once(monkeypatch):
    _configure(monkeypatch)
    _digest_job(
        "Data Director", "High", "Med", 90, 60, salary_min=160000, salary_max=210000, salary_period="year"
    )
    best = _digest_job("Head of Analytics", "High", "High", 95, 90)
    _digest_job("Low Fit Role", "High", "Low")
    _digest_job("Med Interest Role", "Med", "High")

    result = email_digest.send_new_jobs_email()
    assert result == {"sent": True, "count": 2, "to": "me@example.com"}
    message = SENT[0]
    assert message["Subject"] == "2 new job matches: Head of Analytics at Acme, +1 more"
    body = message.get_body(("plain",)).get_content()
    assert body.index("Head of Analytics") < body.index("Data Director")  # best first
    assert "$160k-210k" in body and "/yr" not in body
    assert f"https://jobhunt.example.run.app/jobs/{best}" in body
    assert "Low Fit Role" not in body and "Med Interest Role" not in body
    assert "View posting" in message.get_body(("html",)).get_content()

    # Already emailed: nothing new the second time.
    assert email_digest.send_new_jobs_email() == {"sent": False, "reason": "nothing new"}
    assert len(SENT) == 1


def test_empty_day_sends_nothing(monkeypatch):
    _configure(monkeypatch)
    _digest_job("Med Interest Role", "Med", "High")
    assert email_digest.send_new_jobs_email() == {"sent": False, "reason": "nothing new"}
    assert SENT == []


def test_email_failure_does_not_fail_the_daily_run(monkeypatch):
    def boom():
        raise OSError("smtp down")

    monkeypatch.setattr(daily_run, "send_new_jobs_email", boom)
    summary = daily_run.run_daily()
    assert summary["email"] == {"sent": False, "reason": "error: smtp down"}


def test_test_email_route(monkeypatch):
    monkeypatch.delenv("EMAIL_ADDRESS", raising=False)
    with TestClient(app) as client:
        assert client.get("/api/settings/email-status").json() == {"configured": False, "to": None}
        assert client.post("/api/settings/test-email").status_code == 400

    _configure(monkeypatch)
    _digest_job("Data Director", "High", "High")
    with TestClient(app) as client:
        assert client.get("/api/settings/email-status").json() == {"configured": True, "to": "me@example.com"}
        resp = client.post("/api/settings/test-email")
    assert resp.json()["count"] == 1
    assert SENT[0]["Subject"].startswith("[Test] ")
    # A test send doesn't use up the real daily email.
    assert email_digest.send_new_jobs_email()["sent"] is True
