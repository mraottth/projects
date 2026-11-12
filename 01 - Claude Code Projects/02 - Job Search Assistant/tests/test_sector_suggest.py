"""Sector suggestions from the resume (core/sector_suggest.py, asked 2026-11-12): a few names for
"Desired sectors"' hint, never saved to the person's sectors, and not asked for again until the
current resume changes."""

from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import sector_suggest
from jobhunt.core.sector_suggest import Suggestions
from jobhunt.core.settings import get_settings, update_settings
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeParseResponse


def _resume(text="Data director at a voter-file analytics firm."):
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO resumes (user_id, kind, file_name, file_type, stored_path, raw_text, is_template) "
            "VALUES (1, 'resume', 'r.docx', 'docx', '/x', ?, 1)",
            (text,),
        ).lastrowid


def test_suggestions_come_from_the_resume_once_and_are_never_saved_as_sectors(monkeypatch):
    _resume()
    update_settings({"prefilter_sectors": ["Civic tech"]})
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return FakeParseResponse(
            Suggestions(sectors=["Political data and campaigns", "civic tech", "Nonprofit"])
        )

    client = FakeAnthropicClient()
    client.messages.parse = parse
    monkeypatch.setattr(sector_suggest, "get_client", lambda: client)
    with TestClient(app) as http:
        first = http.post("/api/onboarding/sector-suggestions").json()
        again = http.post("/api/onboarding/sector-suggestions").json()
    assert first == again == {"sectors": ["Political data and campaigns", "Nonprofit"]}  # minus listed
    assert len(calls) == 1 and calls[0]["model"] == "claude-haiku-5-5"
    assert "Already listed: civic tech" in calls[0]["messages"][0]["content"]
    assert get_settings()["prefilter_sectors"] == ["Civic tech"]  # nothing added for them

    with get_db() as conn:  # a new current resume demotes the old one, as an upload does
        conn.execute("UPDATE resumes SET is_template = 0, is_archived = 1")
    _resume("A new resume.")  # a new current resume: asked again
    with TestClient(app) as http:
        http.post("/api/onboarding/sector-suggestions")
    assert len(calls) == 2


def test_no_resume_means_no_call(monkeypatch):
    def no_client():
        raise AssertionError("no call without a resume")

    monkeypatch.setattr(sector_suggest, "get_client", no_client)
    with TestClient(app) as http:
        assert http.post("/api/onboarding/sector-suggestions").json() == {"sectors": []}
