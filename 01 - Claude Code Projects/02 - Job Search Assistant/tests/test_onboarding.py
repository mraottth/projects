from docx import Document
from fastapi.testclient import TestClient

from jobhunt.core import users
from jobhunt.api.main import app
from jobhunt.core import generate, profile
from jobhunt.core.docx_writer import CoverLetterContent
from jobhunt.core.settings import update_settings
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeCreateResponse, FakeParseResponse


def _insert_doc(kind="resume", raw_text="Jane Doe. Analytics lead, 10 years.", is_template=1):
    with get_db() as conn:
        return conn.execute(
            """INSERT INTO resumes (user_id, kind, label, file_name, file_type, stored_path, raw_text,
                is_template)
               VALUES (1, ?, 'x', 'x.txt', 'txt', '/tmp/x.txt', ?, ?)""",
            (kind, raw_text, is_template),
        ).lastrowid


def _status(client):
    return client.get("/api/onboarding/status").json()


def test_status_needs_onboarding_on_empty_db():
    with TestClient(app) as client:
        status = _status(client)
    assert status["needs_onboarding"] is True
    assert status["step"] == "resume"
    assert status["has_resume"] is False
    assert status["has_profile"] is False


def test_status_skips_onboarding_once_any_doc_exists():
    _insert_doc(kind="cover_letter")
    with TestClient(app) as client:
        status = _status(client)
    assert status["needs_onboarding"] is False
    assert status["has_cover_letter"] is True
    assert status["has_resume"] is False


def test_status_skips_onboarding_once_completed_even_with_no_docs():
    with TestClient(app) as client:
        update_settings({"onboarding_completed": True})
        assert _status(client)["needs_onboarding"] is False


def _fake_cover_letter_client(paragraphs):
    return FakeAnthropicClient(parse_result=FakeParseResponse(CoverLetterContent(paragraphs=paragraphs)))


def test_starter_cover_letter_becomes_current_and_demotes_previous(monkeypatch, tmp_path):
    _insert_doc(kind="resume")
    previous = _insert_doc(kind="cover_letter")
    monkeypatch.setattr(users, "RESUME_STORAGE_DIR", tmp_path / "resumes")
    paragraphs = ["Dear Hiring Manager,", "Body one.", "Body two.", "Sincerely,", "Jane Doe"]
    monkeypatch.setattr(generate, "get_client", lambda: _fake_cover_letter_client(paragraphs))

    with TestClient(app) as client:
        resp = client.post("/api/onboarding/generate-cover-letter")
    assert resp.status_code == 200
    body = resp.json()
    assert body["content_text"] == "\n\n".join(paragraphs)

    with get_db() as conn:
        new_row = dict(conn.execute("SELECT * FROM resumes WHERE id = ?", (body["id"],)).fetchone())
        old_row = dict(conn.execute("SELECT * FROM resumes WHERE id = ?", (previous,)).fetchone())
    assert new_row["kind"] == "cover_letter"
    assert new_row["is_template"] == 1
    assert new_row["file_type"] == "docx"
    assert old_row["is_template"] == 0
    assert old_row["is_archived"] == 1
    # A real .docx on disk, one paragraph per element -- usable as the positional template that
    # later per-job cover-letter generation fills in.
    saved = Document(new_row["stored_path"])
    assert [p.text for p in saved.paragraphs] == paragraphs


def test_starter_cover_letter_requires_a_resume():
    with TestClient(app) as client:
        resp = client.post("/api/onboarding/generate-cover-letter")
    assert resp.status_code == 400


def test_revise_profile_inserts_a_new_row(monkeypatch):
    profile.set_candidate_profile("Original profile.")
    client = FakeAnthropicClient(create_result=FakeCreateResponse("Revised profile."))
    captured = {}
    orig_create = client.messages.create

    def spy_create(**kwargs):
        captured["content"] = kwargs["messages"][0]["content"]
        return orig_create(**kwargs)

    client.messages.create = spy_create
    monkeypatch.setattr(profile, "get_client", lambda: client)

    with TestClient(app) as http:
        resp = http.post("/api/resumes/profile/revise", json={"instructions": "Emphasize policy work"})
    assert resp.status_code == 200
    assert resp.json()["text"] == "Revised profile."
    assert profile.get_current_candidate_profile() == "Revised profile."
    assert "Original profile." in captured["content"]
    assert "Emphasize policy work" in captured["content"]


def test_revise_profile_rejects_missing_profile_or_blank_instructions():
    with TestClient(app) as http:
        assert http.post("/api/resumes/profile/revise", json={"instructions": "x"}).status_code == 400
        profile.set_candidate_profile("Original profile.")
        assert http.post("/api/resumes/profile/revise", json={"instructions": "  "}).status_code == 400


def _bad_request(message):
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(400, request=request)
    return anthropic.BadRequestError(message, response=response, body=None)


class _RaisingClient:
    def __init__(self, exc):
        self.messages = self
        self._exc = exc

    def parse(self, **kwargs):
        raise self._exc


def test_out_of_credits_error_says_what_to_do(monkeypatch):
    # Caught live: a claude.ai plan doesn't fund the API, and the raw 400 read like an app bug.
    _insert_doc(kind="resume")
    exc = _bad_request(
        "Error code: 400 - Your credit balance is too low to access the Anthropic API. "
        "Please go to Plans & Billing to upgrade or purchase credits."
    )
    monkeypatch.setattr(generate, "get_client", lambda: _RaisingClient(exc))

    with TestClient(app) as client:
        resp = client.post("/api/onboarding/generate-cover-letter")
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "out of credits" in detail
    assert "platform.claude.com" in detail


def test_other_bad_requests_keep_the_raw_error(monkeypatch):
    _insert_doc(kind="resume")
    monkeypatch.setattr(generate, "get_client", lambda: _RaisingClient(_bad_request("max_tokens too large")))

    with TestClient(app) as client:
        resp = client.post("/api/onboarding/generate-cover-letter")
    assert resp.status_code == 502
    assert "max_tokens too large" in resp.json()["detail"]
    assert "out of credits" not in resp.json()["detail"]


def test_the_tour_flag_round_trips_per_person():
    # Onboarding sets tour_pending; the guided tour (frontend Tour.tsx) clears it when it ends.
    from fastapi.testclient import TestClient

    from jobhunt.api.main import app

    with TestClient(app) as client:
        assert client.get("/api/settings").json().get("tour_pending") is None  # set up before the tour
        client.put("/api/settings", json={"onboarding_completed": True, "tour_pending": True})
        assert client.get("/api/settings").json()["tour_pending"] is True
        client.put("/api/settings", json={"tour_pending": False})
        assert client.get("/api/settings").json()["tour_pending"] is False
