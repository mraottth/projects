from docx import Document
from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import generate, profile
from jobhunt.core.docx_writer import CoverLetterContent
from jobhunt.core.settings import update_settings
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeCreateResponse, FakeParseResponse


def _insert_doc(kind="resume", raw_text="Jane Doe. Analytics lead, 10 years.", is_template=1):
    with get_db() as conn:
        return conn.execute(
            """INSERT INTO resumes (kind, label, file_name, file_type, stored_path, raw_text, is_template)
               VALUES (?, 'x', 'x.txt', 'txt', '/tmp/x.txt', ?, ?)""",
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
    monkeypatch.setattr(generate, "RESUME_STORAGE_DIR", tmp_path / "resumes")
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
