from docx import Document
from fastapi.testclient import TestClient

from jobhunt.api.main import app
from jobhunt.core import generate
from jobhunt.core.docx_writer import CoverLetterContent, ResumeContent, ResumeLine, ResumeSection
from jobhunt.db.connection import get_db
from tests.fakes import FakeAnthropicClient, FakeCreateResponse, FakeParseResponse


def _insert_job() -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company, description_text) "
            "VALUES ('k1', 'Engineer', 'Acme', 'Build things')"
        )
        return cursor.lastrowid


def _insert_resume(text="Experienced engineer. Built many things.") -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO resumes (kind, file_name, file_type, stored_path, raw_text) "
            "VALUES ('resume', 'r.md', 'md', '/tmp/r.md', ?)",
            (text,),
        )
        return cursor.lastrowid


def _make_resume_docx(path):
    doc = Document()
    doc.add_paragraph("OLD NAME", style="Title")
    doc.add_paragraph("old@example.com", style="Normal")
    doc.add_paragraph("WORK EXPERIENCE", style="Heading 1")
    doc.add_paragraph("", style="Normal")
    p = doc.add_paragraph(style="Normal")
    r1 = p.add_run("OLD CORP | Old Title")
    r1.bold = True
    r2 = p.add_run(" | 2020 - 2021")
    r2.bold = False
    doc.add_paragraph("Did an old thing.", style="List Paragraph")
    doc.save(str(path))


def _make_cover_letter_docx(path):
    doc = Document()
    doc.add_paragraph("To Whom It May Concern:", style="Normal")
    doc.add_paragraph("", style="Normal")
    doc.add_paragraph("Old body paragraph.", style="Normal")
    doc.add_paragraph("", style="Normal")
    doc.add_paragraph("Sincerely, Old Person", style="Normal")
    doc.save(str(path))


def _insert_resume_with_docx(tmp_path, kind="resume", is_template=0, make=_make_resume_docx):
    docx_path = tmp_path / f"{kind}_{is_template}.docx"
    make(docx_path)
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO resumes
               (kind, file_name, file_type, stored_path, raw_text, source_path, is_template)
               VALUES (?, ?, 'docx', ?, 'reference text', ?, ?)""",
            (kind, docx_path.name, str(docx_path), str(docx_path), is_template),
        )
        return cursor.lastrowid


def test_generate_resume_requires_a_base_resume():
    job_id = _insert_job()
    with TestClient(app) as client:
        resp = client.post("/api/generate/resume", json={"job_id": job_id})
        assert resp.status_code == 400
        assert "upload one" in resp.json()["detail"]


def test_generate_resume_and_cover_letter_flow(monkeypatch):
    job_id = _insert_job()
    _insert_resume()
    monkeypatch.setattr(
        generate,
        "get_client",
        lambda: FakeAnthropicClient(create_result=FakeCreateResponse("# Tailored Resume\n...")),
    )

    with TestClient(app) as client:
        app_resp = client.post("/api/applications", json={"job_id": job_id})
        application_id = app_resp.json()["id"]

        resp = client.post(
            "/api/generate/resume", json={"job_id": job_id, "application_id": application_id}
        )
        assert resp.status_code == 200
        assert "Tailored Resume" in resp.json()["content_text"]

        monkeypatch.setattr(
            generate,
            "get_client",
            lambda: FakeAnthropicClient(create_result=FakeCreateResponse("Dear Hiring Manager...")),
        )
        resp = client.post(
            "/api/generate/cover-letter", json={"job_id": job_id, "application_id": application_id}
        )
        assert resp.status_code == 200
        assert "Dear Hiring Manager" in resp.json()["content_text"]

        app_after = client.get(f"/api/applications/{application_id}").json()
        assert app_after["resume_doc_id"] is not None
        assert app_after["cover_letter_doc_id"] is not None

        history = client.get(f"/api/generate/history?job_id={job_id}").json()
        assert len(history) == 2


def test_generate_resume_with_explicit_resume_id(monkeypatch):
    job_id = _insert_job()
    _insert_resume("Old resume")
    newer_id = _insert_resume("Newer resume")
    monkeypatch.setattr(
        generate, "get_client", lambda: FakeAnthropicClient(create_result=FakeCreateResponse("tailored"))
    )

    with TestClient(app) as client:
        resp = client.post("/api/generate/resume", json={"job_id": job_id, "resume_id": newer_id})
        assert resp.status_code == 200
        with get_db() as conn:
            row = conn.execute(
                "SELECT based_on_resume_id FROM generated_documents WHERE id = ?",
                (resp.json()["id"],),
            ).fetchone()
        assert row["based_on_resume_id"] == newer_id


def test_generate_resume_uses_docx_template_when_available(tmp_path, monkeypatch):
    job_id = _insert_job()
    resume_id = _insert_resume_with_docx(tmp_path)

    new_content = ResumeContent(
        name="JANE TESTER",
        contact_line="jane@example.com",
        sections=[
            ResumeSection(
                heading="WORK EXPERIENCE",
                lines=[
                    ResumeLine(kind="normal", bold_part="ACME | Senior Engineer", text=" | 2024 - Present"),
                    ResumeLine(kind="bullet", text="Shipped a tailored thing."),
                ],
            )
        ],
    )
    monkeypatch.setattr(
        generate, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(new_content))
    )
    monkeypatch.setattr(generate, "_output_dir", lambda: tmp_path / "out")

    with TestClient(app) as client:
        resp = client.post("/api/generate/resume", json={"job_id": job_id, "resume_id": resume_id})
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_path"] is not None
        assert body["based_on_resume_id"] == resume_id
        assert "Shipped a tailored thing." in body["content_text"]

    doc = Document(body["file_path"])
    assert "JANE TESTER" in [p.text for p in doc.paragraphs]
    assert "OLD NAME" not in " ".join(p.text for p in doc.paragraphs)


def test_generate_resume_falls_back_to_plain_text_when_only_pdf_on_file(monkeypatch):
    # A resume with no usable .docx (e.g. only ever uploaded as a PDF) still works, just without
    # a saved file.
    job_id = _insert_job()
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO resumes (kind, file_name, file_type, stored_path, raw_text) "
            "VALUES ('resume', 'r.pdf', 'pdf', '/tmp/r.pdf', 'pdf resume text')"
        )
        resume_id = cursor.lastrowid

    fake_response = FakeCreateResponse("plain tailored")
    monkeypatch.setattr(generate, "get_client", lambda: FakeAnthropicClient(create_result=fake_response))

    with TestClient(app) as client:
        resp = client.post("/api/generate/resume", json={"job_id": job_id, "resume_id": resume_id})
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_path"] is None
        assert body["content_text"] == "plain tailored"


def test_generate_cover_letter_uses_docx_template_when_available(tmp_path, monkeypatch):
    job_id = _insert_job()
    _insert_resume_with_docx(tmp_path, kind="cover_letter", make=_make_cover_letter_docx)

    new_content = CoverLetterContent(
        paragraphs=["To Whom It May Concern:", "New tailored body.", "Sincerely, Jane"]
    )
    monkeypatch.setattr(
        generate, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(new_content))
    )
    monkeypatch.setattr(generate, "_output_dir", lambda: tmp_path / "out")

    with TestClient(app) as client:
        resp = client.post("/api/generate/cover-letter", json={"job_id": job_id})
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_path"] is not None
        assert "New tailored body." in body["content_text"]

    doc = Document(body["file_path"])
    full_text = " ".join(p.text for p in doc.paragraphs)
    assert "New tailored body." in full_text
    assert "Old body paragraph." not in full_text


def test_generate_resume_prefers_is_template_flag_over_recency(tmp_path, monkeypatch):
    job_id = _insert_job()
    _insert_resume_with_docx(tmp_path, kind="resume", is_template=0)
    template_id = _insert_resume_with_docx(tmp_path, kind="resume", is_template=1)

    new_content = ResumeContent(name="X", contact_line="x", sections=[])
    monkeypatch.setattr(
        generate, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(new_content))
    )
    monkeypatch.setattr(generate, "_output_dir", lambda: tmp_path / "out")

    with TestClient(app) as client:
        resp = client.post("/api/generate/resume", json={"job_id": job_id})
        assert resp.json()["based_on_resume_id"] == template_id
