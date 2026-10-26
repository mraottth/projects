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


def test_generate_resume_uses_docx_template_from_web_upload_with_no_source_path(tmp_path, monkeypatch):
    # Regression: _usable_template_path() used to check only source_path, which is populated for
    # bulk-ingested resumes but is NULL for anything uploaded through the web UI (stored_path is
    # the one column always set, regardless of upload path) -- a web-uploaded "current" resume
    # silently fell back to plain-text generation even with a perfectly good .docx on disk.
    job_id = _insert_job()
    docx_path = tmp_path / "web_uploaded.docx"
    _make_resume_docx(docx_path)
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO resumes (kind, file_name, file_type, stored_path, raw_text, is_template)
               VALUES ('resume', ?, 'docx', ?, 'reference text', 1)""",
            (docx_path.name, str(docx_path)),
        )
        resume_id = cursor.lastrowid

    new_content = ResumeContent(name="Web Upload Test", contact_line="x@example.com", sections=[])
    monkeypatch.setattr(
        generate, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(new_content))
    )
    monkeypatch.setattr(generate, "_output_dir", lambda: tmp_path / "out")

    with TestClient(app) as client:
        resp = client.post("/api/generate/resume", json={"job_id": job_id, "resume_id": resume_id})
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_path"] is not None, "should have used the docx template, not fallen back to text"


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


def test_generate_resume_returns_clean_502_on_truncated_llm_response(tmp_path, monkeypatch):
    # Regression: a truncated/malformed structured-output response from Claude raises
    # pydantic.ValidationError, which subclasses ValueError -- this used to be mis-caught by the
    # "no base resume uploaded" handler, returning a confusing 400 with the wrong message.
    job_id = _insert_job()
    _insert_resume_with_docx(tmp_path)

    class _BadMessages:
        def parse(self, **kwargs):
            ResumeContent.model_validate({})  # raises a real pydantic.ValidationError

    class _BadClient:
        messages = _BadMessages()

    monkeypatch.setattr(generate, "get_client", lambda: _BadClient())

    with TestClient(app) as client:
        resp = client.post("/api/generate/resume", json={"job_id": job_id})
        assert resp.status_code == 502
        assert "couldn't be parsed" in resp.json()["detail"]
        assert "upload one" not in resp.json()["detail"]


def test_generate_cover_letter_returns_clean_502_on_truncated_llm_response(tmp_path, monkeypatch):
    # Regression: this used to be an unhandled exception (no except clause caught
    # ValidationError), surfacing as a raw 500 Internal Server Error with no useful detail.
    job_id = _insert_job()
    _insert_resume_with_docx(tmp_path, kind="cover_letter", make=_make_cover_letter_docx)

    class _BadMessages:
        def parse(self, **kwargs):
            CoverLetterContent.model_validate({})

    class _BadClient:
        messages = _BadMessages()

    monkeypatch.setattr(generate, "get_client", lambda: _BadClient())

    with TestClient(app) as client:
        resp = client.post("/api/generate/cover-letter", json={"job_id": job_id})
        assert resp.status_code == 502
        assert "couldn't be parsed" in resp.json()["detail"]


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


def test_generate_resume_with_stale_application_id_still_saves_the_document(monkeypatch):
    # Regression: generated_documents.application_id has a foreign-key constraint. If the frontend
    # sends an application_id for a job that's since been removed from My Jobs (DELETE
    # /api/applications/{id}), the INSERT used to raise an unhandled sqlite3.IntegrityError --
    # the expensive, already-generated content would be thrown away behind a raw 500.
    job_id = _insert_job()
    _insert_resume()
    monkeypatch.setattr(
        generate, "get_client", lambda: FakeAnthropicClient(create_result=FakeCreateResponse("tailored"))
    )

    with TestClient(app) as client:
        nonexistent_application_id = 999999
        resp = client.post(
            "/api/generate/resume",
            json={"job_id": job_id, "application_id": nonexistent_application_id},
        )
        assert resp.status_code == 200
        assert resp.json()["content_text"] == "tailored"

    with get_db() as conn:
        row = conn.execute(
            "SELECT application_id FROM generated_documents WHERE id = ?", (resp.json()["id"],)
        ).fetchone()
    assert row["application_id"] is None


def test_all_generated_documents_lists_across_jobs_with_job_info(monkeypatch):
    job_a = _insert_job()
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company, description_text) "
            "VALUES ('k2', 'Designer', 'Beta', 'Design things')"
        )
        job_b = cursor.lastrowid

    _insert_resume()
    monkeypatch.setattr(
        generate, "get_client", lambda: FakeAnthropicClient(create_result=FakeCreateResponse("tailored"))
    )

    with TestClient(app) as client:
        client.post("/api/generate/resume", json={"job_id": job_a})
        client.post("/api/generate/cover-letter", json={"job_id": job_b})

        docs = client.get("/api/generate/documents").json()
        assert len(docs) == 2
        by_company = {d["job_company"]: d for d in docs}
        assert by_company["Acme"]["job_title"] == "Engineer"
        assert by_company["Acme"]["kind"] == "resume"
        assert by_company["Beta"]["job_title"] == "Designer"
        assert by_company["Beta"]["kind"] == "cover_letter"


def test_download_text_only_document_returns_markdown(monkeypatch):
    job_id = _insert_job()
    _insert_resume()
    fake_response = FakeCreateResponse("tailored content")
    monkeypatch.setattr(generate, "get_client", lambda: FakeAnthropicClient(create_result=fake_response))

    with TestClient(app) as client:
        doc_id = client.post("/api/generate/resume", json={"job_id": job_id}).json()["id"]
        resp = client.get(f"/api/generate/documents/{doc_id}/download")
        assert resp.status_code == 200
        assert resp.text == "tailored content"
        assert resp.headers["content-type"].startswith("text/markdown")
        assert "attachment" in resp.headers["content-disposition"]
        assert ".md" in resp.headers["content-disposition"]


def test_download_docx_document_returns_the_file(tmp_path, monkeypatch):
    job_id = _insert_job()
    resume_id = _insert_resume_with_docx(tmp_path)

    new_content = ResumeContent(name="Jane Tester", contact_line="jane@example.com", sections=[])
    monkeypatch.setattr(
        generate, "get_client", lambda: FakeAnthropicClient(parse_result=FakeParseResponse(new_content))
    )
    monkeypatch.setattr(generate, "_output_dir", lambda: tmp_path / "out")

    with TestClient(app) as client:
        doc_id = client.post(
            "/api/generate/resume", json={"job_id": job_id, "resume_id": resume_id}
        ).json()["id"]
        resp = client.get(f"/api/generate/documents/{doc_id}/download")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert resp.content[:2] == b"PK"  # docx files are zip archives
        assert ".docx" in resp.headers["content-disposition"]


def test_download_nonexistent_document_404s():
    with TestClient(app) as client:
        resp = client.get("/api/generate/documents/999999/download")
        assert resp.status_code == 404
