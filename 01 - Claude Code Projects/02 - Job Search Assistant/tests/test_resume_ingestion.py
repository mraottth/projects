import pytest
from docx import Document

from jobhunt.core.resume_parse import extract_text
from jobhunt.db.connection import get_db


def test_extract_text_md(tmp_path):
    path = tmp_path / "resume.md"
    path.write_text("# Jane Doe\n\nSenior Engineer at Acme.")
    assert "Senior Engineer" in extract_text(path)


def test_extract_text_docx_reads_table_based_layout(tmp_path):
    # Regression: several real resumes use a table-based two-column layout (dates/company in one
    # cell, bullets in another) -- `.paragraphs` alone misses table content entirely, confirmed
    # live against a real file that extracted 111 chars instead of ~5000.
    doc = Document()
    doc.add_paragraph("JANE TESTER")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "ACME CORP\nJan 2020 - Present"
    table.rows[0].cells[1].text = "Shipped a thing that mattered to the business."
    doc.add_paragraph("EDUCATION")
    path = tmp_path / "resume.docx"
    doc.save(str(path))

    text = extract_text(path)
    assert "JANE TESTER" in text
    assert "ACME CORP" in text
    assert "Shipped a thing that mattered" in text
    assert "EDUCATION" in text


def test_extract_text_rejects_unsupported(tmp_path):
    path = tmp_path / "resume.xyz"
    path.write_text("nope")
    with pytest.raises(ValueError):
        extract_text(path)


def test_resume_row_roundtrip():
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO resumes (user_id, kind, label, file_name, file_type, stored_path, raw_text)
               VALUES (1, ?, ?, ?, ?, ?, ?)""",
            ("resume", "2024 resume", "resume.md", "md", "/tmp/fake.md", "some text"),
        )
        resume_id = cursor.lastrowid

    with get_db() as conn:
        row = conn.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,)).fetchone()
    assert row["kind"] == "resume"
    assert row["raw_text"] == "some text"


def test_settings_defaults_seeded():
    with get_db() as conn:
        row = conn.execute(
            "SELECT value_json FROM settings WHERE key = 'daily_llm_cap'"
        ).fetchone()
    assert row is not None
    assert row["value_json"] == "50"
