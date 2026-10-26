import pytest

from jobhunt.core.resume_parse import extract_text
from jobhunt.db.connection import get_db


def test_extract_text_md(tmp_path):
    path = tmp_path / "resume.md"
    path.write_text("# Jane Doe\n\nSenior Engineer at Acme.")
    assert "Senior Engineer" in extract_text(path)


def test_extract_text_rejects_unsupported(tmp_path):
    path = tmp_path / "resume.xyz"
    path.write_text("nope")
    with pytest.raises(ValueError):
        extract_text(path)


def test_resume_row_roundtrip():
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO resumes (kind, label, file_name, file_type, stored_path, raw_text)
               VALUES (?, ?, ?, ?, ?, ?)""",
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
