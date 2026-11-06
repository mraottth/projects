from docx import Document

from jobhunt.cli.ingest_resumes import _discover_files, _is_archived, _prefer_docx_over_pdf, ingest_folder
from jobhunt.core.settings import get_setting
from jobhunt.db.connection import get_db


def _write_docx(path, text: str) -> None:
    doc = Document()
    doc.add_paragraph(text)
    doc.save(str(path))


def test_is_archived_requires_whole_word_match():
    assert _is_archived(("Old Resumes",)) is True
    assert _is_archived(("old",)) is True
    assert _is_archived(("Cold Outreach",)) is False  # substring trap: "old" inside "Cold"
    assert _is_archived(("Current",)) is False


def test_discover_files_skips_lock_files_and_ds_store_and_generated_folder(tmp_path):
    (tmp_path / "resume.docx").write_text("x")
    (tmp_path / "~$resume.docx").write_text("x")
    (tmp_path / ".DS_Store").write_text("x")
    (tmp_path / "notes.txt.bak").write_text("x")  # unsupported suffix
    generated = tmp_path / "AI Generated"
    generated.mkdir()
    (generated / "tailored.docx").write_text("x")

    found = {p.name for p in _discover_files(tmp_path)}
    assert found == {"resume.docx"}


def test_prefer_docx_over_pdf_same_stem(tmp_path):
    (tmp_path / "resume.docx").write_text("x")
    (tmp_path / "resume.pdf").write_text("x")
    (tmp_path / "other.pdf").write_text("x")  # no docx sibling -- kept

    candidates = [tmp_path / "resume.docx", tmp_path / "resume.pdf", tmp_path / "other.pdf"]
    selected = {p.name for p in _prefer_docx_over_pdf(candidates)}
    assert selected == {"resume.docx", "other.pdf"}


def test_ingest_folder_classifies_kind_archived_template(tmp_path):
    root = tmp_path / "Resumes"
    root.mkdir()
    _write_docx(root / "Matt_Resume_2026.docx", "Current resume content, quite long indeed yes.")
    _write_docx(
        root / "cover letter template.docx",
        "Dear Sir or Madam, this is my template cover letter body text here.",
    )
    old_dir = root / "Old Resumes"
    old_dir.mkdir()
    _write_docx(old_dir / "ancient_resume.docx", "Ancient resume content from long ago, quite old.")

    imported = ingest_folder(root)
    assert len(imported) == 3

    with get_db() as conn:
        rows = {r["file_name"]: dict(r) for r in conn.execute("SELECT * FROM resumes")}

    assert rows["Matt_Resume_2026.docx"]["kind"] == "resume"
    assert rows["Matt_Resume_2026.docx"]["is_archived"] == 0
    assert rows["Matt_Resume_2026.docx"]["is_template"] == 0

    assert rows["cover letter template.docx"]["kind"] == "cover_letter"
    assert rows["cover letter template.docx"]["is_template"] == 1

    assert rows["ancient_resume.docx"]["is_archived"] == 1

    assert get_setting("resume_library_path") == str(root.resolve())


def test_ingest_folder_dedupes_by_content_across_reruns(tmp_path):
    root = tmp_path / "Resumes"
    root.mkdir()
    text = "Some resume content that is long enough to matter here."
    _write_docx(root / "resume.docx", text)
    _write_docx(root / "resume copy.docx", text)

    first_run = ingest_folder(root)
    assert len(first_run) == 1  # the "copy" duplicate is skipped by content hash

    second_run = ingest_folder(root)
    assert second_run == []  # nothing new on a rerun

    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM resumes").fetchone()["n"]
    assert count == 1


def test_ingest_folder_skips_unextractable_and_empty_files(tmp_path):
    root = tmp_path / "Resumes"
    root.mkdir()
    _write_docx(root / "empty.docx", "")
    (root / "bad.pdf").write_bytes(b"not a real pdf")

    imported = ingest_folder(root)
    assert imported == []
