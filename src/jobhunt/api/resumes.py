import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from jobhunt.config import RESUME_STORAGE_DIR
from jobhunt.core.llm import LLMError
from jobhunt.core.profile import get_current_candidate_profile, rebuild_candidate_profile
from jobhunt.core.resume_parse import extract_text
from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/resumes", tags=["resumes"])


@router.get("")
def list_resumes():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, kind, label, file_name, file_type, uploaded_at, is_archived, is_template, "
            "length(raw_text) AS text_length FROM resumes "
            "ORDER BY is_template DESC, is_archived ASC, "
            "COALESCE(source_modified_at, uploaded_at) DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@router.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    kind: str = Form(...),
    label: str | None = Form(None),
):
    if kind not in ("resume", "cover_letter"):
        raise HTTPException(400, "kind must be 'resume' or 'cover_letter'")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".docx", ".md", ".txt"):
        raise HTTPException(400, f"unsupported file type: {suffix}")

    RESUME_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{suffix}"
    stored_path = RESUME_STORAGE_DIR / stored_name
    contents = await file.read()
    stored_path.write_bytes(contents)

    try:
        raw_text = extract_text(stored_path)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(400, f"could not extract text: {exc}") from exc

    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO resumes (kind, label, file_name, file_type, stored_path, raw_text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (kind, label, file.filename, suffix.lstrip("."), str(stored_path), raw_text),
        )
        resume_id = cursor.lastrowid

    return {"id": resume_id, "file_name": file.filename, "text_length": len(raw_text)}


@router.delete("/{resume_id}")
def delete_resume(resume_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT stored_path FROM resumes WHERE id = ?", (resume_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "resume not found")
        conn.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))
    Path(row["stored_path"]).unlink(missing_ok=True)
    return {"ok": True}


@router.get("/profile")
def get_profile():
    text = get_current_candidate_profile()
    if text is None:
        raise HTTPException(404, "no candidate profile yet — upload resumes and rebuild")
    return {"text": text}


@router.post("/profile/rebuild")
def rebuild_profile():
    try:
        text = rebuild_candidate_profile()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"text": text}
