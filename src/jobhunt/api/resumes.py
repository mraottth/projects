import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from jobhunt.config import RESUME_STORAGE_DIR
from jobhunt.core.llm import LLMError
from jobhunt.core.profile import (
    get_current_candidate_profile,
    rebuild_candidate_profile,
    revise_candidate_profile,
    set_candidate_profile,
)
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
    is_archived: bool = Form(False),
):
    """By default, an upload becomes the new "current" resume/cover letter for its kind --
    demoting whatever was current before to Old, so "drag the old one to Old, then upload the new
    one" and "just upload a replacement" both land in the same state without extra steps. Pass
    is_archived=true to add straight into Old instead (e.g. the Old section's own upload form)."""
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
        if not is_archived:
            conn.execute(
                "UPDATE resumes SET is_template = 0, is_archived = 1 WHERE kind = ? AND is_template = 1",
                (kind,),
            )
        cursor = conn.execute(
            """INSERT INTO resumes
               (kind, label, file_name, file_type, stored_path, raw_text, is_archived, is_template)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                kind,
                label,
                file.filename,
                suffix.lstrip("."),
                str(stored_path),
                raw_text,
                int(is_archived),
                int(not is_archived),
            ),
        )
        resume_id = cursor.lastrowid

    return {"id": resume_id, "file_name": file.filename, "text_length": len(raw_text)}


class ResumeUpdateRequest(BaseModel):
    is_archived: bool | None = None
    is_template: bool | None = None


@router.patch("/{resume_id}")
def update_resume(resume_id: int, payload: ResumeUpdateRequest):
    """Moves a document between "current" and "Old". Setting is_template=true promotes it to
    current for its kind (demoting whatever was current before, same as an upload); setting
    is_archived=true archives it into Old (a document can't be both current and archived)."""
    with get_db() as conn:
        row = conn.execute("SELECT kind FROM resumes WHERE id = ?", (resume_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "resume not found")

        if payload.is_template:
            conn.execute(
                "UPDATE resumes SET is_template = 0, is_archived = 1 "
                "WHERE kind = ? AND id != ? AND is_template = 1",
                (row["kind"], resume_id),
            )
            conn.execute(
                "UPDATE resumes SET is_template = 1, is_archived = 0 WHERE id = ?", (resume_id,)
            )
        elif payload.is_archived:
            conn.execute(
                "UPDATE resumes SET is_archived = 1, is_template = 0 WHERE id = ?", (resume_id,)
            )
        elif payload.is_archived is False:
            conn.execute("UPDATE resumes SET is_archived = 0 WHERE id = ?", (resume_id,))

    return {"ok": True}


@router.get("/{resume_id}/download")
def download_resume(resume_id: int):
    """The uploaded file itself (resume or cover letter), under its original file name -- backs the
    Profile page's download links. Generated documents have their own download route
    (api/generate.py)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT file_name, stored_path FROM resumes WHERE id = ?", (resume_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "document not found")
    path = Path(row["stored_path"])
    if not path.exists():
        raise HTTPException(404, "the file is no longer on disk")
    return FileResponse(path, filename=row["file_name"])


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


class ProfileUpdateRequest(BaseModel):
    text: str


@router.put("/profile")
def update_profile(payload: ProfileUpdateRequest):
    if not payload.text.strip():
        raise HTTPException(400, "profile text cannot be empty")
    text = set_candidate_profile(payload.text)
    return {"text": text}


class ProfileReviseRequest(BaseModel):
    instructions: str


@router.post("/profile/revise")
def revise_profile(payload: ProfileReviseRequest):
    try:
        text = revise_candidate_profile(payload.instructions)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"text": text}
