from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.core.generate import generate_cover_letter, generate_resume
from jobhunt.core.llm import LLMError
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/generate", tags=["generate"])


class GenerateRequest(BaseModel):
    job_id: int
    application_id: int | None = None
    resume_id: int | None = None  # resume to tailor from; defaults to the most recently uploaded


def _get_job(job_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "job not found")
    return dict(row)


def _link_to_application(application_id: int | None, column: str, doc_id: int) -> None:
    if application_id is None:
        return
    with get_db() as conn:
        conn.execute(f"UPDATE applications SET {column} = ? WHERE id = ?", (doc_id, application_id))


@router.post("/resume")
def generate_resume_route(payload: GenerateRequest):
    job = _get_job(payload.job_id)

    with get_db() as conn:
        if payload.resume_id is not None:
            resume = conn.execute(
                "SELECT * FROM resumes WHERE id = ? AND kind = 'resume'", (payload.resume_id,)
            ).fetchone()
        else:
            resume = conn.execute(
                "SELECT * FROM resumes WHERE kind = 'resume' ORDER BY uploaded_at DESC LIMIT 1"
            ).fetchone()
    if resume is None:
        raise HTTPException(400, "no base resume available — upload one on the Resumes page first")

    try:
        content = generate_resume(job, get_current_candidate_profile(), resume["raw_text"] or "")
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc

    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO generated_documents
               (job_id, application_id, kind, based_on_resume_id, content_text)
               VALUES (?, ?, 'resume', ?, ?)""",
            (payload.job_id, payload.application_id, resume["id"], content),
        )
        doc_id = cursor.lastrowid
    _link_to_application(payload.application_id, "resume_doc_id", doc_id)

    return {"id": doc_id, "content_text": content}


@router.post("/cover-letter")
def generate_cover_letter_route(payload: GenerateRequest):
    job = _get_job(payload.job_id)

    try:
        content = generate_cover_letter(job, get_current_candidate_profile())
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc

    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO generated_documents (job_id, application_id, kind, content_text)
               VALUES (?, ?, 'cover_letter', ?)""",
            (payload.job_id, payload.application_id, content),
        )
        doc_id = cursor.lastrowid
    _link_to_application(payload.application_id, "cover_letter_doc_id", doc_id)

    return {"id": doc_id, "content_text": content}


@router.get("/history")
def generation_history(job_id: int):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM generated_documents WHERE job_id = ? ORDER BY created_at DESC", (job_id,)
        ).fetchall()
    return [dict(r) for r in rows]
