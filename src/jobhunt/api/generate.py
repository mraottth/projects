from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from jobhunt.core.generate import generate_cover_letter, generate_resume
from jobhunt.core.llm import LLMError
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/generate", tags=["generate"])


class GenerateRequest(BaseModel):
    job_id: int
    application_id: int | None = None
    resume_id: int | None = None  # base resume to tailor from; None lets generate_resume auto-pick


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

    try:
        result = generate_resume(job, get_current_candidate_profile(), payload.resume_id)
    except ValidationError as exc:
        # Pydantic's ValidationError subclasses ValueError, so this must be caught *before* the
        # plain ValueError below -- otherwise a truncated/malformed structured-output response
        # from Claude gets mislabeled as "no resume uploaded" instead of what it actually is.
        raise HTTPException(
            502, f"Claude's response couldn't be parsed (likely cut off mid-output) — try again: {exc}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, f"{exc} — upload one on the Profile page first") from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc

    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO generated_documents
               (job_id, application_id, kind, based_on_resume_id, content_text, file_path)
               VALUES (?, ?, 'resume', ?, ?, ?)""",
            (
                payload.job_id,
                payload.application_id,
                result["based_on_resume_id"],
                result["content_text"],
                result["file_path"],
            ),
        )
        doc_id = cursor.lastrowid
    _link_to_application(payload.application_id, "resume_doc_id", doc_id)

    return {"id": doc_id, **result}


@router.post("/cover-letter")
def generate_cover_letter_route(payload: GenerateRequest):
    job = _get_job(payload.job_id)

    try:
        result = generate_cover_letter(job, get_current_candidate_profile())
    except ValidationError as exc:
        raise HTTPException(
            502, f"Claude's response couldn't be parsed (likely cut off mid-output) — try again: {exc}"
        ) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc

    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO generated_documents (job_id, application_id, kind, content_text, file_path)
               VALUES (?, ?, 'cover_letter', ?, ?)""",
            (payload.job_id, payload.application_id, result["content_text"], result["file_path"]),
        )
        doc_id = cursor.lastrowid
    _link_to_application(payload.application_id, "cover_letter_doc_id", doc_id)

    return {"id": doc_id, **result}


@router.get("/history")
def generation_history(job_id: int):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM generated_documents WHERE job_id = ? ORDER BY created_at DESC", (job_id,)
        ).fetchall()
    return [dict(r) for r in rows]
