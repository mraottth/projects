import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ValidationError

from jobhunt.core.docx_writer import AmbiguousAnchorError, AnchorNotFoundError, DisallowedAnchorError
from jobhunt.core.generate import generate_cover_letter, generate_resume
from jobhunt.core.llm import LLMError
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.db.connection import get_db
from jobhunt.core.users import VISIBLE_JOB_SQL, current_user_id

router = APIRouter(prefix="/api/generate", tags=["generate"])

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class GenerateRequest(BaseModel):
    job_id: int
    application_id: int | None = None
    resume_id: int | None = None  # base resume to tailor from; None lets generate_resume auto-pick


def _get_job(job_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute(
            f"SELECT j.* FROM jobs j WHERE j.id = :job AND {VISIBLE_JOB_SQL}",
            {"job": job_id, "uid": current_user_id()},
        ).fetchone()
    if row is None:
        raise HTTPException(404, "job not found")
    return dict(row)


def _link_to_application(application_id: int | None, column: str, doc_id: int) -> None:
    if application_id is None:
        return
    with get_db() as conn:
        conn.execute(
            f"UPDATE applications SET {column} = ? WHERE id = ? AND user_id = ?",
            (doc_id, application_id, current_user_id()),
        )


def _resolve_application_id(application_id: int | None) -> int | None:
    """Drop a stale application_id rather than letting the INSERT below hit the foreign-key
    constraint -- e.g. the job was removed from My Jobs (DELETE /api/applications/{id}) between
    the frontend loading its state and the user clicking Generate. The document itself (the
    expensive part) is still saved either way, just without a tracker link."""
    if application_id is None:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM applications WHERE id = ? AND user_id = ?", (application_id, current_user_id())
        ).fetchone()
    return application_id if row is not None else None


def _save_generated_document(kind: str, payload: GenerateRequest, result: dict) -> int:
    application_id = _resolve_application_id(payload.application_id)
    try:
        with get_db() as conn:
            if kind == "resume":
                cursor = conn.execute(
                    """INSERT INTO generated_documents
                       (user_id, job_id, application_id, kind, based_on_resume_id, content_text, file_path)
                       VALUES (?, ?, ?, 'resume', ?, ?, ?)""",
                    (
                        current_user_id(),
                        payload.job_id,
                        application_id,
                        result["based_on_resume_id"],
                        result["content_text"],
                        result["file_path"],
                    ),
                )
            else:
                cursor = conn.execute(
                    """INSERT INTO generated_documents
                       (user_id, job_id, application_id, kind, content_text, file_path)
                       VALUES (?, ?, ?, 'cover_letter', ?, ?)""",
                    (
                        current_user_id(),
                        payload.job_id,
                        application_id,
                        result["content_text"],
                        result["file_path"],
                    ),
                )
            doc_id = cursor.lastrowid
    except sqlite3.Error as exc:
        # The content itself was already generated successfully at this point -- this is purely a
        # bookkeeping failure, but it must not surface as a bare, undiagnosable 500.
        raise HTTPException(500, f"generated the document but failed to save it: {exc}") from exc
    column = "resume_doc_id" if kind == "resume" else "cover_letter_doc_id"
    _link_to_application(application_id, column, doc_id)
    return doc_id


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
    except (AnchorNotFoundError, AmbiguousAnchorError, DisallowedAnchorError) as exc:
        raise HTTPException(
            502, f"the tailored edits didn't line up with your resume template ({exc}) — try again"
        ) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc

    doc_id = _save_generated_document("resume", payload, result)
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

    doc_id = _save_generated_document("cover_letter", payload, result)
    return {"id": doc_id, **result}


@router.get("/history")
def generation_history(job_id: int):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM generated_documents WHERE job_id = ? AND user_id = ? ORDER BY created_at DESC",
            (job_id, current_user_id()),
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/documents")
def all_generated_documents():
    """Every resume/cover letter ever generated, across all jobs -- backs the Profile page's
    Generated Documents section, which links each one back to the posting it was written for."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT gd.*, j.title AS job_title, j.company AS job_company
               FROM generated_documents gd JOIN jobs j ON j.id = gd.job_id
               WHERE gd.user_id = ?
               ORDER BY gd.created_at DESC""",
            (current_user_id(),),
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/documents/{doc_id}/download")
def download_generated_document(doc_id: int):
    with get_db() as conn:
        row = conn.execute(
            """SELECT gd.*, j.title AS job_title, j.company AS job_company
               FROM generated_documents gd JOIN jobs j ON j.id = gd.job_id
               WHERE gd.id = ? AND gd.user_id = ?""",
            (doc_id, current_user_id()),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "document not found")
    doc = dict(row)

    safe_name = f"{doc['job_company']}_{doc['job_title']}_{doc['kind']}".replace(" ", "_").replace("/", "_")

    file_path = Path(doc["file_path"]) if doc["file_path"] else None
    if file_path is not None and file_path.exists():
        return FileResponse(file_path, filename=f"{safe_name}{file_path.suffix}", media_type=DOCX_MEDIA_TYPE)

    # No docx on disk (plain-text fallback generation, or the file's since been moved/deleted) --
    # the markdown content is still downloadable, just not as a formatted .docx.
    return Response(
        content=doc["content_text"],
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.md"'},
    )
