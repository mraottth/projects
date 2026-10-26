from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.core.digest import add_to_digest
from jobhunt.core.llm import LLMError
from jobhunt.core.preference import get_current_preference_profile
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.scoring import score_job, store_score
from jobhunt.db.connection import get_db
from jobhunt.ingest.manual_import import ManualImportError, parse_job_posting

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class ManualImportRequest(BaseModel):
    url: str | None = None
    raw_text: str | None = None


class FeedbackRequest(BaseModel):
    interested: str
    note: str | None = None


class JobUpdateRequest(BaseModel):
    title: str | None = None
    company: str | None = None
    location: str | None = None
    remote_type: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    description_text: str | None = None


def _row_to_dict(row) -> dict:
    return dict(row)


@router.get("")
def list_jobs(status: str | None = None):
    query = "SELECT * FROM jobs"
    params: tuple = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY first_seen_at DESC"
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows]


@router.get("/{job_id}")
def get_job(job_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "job not found")
        scores = conn.execute(
            "SELECT * FROM job_scores WHERE job_id = ? ORDER BY id DESC", (job_id,)
        ).fetchall()
    job = _row_to_dict(row)
    job["scores"] = [_row_to_dict(s) for s in scores]
    return job


@router.post("/manual")
def create_manual_job(payload: ManualImportRequest):
    if not payload.url and not payload.raw_text:
        raise HTTPException(400, "provide url or raw_text")
    try:
        parsed = parse_job_posting(url=payload.url, raw_text=payload.raw_text)
    except ManualImportError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc

    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM jobs WHERE dedupe_key = ?", (parsed["dedupe_key"],)
        ).fetchone()
        if existing:
            raise HTTPException(409, f"this job already exists (id={existing['id']})")

        cursor = conn.execute(
            """INSERT INTO jobs
               (dedupe_key, title, company, location, remote_type, salary_min, salary_max,
                salary_currency, salary_period, description_text, url, posted_at, needs_review)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                parsed["dedupe_key"],
                parsed["title"],
                parsed["company"],
                parsed["location"],
                parsed["remote_type"],
                parsed["salary_min"],
                parsed["salary_max"],
                parsed["salary_currency"],
                parsed["salary_period"],
                parsed["description_text"],
                parsed["url"],
                parsed["posted_at"],
            ),
        )
        job_id = cursor.lastrowid

    return get_job(job_id)


@router.patch("/{job_id}")
def update_job(job_id: int, payload: JobUpdateRequest):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        return get_job(job_id)
    with get_db() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "job not found")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", (*updates.values(), job_id))
    return get_job(job_id)


@router.post("/{job_id}/confirm")
def confirm_job(job_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "job not found")
        job = _row_to_dict(row)
        conn.execute("UPDATE jobs SET needs_review = 0 WHERE id = ?", (job_id,))

    try:
        score = score_job(job, get_current_candidate_profile(), get_current_preference_profile())
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    store_score(job_id, score)
    add_to_digest(job_id)

    return get_job(job_id)


@router.post("/{job_id}/feedback")
def submit_feedback(job_id: int, payload: FeedbackRequest):
    if payload.interested not in ("interested", "not_interested", "maybe"):
        raise HTTPException(400, "interested must be one of: interested, not_interested, maybe")
    with get_db() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "job not found")
        conn.execute(
            "INSERT INTO feedback (job_id, interested, note) VALUES (?, ?, ?)",
            (job_id, payload.interested, payload.note),
        )
    return {"ok": True}
