from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.api.applications import CreateApplicationRequest, create_application
from jobhunt.core.digest import IS_FAVORITE_SQL
from jobhunt.core.llm import LLMError
from jobhunt.core.preference import get_current_preference_profile
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.scoring import score_job, store_score
from jobhunt.core.settings import get_setting
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


@router.get("/ranked")
def list_ranked_jobs(status: str = "active"):
    """All jobs discovered from a tracked source (company board or job board), regardless of
    whether they made it into a capped daily digest or the 90-day/not-interested cuts that gate
    "New & Noteworthy" -- ranked best-first by score, with unscored jobs (not yet reached by the
    daily cap) sorted after scored ones. Backs the All Jobs page's full browse view. Manually
    pasted jobs (no source_id) are excluded -- those go straight to My Jobs, not here. Same rank
    order as the digest (favorite, then score, then recency, then salary) for consistency."""
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT j.*, js.interest_fit, js.interest_score, js.candidate_fit, js.candidate_fit_score,
                       js.summary_text, js.rationale_text, s.name AS source_name, {IS_FAVORITE_SQL}
                FROM jobs j
                LEFT JOIN job_scores js ON js.id = (
                    SELECT id FROM job_scores WHERE job_id = j.id ORDER BY id DESC LIMIT 1
                )
                LEFT JOIN sources s ON s.id = j.source_id
                WHERE j.status = ? AND j.source_id IS NOT NULL
                ORDER BY is_favorite DESC,
                         (COALESCE(js.interest_score, 0) + COALESCE(js.candidate_fit_score, 0)) DESC,
                         COALESCE(j.posted_at, j.first_seen_at) DESC,
                         COALESCE(j.salary_max, j.salary_min, 0) DESC""",
            (status,),
        ).fetchall()
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
        score = score_job(
            job,
            get_current_candidate_profile(),
            get_current_preference_profile(),
            get_setting("prefilter_sectors", []),
            get_setting("prefilter_titles_include", []),
            get_setting("salary_floor", None),
        )
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    store_score(job_id, score)

    # Confirming a manually pasted job IS the interest signal -- it goes straight to My Jobs, not
    # through the Digest's "new and noteworthy" ranking (that's for source-discovered postings the
    # user hasn't seen yet). Swallow a 409 so re-confirming an already-tracked job is a no-op.
    try:
        create_application(CreateApplicationRequest(job_id=job_id))
    except HTTPException as exc:
        if exc.status_code != 409:
            raise

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
