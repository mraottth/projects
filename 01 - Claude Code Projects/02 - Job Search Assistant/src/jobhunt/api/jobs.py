from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.api.applications import CreateApplicationRequest, create_application
from jobhunt.core.llm import LLMError
from jobhunt.core.preference import get_current_preference_profile
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.scoring import score_job, store_score
from jobhunt.core import dealbreakers, job_merge
from jobhunt.core.digest import SOURCE_IS_BOARD_SQL
from jobhunt.core.settings import get_setting, get_settings
from jobhunt.db.connection import get_db
from jobhunt.ingest.manual_import import ManualImportError, parse_job_posting
from jobhunt.core.generic_postings import is_generic_posting
from jobhunt.core.users import VISIBLE_JOB_SQL, current_user_id, job_is_visible

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
    query = f"SELECT j.* FROM jobs j WHERE {VISIBLE_JOB_SQL}"
    params: dict = {"uid": current_user_id()}
    if status:
        query += " AND j.status = :status"
        params["status"] = status
    query += " ORDER BY j.first_seen_at DESC"
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows]


# What a job card and the filters use. Not raw_json (each source's raw copy of the posting, never
# shown) or the description (its own route, below): the list is sent whole, every job at once.
RANKED_JOB_COLUMNS = ", ".join(
    f"j.{c}"
    for c in (
        "id", "source_id", "title", "company", "location", "remote_type", "salary_min", "salary_max",
        "salary_currency", "salary_period", "url", "posted_at", "first_seen_at", "last_seen_at", "status",
        "user_id",
    )
)  # fmt: skip
RANKED_WHERE = (
    "j.status = :status AND j.source_id IS NOT NULL "
    "AND j.source_id IN (SELECT source_id FROM source_follows WHERE user_id = :uid)"
)


@router.get("/ranked")
def list_ranked_jobs(status: str = "active"):
    """All jobs discovered from a tracked source (company board or job board), regardless of
    whether they made it into a capped daily digest or the 90-day/not-interested cuts that gate
    "New & Noteworthy" -- ranked best-first by score, with unscored jobs (not yet reached by the
    daily cap) sorted after scored ones. Backs the All Jobs page's full browse view. Manually
    pasted jobs (no source_id) are excluded -- those go straight to My Jobs, not here, and so are
    general-application/talent-pool "postings" (core/generic_postings.py). Same rank order as the
    digest (score, then recency, then salary) for consistency."""
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT {RANKED_JOB_COLUMNS},
                       js.interest_fit, js.interest_score, js.candidate_fit, js.candidate_fit_score,
                       js.summary_text, js.rationale_text, s.name AS source_name,
                       {SOURCE_IS_BOARD_SQL} AS source_is_board,
                       a.status AS application_status,
                       q.screen = 'drop' AS screened_out
                FROM jobs j
                LEFT JOIN job_scores js ON js.id = (
                    SELECT id FROM job_scores WHERE user_id = :uid AND job_id = j.id ORDER BY id DESC LIMIT 1
                )
                LEFT JOIN sources s ON s.id = j.source_id
                -- Its status on the person's My Jobs, if it's there (the card shows it; asked 2026-11-10).
                LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = :uid
                -- Set aside by the title screen (core/screening.py): hidden unless asked for.
                LEFT JOIN scoring_queue q ON q.job_id = j.id AND q.user_id = :uid AND q.status = 'pending'
                WHERE {RANKED_WHERE}
                ORDER BY (COALESCE(js.interest_score, 0) + COALESCE(js.candidate_fit_score, 0)) DESC,
                         COALESCE(j.posted_at, j.first_seen_at) DESC,
                         COALESCE(j.salary_max, j.salary_min, 0) DESC""",
            {"status": status, "uid": current_user_id()},
        ).fetchall()
    # The page hides jobs a dealbreaker rules out unless asked (a filter toggle), so it can say how
    # many are hidden; every job carries which dealbreakers apply ([] for none).
    settings = get_settings()
    jobs = [_row_to_dict(r) for r in rows if not is_generic_posting(r["title"])]
    for job in jobs:
        job["dealbreakers"] = dealbreakers.reasons(job, settings)
        job["screened_out"] = bool(job.get("screened_out"))
        job["description_text"] = None  # sent by /ranked/descriptions, after the list
    return jobs


@router.get("/ranked/descriptions")
def ranked_descriptions(status: str = "active"):
    """{job id: description} for the same jobs as /ranked. All Jobs loads it after the list, for
    its search box, which also searches descriptions: sent with the list they were most of its
    size (17.6 MB for 4,000 jobs, measured 2026-11-10) and held up the first paint."""
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT j.id, j.description_text FROM jobs j "
            f"WHERE {RANKED_WHERE} AND j.description_text IS NOT NULL",
            {"status": status, "uid": current_user_id()},
        ).fetchall()
    return {str(r["id"]): r["description_text"] for r in rows}


@router.get("/{job_id}")
def get_job(job_id: int):
    params = {"job": job_id, "uid": current_user_id()}
    with get_db() as conn:
        row = conn.execute(
            f"SELECT j.* FROM jobs j WHERE j.id = :job AND {VISIBLE_JOB_SQL}", params
        ).fetchone()
        if row is None:
            raise HTTPException(404, "job not found")
        scores = conn.execute(
            "SELECT * FROM job_scores WHERE job_id = :job AND user_id = :uid ORDER BY id DESC", params
        ).fetchall()
    job = _row_to_dict(row)
    job["scores"] = [_row_to_dict(s) for s in scores]
    return job


def _track_existing(job_id: int) -> dict:
    """A pasted job one of your sources already has (D-035): that job goes on My Jobs instead of a
    second copy, and the reply says so (`already_in_sources`)."""
    try:
        create_application(CreateApplicationRequest(job_id=job_id))
    except HTTPException as exc:
        if exc.status_code != 409:
            raise
    return {**get_job(job_id), "already_in_sources": True}


@router.post("/manual")
def create_manual_job(payload: ManualImportRequest):
    if not payload.url and not payload.raw_text:
        raise HTTPException(400, "provide url or raw_text")
    user_id = current_user_id()
    if payload.url:
        # Checked before the paid read: the address alone can show it's a job we already have.
        with get_db() as conn:
            identity = job_merge.posting_identity(payload.url)
            same = [j for j in job_merge.source_jobs_for(conn, user_id)
                    if identity and job_merge.posting_identity(j["url"]) == identity]  # fmt: skip
        if same:
            return _track_existing(job_merge.find_match({"url": payload.url}, same)["id"])
    try:
        parsed = parse_job_posting(url=payload.url, raw_text=payload.raw_text)
    except ManualImportError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc

    with get_db() as conn:
        existing_source_job = job_merge.match_for_pasted(conn, user_id, parsed)
    if existing_source_job is not None:
        return _track_existing(existing_source_job)

    # A pasted-in job is private to whoever pasted it, so its key is too: two people pasting the
    # same posting each get their own copy.
    dedupe_key = f"u{user_id}:{parsed['dedupe_key']}"
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM jobs WHERE dedupe_key IN (?, ?) AND user_id = ?",
            (dedupe_key, parsed["dedupe_key"], user_id),
        ).fetchone()
        if existing:
            raise HTTPException(409, f"this job already exists (id={existing['id']})")

        cursor = conn.execute(
            """INSERT INTO jobs
               (user_id, dedupe_key, title, company, location, remote_type, salary_min, salary_max,
                salary_currency, salary_period, description_text, url, posted_at, needs_review)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                user_id,
                dedupe_key,
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
        # Only a pasted-in job is editable, and only by its owner: a source's job is shared.
        row = conn.execute(
            "SELECT id FROM jobs WHERE id = ? AND user_id = ?", (job_id, current_user_id())
        ).fetchone()
        if row is None:
            raise HTTPException(404, "job not found")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", (*updates.values(), job_id))
    return get_job(job_id)


@router.post("/{job_id}/confirm")
def confirm_job(job_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, current_user_id())
        ).fetchone()
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
            get_setting("looking_for_notes", ""),
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


@router.post("/{job_id}/score")
def score_one_job(job_id: int):
    """Scores one job now, on request (asked 2026-11-12: a "Score this job" button on a card).
    Live, about a cent, and outside the daily cap, like a rescore: it's an explicit ask. The
    monthly budget still applies (get_client raises BudgetExceeded, an LLMError: a 502 here). Works
    on a job the title screen set aside too, and on one already scored (a fresh score)."""
    from jobhunt.core import queue
    from jobhunt.pipeline.daily_run import _backfill_description

    if not job_is_visible(job_id):
        raise HTTPException(404, "job not found")
    with get_db() as conn:
        job = _row_to_dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())
    job = _backfill_description(job)  # a web-board job's posting is fetched before its first score
    try:
        score = score_job(
            job,
            get_current_candidate_profile(),
            get_current_preference_profile(),
            get_setting("prefilter_sectors", []),
            get_setting("prefilter_titles_include", []),
            get_setting("salary_floor", None),
            get_setting("looking_for_notes", ""),
        )
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    store_score(job_id, score)
    queue.mark_scored(job_id)
    return {
        "interest_fit": score.interest_fit,
        "interest_score": score.interest_score,
        "candidate_fit": score.candidate_fit,
        "candidate_fit_score": score.candidate_fit_score,
        "summary_text": score.summary_text,
        "rationale_text": score.rationale_text,
    }


@router.post("/{job_id}/feedback")
def submit_feedback(job_id: int, payload: FeedbackRequest):
    if payload.interested not in ("interested", "not_interested", "maybe"):
        raise HTTPException(400, "interested must be one of: interested, not_interested, maybe")
    if not job_is_visible(job_id):
        raise HTTPException(404, "job not found")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, job_id, interested, note) VALUES (?, ?, ?, ?)",
            (current_user_id(), job_id, payload.interested, payload.note),
        )
    return {"ok": True}
