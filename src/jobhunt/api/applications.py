import csv
import io
from datetime import date

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from jobhunt.core import app_insights
from jobhunt.core.llm import LLMError
from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/applications", tags=["applications"])

VALID_STATUSES = {
    "saved",
    "waiting",
    "interviewing",
    "offer",
    "rejected_no_interview",
    "rejected_interviewed",
    "withdrawn",
}

# Display labels, as on the My Jobs page (frontend api.ts APPLICATION_STATUS_LABELS), for the CSV.
STATUS_LABELS = {
    "saved": "Drafting",
    "waiting": "Waiting",
    "interviewing": "Interviewing",
    "offer": "Offer",
    "rejected_no_interview": "Rejected - no interview",
    "rejected_interviewed": "Rejected - interviewed",
    "withdrawn": "Withdrawn",
}

# The job's latest score, for My Jobs' fit chips and default order.
_APPLICATION_SELECT = """SELECT a.*, j.title, j.company, j.location, j.url,
       j.salary_min, j.salary_max, j.salary_currency, j.salary_period,
       s.interest_fit, s.interest_score, s.candidate_fit, s.candidate_fit_score
  FROM applications a JOIN jobs j ON j.id = a.job_id
  LEFT JOIN job_scores s ON s.id = (
      SELECT id FROM job_scores WHERE job_id = j.id ORDER BY scored_at DESC, id DESC LIMIT 1)"""


class CreateApplicationRequest(BaseModel):
    job_id: int
    status: str = "saved"


class UpdateApplicationRequest(BaseModel):
    status: str | None = None
    notes: str | None = None
    next_action: str | None = None
    next_action_date: str | None = None
    applied_at: str | None = None


@router.get("")
def list_applications():
    with get_db() as conn:
        rows = conn.execute(f"{_APPLICATION_SELECT} ORDER BY a.updated_at DESC").fetchall()
        history_rows = conn.execute("SELECT * FROM application_status_history ORDER BY changed_at").fetchall()

    history_by_app: dict[int, list[dict]] = {}
    for h in history_rows:
        history_by_app.setdefault(h["application_id"], []).append(dict(h))

    apps = [dict(row) for row in rows]
    for app in apps:
        app["history"] = history_by_app.get(app["id"], [])
    return apps


@router.post("")
def create_application(payload: CreateApplicationRequest):
    if payload.status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of: {sorted(VALID_STATUSES)}")
    with get_db() as conn:
        job = conn.execute("SELECT id FROM jobs WHERE id = ?", (payload.job_id,)).fetchone()
        if job is None:
            raise HTTPException(404, "job not found")
        existing = conn.execute("SELECT id FROM applications WHERE job_id = ?", (payload.job_id,)).fetchone()
        if existing:
            raise HTTPException(409, f"an application for this job already exists (id={existing['id']})")

        applied_at = "datetime('now')" if payload.status != "saved" else "NULL"
        cursor = conn.execute(
            f"INSERT INTO applications (job_id, status, applied_at) VALUES (?, ?, {applied_at})",
            (payload.job_id, payload.status),
        )
        app_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO application_status_history (application_id, status) VALUES (?, ?)",
            (app_id, payload.status),
        )
    return _get_application(app_id)


def _get_application(app_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute(f"{_APPLICATION_SELECT} WHERE a.id = ?", (app_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "application not found")
        history = conn.execute(
            "SELECT * FROM application_status_history WHERE application_id = ? ORDER BY changed_at",
            (app_id,),
        ).fetchall()
    app = dict(row)
    app["history"] = [dict(h) for h in history]
    return app


@router.get("/export.csv")
def export_csv():
    """My Jobs as a spreadsheet: one row per tracked job, newest application first."""
    with get_db() as conn:
        rows = conn.execute(f"{_APPLICATION_SELECT} ORDER BY a.applied_at DESC, a.id DESC").fetchall()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        [
            "Title",
            "Organization",
            "Location",
            "Salary min",
            "Salary max",
            "Salary period",
            "Applied",
            "Status",
            "Interest",
            "Fit",
            "Posting URL",
            "Notes",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r["title"],
                r["company"],
                r["location"],
                r["salary_min"],
                r["salary_max"],
                r["salary_period"],
                (r["applied_at"] or "")[:10],
                STATUS_LABELS.get(r["status"], r["status"]),
                r["interest_fit"],
                r["candidate_fit"],
                r["url"],
                r["notes"],
            ]
        )
    name = f"my-jobs-{date.today().isoformat()}.csv"
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get("/insights")
def get_insights():
    """The last "What's working?" analysis, or null."""
    return app_insights.get_saved()


@router.post("/insights")
def build_insights():
    try:
        return app_insights.build()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/{app_id}")
def get_application(app_id: int):
    return _get_application(app_id)


@router.patch("/{app_id}")
def update_application(app_id: int, payload: UpdateApplicationRequest):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    with get_db() as conn:
        row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "application not found")

        if "status" in updates and updates["status"] not in VALID_STATUSES:
            raise HTTPException(400, f"status must be one of: {sorted(VALID_STATUSES)}")

        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates) + ", updated_at = datetime('now')"
            conn.execute(f"UPDATE applications SET {set_clause} WHERE id = ?", (*updates.values(), app_id))

        if "status" in updates and updates["status"] != row["status"]:
            conn.execute(
                "INSERT INTO application_status_history (application_id, status) VALUES (?, ?)",
                (app_id, updates["status"]),
            )
            # Don't clobber an applied_at the caller explicitly set in this same request.
            if "applied_at" not in updates and row["applied_at"] is None and updates["status"] != "saved":
                conn.execute("UPDATE applications SET applied_at = datetime('now') WHERE id = ?", (app_id,))

    return _get_application(app_id)


@router.delete("/{app_id}")
def delete_application(app_id: int):
    """Removes a job from My Jobs. Any resume/cover letter already generated for it is kept
    (just detached from the application, since it's still a real document the user may want),
    not deleted -- only the tracking row and its status history go."""
    with get_db() as conn:
        row = conn.execute("SELECT id FROM applications WHERE id = ?", (app_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "application not found")
        conn.execute(
            "UPDATE generated_documents SET application_id = NULL WHERE application_id = ?", (app_id,)
        )
        conn.execute("DELETE FROM application_status_history WHERE application_id = ?", (app_id,))
        conn.execute("DELETE FROM applications WHERE id = ?", (app_id,))
    return {"ok": True}
