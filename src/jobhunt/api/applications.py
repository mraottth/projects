from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/applications", tags=["applications"])

VALID_STATUSES = {
    "saved",
    "applied",
    "phone_screen",
    "interview",
    "offer",
    "rejected",
    "withdrawn",
}


class CreateApplicationRequest(BaseModel):
    job_id: int
    status: str = "saved"


class UpdateApplicationRequest(BaseModel):
    status: str | None = None
    notes: str | None = None
    next_action: str | None = None
    next_action_date: str | None = None


@router.get("")
def list_applications():
    with get_db() as conn:
        rows = conn.execute(
            """SELECT a.*, j.title, j.company, j.location, j.url
               FROM applications a JOIN jobs j ON j.id = a.job_id
               ORDER BY a.updated_at DESC"""
        ).fetchall()
        history_rows = conn.execute(
            "SELECT * FROM application_status_history ORDER BY changed_at"
        ).fetchall()

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
        existing = conn.execute(
            "SELECT id FROM applications WHERE job_id = ?", (payload.job_id,)
        ).fetchone()
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
        row = conn.execute(
            """SELECT a.*, j.title, j.company, j.location, j.url
               FROM applications a JOIN jobs j ON j.id = a.job_id WHERE a.id = ?""",
            (app_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "application not found")
        history = conn.execute(
            "SELECT * FROM application_status_history WHERE application_id = ? ORDER BY changed_at",
            (app_id,),
        ).fetchall()
    app = dict(row)
    app["history"] = [dict(h) for h in history]
    return app


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
            conn.execute(
                f"UPDATE applications SET {set_clause} WHERE id = ?", (*updates.values(), app_id)
            )

        if "status" in updates and updates["status"] != row["status"]:
            conn.execute(
                "INSERT INTO application_status_history (application_id, status) VALUES (?, ?)",
                (app_id, updates["status"]),
            )
            if row["applied_at"] is None and updates["status"] != "saved":
                conn.execute(
                    "UPDATE applications SET applied_at = datetime('now') WHERE id = ?", (app_id,)
                )

    return _get_application(app_id)
