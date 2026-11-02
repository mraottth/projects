"""The admin page's API (D-021): access requests, everyone's status, usage and limits."""

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from jobhunt.core.email_digest import send_plain_email
from jobhunt.core.reset import AdminAccountError, delete_account
from jobhunt.core.users import STATUSES, current_user_id, get_user, insert_default_settings, is_admin
from jobhunt.db.connection import get_db


def require_admin() -> None:
    if not is_admin():
        raise HTTPException(403, "only the admin can do this")


router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/users")
def list_users():
    """Everyone who has asked, newest request first among pending, with today's scoring against
    their cap, this month's estimated spend against their budget, and how many sources they follow."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT u.*,
                   (SELECT COUNT(*) FROM job_scores s WHERE s.user_id = u.id
                     AND date(s.scored_at) = date('now')) AS scored_today,
                   (SELECT COALESCE(SUM(c.estimated_cost_usd), 0) FROM llm_calls c WHERE c.user_id = u.id
                     AND c.created_at >= date('now', 'start of month')) AS spend_this_month,
                   (SELECT COUNT(*) FROM source_follows f WHERE f.user_id = u.id) AS sources_followed,
                   (SELECT COUNT(*) FROM applications a WHERE a.user_id = u.id) AS applications
               FROM users u
               ORDER BY CASE u.status WHEN 'pending' THEN 0 WHEN 'approved' THEN 1 ELSE 2 END,
                        u.is_admin DESC, u.created_at DESC"""
        ).fetchall()
        shared = conn.execute(
            """SELECT COALESCE(SUM(estimated_cost_usd), 0) AS spend FROM llm_calls
               WHERE user_id IS NULL AND created_at >= date('now', 'start of month')"""
        ).fetchone()
    return {"users": [dict(r) for r in rows], "shared_spend_this_month": shared["spend"]}


class UserUpdate(BaseModel):
    status: str | None = None
    daily_score_cap: int | None = Field(default=None, ge=0, le=500)
    monthly_budget_usd: float | None = Field(default=None, ge=0, le=1000)


@router.patch("/users/{user_id}")
def update_user(user_id: int, payload: UserUpdate):
    user = get_user(user_id)
    if user is None:
        raise HTTPException(404, "user not found")
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if "status" in updates:
        if updates["status"] not in STATUSES:
            raise HTTPException(400, f"status must be one of: {', '.join(STATUSES)}")
        if user_id == current_user_id():
            raise HTTPException(400, "you can't change your own access")
    if not updates:
        return get_user(user_id)
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    if "status" in updates and updates["status"] != user["status"]:
        set_clause += ", decided_at = datetime('now')"
    with get_db() as conn:
        conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", (*updates.values(), user_id))
        if updates.get("status") == "approved":
            insert_default_settings(conn, user_id)

    new_status = updates.get("status")
    if new_status and new_status != user["status"]:
        _tell(user, new_status)
    return get_user(user_id)


def _tell(user: dict, status: str) -> None:
    app_url = (os.environ.get("APP_URL") or "").rstrip("/")
    name = user["name"] or "there"
    if status == "approved":
        send_plain_email(
            user["email"],
            "You're in: Job Search Assistant",
            f"Hi {name},\n\nYour access to Job Search Assistant was approved. Sign in with this Google "
            f"account to get started:\n\n{app_url or '(the app link)'}\n",
        )
    elif status == "declined":
        send_plain_email(
            user["email"],
            "Your Job Search Assistant request",
            f"Hi {name},\n\nThanks for your interest. Your request to use Job Search Assistant wasn't "
            "approved this time.\n",
        )


class DeleteUserRequest(BaseModel):
    confirm: str


@router.delete("/users/{user_id}")
def delete_user(user_id: int, payload: DeleteUserRequest):
    """Deletes someone's account and everything in it (core/reset.py's delete_account), and tells
    them by email. Requires {"confirm": "DELETE"}; the admin's own account can't be deleted."""
    if payload.confirm != "DELETE":
        raise HTTPException(400, "type DELETE to confirm")
    user = get_user(user_id)
    if user is None:
        raise HTTPException(404, "user not found")
    try:
        result = delete_account(user_id)
    except AdminAccountError as exc:
        raise HTTPException(400, str(exc)) from exc
    if user["status"] == "approved":
        send_plain_email(
            user["email"],
            "Your Job Search Assistant account was deleted",
            f"Hi {user['name'] or 'there'},\n\nYour Job Search Assistant account and everything in it "
            "were deleted by the app's admin.\n",
        )
    return result

