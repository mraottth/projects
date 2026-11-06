"""Sign-in status and access requests (D-021). Reachable by any signed-in Google account, approved
or not (api/auth.py's OPEN_PATHS); everything else needs the admin's approval first."""

import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from jobhunt.core.email_digest import admin_address, send_plain_email
from jobhunt.core import budget
from jobhunt.core.users import create_user, get_user_by_email
from jobhunt.db.connection import get_db

router = APIRouter(tags=["access"])


def _email(request: Request) -> str:
    return request.state.auth_email


@router.get("/api/me")
def me(request: Request):
    """Who's signed in and whether they're in: status is "approved", "pending", "declined",
    "blocked", or "none" (never asked)."""
    email = _email(request)
    user = get_user_by_email(email)
    if user is None:
        return {"email": email, "status": "none", "is_admin": False, "name": None}
    return {
        "email": email,
        "status": user["status"],
        "is_admin": bool(user["is_admin"]) and user["status"] == "approved",
        "name": user["name"],
        "id": user["id"] if user["status"] == "approved" else None,
        "usage": _usage(user) if user["status"] == "approved" else None,
    }


def _usage(user: dict) -> dict:
    """This month's Claude spend against the budget, and today's scoring against the daily cap
    (step 4 of D-021), for Settings' usage line."""
    with get_db() as conn:
        scored_today = conn.execute(
            "SELECT COUNT(*) FROM job_scores WHERE user_id = ? AND date(scored_at) = date('now')",
            (user["id"],),
        ).fetchone()[0]
    return {
        **budget.status(user["id"]),
        "scored_today": scored_today,
        "daily_score_cap": user["daily_score_cap"],
    }


class AccessRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=1000)


@router.post("/api/access/request")
def request_access(payload: AccessRequest, request: Request):
    """Asks the admin for access: a pending user, and an email to the admin. Asking again while
    pending is a no-op; a declined or blocked person can't ask again."""
    email = _email(request)
    user = get_user_by_email(email)
    if user is not None:
        if user["status"] in ("declined", "blocked"):
            raise HTTPException(403, "this account can't request access")
        return {"status": user["status"]}
    create_user(email, payload.name.strip(), (payload.note or "").strip() or None)
    app_url = (os.environ.get("APP_URL") or "").rstrip("/")
    to = admin_address()
    if to:
        send_plain_email(
            to,
            f"Access request: {payload.name.strip()}",
            f"{payload.name.strip()} ({email}) asked to use Job Search Assistant.\n\n"
            + (f"Their note: {payload.note.strip()}\n\n" if payload.note and payload.note.strip() else "")
            + f"Approve or decline it here: {app_url}/admin\n",
        )
    return {"status": "pending"}
