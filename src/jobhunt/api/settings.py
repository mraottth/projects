from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.core.llm import LLMError
from jobhunt.core.email_digest import EmailNotConfigured, email_status, send_test_email
from jobhunt.core.reset import delete_all_data
from jobhunt.core.preference import get_current_preference_profile, rebuild_preference_profile
from jobhunt.core.settings import get_settings, update_settings
from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/settings", tags=["settings"])


class DeleteAllRequest(BaseModel):
    confirm: str


@router.post("/delete-all-data")
def delete_all_data_route(payload: DeleteAllRequest):
    """Settings' "Delete all my data". Requires {"confirm": "DELETE"} so a stray request can't
    wipe everything; the UI makes the user type it."""
    if payload.confirm != "DELETE":
        raise HTTPException(400, 'type DELETE to confirm')
    return delete_all_data()


@router.get("/email-status")
def get_email_status():
    """Whether the daily email is set up, and who it goes to. Never includes the password."""
    return email_status()


@router.post("/test-email")
def post_test_email():
    try:
        return send_test_email()
    except EmailNotConfigured as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a bad app password or Gmail refusal, shown to the user
        raise HTTPException(502, f"Gmail didn't accept the email: {exc}") from exc


@router.get("")
def get_all_settings():
    return get_settings()


@router.put("")
def put_settings(payload: dict):
    return update_settings(payload)


@router.get("/usage-log")
def usage_log(limit: int = 100):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM llm_calls ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        totals = conn.execute(
            """SELECT
                   COALESCE(SUM(estimated_cost_usd), 0) AS total_cost,
                   COALESCE(SUM(CASE WHEN date(created_at) = date('now') THEN estimated_cost_usd END), 0)
                       AS today_cost,
                   COUNT(*) AS total_calls
               FROM llm_calls"""
        ).fetchone()
    return {"calls": [dict(r) for r in rows], "totals": dict(totals)}


@router.get("/preference-profile")
def get_preference_profile():
    text = get_current_preference_profile()
    if text is None:
        raise HTTPException(404, "no preference profile yet — mark some jobs or track an application")
    return {"text": text}


@router.post("/preference-profile/rebuild")
def rebuild_preference_profile_route():
    try:
        text = rebuild_preference_profile()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"text": text}
