from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException

from jobhunt.api.pipeline import stream_run
from jobhunt.core import budget
from jobhunt.core.digest import get_digest, get_digest_dates, today_str
from jobhunt.core.settings import get_setting, update_settings
from jobhunt.core.users import current_user, current_user_id
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run

router = APIRouter(prefix="/api/digest", tags=["digest"])

# New & Noteworthy's Refresh (step 5 of D-021): the person's own check, at most every
# REFRESH_COOLDOWN, reading only sources nobody has read in the last FRESH_HOURS (their jobs come
# from what's stored), and scoring within what's left of today's cap and the month's budget.
REFRESH_COOLDOWN = timedelta(minutes=30)
FRESH_HOURS = 1.0


@router.get("/today")
def digest_today():
    date = today_str()
    return {"date": date, "entries": get_digest(date)}


@router.get("/dates")
def digest_dates():
    return get_digest_dates()


def _refresh_available_at() -> datetime | None:
    last = get_setting("last_refresh_at")
    if not last:
        return None
    available = datetime.fromisoformat(last) + REFRESH_COOLDOWN
    return available if available > datetime.now(UTC) else None


@router.get("/status")
def digest_status():
    """When this person's matches were last updated (their latest finished check, daily or
    refresh), whether they can refresh now, and how much of today's scoring is left."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT MAX(finished_at) AS last FROM runs WHERE user_id = ? AND finished_at IS NOT NULL",
            (current_user_id(),),
        ).fetchone()
    available = _refresh_available_at()
    return {
        "last_updated": row["last"],
        "can_refresh": available is None,
        "refresh_available_at": available.isoformat(timespec="seconds") if available else None,
        "scored_today": daily_run.scored_today(),
        "daily_score_cap": current_user()["daily_score_cap"],
        "over_budget": budget.status(current_user_id())["over"],
    }


@router.post("/refresh/stream")
def refresh_stream():
    """Runs this person's check now, streaming progress like /api/pipeline/run-daily/stream."""
    available = _refresh_available_at()
    if available is not None:
        minutes = max(1, round((available - datetime.now(UTC)).total_seconds() / 60))
        raise HTTPException(429, f"You can refresh again in {minutes} minute{'s' if minutes != 1 else ''}.")
    update_settings({"last_refresh_at": datetime.now(UTC).isoformat(timespec="seconds")})
    return stream_run(daily_run.start_run_daily(only_user=current_user_id(), fresh_hours=FRESH_HOURS))


@router.get("/{digest_date}")
def digest_by_date(digest_date: str):
    return {"date": digest_date, "entries": get_digest(digest_date)}
