import os

from fastapi import APIRouter, Header, HTTPException

from jobhunt.pipeline.daily_run import run_daily

router = APIRouter(prefix="/internal", tags=["internal"])


@router.post("/run-daily")
def trigger_run_daily(x_scheduler_secret: str | None = Header(default=None)):
    """Cloud Scheduler's target (see Makefile's scheduler-setup). Gated by its own secret header,
    independent of whatever locks down the UI (IAP), since Scheduler can't do interactive sign-in.
    If SCHEDULER_SECRET isn't set (local dev), the check is skipped entirely."""
    expected = os.environ.get("SCHEDULER_SECRET")
    if expected and x_scheduler_secret != expected:
        raise HTTPException(403, "invalid or missing scheduler secret")
    return run_daily()
