from fastapi import APIRouter, Query

from jobhunt.pipeline import daily_run

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/run-daily")
def run_daily_now(cap: int | None = Query(default=None, ge=1, le=200)):
    """The UI's "Run now" (Manage Sources, and onboarding's last step). Deliberately separate from
    /internal/run-daily: that one checks X-Scheduler-Secret because Cloud Scheduler can't do an
    interactive IAP sign-in, so a browser call there always gets 403 in production. Like every other
    /api/* route, this one relies on IAP for access control instead.

    `cap` scores at most that many jobs this run (onboarding passes a small one so the user isn't
    kept waiting); the rest stay queued for the next run."""
    return daily_run.run_daily(cap=cap)


@router.get("/progress")
def run_progress():
    """Progress of the run in flight (or the last one), polled by the UI while its run-daily
    request is still open."""
    return daily_run.get_progress()
