from fastapi import APIRouter

from jobhunt.pipeline.daily_run import run_daily

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/run-daily")
def run_daily_now():
    """The UI's "Run now" (Manage Sources, and onboarding's last step). Deliberately separate from
    /internal/run-daily: that one checks X-Scheduler-Secret because Cloud Scheduler can't do an
    interactive IAP sign-in, so a browser call there always gets 403 in production. Like every other
    /api/* route, this one relies on IAP for access control instead."""
    return run_daily()
