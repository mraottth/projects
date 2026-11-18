import os

from fastapi import APIRouter, Header, HTTPException

from jobhunt.pipeline.daily_run import run_daily

router = APIRouter(prefix="/internal", tags=["internal"])


def _check_secret(given: str | None) -> None:
    """Cloud Scheduler's targets are gated by their own secret header, independent of whatever
    locks down the UI (IAP), since Scheduler can't do interactive sign-in. If SCHEDULER_SECRET
    isn't set (local dev), the check is skipped entirely."""
    expected = os.environ.get("SCHEDULER_SECRET")
    if expected and given != expected:
        raise HTTPException(403, "invalid or missing scheduler secret")


@router.post("/run-daily")
def trigger_run_daily(x_scheduler_secret: str | None = Header(default=None)):
    """Cloud Scheduler's 2pm target (see Makefile's scheduler-setup): read, screen, score, email."""
    _check_secret(x_scheduler_secret)
    return run_daily()


@router.post("/read-sources")
def trigger_read_sources(x_scheduler_secret: str | None = Header(default=None)):
    """Cloud Scheduler's 10am target (asked 2026-11-18; Makefile's scheduler-setup): read every
    source and run the relevance check, without scoring or email. The jobs it finds wait on the
    run audit's "Waiting to be scored" for the 2pm run or the person's "Score this job now"."""
    _check_secret(x_scheduler_secret)
    return run_daily(score=False)


@router.get("/ping")
def ping():
    """Does nothing; Cloud Scheduler calls it every 10 minutes during the warm window (Makefile's
    keep-warm, asked 2026-11-10), so the server stays up and nobody waits for a cold start. A
    request keeps a Cloud Run instance up for about 15 minutes. No secret: it reads and writes
    nothing, and IAP still guards it like every other path."""
    return {"ok": True}
