import json
import time

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

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


PROGRESS_INTERVAL_SECONDS = 2.0


@router.post("/run-daily/stream")
def run_daily_stream(cap: int | None = Query(default=None, ge=1, le=200)):
    """What the UI actually calls for a run: starts it in the background (daily_run.start_run_daily)
    and streams progress as server-sent events every couple of seconds until it finishes --
    {"type": "started", "run_id"}, then {"type": "progress", ...}, then {"type": "done", "summary"}
    or {"type": "error", "message"}. Data keeps flowing, so the connection isn't dropped as idle,
    and Cloud Run keeps CPU allocated to the instance while this request is open. If the page loses
    the stream anyway, the run carries on, and the page follows it through GET /progress."""
    run_id = daily_run.start_run_daily(cap=cap)

    def events():
        yield _sse({"type": "started", "run_id": run_id})
        while True:
            progress = daily_run.get_progress()
            if progress["finished_run_id"] == run_id:
                if progress["last_error"]:
                    yield _sse({"type": "error", "message": progress["last_error"]})
                else:
                    yield _sse({"type": "done", "summary": progress["last_summary"]})
                return
            yield _sse({"type": "progress", **progress})
            time.sleep(PROGRESS_INTERVAL_SECONDS)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.get("/progress")
def run_progress():
    """Progress of the run in flight (or the last one), polled by the UI while its run-daily
    request is still open."""
    return daily_run.get_progress()
