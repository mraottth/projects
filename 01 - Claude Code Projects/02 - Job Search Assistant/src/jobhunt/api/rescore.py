"""Rescoring My Jobs or All Jobs on request (core/rescore.py)."""

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from jobhunt.core import rescore

router = APIRouter(prefix="/api/rescore", tags=["rescore"])


def _scope(scope: str) -> str:
    if scope not in rescore.SCOPES:
        raise HTTPException(404, "unknown list")
    return scope


@router.get("/{scope}")
def rescore_status(scope: str):
    """How many jobs a rescore of that list would score, about what it would cost, and when it's
    next available (null: now)."""
    return rescore.status(_scope(scope))


def _wait_text(ready_iso: str) -> str:
    hours = (datetime.fromisoformat(ready_iso) - datetime.now(timezone.utc)).total_seconds() / 3600
    if hours >= 1:
        n = round(hours)
        return f"You can rescore this list again in {n} hour{'s' if n != 1 else ''}."
    n = max(1, round(hours * 60))
    return f"You can rescore this list again in {n} minute{'s' if n != 1 else ''}."


@router.post("/{scope}")
def run_rescore(scope: str):
    """Starts (or follows) a rescore and streams {"type": "progress", done, total}, then
    {"type": "done", scored, failed, stopped} or {"type": "error", message}. 429 during the
    once-a-day cooldown (anyone but the admin)."""
    _scope(scope)
    try:
        rescore.start(scope)
    except rescore.RescoreUnavailable as exc:
        raise HTTPException(429, _wait_text(exc.available_at) if exc.available_at else str(exc)) from exc

    def events():
        while True:
            state = rescore.get_state()
            if not state["running"]:
                if state.get("error"):
                    yield _sse({"type": "error", "message": state["error"]})
                else:
                    yield _sse(
                        {
                            "type": "done",
                            "scored": state["scored"],
                            "failed": state["failed"],
                            "stopped": state["stopped"],
                        }
                    )
                return
            yield _sse({"type": "progress", "done": state["done"], "total": state["total"]})
            time.sleep(1.0)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"
