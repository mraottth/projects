import json
import tempfile
import time
from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from jobhunt.core import backup, retag

from jobhunt.core.llm import CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER, PRICING_PER_MTOK, LLMError
from jobhunt.core.email_digest import EmailNotConfigured, email_status, send_test_email
from jobhunt.core.reset import delete_all_data
from jobhunt.core.preference import get_current_preference_profile, rebuild_preference_profile
from jobhunt.core.settings import get_settings, update_settings
from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/settings", tags=["settings"])


class DeleteAllRequest(BaseModel):
    confirm: str


@router.get("/backup")
def download_backup():
    """Settings' "Download a backup": a fresh, consistent copy of the whole database (SQLite's
    online backup API), for a copy kept off Google Cloud. Deleted once sent."""
    tmp = Path(tempfile.mkdtemp()) / "jobhunt.db"
    backup.copy_database(tmp)
    return FileResponse(
        tmp,
        media_type="application/vnd.sqlite3",
        filename=f"jobhunt-backup-{date.today().isoformat()}.db",
        background=BackgroundTask(lambda: (tmp.unlink(missing_ok=True), tmp.parent.rmdir())),
    )


@router.get("/backups")
def list_backups():
    """The snapshots in backups/ (core/backup.py), newest first, for Settings' "Your data" card."""
    folder = backup.backup_dir()
    files = sorted(folder.glob("jobhunt-*.db"), reverse=True) if folder.exists() else []
    return {"backups": [{"name": f.name, "bytes": f.stat().st_size} for f in files]}


@router.post("/delete-all-data")
def delete_all_data_route(payload: DeleteAllRequest):
    """Settings' "Delete all my data". Requires {"confirm": "DELETE"} so a stray request can't
    wipe everything; the UI makes the user type it."""
    if payload.confirm != "DELETE":
        raise HTTPException(400, "type DELETE to confirm")
    return delete_all_data()


@router.get("/retag-highlights")
def retag_highlights_status():
    """How many job summaries still have uncolored (untagged) highlights, and a cost estimate."""
    return retag.pending_summary()


@router.post("/retag-highlights")
def retag_highlights():
    """Colors older summaries' highlights (core/retag.py) in a background thread, streaming
    progress as server-sent events -- {"type": "progress", done, total}, then {"type": "done",
    retagged, skipped} or {"type": "error", message} -- like the pipeline's run stream, so a long
    pass isn't a silent request that gets cut. A second request follows the pass in progress."""
    retag.start()

    def events():
        while True:
            state = retag.get_state()
            if not state["running"]:
                if state["error"]:
                    yield _sse({"type": "error", "message": state["error"]})
                else:
                    yield _sse({"type": "done", "retagged": state["retagged"], "skipped": state["skipped"]})
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
        rows = conn.execute("SELECT * FROM llm_calls ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        totals = conn.execute(
            """SELECT
                   COALESCE(SUM(estimated_cost_usd), 0) AS total_cost,
                   COALESCE(SUM(CASE WHEN date(created_at) = date('now') THEN estimated_cost_usd END), 0)
                       AS today_cost,
                   COUNT(*) AS total_calls
               FROM llm_calls"""
        ).fetchone()
        cache_rows = conn.execute(
            """SELECT model, SUM(input_tokens) AS input, SUM(cache_read_tokens) AS reads,
                      SUM(cache_write_tokens) AS writes
               FROM llm_calls WHERE created_at >= datetime('now', '-7 days') GROUP BY model"""
        ).fetchall()
    return {"calls": [dict(r) for r in rows], "totals": dict(totals), "cache": _cache_summary(cache_rows)}


def _cache_summary(rows) -> dict:
    """The last 7 days of prompt caching: the share of input tokens read from cache, and what that
    saved after paying for the cache writes."""
    total_input = reads = 0
    saved = 0.0
    for r in rows:
        r_input, r_reads, r_writes = r["input"] or 0, r["reads"] or 0, r["writes"] or 0
        total_input += r_input + r_reads + r_writes
        reads += r_reads
        rates = PRICING_PER_MTOK.get(r["model"])
        if rates:
            per_token = rates[0] / 1_000_000
            saved += r_reads * per_token * (1 - CACHE_READ_MULTIPLIER)
            saved -= r_writes * per_token * (CACHE_WRITE_MULTIPLIER - 1)
    return {"hit_rate": reads / total_input if total_input else 0.0, "saved_usd": saved}


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
