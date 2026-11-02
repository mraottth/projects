import json
import threading
import time
import uuid
from collections.abc import Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from jobhunt.core.discovery import suggest_companies, suggest_job_boards
from jobhunt.core.llm import LLMError
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.settings import get_settings
from jobhunt.db.connection import get_db
from jobhunt.core.users import carry, current_user_id

router = APIRouter(prefix="/api/discovery", tags=["discovery"])


def _already_monitored_company_names() -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT s.company_name FROM sources s JOIN source_follows f ON f.source_id = s.id "
            "WHERE f.user_id = ? AND s.company_name IS NOT NULL",
            (current_user_id(),),
        ).fetchall()
    return sorted(r["company_name"] for r in rows)


def _already_monitored_board_urls() -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT s.config_json FROM sources s JOIN source_follows f ON f.source_id = s.id "
            "WHERE f.user_id = ? AND s.kind = 'custom_board'",
            (current_user_id(),),
        ).fetchall()
    urls = []
    for r in rows:
        config = json.loads(r["config_json"] or "{}")
        if config.get("url"):
            urls.append(config["url"])
    return urls


class DiscoveryRequest(BaseModel):
    """Optional per-search steering. `guidance` is free text ("mission-driven tech startups");
    `like` describes a source or suggestion to find more of ("Nava PBC (https://...)")."""

    guidance: str | None = Field(default=None, max_length=500)
    like: str | None = Field(default=None, max_length=500)


@router.post("/suggest")
def suggest(payload: DiscoveryRequest | None = None):
    """Returns suggested companies for the user to review -- nothing is persisted here. Approving
    one calls POST /api/sources/resolve from the frontend. Companies already monitored are
    excluded so re-running discovery doesn't keep re-suggesting the same ones."""
    try:
        suggestions = suggest_companies(
            get_current_candidate_profile(),
            get_settings(),
            exclude_names=_already_monitored_company_names(),
            guidance=payload.guidance if payload else None,
            like=payload.like if payload else None,
        )
    except ValueError as exc:
        # _extract_json raises a plain ValueError if Claude's response has no parseable JSON
        # (e.g. cut off mid-output after a long web-search tool-use turn); pydantic.ValidationError
        # from a malformed suggestion also subclasses ValueError, so this catches both -- without
        # it, either case propagates as an unhandled exception and surfaces as a raw 500.
        raise HTTPException(
            502, f"Claude's response couldn't be parsed (likely cut off mid-output) — try again: {exc}"
        ) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return [s.model_dump() for s in suggestions]


@router.post("/suggest-boards")
def suggest_boards(payload: DiscoveryRequest | None = None):
    """Returns suggested multi-employer job boards for the user to review -- nothing is persisted
    here. Approving one calls POST /api/sources directly (kind=custom_board) from the frontend,
    since a board's URL is already known -- no resolution step needed. Boards already monitored
    are excluded by hostname."""
    try:
        suggestions = suggest_job_boards(
            get_current_candidate_profile(),
            get_settings(),
            exclude_urls=_already_monitored_board_urls(),
            guidance=payload.guidance if payload else None,
            like=payload.like if payload else None,
        )
    except ValueError as exc:
        raise HTTPException(
            502, f"Claude's response couldn't be parsed (likely cut off mid-output) — try again: {exc}"
        ) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return [s.model_dump() for s in suggestions]


# A discovery search takes 1-2 minutes and sends nothing back until it's done. Held open as one
# request, that silent wait got the connection dropped before the answer arrived (caught live in
# Safari: "Load failed" on "Find more like this"). So the UI starts a search in the background and
# polls for its result with short requests instead. Tasks live in this process's memory, which is
# fine with Cloud Run's single instance; a restart mid-search just 404s the poll.
TASK_TTL_SECONDS = 3600
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()


def _start_task(handler: Callable[[DiscoveryRequest | None], list], payload: DiscoveryRequest | None) -> str:
    task_id = uuid.uuid4().hex
    with _tasks_lock:
        cutoff = time.time() - TASK_TTL_SECONDS
        for old in [k for k, t in _tasks.items() if t["started"] < cutoff]:
            del _tasks[old]
        _tasks[task_id] = {"status": "running", "started": time.time(), "user_id": current_user_id()}

    def work():
        try:
            update = {"status": "done", "result": handler(payload)}
        except HTTPException as exc:
            update = {"status": "error", "error": exc.detail}
        except Exception as exc:  # noqa: BLE001 -- surfaced to the UI rather than lost in a thread
            update = {"status": "error", "error": f"Discovery failed: {exc}"}
        with _tasks_lock:
            _tasks[task_id].update(update)

    threading.Thread(target=carry(work), daemon=True).start()
    return task_id


@router.post("/suggest/start")
def start_suggest(payload: DiscoveryRequest | None = None):
    return {"task_id": _start_task(suggest, payload)}


@router.post("/suggest-boards/start")
def start_suggest_boards(payload: DiscoveryRequest | None = None):
    return {"task_id": _start_task(suggest_boards, payload)}


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    """{"status": "running"} until the search finishes, then "done" with `result` (the same list the
    synchronous routes return) or "error" with `error`."""
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None or task["user_id"] != current_user_id():
            raise HTTPException(404, "This search was interrupted (the server restarted) — try again.")
        return {k: v for k, v in task.items() if k not in ("started", "user_id")}
