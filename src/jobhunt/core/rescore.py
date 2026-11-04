"""Rescoring a whole list on request (asked 2026-11-04): My Jobs (every tracked job) or All Jobs
(every active job from the sources you follow). Scores normally happen once, when a job is new, so
a changed profile or "What are you looking for?" never reaches older jobs.

A rescore scores each job again with the current settings and stores a new `job_scores` row, the
same as a first score (the latest row is the one every page shows, so history is kept). It isn't
held to the daily cap, since it's an explicit request whose cost is shown first, but the monthly
budget still applies: `llm.get_client()` raises BudgetExceeded and the run stops there. Anyone but
the admin can rescore each list once a day (`last_rescore_<scope>` setting). One rescore at a time
per person; it runs in a background thread and the route streams its progress."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from jobhunt.core import budget, dealbreakers, queue
from jobhunt.core.generic_postings import is_generic_posting
from jobhunt.core.llm import BudgetExceeded
from jobhunt.core.scoring import score_job, store_score
from jobhunt.core.settings import get_setting, get_settings, update_settings
from jobhunt.core.users import carry, current_user_id, is_admin
from jobhunt.db.connection import get_db

logger = logging.getLogger(__name__)

SCOPES = ("my_jobs", "all_jobs")
COOLDOWN = timedelta(days=1)
CONCURRENCY = 4
# Used for the estimate until the person has scores of their own to average (measured 2026-11-03:
# 0.9 cents a job at the current settings).
DEFAULT_COST_PER_JOB = 0.01

_lock = threading.Lock()
_states: dict[int, dict] = {}


class RescoreUnavailable(Exception):
    """A rescore can't start now: the cooldown (`available_at` set), or one already running for
    the other list."""

    def __init__(self, message: str, available_at: str | None = None):
        super().__init__(message)
        self.available_at = available_at


def job_ids(scope: str) -> list[int]:
    """The jobs on that list for the current person, the same set the page shows."""
    uid = current_user_id()
    with get_db() as conn:
        if scope == "my_jobs":
            rows = conn.execute(
                "SELECT j.id, j.title FROM applications a JOIN jobs j ON j.id = a.job_id WHERE a.user_id = ?",
                (uid,),
            ).fetchall()
        elif scope == "all_jobs":
            rows = conn.execute(
                """SELECT j.id, j.title, j.location, j.remote_type, j.salary_min, j.salary_max,
                          j.salary_currency, j.salary_period FROM jobs j
                   WHERE j.status = 'active' AND j.source_id IS NOT NULL
                     AND j.source_id IN (SELECT source_id FROM source_follows WHERE user_id = ?)""",
                (uid,),
            ).fetchall()
        else:
            raise ValueError(f"unknown list: {scope}")
    # All Jobs hides jobs a dealbreaker rules out by default, so they aren't paid for.
    # My Jobs is never filtered.
    settings = get_settings() if scope == "all_jobs" else {}
    return [
        r["id"]
        for r in rows
        if not is_generic_posting(r["title"]) and not dealbreakers.blocks(dict(r), settings)
    ]


def cost_per_job() -> float:
    """This person's average scoring cost over their last 50 scores, or the default."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT AVG(estimated_cost_usd) AS avg, COUNT(*) AS n FROM (
                   SELECT estimated_cost_usd FROM llm_calls
                   WHERE user_id = ? AND call_type = 'scoring' AND estimated_cost_usd IS NOT NULL
                   ORDER BY id DESC LIMIT 50)""",
            (current_user_id(),),
        ).fetchone()
    return float(row["avg"]) if row["n"] >= 5 else DEFAULT_COST_PER_JOB


def available_at(scope: str) -> str | None:
    """When this person can next rescore this list, or None if they can now."""
    if is_admin():
        return None
    last = get_setting(f"last_rescore_{scope}")
    if not last:
        return None
    ready = datetime.fromisoformat(last) + COOLDOWN
    return ready.isoformat(timespec="seconds") if ready > datetime.now(timezone.utc) else None


def status(scope: str) -> dict:
    count = len(job_ids(scope))
    spend = budget.status(current_user_id())
    return {
        "scope": scope,
        "count": count,
        "estimate_usd": round(count * cost_per_job(), 2),
        "available_at": available_at(scope),
        "budget_remaining_usd": None if spend["exempt"] else round(spend["remaining"], 2),
        "running": get_state()["running"],
    }


def get_state() -> dict:
    with _lock:
        return dict(_states.get(current_user_id()) or {"running": False})


def start(scope: str) -> bool:
    """Starts a rescore of that list. False if this person's same rescore is already running (the
    caller follows it). Raises RescoreUnavailable during the cooldown or while another list's
    rescore is running."""
    uid = current_user_id()
    with _lock:
        state = _states.get(uid)
        if state and state["running"]:
            if state["scope"] == scope:
                return False
            raise RescoreUnavailable("Your other list is still being rescored. Try again when it's done.")
    ready = available_at(scope)
    if ready:
        raise RescoreUnavailable("cooling down", available_at=ready)
    ids = job_ids(scope)
    with _lock:
        _states[uid] = {
            "running": True,
            "scope": scope,
            "done": 0,
            "total": len(ids),
            "scored": 0,
            "failed": 0,
            "stopped": None,
            "error": None,
        }
    update_settings({f"last_rescore_{scope}": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    threading.Thread(target=carry(lambda: _run(ids)), daemon=True).start()
    return True


def _run(ids: list[int]) -> None:
    from jobhunt.pipeline.daily_run import _backfill_description, _score_args

    uid = current_user_id()
    stop = threading.Event()

    def one(job_id: int) -> None:
        if stop.is_set():
            return
        try:
            with get_db() as conn:
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is not None:
                score = score_job(_backfill_description(dict(row)), *args)
                store_score(job_id, score)
                queue.mark_scored(job_id)  # a queued job just scored here isn't paid for again
                outcome = "scored"
            else:
                outcome = None
        except BudgetExceeded as exc:
            stop.set()
            with _lock:
                _states[uid]["stopped"] = str(exc)
            return
        except Exception:  # noqa: BLE001 - one job's failure doesn't end the rescore
            logger.exception("rescoring job %s failed", job_id)
            outcome = "failed"
        with _lock:
            _states[uid]["done"] += 1
            if outcome:
                _states[uid][outcome] += 1

    try:
        args = _score_args()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            list(pool.map(carry(one), ids))
    except Exception as exc:  # noqa: BLE001 - reported to the page
        logger.exception("rescore failed")
        with _lock:
            _states[uid]["error"] = str(exc) or type(exc).__name__
    finally:
        with _lock:
            _states[uid]["running"] = False
