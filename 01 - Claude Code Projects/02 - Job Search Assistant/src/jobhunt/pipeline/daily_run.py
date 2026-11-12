"""Orchestrates the daily cycle: poll enabled sources -> dedupe/upsert jobs -> pre-filter and
queue new ones -> score up to the daily cap (highest-priority first) -> build today's digest ->
conditionally refresh the preference profile -> record a run summary. Triggered by Cloud Scheduler
via POST /internal/run-daily (see api/internal.py); safe to run manually/repeatedly."""

import contextvars
import itertools
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, date, datetime

from jobhunt.core import backup, budget, prefilter, queue
from jobhunt.core import settings as settings_module
from jobhunt.core.dedupe import compute_dedupe_key
from jobhunt.core.digest import add_to_digest
from jobhunt.core.email_digest import record_outcome as record_email_outcome
from jobhunt.core.email_digest import send_new_jobs_email, send_plain_email
from jobhunt.core.generic_postings import is_generic_posting
from jobhunt.core.llm import LLMError
from jobhunt.core.html_extract import html_to_markdown
from jobhunt.core.page_fetch import PageFetchError, fetch_page
from jobhunt.core.preference import get_current_preference_profile, rebuild_preference_profile
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.salary_extract import extract_salary
from jobhunt.core.scoring import score_job, score_jobs_batch, store_score
from jobhunt.core.screening import screen_pending
from jobhunt.core.users import (
    acting_as,
    active_user_ids,
    mark_paused_email_sent,
    newly_inactive_users,
    current_user_id,
    get_user,
    optional_user_id,
    reset_current_user,
    set_current_user,
)
from jobhunt.db.connection import get_db
from jobhunt.ingest.adapters import adzuna, ashby, custom_board, greenhouse, lever, remoteok

logger = logging.getLogger(__name__)

ADAPTERS = {
    "ats_greenhouse": greenhouse.fetch,
    "ats_lever": lever.fetch,
    "ats_ashby": ashby.fetch,
    "aggregator_adzuna": adzuna.fetch,
    "aggregator_remoteok": remoteok.fetch,
    "custom_board": custom_board.fetch,
}

STALE_AFTER_DAYS = 14
REBUILD_FEEDBACK_THRESHOLD = 5
REBUILD_MAX_DAYS = 7

# Sources are polled, and jobs scored, concurrently -- done one at a time, a run took many minutes
# (onboarding's last step waits on a whole run). Scoring is capped at 4 in flight because a new API
# account's per-minute limits are low; the SDK retries an occasional 429 with backoff. Polling is
# mostly network waits (page fetches, sometimes Playwright plus a Claude extraction), so more is fine.
SCORING_CONCURRENCY = 4
POLL_CONCURRENCY = 6

# Live progress of a run, for the UI to poll (GET /api/pipeline/progress) while its run-daily
# request is still open. In-process state is the real state: Cloud Run runs a single instance
# (--max-instances 1). Kept per person who started a run (None for the scheduler's), so each person
# follows their own; a run's dict travels with it in a context variable, through its worker threads.
_progress_lock = threading.Lock()
_progress_by_owner: dict[int | None, dict] = {}
_run_progress: contextvars.ContextVar[dict | None] = contextvars.ContextVar("run_progress", default=None)


def _new_progress() -> dict:
    return {
        "stage": "idle",
        "sources_done": 0,
        "sources_total": 0,
        "jobs_done": 0,
        "jobs_total": 0,
        # Names of sources being read right now, so a slow one is visible ("still reading: ...").
        "sources_pending": [],
        # Set by start_run_daily()'s background runs, so a page can tell which run finished and how.
        "run_id": None,
        "finished_run_id": None,
        "last_summary": None,
        "last_error": None,
    }


def _owner_progress(owner: int | None) -> dict:
    with _progress_lock:
        return _progress_by_owner.setdefault(owner, _new_progress())


def _current_progress() -> dict:
    return _run_progress.get() or _owner_progress(optional_user_id())


def get_progress() -> dict:
    """The signed-in person's latest run."""
    progress = _owner_progress(optional_user_id())
    with _progress_lock:
        return dict(progress)


def _set_progress(**updates) -> None:
    progress = _current_progress()
    with _progress_lock:
        progress.update(updates)


def _bump_progress(key: str) -> None:
    progress = _current_progress()
    with _progress_lock:
        progress[key] += 1


def _map_in_context(pool: ThreadPoolExecutor, fn, items: list) -> list:
    """pool.map, with each call running in a copy of this thread's context (the person the run is
    working for, and the run's progress); worker threads otherwise start with neither."""
    futures = [pool.submit(contextvars.copy_context().run, fn, item) for item in items]
    return [f.result() for f in futures]


def _fetch_source(source: dict) -> tuple[list[dict] | None, str | None]:
    """Runs one source's adapter (in a worker thread). Returns (postings, error); postings is None
    for a failed fetch or a kind with no adapter."""
    progress = _current_progress()
    with _progress_lock:
        progress["sources_pending"] = [*progress["sources_pending"], source["name"]]
    try:
        adapter = ADAPTERS.get(source["kind"])
        if adapter is None:
            return None, None
        try:
            return adapter(source), None
        except Exception as exc:  # noqa: BLE001 - one bad source must not abort the whole run
            with get_db() as conn:
                conn.execute(
                    "UPDATE sources SET last_polled_at = datetime('now'), last_error = ? WHERE id = ?",
                    (str(exc), source["id"]),
                )
            return None, f"{source['name']}: {exc}"
    finally:
        with _progress_lock:
            pending = list(progress["sources_pending"])
            if source["name"] in pending:
                pending.remove(source["name"])
            progress["sources_pending"] = pending
        _bump_progress("sources_done")


# The sources worth reading: those followed (and switched on) by at least one approved person.
_FOLLOWED_SOURCES_SQL = """SELECT s.* FROM sources s WHERE EXISTS (
    SELECT 1 FROM source_follows f JOIN users u ON u.id = f.user_id
     WHERE f.source_id = s.id AND f.enabled = 1 AND u.status = 'approved' {who})"""


def _poll_sources(
    source_ids: list[int] | None = None, user_ids: list[int] | None = None, fresh_hours: float | None = None
) -> dict:
    """Reads each source once, however many people follow it: the given sources, or else every
    source followed by `user_ids` (default: anyone approved). With `fresh_hours`, a source read
    successfully within that many hours (by anyone's run) isn't read again: its recent jobs are
    taken from what's already stored, so the person still gets them queued and scored."""
    who = ""
    if user_ids is not None:
        who = f"AND f.user_id IN ({','.join(str(int(u)) for u in user_ids) or 'NULL'})"
    with get_db() as conn:
        if source_ids is not None:
            marks = ",".join("?" * len(source_ids))
            rows = (
                conn.execute(f"SELECT * FROM sources WHERE id IN ({marks})", source_ids) if source_ids else []
            )
        else:
            rows = conn.execute(_FOLLOWED_SOURCES_SQL.format(who=who))
        sources = [dict(row) for row in rows]
    for source in sources:
        source["config"] = json.loads(source.pop("config_json") or "{}")
    jobs_by_source: dict[int, list[int]] = {}
    fresh = [s for s in sources if fresh_hours and _read_within(s, fresh_hours)]
    if fresh:
        sources = [s for s in sources if s not in fresh]
        jobs_by_source.update(_recent_jobs_for_sources([s["id"] for s in fresh]))
    _set_progress(stage="polling", sources_done=0, sources_total=len(sources), sources_pending=[])

    fetched = 0
    upserted = 0
    errors = []
    job_ids = []

    # Fetches run concurrently; upserts stay sequential here, in source order -- two sources can
    # produce the same URL-based dedupe key, and _upsert_job's select-then-insert would race.
    with ThreadPoolExecutor(max_workers=POLL_CONCURRENCY) as pool:
        results = _map_in_context(pool, _fetch_source, sources)

    for source, (postings, error) in zip(sources, results, strict=True):
        if error:
            errors.append(error)
        if postings is None:
            continue

        if not postings:
            # A source with active jobs that suddenly lists none is far more likely a read that went
            # wrong (a board that didn't render in time) than a board emptied overnight: say so,
            # rather than count it a success and leave its jobs as they were (asked 2026-11-12).
            with get_db() as conn:
                had = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE source_id = ? AND status = 'active'", (source["id"],)
                ).fetchone()[0]
                if had:
                    message = f"found no postings this time (it had {had}); will try again next check"
                    conn.execute(
                        "UPDATE sources SET last_polled_at = datetime('now'), last_error = ? WHERE id = ?",
                        (message, source["id"]),
                    )
                    errors.append(f"{source['name']}: {message}")
                    continue

        fetched += len(postings)
        seen: list[int] = []
        for posting in postings:
            job_id = _upsert_job(source["id"], posting)
            job_ids.append(job_id)
            seen.append(job_id)
            jobs_by_source.setdefault(source["id"], []).append(job_id)
            upserted += 1
        if postings and (source["kind"] in COMPLETE_KINDS or source.get("read_complete")):
            _close_unlisted(source["id"], seen)

        with get_db() as conn:
            conn.execute(
                "UPDATE sources SET last_polled_at = datetime('now'), last_success_at = datetime('now'), "
                "last_error = NULL WHERE id = ?",
                (source["id"],),
            )

    return {
        "sources_polled": len(sources),
        "sources_fresh": len(fresh),
        "jobs_fetched": fetched,
        "jobs_upserted": upserted,
        "errors": errors,
        "job_ids": job_ids,
        "jobs_by_source": jobs_by_source,
    }


def _read_within(source: dict, hours: float) -> bool:
    if not source.get("last_success_at"):
        return False
    with get_db() as conn:
        row = conn.execute(
            "SELECT julianday('now') - julianday(?) < ? / 24.0 AS fresh", (source["last_success_at"], hours)
        ).fetchone()
    return bool(row["fresh"])


def _recent_jobs_for_sources(source_ids: list[int]) -> dict[int, list[int]]:
    """Active jobs of these sources seen in the last day: what a skipped (just-read) source would
    have returned."""
    marks = ",".join("?" * len(source_ids))
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT id, source_id FROM jobs WHERE source_id IN ({marks}) AND status = 'active' "
            "AND julianday('now') - julianday(last_seen_at) <= 1",
            source_ids,
        ).fetchall()
    found: dict[int, list[int]] = {}
    for row in rows:
        found.setdefault(row["source_id"], []).append(row["id"])
    return found


SALARY_FIELDS = ("salary_min", "salary_max", "salary_currency", "salary_period")


def _with_salary(posting: dict) -> dict:
    """The adapter's own salary if it had one (an ATS pay field, or what Claude read off a board's
    listing page), else whatever the description states (core/salary_extract.py)."""
    if posting.get("salary_min") or posting.get("salary_max"):
        return posting
    found = extract_salary(posting.get("description_text"))
    return {**posting, **found} if found else posting


def _upsert_job(source_id: int, posting: dict) -> int:
    posting = _with_salary(posting)
    dedupe_key = compute_dedupe_key(
        url=posting.get("url"),
        company=posting.get("company") or "",
        title=posting.get("title") or "",
        location=posting.get("location"),
        source_id=source_id,
        external_id=posting.get("external_id"),
    )
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM jobs WHERE dedupe_key = ?", (dedupe_key,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE jobs SET last_seen_at = datetime('now'), status = 'active' WHERE id = ?",
                (existing["id"],),
            )
            if posting.get("description_text"):
                # Keeps a re-seen job's description current, which is also how jobs saved before
                # descriptions kept their formatting (core/html_extract.py) pick it up.
                conn.execute(
                    "UPDATE jobs SET description_text = ? WHERE id = ? AND description_text IS NOT ?",
                    (posting["description_text"], existing["id"], posting["description_text"]),
                )
            if posting.get("location"):
                # Keep where it is current too: an adapter fix (Ashby's secondaryLocations and
                # workplaceType, 2026-11-12) reaches jobs saved before it on their next check.
                conn.execute(
                    "UPDATE jobs SET location = ?, remote_type = ? "
                    "WHERE id = ? AND (location IS NOT ? OR remote_type IS NOT ?)",
                    (
                        posting["location"],
                        posting.get("remote_type"),
                        existing["id"],
                        posting["location"],
                        posting.get("remote_type"),
                    ),
                )
            if posting.get("salary_min") or posting.get("salary_max"):
                # Fill a salary the job was first saved without; never overwrite one it has.
                conn.execute(
                    "UPDATE jobs SET salary_min = ?, salary_max = ?, salary_currency = ?, salary_period = ? "
                    "WHERE id = ? AND salary_min IS NULL AND salary_max IS NULL",
                    (*(posting.get(f) for f in SALARY_FIELDS), existing["id"]),
                )
            return existing["id"]

        cursor = conn.execute(
            """INSERT INTO jobs
               (source_id, external_id, dedupe_key, title, company, location, remote_type,
                salary_min, salary_max, salary_currency, salary_period, description_text,
                raw_json, url, posted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source_id,
                posting.get("external_id"),
                dedupe_key,
                posting["title"],
                posting["company"],
                posting.get("location"),
                posting.get("remote_type"),
                posting.get("salary_min"),
                posting.get("salary_max"),
                posting.get("salary_currency"),
                posting.get("salary_period"),
                posting.get("description_text"),
                json.dumps(posting.get("raw_json")),
                posting.get("url"),
                posting.get("posted_at"),
            ),
        )
        return cursor.lastrowid


# Sources read from an ATS's API, which lists every open posting: a job missing from a read of one
# has been taken down. (A web page read by Claude can list only some, so it never closes anything;
# those age out after STALE_AFTER_DAYS.) A custom board read through its embedded ATS board sets
# read_complete on the source (custom_board.py).
COMPLETE_KINDS = {"ats_greenhouse", "ats_lever", "ats_ashby"}


def _close_unlisted(source_id: int, listed: list[int]) -> None:
    """Closes the source's active jobs that a complete read didn't list (asked 2026-11-12: a posting
    taken down stayed on All Jobs until it aged out). Seen again, a job reopens (_upsert_job)."""
    with get_db() as conn:
        conn.execute(
            f"""UPDATE jobs SET status = 'expired'
                 WHERE source_id = ? AND status = 'active'
                   AND id NOT IN ({",".join("?" * len(listed)) or "NULL"})""",
            (source_id, *listed),
        )


def _expire_stale_jobs() -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'expired' WHERE status = 'active' AND source_id IS NOT NULL "
            "AND julianday('now') - julianday(last_seen_at) > ?",
            (STALE_AFTER_DAYS,),
        )


def _prefilter_and_enqueue(job_ids: list[int]) -> int:
    current_settings = settings_module.get_settings()
    enqueued = 0
    with get_db() as conn:
        for job_id in set(job_ids):
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                continue
            already_scored = conn.execute(
                "SELECT 1 FROM job_scores WHERE user_id = ? AND job_id = ?", (current_user_id(), job_id)
            ).fetchone()
            if already_scored or is_generic_posting(row["title"]):
                continue  # a general application / talent pool isn't a job worth scoring
            score, reasons = prefilter.score_job(dict(row), current_settings)
            queue.enqueue(job_id, score, reasons)
            enqueued += 1
    return enqueued


def _backfill_description(job: dict) -> dict:
    if job.get("description_text") or not job.get("url"):
        return job
    try:
        page = fetch_page(job["url"])
    except PageFetchError:
        return job
    # Formatted from the page's HTML; the plain text is the fallback if that comes out (nearly) empty.
    text = html_to_markdown(page.html, main_content=True)
    if len(text) < len(page.text) // 4:
        text = page.text
    with get_db() as conn:
        conn.execute("UPDATE jobs SET description_text = ? WHERE id = ?", (text, job["id"]))
    job["description_text"] = text
    if not (job.get("salary_min") or job.get("salary_max")) and (found := extract_salary(text)):
        _store_salary(job["id"], found)
        job.update(found)
    return job


def _store_salary(job_id: int, salary: dict) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET salary_min = ?, salary_max = ?, salary_currency = ?, salary_period = ? "
            "WHERE id = ?",
            (*(salary.get(f) for f in SALARY_FIELDS), job_id),
        )


def backfill_salaries() -> int:
    """Reads a salary for every active job still without one: from its description, or failing
    that from its score summary, which states the pay when the posting did (the summary is
    Claude's own reading of the whole posting). Regex only, so it's free; runs at app startup and
    in every daily run, which catches jobs saved before pay was read."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT j.id, j.description_text, js.summary_text
               FROM jobs j
               LEFT JOIN job_scores js ON js.id = (
                   SELECT id FROM job_scores WHERE job_id = j.id ORDER BY id DESC LIMIT 1
               )
               WHERE j.status = 'active' AND j.salary_min IS NULL AND j.salary_max IS NULL
                 AND (j.description_text IS NOT NULL OR js.summary_text IS NOT NULL)"""
        ).fetchall()
    filled = 0
    for row in rows:
        if found := extract_salary(row["description_text"]) or extract_salary(row["summary_text"]):
            _store_salary(row["id"], found)
            filled += 1
    return filled


def _maybe_rebuild_preference_profile() -> bool:
    """For the current person: rebuilds their preference profile once enough new signal (feedback
    marks, application status changes) has come in, or it's a week old."""
    user_id = current_user_id()
    with get_db() as conn:
        last = conn.execute(
            "SELECT based_on_feedback_count, based_on_application_events_count, "
            "(julianday('now') - julianday(updated_at)) AS days_since "
            "FROM preference_profile WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        feedback_count = conn.execute(
            "SELECT COUNT(*) AS n FROM feedback WHERE user_id = ?", (user_id,)
        ).fetchone()["n"]
        app_events = conn.execute(
            "SELECT COUNT(*) AS n FROM application_status_history h JOIN applications a "
            "ON a.id = h.application_id WHERE a.user_id = ?",
            (user_id,),
        ).fetchone()["n"]

    if feedback_count == 0 and app_events == 0:
        return False
    if last is None:
        rebuild_preference_profile()
        return True

    new_signal = (feedback_count - (last["based_on_feedback_count"] or 0)) + (
        app_events - (last["based_on_application_events_count"] or 0)
    )
    if new_signal >= REBUILD_FEEDBACK_THRESHOLD or last["days_since"] >= REBUILD_MAX_DAYS:
        rebuild_preference_profile()
        return True
    return False


def _score_one(job_id: int, score_args: tuple) -> tuple[bool, str | None]:
    """Scores one queued job for the current person (in a worker thread). Returns (scored, error)."""
    try:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return False, None
        job = _backfill_description(dict(row))
        try:
            score = score_job(job, *score_args)
        except LLMError as exc:
            queue.mark_pending_again(job_id)
            return False, f"job {job_id}: {exc}"
        store_score(job_id, score)
        queue.mark_scored(job_id)
        return True, None
    finally:
        _bump_progress("jobs_done")


def _score_args() -> tuple:
    return (
        get_current_candidate_profile(),
        get_current_preference_profile(),
        settings_module.get_setting("prefilter_sectors", []),
        settings_module.get_setting("prefilter_titles_include", []),
        settings_module.get_setting("salary_floor", None),
        settings_module.get_setting("looking_for_notes", ""),
    )


# The 2pm run scores through the Message Batches API at half price (core/scoring.py), within this
# much of the run's time in all (every person's batch shares it), so the run still ends well inside
# Cloud Scheduler's 30-minute deadline. Past it, or for a handful of jobs, scoring is live.
BATCH_TIME_BUDGET_SECONDS = 18 * 60
BATCH_MIN_JOBS = 5


def _score_batch(
    batch: list[int], score_args: tuple, deadline: float
) -> tuple[list[int], list[int], list[str]]:
    """Scores what it can of `batch` in one Message Batch. Returns (scored ids, ids left to
    score live, errors)."""
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM jobs WHERE id IN ({','.join('?' * len(batch))})", batch
        ).fetchall()
    with ThreadPoolExecutor(max_workers=SCORING_CONCURRENCY) as pool:
        jobs = _map_in_context(pool, lambda row: _backfill_description(dict(row)), rows)
    try:
        outcome = score_jobs_batch(
            jobs, score_args, deadline, on_progress=lambda done: _set_progress(jobs_done=done)
        )
    except LLMError as exc:
        return [], list(batch), [f"scoring batch: {exc}"]
    scored = []
    for job_id, score in outcome.scores.items():
        store_score(job_id, score)
        queue.mark_scored(job_id)
        scored.append(job_id)
    return scored, outcome.unfinished, outcome.errors


def _score_and_digest(batch: list[int], batch_deadline: float | None = None) -> tuple[int, list[str]]:
    """Scores the batch for the current person concurrently, then adds each scored job to their
    digest for today. Returns (scored count, errors). One person's jobs are scored together, so
    their (identical) candidate context stays in Claude's prompt cache across the batch."""
    # Queued before general-application postings were recognized: skip rather than pay to score.
    placeholders = ",".join("?" * len(batch))
    with get_db() as conn:
        rows = conn.execute(f"SELECT id, title FROM jobs WHERE id IN ({placeholders})", batch)
        titles = {row["id"]: row["title"] for row in rows}
    generic = [job_id for job_id in batch if is_generic_posting(titles.get(job_id))]
    for job_id in generic:
        queue.mark_skipped(job_id)
    batch = [job_id for job_id in batch if job_id not in generic]

    _set_progress(stage="scoring", jobs_done=0, jobs_total=len(batch))
    score_args = _score_args()
    scoring_errors = []
    scored_ids = []
    if batch_deadline is not None and len(batch) >= BATCH_MIN_JOBS and time.monotonic() < batch_deadline:
        scored_ids, batch, scoring_errors = _score_batch(batch, score_args, batch_deadline)
    if batch:
        with ThreadPoolExecutor(max_workers=min(SCORING_CONCURRENCY, len(batch))) as pool:
            results = _map_in_context(pool, lambda job_id: _score_one(job_id, score_args), batch)
        for job_id, (scored, error) in zip(batch, results, strict=True):
            if scored:
                scored_ids.append(job_id)
            if error:
                scoring_errors.append(error)
    # add_to_digest re-ranks the whole day on every call, so it runs here, sequentially, rather
    # than in the workers where concurrent re-ranks could interleave.
    for job_id in scored_ids:
        add_to_digest(job_id)
    return len(scored_ids), scoring_errors


def scored_today() -> int:
    """How many jobs the current person has had scored today (their daily cap counts these)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM job_scores WHERE user_id = ? AND date(scored_at) = date('now')",
            (current_user_id(),),
        ).fetchone()
    return row["n"]


def remaining_cap(cap: int | None = None) -> int:
    """What's left of the current person's daily cap (users.daily_score_cap, set by the admin),
    further limited by `cap` when a caller asks for fewer (onboarding's first run)."""
    left = max(0, get_user(current_user_id())["daily_score_cap"] - scored_today())
    return min(left, cap) if cap is not None else left


def _followed_source_ids(user_id: int) -> set[int]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT source_id FROM source_follows WHERE user_id = ? AND enabled = 1", (user_id,)
        ).fetchall()
    return {row["source_id"] for row in rows}


def _record_run(summary: dict) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO runs
               (user_id, run_date, started_at, finished_at, sources_polled, jobs_fetched, jobs_deduped,
                prefiltered_count, llm_scored_count, cap_used, carried_over_count, errors_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                optional_user_id(),
                date.today().isoformat(),
                summary["started_at"],
                summary["finished_at"],
                summary["sources_polled"],
                summary["jobs_fetched"],
                summary["jobs_upserted"],
                summary["prefiltered_count"],
                summary["llm_scored_count"],
                summary["cap_used"],
                summary["carried_over_count"],
                json.dumps(summary["errors"]),
            ),
        )


def _run_for_person(
    user_id: int,
    poll_result: dict,
    started_at: str,
    cap: int | None,
    email: bool = True,
    batch_deadline: float | None = None,
) -> dict:
    """Everything after polling, for one person: queue the new jobs from sources they follow,
    score up to what's left of their daily cap, build their New & Noteworthy, refresh their
    preference profile if due, and send their email."""
    with acting_as(user_id):
        followed = _followed_source_ids(user_id)
        job_ids = [
            j
            for source_id, ids in poll_result["jobs_by_source"].items()
            if source_id in followed
            for j in ids
        ]
        enqueued = _prefilter_and_enqueue(job_ids)

        scored_count, scoring_errors, preference_rebuilt = 0, [], False
        allowed = remaining_cap(cap)
        if get_current_candidate_profile() is None:
            # Nothing to score against until they've set up a profile (onboarding builds one).
            allowed = 0
        if allowed and budget.status(user_id)["over"]:
            # This month's budget is used up: skip the batch rather than fail job by job.
            allowed = 0
            scoring_errors.append(budget.exceeded_message(budget.status(user_id)["monthly_budget_usd"]))
        screened = None
        if allowed:
            # Set aside clear title mismatches first, so the cap goes to plausible jobs.
            screened = screen_pending(sorted(followed))
            scoring_errors.extend(screened["errors"])
            batch = queue.select_batch(allowed, source_ids=sorted(followed) or [-1])
            scored_count, errors = _score_and_digest(batch, batch_deadline)
            scoring_errors.extend(errors)
        try:
            preference_rebuilt = _maybe_rebuild_preference_profile()
        except Exception as exc:  # noqa: BLE001 - an optional step; the run carries on without it
            logger.exception("preference profile rebuild for user %s failed", user_id)
            scoring_errors.append(f"preference profile: {describe_error(exc)}")

        email_result = None
        if email:
            # The email goes out once the digest is built. A mail failure must never fail the run.
            try:
                email_result = send_new_jobs_email()
            except Exception as exc:  # noqa: BLE001
                logger.exception("daily email for user %s failed", user_id)
                email_result = {"sent": False, "reason": f"error: {exc}"}
            try:  # what happened, for Settings' Daily email card; never fails the run
                record_email_outcome(email_result)
            except Exception:  # noqa: BLE001
                logger.exception("recording the email outcome for user %s failed", user_id)

        summary = {
            "user_id": user_id,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "sources_polled": len(followed & set(poll_result["jobs_by_source"])),
            "jobs_fetched": len(job_ids),
            "jobs_upserted": len(job_ids),
            "prefiltered_count": enqueued,
            "llm_scored_count": scored_count,
            "cap_used": allowed,
            "carried_over_count": queue.pending_count(),
            "errors": scoring_errors,
            "preference_profile_rebuilt": preference_rebuilt,
            "screened": screened,
            "email": email_result,
            # Today's cap is used up: New & Noteworthy says scoring resumes tomorrow.
            "cap_reached": remaining_cap() == 0,
        }
        _record_run(summary)
        return summary


def run_daily(cap: int | None = None, only_user: int | None = None, fresh_hours: float | None = None) -> dict:
    """The daily check: every source anyone approved follows is read once, then each person's
    jobs are scored (up to what's left of their daily cap), digested and emailed. With
    `only_user`, just that person's sources and jobs ("Check my sources now", onboarding's last
    step). `cap` further limits scoring for this run only -- onboarding scores just the top few so
    the user isn't kept waiting; the rest stay queued for the next run."""
    # Pin this run's progress to whoever started it, before the run switches between people.
    progress_token = _run_progress.set(_current_progress())
    try:
        with _run_lock:
            started_at = datetime.now(UTC).isoformat()
            _set_progress(stage="starting", sources_done=0, sources_total=0, jobs_done=0, jobs_total=0)
            # The everyone run (scheduled, or the admin's "Check every source now") gets a row of
            # its own, written as it starts, so a run that crashes, or whose server is shut down
            # mid-run, still shows on the Admin page (2026-11-02: three days of 500s went unseen).
            system_run = _system_run_started(started_at) if only_user is None else None
            try:
                summary = _run_daily(started_at, cap, only_user, fresh_hours)
            except Exception as exc:
                logger.exception("daily run failed")
                if system_run is not None:
                    _system_run_finished(system_run, None, describe_error(exc))
                raise
            else:
                if system_run is not None:
                    _system_run_finished(system_run, summary, None)
                return summary
            finally:
                progress = _current_progress()
                with _progress_lock:
                    if progress["stage"] != "done":
                        progress["stage"] = "failed"
    finally:
        _run_progress.reset(progress_token)


# The UI's "Check every source now" and onboarding's last step run the pipeline in a thread of its
# own, not in the request: a dropped connection (the page's request used to sit silent for the
# whole multi-minute run and got cut -- "Lost the connection", caught live) must not stop the run,
# and a second click while one is going just follows the one in progress instead of paying for two.
_run_ids = itertools.count(1)
_active_runs: dict[int | None, int] = {}
_active_lock = threading.Lock()


def start_run_daily(
    cap: int | None = None, only_user: int | None = None, fresh_hours: float | None = None
) -> int:
    """Starts a run in the background for whoever's signed in (`only_user` limits it to one
    person), or returns the id of their run already in progress."""
    owner = optional_user_id()
    with _active_lock:
        if owner in _active_runs:
            return _active_runs[owner]
        run_id = next(_run_ids)
        _active_runs[owner] = run_id
    progress = _owner_progress(owner)
    with _progress_lock:
        progress.update(
            run_id=run_id, stage="starting", sources_done=0, sources_total=0, jobs_done=0, jobs_total=0
        )
    context = contextvars.copy_context()
    threading.Thread(
        target=context.run,
        args=(_run_in_background, run_id, cap, only_user, owner, fresh_hours),
        daemon=True,
    ).start()
    return run_id


def _run_in_background(
    run_id: int, cap: int | None, only_user: int | None, owner: int | None, fresh_hours: float | None = None
) -> None:
    progress = _owner_progress(owner)
    _run_progress.set(progress)
    summary, error = None, None
    try:
        summary = run_daily(cap=cap, only_user=only_user, fresh_hours=fresh_hours)
    except Exception as exc:  # noqa: BLE001 - reported to the page instead of lost in a thread
        logger.exception("daily run %s failed", run_id)
        error = str(exc)
    finally:
        with _active_lock:
            _active_runs.pop(owner, None)
        with _progress_lock:
            progress.update(finished_run_id=run_id, last_summary=summary, last_error=error)


# A newly added (or re-enabled) source is checked right away instead of at the next daily run:
# fetch just that source, score up to NEW_SOURCE_SCORE_CAP of its jobs (~3 cents each, within the
# person's daily cap), and add them to New & Noteworthy. Asked for live: new sources sat empty until
# the next morning. Adds in quick succession (Monitoring several suggestions) are gathered into one
# refresh per person by waiting REFRESH_GATHER_SECONDS before starting. Runs never overlap
# (_run_lock), so a refresh started during a daily run waits for it.
NEW_SOURCE_SCORE_CAP = 10
REFRESH_GATHER_SECONDS = 3.0
_run_lock = threading.Lock()
_pending_refresh: dict[int, set[int]] = {}
_pending_lock = threading.Lock()
_refresh_worker: threading.Thread | None = None


def schedule_source_refresh(source_id: int) -> None:
    """Queues a background check of a source the current person just added. Skipped during
    onboarding, whose last step runs the full pipeline over their sources anyway."""
    global _refresh_worker
    if not settings_module.get_setting("onboarding_completed", False):
        return
    with _pending_lock:
        _pending_refresh.setdefault(current_user_id(), set()).add(source_id)
        if _refresh_worker is None or not _refresh_worker.is_alive():
            _refresh_worker = threading.Thread(target=_refresh_loop, daemon=True)
            _refresh_worker.start()


def _refresh_loop() -> None:
    while True:
        time.sleep(REFRESH_GATHER_SECONDS)
        with _pending_lock:
            pending = {user_id: sorted(ids) for user_id, ids in _pending_refresh.items() if ids}
            _pending_refresh.clear()
            if not pending:
                return
        for user_id, source_ids in pending.items():
            try:
                with acting_as(user_id):
                    refresh_sources(source_ids)
            except Exception:  # noqa: BLE001 - a failed refresh just leaves it to the next daily run
                logger.exception("refreshing new sources %s for user %s failed", source_ids, user_id)


def refresh_sources(source_ids: list[int], cap: int = NEW_SOURCE_SCORE_CAP) -> dict:
    """For the current person: polls only these sources, then scores up to `cap` of their new jobs
    (within what's left of the daily cap) into today's digest."""
    progress_token = _run_progress.set(_current_progress())
    try:
        return _refresh_sources(source_ids, cap)
    finally:
        _run_progress.reset(progress_token)


def _refresh_sources(source_ids: list[int], cap: int) -> dict:
    with _run_lock:
        poll_result = _poll_sources(source_ids)
        job_ids = [j for s in source_ids for j in poll_result["jobs_by_source"].get(s, [])]
        enqueued = _prefilter_and_enqueue(job_ids)
        scored, scoring_errors = 0, []
        allowed = remaining_cap(cap)
        if (
            allowed
            and get_current_candidate_profile() is not None
            and not budget.status(current_user_id())["over"]
        ):
            scoring_errors.extend(screen_pending(source_ids)["errors"])
            batch = queue.select_batch(allowed, source_ids=source_ids)
            scored, errors = _score_and_digest(batch)
            scoring_errors.extend(errors)
        _set_progress(stage="done")
    return {
        "sources_polled": poll_result["sources_polled"],
        "jobs_fetched": poll_result["jobs_fetched"],
        "prefiltered_count": enqueued,
        "llm_scored_count": scored,
        "errors": poll_result["errors"] + scoring_errors,
    }


def _run_daily(
    started_at: str, cap: int | None, only_user: int | None = None, fresh_hours: float | None = None
) -> dict:
    everyone = only_user is None
    if everyone:
        # Before the run writes anything, so a run that goes wrong can be rolled back (core/backup.py).
        backup.snapshot()
    queue.requeue_stranded()
    # Everyone's run covers only people active in the last week (users.ACTIVE_DAYS); anyone who just
    # lapsed is told once that their matches are paused.
    people = [only_user] if only_user is not None else active_user_ids()
    paused_emails = _send_paused_emails() if everyone else 0
    # Reading the shared sources is nobody's own spend on the scheduled run (llm_calls.user_id NULL);
    # someone's own "check my sources" is theirs.
    with acting_as(only_user) if only_user is not None else _nobody():
        poll_result = _poll_sources(user_ids=people, fresh_hours=fresh_hours)
    if everyone:
        _expire_stale_jobs()
        backfill_salaries()

    # The everyone run (the 2pm one) scores in batches at half price; its batches share one time
    # budget. Someone's own check, waiting on the page, scores live.
    batch_deadline = time.monotonic() + BATCH_TIME_BUDGET_SECONDS if everyone else None
    per_person = [
        _run_for_person_safely(user_id, poll_result, started_at, cap, batch_deadline) for user_id in people
    ]

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "sources_polled": poll_result["sources_polled"],
        "jobs_fetched": poll_result["jobs_fetched"],
        "jobs_upserted": poll_result["jobs_upserted"],
        "prefiltered_count": sum(p["prefiltered_count"] for p in per_person),
        "llm_scored_count": sum(p["llm_scored_count"] for p in per_person),
        "cap_used": sum(p["cap_used"] for p in per_person),
        "carried_over_count": sum(p["carried_over_count"] for p in per_person),
        "errors": poll_result["errors"] + [e for p in per_person for e in p["errors"]],
        "preference_profile_rebuilt": any(p["preference_profile_rebuilt"] for p in per_person),
        "email": per_person[0]["email"] if len(per_person) == 1 else [p["email"] for p in per_person],
        "people": len(per_person),
        "sources_fresh": poll_result["sources_fresh"],
        "cap_reached": all(p["cap_reached"] for p in per_person) if per_person else False,
        "paused_emails": paused_emails,
    }
    _set_progress(stage="done")
    return summary


@contextmanager
def _nobody():
    token = set_current_user(None)
    try:
        yield
    finally:
        reset_current_user(token)


def _send_paused_emails() -> int:
    """Tells each person who just passed a week away, once, that their daily matches are paused.
    A send that fails or isn't configured leaves them unmarked, so the next run tries again."""
    app_url = (os.environ.get("APP_URL") or "").rstrip("/")
    sent = 0
    for user in newly_inactive_users():
        result = send_plain_email(
            user["email"],
            "Your daily job matches are paused",
            f"Hi {user['name'] or 'there'},\n\nYour daily job matches are paused because you haven't "
            "opened Job Search Assistant in a week. Open it to pick them back up:\n\n"
            f"{app_url or '(the app link)'}\n",
        )
        if result.get("sent"):
            mark_paused_email_sent(user["id"])
            sent += 1
    return sent


def _system_run_started(started_at: str) -> int:
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO runs (user_id, run_date, started_at, status, started_by) "
            "VALUES (NULL, ?, ?, 'running', ?)",
            (date.today().isoformat(), started_at, "scheduled" if optional_user_id() is None else "admin"),
        ).lastrowid


def _system_run_finished(run_id: int, summary: dict | None, error: str | None) -> None:
    summary = summary or {}
    with get_db() as conn:
        conn.execute(
            """UPDATE runs SET finished_at = ?, status = ?, error = ?, sources_polled = ?, jobs_fetched = ?,
                   llm_scored_count = ?, errors_json = ?
               WHERE id = ?""",
            (
                datetime.now(UTC).isoformat(),
                "failed" if error else "ok",
                error,
                summary.get("sources_polled"),
                summary.get("jobs_fetched"),
                summary.get("llm_scored_count"),
                json.dumps({"errors": summary.get("errors", []), "people": summary.get("people")}),
                run_id,
            ),
        )


def describe_error(exc: BaseException) -> str:
    """ "Type: message (file.py:line in function)", from the traceback's last frame, so an error
    with no message (2026-11-03's bare "StopIteration:") still says where it happened."""
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        last = frames[-1]
        text += f" ({os.path.basename(last.filename)}:{last.lineno} in {last.name})"
    return text


def _run_for_person_safely(
    user_id: int, poll_result: dict, started_at: str, cap: int | None, batch_deadline: float | None = None
) -> dict:
    """One person's part of the run. If it fails, the error goes into the run's summary (and so
    onto the Admin page's Daily check card) and the run moves on to the next person, rather than
    one person's problem taking down everyone's run. No `runs` row is written for them: their
    "Last updated" must not claim a check that didn't happen."""
    try:
        return _run_for_person(user_id, poll_result, started_at, cap, batch_deadline=batch_deadline)
    except Exception as exc:  # noqa: BLE001
        logger.exception("daily run for user %s failed", user_id)
        error = f"user {user_id}: {describe_error(exc)}"
        summary = {
            "user_id": user_id,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "sources_polled": 0,
            "jobs_fetched": 0,
            "jobs_upserted": 0,
            "prefiltered_count": 0,
            "llm_scored_count": 0,
            "cap_used": 0,
            "carried_over_count": 0,
            "errors": [error],
            "preference_profile_rebuilt": False,
            "email": None,
            "cap_reached": False,
        }
        return summary
