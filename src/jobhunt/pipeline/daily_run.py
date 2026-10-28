"""Orchestrates the daily cycle: poll enabled sources -> dedupe/upsert jobs -> pre-filter and
queue new ones -> score up to the daily cap (highest-priority first) -> build today's digest ->
conditionally refresh the preference profile -> record a run summary. Triggered by Cloud Scheduler
via POST /internal/run-daily (see api/internal.py); safe to run manually/repeatedly."""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime

from jobhunt.core import prefilter, queue
from jobhunt.core import settings as settings_module
from jobhunt.core.dedupe import compute_dedupe_key
from jobhunt.core.digest import add_to_digest
from jobhunt.core.llm import LLMError
from jobhunt.core.page_fetch import PageFetchError, fetch_page_text
from jobhunt.core.preference import get_current_preference_profile, rebuild_preference_profile
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.scoring import score_job, store_score
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

# Live progress of the current run, for the UI to poll (GET /api/pipeline/progress) while its
# run-daily request is still open. In-process state is the real state: Cloud Run runs a single
# instance (--max-instances 1).
_progress_lock = threading.Lock()
_progress = {"stage": "idle", "sources_done": 0, "sources_total": 0, "jobs_done": 0, "jobs_total": 0}


def get_progress() -> dict:
    with _progress_lock:
        return dict(_progress)


def _set_progress(**updates) -> None:
    with _progress_lock:
        _progress.update(updates)


def _bump_progress(key: str) -> None:
    with _progress_lock:
        _progress[key] += 1


def _fetch_source(source: dict) -> tuple[list[dict] | None, str | None]:
    """Runs one source's adapter (in a worker thread). Returns (postings, error); postings is None
    for a failed fetch or a kind with no adapter."""
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
        _bump_progress("sources_done")


def _poll_sources(source_ids: list[int] | None = None) -> dict:
    with get_db() as conn:
        sources = [dict(row) for row in conn.execute("SELECT * FROM sources WHERE enabled = 1")]
    if source_ids is not None:
        sources = [s for s in sources if s["id"] in source_ids]
    for source in sources:
        source["config"] = json.loads(source.pop("config_json") or "{}")
    _set_progress(stage="polling", sources_done=0, sources_total=len(sources))

    fetched = 0
    upserted = 0
    errors = []
    job_ids = []

    # Fetches run concurrently; upserts stay sequential here, in source order -- two sources can
    # produce the same URL-based dedupe key, and _upsert_job's select-then-insert would race.
    with ThreadPoolExecutor(max_workers=POLL_CONCURRENCY) as pool:
        results = list(pool.map(_fetch_source, sources))

    for source, (postings, error) in zip(sources, results, strict=True):
        if error:
            errors.append(error)
        if postings is None:
            continue

        fetched += len(postings)
        for posting in postings:
            job_ids.append(_upsert_job(source["id"], posting))
            upserted += 1

        with get_db() as conn:
            conn.execute(
                "UPDATE sources SET last_polled_at = datetime('now'), last_success_at = datetime('now'), "
                "last_error = NULL WHERE id = ?",
                (source["id"],),
            )

    return {
        "sources_polled": len(sources),
        "jobs_fetched": fetched,
        "jobs_upserted": upserted,
        "errors": errors,
        "job_ids": job_ids,
    }


def _upsert_job(source_id: int, posting: dict) -> int:
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
            already_scored = conn.execute("SELECT 1 FROM job_scores WHERE job_id = ?", (job_id,)).fetchone()
            if already_scored:
                continue
            score, reasons = prefilter.score_job(dict(row), current_settings)
            queue.enqueue(job_id, score, reasons)
            enqueued += 1
    return enqueued


def _backfill_description(job: dict) -> dict:
    if job.get("description_text") or not job.get("url"):
        return job
    try:
        text = fetch_page_text(job["url"])
    except PageFetchError:
        return job
    with get_db() as conn:
        conn.execute("UPDATE jobs SET description_text = ? WHERE id = ?", (text, job["id"]))
    job["description_text"] = text
    return job


def _maybe_rebuild_preference_profile() -> bool:
    with get_db() as conn:
        last = conn.execute(
            "SELECT based_on_feedback_count, based_on_application_events_count, "
            "(julianday('now') - julianday(updated_at)) AS days_since "
            "FROM preference_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
        feedback_count = conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
        app_events = conn.execute("SELECT COUNT(*) AS n FROM application_status_history").fetchone()["n"]

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
    """Scores one queued job (in a worker thread). Returns (scored, error)."""
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


def run_daily(cap: int | None = None) -> dict:
    """`cap` overrides the daily_llm_cap setting for this run only -- onboarding scores just the top
    few so the user isn't kept waiting; the rest stay queued for the next run."""
    with _run_lock:
        started_at = datetime.now(UTC).isoformat()
        _set_progress(stage="starting", sources_done=0, sources_total=0, jobs_done=0, jobs_total=0)

        try:
            return _run_daily(started_at, cap)
        finally:
            with _progress_lock:
                if _progress["stage"] != "done":
                    _progress["stage"] = "failed"


def _score_args() -> tuple:
    return (
        get_current_candidate_profile(),
        get_current_preference_profile(),
        settings_module.get_setting("prefilter_sectors", []),
        settings_module.get_setting("prefilter_titles_include", []),
        settings_module.get_setting("salary_floor", None),
        settings_module.get_setting("looking_for_notes", ""),
    )


def _score_and_digest(batch: list[int]) -> tuple[int, list[str]]:
    """Scores the batch concurrently, then adds each scored job to today's digest. Returns
    (scored count, errors)."""
    _set_progress(stage="scoring", jobs_done=0, jobs_total=len(batch))
    score_args = _score_args()
    scoring_errors = []
    scored_ids = []
    if batch:
        with ThreadPoolExecutor(max_workers=min(SCORING_CONCURRENCY, len(batch))) as pool:
            results = list(pool.map(lambda job_id: _score_one(job_id, score_args), batch))
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


# A newly added (or re-enabled) source is checked right away instead of at the next daily run:
# fetch just that source, score up to NEW_SOURCE_SCORE_CAP of its jobs (~3 cents each), and add
# them to New & Noteworthy. Asked for live: new sources sat empty until the next morning. Adds in
# quick succession (Monitoring several suggestions) are gathered into one refresh by waiting
# REFRESH_GATHER_SECONDS before starting. Runs never overlap (_run_lock), so a refresh started
# during a daily run waits for it.
NEW_SOURCE_SCORE_CAP = 10
REFRESH_GATHER_SECONDS = 3.0
_run_lock = threading.Lock()
_pending_refresh: set[int] = set()
_pending_lock = threading.Lock()
_refresh_worker: threading.Thread | None = None


def schedule_source_refresh(source_id: int) -> None:
    """Queues a background check of a newly added source. Skipped during onboarding, whose last
    step runs the full pipeline over every source anyway."""
    global _refresh_worker
    if not settings_module.get_setting("onboarding_completed", False):
        return
    with _pending_lock:
        _pending_refresh.add(source_id)
        if _refresh_worker is None or not _refresh_worker.is_alive():
            _refresh_worker = threading.Thread(target=_refresh_loop, daemon=True)
            _refresh_worker.start()


def _refresh_loop() -> None:
    while True:
        time.sleep(REFRESH_GATHER_SECONDS)
        with _pending_lock:
            source_ids = sorted(_pending_refresh)
            _pending_refresh.clear()
            if not source_ids:
                return
        try:
            refresh_sources(source_ids)
        except Exception:  # noqa: BLE001 - a failed refresh just leaves it to the next daily run
            logger.exception("refreshing new sources %s failed", source_ids)


def refresh_sources(source_ids: list[int], cap: int = NEW_SOURCE_SCORE_CAP) -> dict:
    """Polls only these sources, then scores up to `cap` of their new jobs into today's digest."""
    with _run_lock:
        poll_result = _poll_sources(source_ids)
        enqueued = _prefilter_and_enqueue(poll_result["job_ids"])
        batch = queue.select_batch(cap, source_ids=source_ids)
        scored, scoring_errors = _score_and_digest(batch)
        _set_progress(stage="done")
    return {
        "sources_polled": poll_result["sources_polled"],
        "jobs_fetched": poll_result["jobs_fetched"],
        "prefiltered_count": enqueued,
        "llm_scored_count": scored,
        "errors": poll_result["errors"] + scoring_errors,
    }


def _run_daily(started_at: str, cap: int | None) -> dict:
    queue.requeue_stranded()
    poll_result = _poll_sources()
    _expire_stale_jobs()
    enqueued = _prefilter_and_enqueue(poll_result["job_ids"])

    daily_cap = cap if cap is not None else settings_module.get_setting("daily_llm_cap", 50)
    batch = queue.select_batch(daily_cap)
    scored_count, scoring_errors = _score_and_digest(batch)

    preference_rebuilt = _maybe_rebuild_preference_profile()

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "sources_polled": poll_result["sources_polled"],
        "jobs_fetched": poll_result["jobs_fetched"],
        "jobs_upserted": poll_result["jobs_upserted"],
        "prefiltered_count": enqueued,
        "llm_scored_count": scored_count,
        "cap_used": daily_cap,
        "carried_over_count": queue.pending_count(),
        "errors": poll_result["errors"] + scoring_errors,
        "preference_profile_rebuilt": preference_rebuilt,
    }

    with get_db() as conn:
        conn.execute(
            """INSERT INTO runs
               (run_date, started_at, finished_at, sources_polled, jobs_fetched, jobs_deduped,
                prefiltered_count, llm_scored_count, cap_used, carried_over_count, errors_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
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

    _set_progress(stage="done")
    return summary
