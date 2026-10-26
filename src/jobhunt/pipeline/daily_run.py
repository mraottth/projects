"""Orchestrates the daily cycle: poll enabled sources -> dedupe/upsert jobs -> pre-filter and
queue new ones -> score up to the daily cap (highest-priority first) -> build today's digest ->
conditionally refresh the preference profile -> record a run summary. Triggered by Cloud Scheduler
via POST /internal/run-daily (see api/internal.py); safe to run manually/repeatedly."""

import json
from datetime import UTC, date, datetime

from jobhunt.core import favorites, prefilter, queue
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


def _poll_sources() -> dict:
    with get_db() as conn:
        sources = [dict(row) for row in conn.execute("SELECT * FROM sources WHERE enabled = 1")]

    fetched = 0
    upserted = 0
    errors = []
    job_ids = []

    for source in sources:
        source["config"] = json.loads(source.pop("config_json") or "{}")
        adapter = ADAPTERS.get(source["kind"])
        if adapter is None:
            continue

        try:
            postings = adapter(source)
        except Exception as exc:  # noqa: BLE001 - one bad source must not abort the whole run
            errors.append(f"{source['name']}: {exc}")
            with get_db() as conn:
                conn.execute(
                    "UPDATE sources SET last_polled_at = datetime('now'), last_error = ? WHERE id = ?",
                    (str(exc), source["id"]),
                )
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


def _is_favorite_job(job: dict, conn) -> bool:
    """A favorite company says "always show me this company's postings" (matched by name, so it
    still applies even if the same company's job later turns up via a different source). A
    favorite job board says "always show me this board's postings" (matched by source_id, since a
    multi-employer board has no single company name to match on) -- the two are independent, and
    either one is enough to bypass the cap."""
    if favorites.is_favorite_company_name(job.get("company")):
        return True
    source_id = job.get("source_id")
    if source_id is None:
        return False
    row = conn.execute("SELECT is_favorite FROM sources WHERE id = ?", (source_id,)).fetchone()
    return bool(row and row["is_favorite"])


def _prefilter_and_enqueue(job_ids: list[int]) -> int:
    current_settings = settings_module.get_settings()
    enqueued = 0
    with get_db() as conn:
        for job_id in set(job_ids):
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                continue
            already_scored = conn.execute(
                "SELECT 1 FROM job_scores WHERE job_id = ?", (job_id,)
            ).fetchone()
            if already_scored:
                continue
            job = dict(row)
            if _is_favorite_job(job, conn):
                # Bypass pre-filter excludes entirely -- a favorite said "always show me this",
                # not "show me this unless it also matches an exclude keyword".
                queue.enqueue(job_id, 100.0, {"favorite": True}, is_favorite=True)
            else:
                score, reasons = prefilter.score_job(job, current_settings)
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
        app_events = conn.execute(
            "SELECT COUNT(*) AS n FROM application_status_history"
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


def run_daily() -> dict:
    started_at = datetime.now(UTC).isoformat()

    poll_result = _poll_sources()
    _expire_stale_jobs()
    enqueued = _prefilter_and_enqueue(poll_result["job_ids"])

    daily_cap = settings_module.get_setting("daily_llm_cap", 50)
    batch = queue.select_batch(daily_cap)

    candidate_profile = get_current_candidate_profile()
    preference_profile = get_current_preference_profile()
    desired_sectors = settings_module.get_setting("prefilter_sectors", [])
    desired_titles = settings_module.get_setting("prefilter_titles_include", [])
    salary_floor = settings_module.get_setting("salary_floor", None)
    looking_for_notes = settings_module.get_setting("looking_for_notes", "")

    scored_count = 0
    scoring_errors = []
    for job_id in batch:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            continue
        job = _backfill_description(dict(row))
        try:
            score = score_job(
                job,
                candidate_profile,
                preference_profile,
                desired_sectors,
                desired_titles,
                salary_floor,
                looking_for_notes,
            )
        except LLMError as exc:
            scoring_errors.append(f"job {job_id}: {exc}")
            queue.mark_pending_again(job_id)
            continue
        store_score(job_id, score)
        add_to_digest(job_id)
        queue.mark_scored(job_id)
        scored_count += 1

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

    return summary
