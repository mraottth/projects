"""Settings' daily run audit (asked 2026-11-17): for one day, what the person's checks did.

- the checks that ran (their `runs` rows) and which sources they covered (`runs.audit_json`,
  recorded by pipeline/daily_run.py's `_run_for_person` since this was added), and the daily
  checks that failed for them (`failures()`);
- new jobs per source (`jobs.first_seen_at` that day, sources they follow);
- the jobs scored that day (`job_scores.scored_at`), with a rescore marked;
- the jobs still waiting to be scored now, in the order they'd be taken (the queue isn't kept per
  day, so this is always the current list).

A day is midnight to midnight in the person's time zone (the browser's), as UTC bounds.

`calendar()` colors a month's days for the card's calendar (asked 2026-11-18): "ran" (a check of
theirs that scores ran), "read" (only the 10am read ran, so nothing was scored yet), "failed" (a
daily check failed for them; this wins over the others, so a failure stays visible after a later
Refresh) or "none"."""

import json
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jobhunt.core import dealbreakers
from jobhunt.core.digest import SOURCE_IS_BOARD_SQL
from jobhunt.core.settings import get_settings
from jobhunt.core.users import current_user_id, get_user, is_admin
from jobhunt.db.connection import get_db

DEFAULT_TZ = "America/New_York"
# A run still 'running' after this long died without finishing (its server was shut down or
# crashed), since the request's own limit is an hour (--timeout 3600). The Admin page uses it too.
STUCK_AFTER_MINUTES = 70


def _zone(tz: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(tz or DEFAULT_TZ)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(DEFAULT_TZ)


def _bounds(day: date, zone: ZoneInfo) -> tuple[str, str]:
    """The day's start and end as SQLite UTC times ("YYYY-MM-DD HH:MM:SS")."""
    start = datetime.combine(day, time.min, zone).astimezone(ZoneInfo("UTC"))
    end = datetime.combine(day + timedelta(days=1), time.min, zone).astimezone(ZoneInfo("UTC"))
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


def _utc(value: str) -> str:
    """A stored time as SQLite's format; runs keep ISO times with a "T" and an offset."""
    if "T" not in value:
        return value
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo:
        parsed = parsed.astimezone(ZoneInfo("UTC"))
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def today(tz: str | None) -> date:
    return datetime.now(_zone(tz)).date()


def _local_day(stamp: str, zone: ZoneInfo) -> str:
    return (
        datetime.strptime(_utc(stamp), "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=ZoneInfo("UTC"))
        .astimezone(zone)
        .date()
        .isoformat()
    )


def failures(conn, uid: int) -> list[dict]:
    """The everyone runs (`user_id` NULL) that failed this person, as {started_at, message}:
    their own part failed (`_run_for_person_safely` puts "user N: ..." in the run's errors and
    writes no `runs` row for them), or the whole run failed or never finished before their part
    was recorded (a person's row shares the run's `started_at`). A person's own Refresh that fails
    leaves no record, so it isn't here."""
    own = {r[0] for r in conn.execute("SELECT started_at FROM runs WHERE user_id = ?", (uid,))}
    prefix = f"user {uid}: "
    stuck = datetime.now(ZoneInfo("UTC")) - timedelta(minutes=STUCK_AFTER_MINUTES)
    out = []
    for r in conn.execute(
        "SELECT started_at, status, error, errors_json FROM runs "
        "WHERE user_id IS NULL AND status IS NOT NULL ORDER BY id"
    ):
        details = json.loads(r["errors_json"] or "{}")
        errors = details.get("errors", []) if isinstance(details, dict) else details
        mine = [e[len(prefix) :] for e in errors if isinstance(e, str) and e.startswith(prefix)]
        if mine:
            out.append({"started_at": r["started_at"], "message": f"Your part failed: {mine[0]}"})
        elif r["started_at"] in own:
            continue
        elif r["status"] == "failed":
            # The error is about the app, not them; only the admin sees it.
            message = "The daily check failed"
            if is_admin() and r["error"]:
                message += f": {r['error']}"
            out.append({"started_at": r["started_at"], "message": message})
        elif r["status"] == "running" and _utc(r["started_at"]) < stuck.strftime("%Y-%m-%d %H:%M:%S"):
            out.append({"started_at": r["started_at"], "message": "The daily check didn't finish"})
    return out


def calendar(month: date, tz: str | None = None) -> dict:
    """Each day of `month` up to today: "ran", "read", "failed" or "none". Days before the person's
    account and days still to come are left out, and `first` (their account's first day) bounds
    the calendar's Back."""
    zone = _zone(tz)
    uid = current_user_id()
    first = _local_day((get_user(uid) or {}).get("created_at") or "1970-01-01 00:00:00", zone)
    with get_db() as conn:
        ran, read = set(), set()
        for r in conn.execute("SELECT started_at, kind FROM runs WHERE user_id = ?", (uid,)):
            (read if r["kind"] == "read" else ran).add(_local_day(r["started_at"], zone))
        failed = {_local_day(f["started_at"], zone) for f in failures(conn, uid)}
    last = today(tz).isoformat()
    days = {}
    day = month.replace(day=1)
    while day.month == month.month:
        iso = day.isoformat()
        if first <= iso <= last:
            days[iso] = (
                "failed" if iso in failed else "ran" if iso in ran else "read" if iso in read else "none"
            )
        day += timedelta(days=1)
    return {"month": month.strftime("%Y-%m"), "first": first, "today": last, "days": days}


def audit(day: date, tz: str | None = None) -> dict:
    zone = _zone(tz)
    start, end = _bounds(day, zone)
    uid = current_user_id()
    with get_db() as conn:
        follows = conn.execute(
            f"""SELECT s.id, s.name, {SOURCE_IS_BOARD_SQL} AS is_board, f.enabled
                  FROM source_follows f JOIN sources s ON s.id = f.source_id
                 WHERE f.user_id = ?""",
            (uid,),
        ).fetchall()
        runs = [
            dict(r)
            for r in conn.execute(
                "SELECT started_at, finished_at, errors_json, audit_json, kind, llm_scored_count "
                "FROM runs WHERE user_id = ? "
                "ORDER BY id",
                (uid,),
            )
        ]
        runs = [r for r in runs if start <= _utc(r["started_at"]) < end]
        new_jobs = conn.execute(
            """SELECT s.name AS source, COUNT(*) AS jobs
                 FROM jobs j JOIN sources s ON s.id = j.source_id
                 JOIN source_follows f ON f.source_id = s.id AND f.user_id = ?
                WHERE j.first_seen_at >= ? AND j.first_seen_at < ?
                GROUP BY s.id ORDER BY jobs DESC, s.name""",
            (uid, start, end),
        ).fetchall()
        scored = conn.execute(
            """SELECT j.id, j.title, j.company, sc.interest_fit, sc.candidate_fit, sc.scored_at,
                      sc.interest_score, sc.candidate_fit_score,
                      EXISTS (SELECT 1 FROM job_scores e WHERE e.user_id = sc.user_id
                              AND e.job_id = sc.job_id AND e.id < sc.id) AS rescored
                 FROM job_scores sc JOIN jobs j ON j.id = sc.job_id
                WHERE sc.user_id = ? AND sc.scored_at >= ? AND sc.scored_at < ?
                -- Best first, as New & Noteworthy ranks (asked 2026-11-17), then in the order scored.
                ORDER BY COALESCE(sc.interest_score, 0) + COALESCE(sc.candidate_fit_score, 0) DESC, sc.id""",
            (uid, start, end),
        ).fetchall()
        waiting = conn.execute(
            """SELECT j.id, j.title, j.company, j.location, j.remote_type, j.salary_min, j.salary_max,
                      j.salary_currency, j.salary_period, j.source_id, s.name AS source, q.screen
                 FROM scoring_queue q JOIN jobs j ON j.id = q.job_id
                 JOIN sources s ON s.id = j.source_id
                 JOIN source_follows f ON f.source_id = s.id AND f.user_id = q.user_id AND f.enabled = 1
                WHERE q.user_id = ? AND q.status = 'pending'
                ORDER BY q.priority DESC, q.id""",
            (uid,),
        ).fetchall()
        failed_checks = [f for f in failures(conn, uid) if start <= _utc(f["started_at"]) < end]

    by_id = {r["id"]: r for r in follows}
    checked: set[int] = set()
    failed: dict[int, str] = {}
    recorded = False
    for run in runs:
        if not run["audit_json"]:
            continue
        recorded = True
        info = json.loads(run["audit_json"])
        checked.update(i for i in info.get("checked", []) if i in by_id)
        for sid, err in info.get("failed", {}).items():
            if int(sid) in by_id:
                failed[int(sid)] = err
    # A later successful read of a source clears an earlier failure that day.
    failed = {sid: err for sid, err in failed.items() if sid not in checked}

    def side(is_board: bool) -> dict:
        mine = [r for r in follows if bool(r["is_board"]) == is_board and r["enabled"]]
        return {"checked": sum(1 for r in mine if r["id"] in checked), "total": len(mine)}

    settings = get_settings()
    hide = dealbreakers.any_on(settings)
    ranked, set_aside, hidden = [], 0, 0
    per_source: dict[int, int] = {}
    for n, r in enumerate(waiting):  # highest priority first
        if r["screen"] == "drop":
            set_aside += 1
        elif hide and dealbreakers.blocks(dict(r), settings):
            hidden += 1
        else:
            per_source[r["source_id"]] = rank = per_source.get(r["source_id"], 0) + 1
            ranked.append((rank, n, r))
    # The order the next scoring takes them, like queue.select_batch: every source's best job, then
    # every source's second best, and so on.
    queue = [
        {"id": r["id"], "title": r["title"], "company": r["company"], "source": r["source"]}
        for _, _, r in sorted(ranked, key=lambda t: (t[0], t[1]))
    ]

    return {
        "date": day.isoformat(),
        "checks": [
            {
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "errors": json.loads(r["errors_json"] or "[]"),
                # 'read' for the 10am read (no scoring), else 'score' (asked 2026-11-18).
                "kind": r["kind"] or "score",
                "scored": r["llm_scored_count"] or 0,
            }
            for r in runs
        ],
        "failures": failed_checks,
        "sources": {
            "recorded": recorded,
            "organizations": side(False),
            "boards": side(True),
            "failed": [{"name": by_id[sid]["name"], "error": err} for sid, err in failed.items()],
        },
        "new_jobs": [{"source": r["source"], "jobs": r["jobs"]} for r in new_jobs],
        "scored": [
            {
                "id": r["id"],
                "title": r["title"],
                "company": r["company"],
                "interest_fit": r["interest_fit"],
                "candidate_fit": r["candidate_fit"],
                "interest_score": r["interest_score"],
                "candidate_fit_score": r["candidate_fit_score"],
                "rescored": bool(r["rescored"]),
            }
            for r in scored
        ],
        "queue": {"jobs": queue, "set_aside": set_aside, "hidden": hidden},
    }
