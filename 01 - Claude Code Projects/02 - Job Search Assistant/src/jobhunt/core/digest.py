from datetime import date

from jobhunt.core import dealbreakers
from jobhunt.core.generic_postings import is_generic_posting
from jobhunt.core.settings import get_settings
from jobhunt.db.connection import get_db
from jobhunt.core.users import current_user_id

MAX_AGE_DAYS = 90

# "New and noteworthy" excludes a job the user already dismissed (once marked not interested,
# it's done surfacing here -- still reachable from All Jobs, just not pushed at the user again)
# and a posting old enough that it isn't really "new" -- unless we simply don't know its age, in
# which case we'd rather show it than silently drop it.
# Both take the user's id as a named parameter, :uid.
NOT_DISMISSED_SQL = """NOT EXISTS (
    SELECT 1 FROM feedback f WHERE f.user_id = :uid AND f.job_id = j.id AND f.interested = 'not_interested'
)"""
# Nor a job already on the person's My Jobs (asked 2026-11-04): it's no longer a discovery. Checked
# when read, so it covers every day's list and the daily email; removing the job from My Jobs
# brings it back.
NOT_TRACKED_SQL = """NOT EXISTS (
    SELECT 1 FROM applications a WHERE a.user_id = :uid AND a.job_id = j.id
)"""
LATEST_SCORE_SQL = """js.id = (
    SELECT id FROM job_scores WHERE user_id = :uid AND job_id = {job} ORDER BY id DESC LIMIT 1
)"""
RECENT_OR_UNKNOWN_SQL = f"""(
    j.posted_at IS NULL OR julianday('now') - julianday(j.posted_at) <= {MAX_AGE_DAYS}
)"""


def today_str() -> str:
    return date.today().isoformat()


def _rank_key(row) -> tuple:
    """Shared sort key: interest+candidate fit score, then how recently it was posted, then
    salary. Used both to persist `digest_entries.rank` (informational/historical) and, more
    importantly, to sort get_digest()'s results live -- the stored rank only gets recomputed
    when a *new* job is added to that date's digest, so trusting it alone goes stale the moment a
    feedback mark or anything else changes what should be excluded/reordered without a new insert."""
    return (
        (row["interest_score"] or 0) + (row["candidate_fit_score"] or 0),
        row["posted_at"] or row["first_seen_at"] or "",
        row["salary_max"] or row["salary_min"] or 0,
    )


def add_to_digest(job_id: int, digest_date: str | None = None) -> None:
    """Insert job_id into the given day's digest (default today) and re-rank that day. A posting
    older than MAX_AGE_DAYS (and not of unknown age) never enters the digest at all -- "new and
    noteworthy" means new."""
    digest_date = digest_date or today_str()
    uid = current_user_id()
    with get_db() as conn:
        is_recent = conn.execute(
            f"SELECT {RECENT_OR_UNKNOWN_SQL} AS ok, j.title FROM jobs j WHERE j.id = ?", (job_id,)
        ).fetchone()
        if is_recent is None or not is_recent["ok"] or is_generic_posting(is_recent["title"]):
            return

        conn.execute(
            "INSERT OR IGNORE INTO digest_entries (user_id, digest_date, job_id, rank) VALUES (?, ?, ?, 0)",
            (uid, digest_date, job_id),
        )
        rows = conn.execute(
            f"""SELECT de.job_id, js.interest_score, js.candidate_fit_score,
                      j.posted_at, j.first_seen_at, j.salary_max, j.salary_min
               FROM digest_entries de
               JOIN jobs j ON j.id = de.job_id
               LEFT JOIN job_scores js ON {LATEST_SCORE_SQL.format(job="de.job_id")}
               WHERE de.user_id = :uid AND de.digest_date = :day""",
            {"uid": uid, "day": digest_date},
        ).fetchall()
        ranked = sorted(rows, key=_rank_key, reverse=True)
        conn.executemany(
            "UPDATE digest_entries SET rank = ? WHERE user_id = ? AND digest_date = ? AND job_id = ?",
            [(i + 1, uid, digest_date, r["job_id"]) for i, r in enumerate(ranked)],
        )


# Whether a job's source is a job board (several employers) rather than one organization's own
# careers page: the same rule Manage Sources files them by (isOrganization() there). Only a board's
# name is worth showing on a job card, since an organization's source just repeats the employer.
SOURCE_IS_BOARD_SQL = (
    "(s.id IS NOT NULL AND s.kind NOT LIKE 'ats\\_%' ESCAPE '\\' AND COALESCE(s.company_name, '') = '')"
)


def get_digest(digest_date: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT j.*, js.interest_fit, js.interest_score, js.candidate_fit, js.candidate_fit_score,
                      js.summary_text, js.rationale_text, de.rank, de.emailed_at,
                      de.digest_date, s.name AS source_name,
                      {SOURCE_IS_BOARD_SQL} AS source_is_board
               FROM digest_entries de
               JOIN jobs j ON j.id = de.job_id
               LEFT JOIN job_scores js ON {LATEST_SCORE_SQL.format(job="de.job_id")}
               LEFT JOIN sources s ON s.id = j.source_id
               WHERE de.user_id = :uid AND de.digest_date = :day
                 AND {NOT_DISMISSED_SQL} AND {NOT_TRACKED_SQL} AND {RECENT_OR_UNKNOWN_SQL}""",
            {"uid": current_user_id(), "day": digest_date},
        ).fetchall()
    # Sort live rather than trusting the stored `rank` column -- see _rank_key's docstring. A
    # general-application posting already in an older digest is dropped here too.
    # The dealbreakers (location, salary) are applied here, when read, so turning it on or off (or changing
    # the locations) takes effect at once, on every day's list and the daily email alike.
    settings = get_settings()
    rows = [
        row
        for row in rows
        if not is_generic_posting(row["title"]) and not dealbreakers.blocks(dict(row), settings)
    ]
    return [dict(row) for row in sorted(rows, key=_rank_key, reverse=True)]


def get_digest_dates() -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT digest_date FROM digest_entries WHERE user_id = ? ORDER BY digest_date DESC",
            (current_user_id(),),
        ).fetchall()
    return [row["digest_date"] for row in rows]
