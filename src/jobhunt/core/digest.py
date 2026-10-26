from datetime import date

from jobhunt.db.connection import get_db

MAX_AGE_DAYS = 90

# A job is "favorite" if either its company is a resolved favorite company (matched by name, so
# it applies even if the same company later turns up via a different source) or its own source
# was favorited directly (matched by source_id -- a multi-employer board has no single company
# name to match on, so this is the only way a favorite job board's postings get flagged).
IS_FAVORITE_SQL = """(
    EXISTS (
        SELECT 1 FROM favorite_companies fc
        WHERE lower(fc.company_name) = lower(j.company) AND fc.resolution_status = 'resolved'
    )
    OR EXISTS (SELECT 1 FROM sources s WHERE s.id = j.source_id AND s.is_favorite = 1)
) AS is_favorite"""

# "New and noteworthy" excludes a job the user already dismissed (once marked not interested,
# it's done surfacing here -- still reachable from All Jobs, just not pushed at the user again)
# and a posting old enough that it isn't really "new" -- unless we simply don't know its age, in
# which case we'd rather show it than silently drop it.
NOT_DISMISSED_SQL = """NOT EXISTS (
    SELECT 1 FROM feedback f WHERE f.job_id = j.id AND f.interested = 'not_interested'
)"""
RECENT_OR_UNKNOWN_SQL = f"""(
    j.posted_at IS NULL OR julianday('now') - julianday(j.posted_at) <= {MAX_AGE_DAYS}
)"""


def today_str() -> str:
    return date.today().isoformat()


def add_to_digest(job_id: int, digest_date: str | None = None) -> None:
    """Insert job_id into the given day's digest (default today) and re-rank that day. A posting
    older than MAX_AGE_DAYS (and not of unknown age) never enters the digest at all -- "new and
    noteworthy" means new. Default rank order: interest+candidate fit score, then how recently it
    was posted, then salary -- favorites always rank above all of that, "favorite" meaning "always
    show me this," not "weight it a bit higher"."""
    digest_date = digest_date or today_str()
    with get_db() as conn:
        is_recent = conn.execute(
            f"SELECT {RECENT_OR_UNKNOWN_SQL} AS ok FROM jobs j WHERE j.id = ?", (job_id,)
        ).fetchone()
        if is_recent is None or not is_recent["ok"]:
            return

        conn.execute(
            "INSERT OR IGNORE INTO digest_entries (digest_date, job_id, rank) VALUES (?, ?, 0)",
            (digest_date, job_id),
        )
        rows = conn.execute(
            f"""SELECT de.job_id, js.interest_score, js.candidate_fit_score,
                      j.posted_at, j.first_seen_at, j.salary_max, j.salary_min, {IS_FAVORITE_SQL}
               FROM digest_entries de
               JOIN jobs j ON j.id = de.job_id
               LEFT JOIN job_scores js ON js.id = (
                   SELECT id FROM job_scores WHERE job_id = de.job_id ORDER BY id DESC LIMIT 1
               )
               WHERE de.digest_date = ?""",
            (digest_date,),
        ).fetchall()
        ranked = sorted(
            rows,
            key=lambda r: (
                r["is_favorite"],
                (r["interest_score"] or 0) + (r["candidate_fit_score"] or 0),
                r["posted_at"] or r["first_seen_at"] or "",
                r["salary_max"] or r["salary_min"] or 0,
            ),
            reverse=True,
        )
        conn.executemany(
            "UPDATE digest_entries SET rank = ? WHERE digest_date = ? AND job_id = ?",
            [(i + 1, digest_date, r["job_id"]) for i, r in enumerate(ranked)],
        )


def get_digest(digest_date: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT j.*, js.interest_fit, js.interest_score, js.candidate_fit, js.candidate_fit_score,
                      js.summary_text, js.rationale_text, de.rank, s.name AS source_name, {IS_FAVORITE_SQL}
               FROM digest_entries de
               JOIN jobs j ON j.id = de.job_id
               LEFT JOIN job_scores js ON js.id = (
                   SELECT id FROM job_scores WHERE job_id = de.job_id ORDER BY id DESC LIMIT 1
               )
               LEFT JOIN sources s ON s.id = j.source_id
               WHERE de.digest_date = ? AND {NOT_DISMISSED_SQL} AND {RECENT_OR_UNKNOWN_SQL}
               ORDER BY de.rank""",
            (digest_date,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_digest_dates() -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT digest_date FROM digest_entries ORDER BY digest_date DESC"
        ).fetchall()
    return [row["digest_date"] for row in rows]
