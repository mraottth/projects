from datetime import date

from jobhunt.db.connection import get_db


def today_str() -> str:
    return date.today().isoformat()


def add_to_digest(job_id: int, digest_date: str | None = None) -> None:
    """Insert job_id into the given day's digest (default today) and re-rank that day by combined score."""
    digest_date = digest_date or today_str()
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO digest_entries (digest_date, job_id, rank) VALUES (?, ?, 0)",
            (digest_date, job_id),
        )
        rows = conn.execute(
            """SELECT de.job_id, js.interest_score, js.candidate_fit_score,
                      EXISTS (
                          SELECT 1 FROM favorite_companies fc
                          WHERE lower(fc.company_name) = lower(j.company)
                            AND fc.resolution_status = 'resolved'
                      ) AS is_favorite_company
               FROM digest_entries de
               JOIN jobs j ON j.id = de.job_id
               LEFT JOIN job_scores js ON js.id = (
                   SELECT id FROM job_scores WHERE job_id = de.job_id ORDER BY id DESC LIMIT 1
               )
               WHERE de.digest_date = ?""",
            (digest_date,),
        ).fetchall()
        # Favorite-company jobs always rank first, regardless of score -- "favorite" means
        # "always show me this", not "weight it a bit higher".
        ranked = sorted(
            rows,
            key=lambda r: (
                r["is_favorite_company"],
                (r["interest_score"] or 0) + (r["candidate_fit_score"] or 0),
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
            """SELECT j.*, js.interest_fit, js.interest_score, js.candidate_fit, js.candidate_fit_score,
                      js.summary_text, js.rationale_text, de.rank,
                      EXISTS (
                          SELECT 1 FROM favorite_companies fc
                          WHERE lower(fc.company_name) = lower(j.company)
                            AND fc.resolution_status = 'resolved'
                      ) AS is_favorite_company
               FROM digest_entries de
               JOIN jobs j ON j.id = de.job_id
               LEFT JOIN job_scores js ON js.id = (
                   SELECT id FROM job_scores WHERE job_id = de.job_id ORDER BY id DESC LIMIT 1
               )
               WHERE de.digest_date = ?
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
