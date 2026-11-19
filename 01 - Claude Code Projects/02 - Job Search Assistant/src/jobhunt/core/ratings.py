"""The person's own Interest and Fit ratings (asked 2026-11-19).

Clicking a job's Interest or Fit chip lets the person change it. The new rating is stored as a
`job_scores` row with model `USER_MODEL`, so every page, filter, New & Noteworthy and the daily email
follow it at once (they all read a job's latest score). What it was changed from, and why, goes in
`rating_corrections`, which scoring (`scoring._corrections_text`) and the preference profile learn
from. Rescores and "Score this job" leave a re-rated job alone (`rated_by_person`).
"""

import json

from jobhunt.core.users import current_user_id, job_is_visible
from jobhunt.db.connection import get_db

USER_MODEL = "you"
LEVELS = ("High", "Med", "Low")
# A changed label gets its band's middle; an unchanged one keeps its number.
BAND_MIDDLE = {"High": 83.0, "Med": 50.0, "Low": 17.0}
NOTE_LIMIT = 1000


class NotScored(ValueError):
    """There's no rating to change yet."""


class NotVisible(LookupError):
    """Not a job this person can see."""


def correct(job_id: int, interest: str, fit: str, note: str | None = None) -> dict:
    """Stores the person's Interest and Fit for a job. Returns the new score row."""
    if interest not in LEVELS or fit not in LEVELS:
        raise ValueError("ratings must be High, Med or Low")
    if not job_is_visible(job_id):
        raise NotVisible(job_id)
    uid = current_user_id()
    note = (note or "").strip()[:NOTE_LIMIT] or None
    with get_db() as conn:
        latest = conn.execute(
            "SELECT * FROM job_scores WHERE user_id = ? AND job_id = ? ORDER BY id DESC LIMIT 1",
            (uid, job_id),
        ).fetchone()
        if latest is None:
            raise NotScored(job_id)
        interest_score = (
            latest["interest_score"] if latest["interest_fit"] == interest else BAND_MIDDLE[interest]
        )
        fit_score = latest["candidate_fit_score"] if latest["candidate_fit"] == fit else BAND_MIDDLE[fit]
        score_id = conn.execute(
            """INSERT INTO job_scores
               (user_id, job_id, model, interest_fit, interest_score, candidate_fit, candidate_fit_score,
                summary_text, rationale_text, raw_response_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uid,
                job_id,
                USER_MODEL,
                interest,
                interest_score,
                fit,
                fit_score,
                latest["summary_text"],
                latest["rationale_text"],
                json.dumps({"changed_from_score_id": latest["id"], "note": note}),
            ),
        ).lastrowid
        # What the model (or their own earlier rating) said, for learning.
        conn.execute(
            """INSERT INTO rating_corrections
               (user_id, job_id, from_interest, from_fit, to_interest, to_fit, note)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uid, job_id, latest["interest_fit"], latest["candidate_fit"], interest, fit, note),
        )
        row = conn.execute("SELECT * FROM job_scores WHERE id = ?", (score_id,)).fetchone()
    return {**dict(row), "rated_by_you": True}


def rated_by_person(job_id: int) -> bool:
    """Whether the job's latest score is the person's own rating (rescores leave those alone)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT model FROM job_scores WHERE user_id = ? AND job_id = ? ORDER BY id DESC LIMIT 1",
            (current_user_id(), job_id),
        ).fetchone()
    return bool(row) and row["model"] == USER_MODEL


def rated_job_ids() -> set[int]:
    """Jobs whose latest score is the person's own rating."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT js.job_id FROM job_scores js
               WHERE js.user_id = :uid AND js.model = :model AND js.id = (
                   SELECT MAX(id) FROM job_scores WHERE user_id = :uid AND job_id = js.job_id)""",
            {"uid": current_user_id(), "model": USER_MODEL},
        ).fetchall()
    return {r[0] for r in rows}


def correction_count() -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM rating_corrections WHERE user_id = ?", (current_user_id(),)
        ).fetchone()[0]
