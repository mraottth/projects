"""Who the app is acting for (D-021: multi-user, up to ~20 people the admin approves).

Every per-person table carries a `user_id`, and every query on one filters by `current_user_id()`.
The id is held in a context variable, not passed through every function:
- `api/auth.py`'s middleware sets it for each request, from the verified sign-in;
- the daily run sets it while it works on each person (`acting_as`);
- a background thread doesn't inherit it, so it's handed over explicitly (`carry`).

It fails closed: with nobody set, `current_user_id()` raises rather than reading anyone's data.
"""

import contextvars
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

from jobhunt.config import DEFAULT_SETTINGS, RESUME_STORAGE_DIR
from jobhunt.db.connection import get_db

_current: contextvars.ContextVar[int | None] = contextvars.ContextVar("jobhunt_user_id", default=None)

STATUSES = ("pending", "approved", "declined", "blocked")


class NoUserError(RuntimeError):
    """Per-person data was touched with nobody signed in (a bug, never a normal path)."""


def current_user_id() -> int:
    user_id = _current.get()
    if user_id is None:
        raise NoUserError("no user is set for this request or task")
    return user_id


def optional_user_id() -> int | None:
    return _current.get()


def set_current_user(user_id: int | None) -> contextvars.Token:
    return _current.set(user_id)


def reset_current_user(token: contextvars.Token) -> None:
    _current.reset(token)


@contextmanager
def acting_as(user_id: int) -> Iterator[None]:
    token = _current.set(user_id)
    try:
        yield
    finally:
        _current.reset(token)


def carry(fn: Callable) -> Callable:
    """`fn` wrapped to run as the current user in another thread (thread pools and
    threading.Thread start with no user set)."""
    user_id = _current.get()

    @wraps(fn)
    def run(*args, **kwargs):
        with acting_as(user_id):
            return fn(*args, **kwargs)

    return run


def get_user(user_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip(),)).fetchone()
    return dict(row) if row else None


def current_user() -> dict:
    user = get_user(current_user_id())
    if user is None:
        raise NoUserError("the current user no longer exists")
    return user


def is_admin() -> bool:
    user_id = _current.get()
    return user_id is not None and bool((get_user(user_id) or {}).get("is_admin"))


def create_user(
    email: str, name: str | None = None, note: str | None = None, status: str = "pending"
) -> dict:
    with get_db() as conn:
        user_id = conn.execute(
            "INSERT INTO users (email, name, note, status) VALUES (?, ?, ?, ?)",
            (email.strip(), name, note, status),
        ).lastrowid
        insert_default_settings(conn, user_id)
    return get_user(user_id)


def insert_default_settings(conn, user_id: int) -> None:
    for key, value in DEFAULT_SETTINGS.items():
        conn.execute(
            "INSERT OR IGNORE INTO settings (user_id, key, value_json) VALUES (?, ?, ?)",
            (user_id, key, json.dumps(value)),
        )


def approved_user_ids() -> list[int]:
    with get_db() as conn:
        rows = conn.execute("SELECT id FROM users WHERE status = 'approved' ORDER BY id").fetchall()
    return [row["id"] for row in rows]


# The daily run only scores people who opened the app in the last ACTIVE_DAYS (step 4 of D-021):
# nobody pays for matches no one reads. Someone approved but never seen yet counts from their
# approval, so a new person isn't paused before their first visit.
ACTIVE_DAYS = 7
_LAST_ACTIVE = "COALESCE(last_seen_at, decided_at, created_at)"


def active_user_ids() -> list[int]:
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT id FROM users WHERE status = 'approved' "
            f"AND julianday('now') - julianday({_LAST_ACTIVE}) <= ? ORDER BY id",
            (ACTIVE_DAYS,),
        ).fetchall()
    return [row["id"] for row in rows]


def newly_inactive_users() -> list[dict]:
    """Approved people past ACTIVE_DAYS who haven't been told their matches are paused."""
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM users WHERE status = 'approved' AND paused_email_sent_at IS NULL "
            f"AND julianday('now') - julianday({_LAST_ACTIVE}) > ? ORDER BY id",
            (ACTIVE_DAYS,),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_paused_email_sent(user_id: int) -> None:
    with get_db() as conn:
        conn.execute("UPDATE users SET paused_email_sent_at = datetime('now') WHERE id = ?", (user_id,))


def touch_last_seen(user_id: int) -> None:
    """Records a visit, at most once an hour (the 7-day activity rule reads it). A visit also
    clears the "matches paused" email flag, so it can go out again after a later lapse."""
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET last_seen_at = ?, paused_email_sent_at = NULL WHERE id = ? "
            "AND (last_seen_at IS NULL OR julianday('now') - julianday(last_seen_at) > 1.0 / 24)",
            (datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"), user_id),
        )


# Which jobs someone can see, for `jobs j` with the user's id bound as :uid. Jobs from sources
# are shared, but each person only sees those from sources they follow, plus any they've scored or
# tracked (a job stays visible after unfollowing its source). A pasted-in job is its owner's alone.
VISIBLE_JOB_SQL = """(j.user_id = :uid OR (j.user_id IS NULL AND (
    EXISTS (SELECT 1 FROM source_follows vf WHERE vf.user_id = :uid AND vf.source_id = j.source_id)
    OR EXISTS (SELECT 1 FROM applications va WHERE va.user_id = :uid AND va.job_id = j.id)
    OR EXISTS (SELECT 1 FROM job_scores vs WHERE vs.user_id = :uid AND vs.job_id = j.id)
)))"""


def job_is_visible(job_id: int) -> bool:
    with get_db() as conn:
        row = conn.execute(
            f"SELECT 1 FROM jobs j WHERE j.id = :job AND {VISIBLE_JOB_SQL}",
            {"job": job_id, "uid": current_user_id()},
        ).fetchone()
    return row is not None


def storage_dir() -> Path:
    """Where the current person's uploaded and generated files go: RESUME_STORAGE_DIR/<user id>.
    (Files from before multi-user stay where they were; the database holds each one's path.)"""
    return RESUME_STORAGE_DIR / str(current_user_id())
