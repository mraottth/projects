import json

from jobhunt.core.company_resolver import resolve_company
from jobhunt.core.llm import LLMError
from jobhunt.db.connection import get_db


def list_favorites() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT f.*, s.kind AS source_kind, s.config_json AS source_config_json,
                      s.name AS source_name
               FROM favorite_companies f LEFT JOIN sources s ON s.id = f.source_id
               ORDER BY f.created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def _get_or_create_favorite_row(company_name: str) -> int:
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM favorite_companies WHERE company_name = ?", (company_name,)
        ).fetchone()
        if existing:
            return existing["id"]
        cursor = conn.execute(
            "INSERT INTO favorite_companies (company_name) VALUES (?)", (company_name,)
        )
        return cursor.lastrowid


def _link_resolved_source(favorite_id: int, company_name: str, kind: str, config: dict, note: str) -> None:
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO sources (kind, name, company_name, config_json, is_favorite)
               VALUES (?, ?, ?, ?, 1)""",
            # No "(favorite)" suffix on the name -- the star badge already conveys that
            # everywhere a source/job card renders, and baking it into the name just gets
            # displayed redundantly (and awkwardly) in every list that shows source names.
            (kind, company_name, company_name, json.dumps(config)),
        )
        source_id = cursor.lastrowid
        conn.execute(
            "UPDATE favorite_companies SET resolution_status = 'resolved', "
            "resolution_note = ?, source_id = ? WHERE id = ?",
            (note, source_id, favorite_id),
        )


def add_favorite(company_name: str) -> dict:
    """Insert the favorite (status=pending), attempt resolution immediately via company_resolver
    (slug probing, then a Claude + web-search fallback), and link a source on success. Used when
    all we have is a company name -- e.g. approving a Discovery suggestion. Resolution failure --
    whether "not found" or an LLM-call error from the web-search fallback -- is not an error from
    the caller's perspective: the favorite is still recorded as 'failed' with a note, so the UI can
    show it and let the user retry, rather than leaving the row stuck at 'pending' forever."""
    favorite_id = _get_or_create_favorite_row(company_name)

    try:
        resolved = resolve_company(company_name)
    except LLMError as exc:
        resolved = None
        failure_note = str(exc)
    else:
        failure_note = "could not find a job board for this company"

    if resolved is None:
        with get_db() as conn:
            conn.execute(
                "UPDATE favorite_companies SET resolution_status = 'failed', "
                "resolution_note = ? WHERE id = ?",
                (failure_note, favorite_id),
            )
    else:
        _link_resolved_source(favorite_id, company_name, resolved.kind, resolved.config, resolved.note)

    return get_favorite(favorite_id)


def add_favorite_with_url(company_name: str, kind: str, config: dict) -> dict:
    """Add a favorite directly from an already-known job-board URL -- no company_resolver call,
    no guessing, no possible 'failed' state. Used when the user supplies the board URL themselves
    (the Sources page's favorites form), where `kind`/`config` come from the same client-side
    URL-kind detection the plain 'add a source' form uses."""
    favorite_id = _get_or_create_favorite_row(company_name)
    _link_resolved_source(favorite_id, company_name, kind, config, "added directly via job board URL")
    return get_favorite(favorite_id)


def get_favorite(favorite_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute(
            """SELECT f.*, s.kind AS source_kind, s.config_json AS source_config_json
               FROM favorite_companies f LEFT JOIN sources s ON s.id = f.source_id
               WHERE f.id = ?""",
            (favorite_id,),
        ).fetchone()
    return dict(row) if row else None


def retry_favorite(favorite_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute(
            "SELECT company_name FROM favorite_companies WHERE id = ?", (favorite_id,)
        ).fetchone()
    if row is None:
        raise ValueError("favorite not found")
    return add_favorite(row["company_name"])


def remove_favorite(favorite_id: int) -> None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT source_id FROM favorite_companies WHERE id = ?", (favorite_id,)
        ).fetchone()
        if row is None:
            return
        if row["source_id"] is not None:
            conn.execute("UPDATE sources SET is_favorite = 0 WHERE id = ?", (row["source_id"],))
        conn.execute("DELETE FROM favorite_companies WHERE id = ?", (favorite_id,))


def is_favorite_company_name(company_name: str | None) -> bool:
    if not company_name:
        return False
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM favorite_companies WHERE lower(company_name) = lower(?) "
            "AND resolution_status = 'resolved'",
            (company_name,),
        ).fetchone()
    return row is not None
