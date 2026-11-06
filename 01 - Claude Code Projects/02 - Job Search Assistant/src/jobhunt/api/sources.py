import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.core.company_resolver import resolve_company
from jobhunt.core.llm import LLMError
from jobhunt.db.connection import get_db
from jobhunt.ingest.adapters import custom_board
from jobhunt.pipeline.daily_run import ADAPTERS, schedule_source_refresh
from jobhunt.core.users import current_user_id, is_admin

router = APIRouter(prefix="/api/sources", tags=["sources"])

VALID_KINDS = set(ADAPTERS.keys())


class SourceRequest(BaseModel):
    kind: str
    name: str
    company_name: str | None = None
    config: dict = {}
    enabled: bool = True
    poll_frequency_hours: int = 24


class ResolveSourceRequest(BaseModel):
    company_name: str


class SourceUpdateRequest(BaseModel):
    name: str | None = None
    company_name: str | None = None
    config: dict | None = None
    enabled: bool | None = None
    poll_frequency_hours: int | None = None


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["config"] = json.loads(d.pop("config_json") or "{}")
    return d


# Sources are shared (D-021): one row per board, read once per run however many people follow it.
# Each person sees the ones they follow, with `enabled` being their own follow's switch, and
# `followers` how many people follow it (shared settings are only editable by a sole follower).
_FOLLOWED_SELECT = """SELECT s.id, s.kind, s.name, s.company_name, s.config_json, s.poll_frequency_hours,
       s.last_polled_at, s.last_success_at, s.last_error, s.created_at,
       f.enabled AS enabled, f.created_at AS followed_at,
       (SELECT COUNT(*) FROM jobs j WHERE j.source_id = s.id AND j.status = 'active') AS job_count,
       (SELECT COUNT(*) FROM source_follows f2 WHERE f2.source_id = s.id) AS followers
  FROM sources s JOIN source_follows f ON f.source_id = s.id AND f.user_id = :uid"""


def _canonical(kind: str, config: dict) -> str:
    """What makes two sources the same board: the kind, plus its config with any URL compared
    without case, protocol or trailing slash."""
    normalized = {}
    for key, value in config.items():
        if isinstance(value, str) and key in ("url", "board_url"):
            value = value.strip().lower().removeprefix("https://").removeprefix("http://").rstrip("/")
        elif isinstance(value, str):
            value = value.strip().lower()
        normalized[key] = value
    return kind + json.dumps(normalized, sort_keys=True)


def _find_existing(kind: str, config: dict) -> int | None:
    wanted = _canonical(kind, config)
    with get_db() as conn:
        rows = conn.execute("SELECT id, config_json FROM sources WHERE kind = ?", (kind,)).fetchall()
    for row in rows:
        if _canonical(kind, json.loads(row["config_json"] or "{}")) == wanted:
            return row["id"]
    return None


def _follow(source_id: int, enabled: bool = True) -> bool:
    """Follows the source; True if this person wasn't already following it."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO source_follows (user_id, source_id, enabled) VALUES (?, ?, ?)",
            (current_user_id(), source_id, int(enabled)),
        )
    return cursor.rowcount > 0


def _add_or_follow(
    kind: str,
    name: str,
    company_name: str | None,
    config: dict,
    enabled: bool = True,
    poll_frequency_hours: int = 24,
) -> dict:
    source_id = _find_existing(kind, config)
    if source_id is None:
        with get_db() as conn:
            source_id = conn.execute(
                """INSERT INTO sources
                   (kind, name, company_name, config_json, enabled, poll_frequency_hours)
                   VALUES (?, ?, ?, ?, 1, ?)""",
                (kind, name, company_name, json.dumps(config), poll_frequency_hours),
            ).lastrowid
    if _follow(source_id, enabled) and enabled:
        schedule_source_refresh(source_id)
    return get_source(source_id)


@router.get("")
def list_sources():
    """The sources you follow, each with `job_count`, its active jobs (shown on Manage Sources)."""
    with get_db() as conn:
        rows = conn.execute(
            f"{_FOLLOWED_SELECT} ORDER BY f.created_at DESC, s.id DESC", {"uid": current_user_id()}
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


@router.post("")
def create_source(payload: SourceRequest):
    if payload.kind not in VALID_KINDS:
        raise HTTPException(400, f"kind must be one of: {sorted(VALID_KINDS)}")
    return _add_or_follow(
        payload.kind,
        payload.name,
        payload.company_name,
        payload.config,
        payload.enabled,
        payload.poll_frequency_hours,
    )


@router.post("/resolve")
def resolve_source(payload: ResolveSourceRequest):
    """Resolve a company name to an actual job board and add it as a source --
    used when approving a discovery suggestion, or adding a company by name directly."""
    try:
        resolved = resolve_company(payload.company_name)
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    if resolved is None:
        raise HTTPException(404, f"could not find a job board for '{payload.company_name}'")

    return _add_or_follow(resolved.kind, payload.company_name, payload.company_name, resolved.config)


@router.get("/{source_id}")
def get_source(source_id: int):
    with get_db() as conn:
        row = conn.execute(
            f"{_FOLLOWED_SELECT} WHERE s.id = :source", {"uid": current_user_id(), "source": source_id}
        ).fetchone()
    if row is None:
        raise HTTPException(404, "source not found")
    return _row_to_dict(row)


@router.patch("/{source_id}")
def update_source(source_id: int, payload: SourceUpdateRequest):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    current = get_source(source_id)  # 404 unless you follow it
    enabled = updates.pop("enabled", None)
    if "config" in updates:
        updates["config_json"] = json.dumps(updates.pop("config"))
    if updates:
        # Name, employer and config are shared by everyone following the source.
        if current["followers"] > 1 and not is_admin():
            raise HTTPException(
                403, "other people follow this source too, so its details can't be changed here"
            )
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        with get_db() as conn:
            conn.execute(f"UPDATE sources SET {set_clause} WHERE id = ?", (*updates.values(), source_id))
    if enabled is not None:
        with get_db() as conn:
            conn.execute(
                "UPDATE source_follows SET enabled = ? WHERE user_id = ? AND source_id = ?",
                (int(enabled), current_user_id(), source_id),
            )
        if enabled and not current["enabled"]:
            schedule_source_refresh(source_id)  # re-enabled: check it now, like a new source
    return get_source(source_id)


@router.delete("/{source_id}")
def delete_source(source_id: int):
    """Stops following the source. The source itself stays for anyone else who follows it (one
    nobody follows is simply no longer read), as do its jobs you've scored or tracked."""
    get_source(source_id)  # 404 unless you follow it
    with get_db() as conn:
        conn.execute(
            "DELETE FROM source_follows WHERE user_id = ? AND source_id = ?", (current_user_id(), source_id)
        )
    return {"ok": True}


@router.post("/{source_id}/test")
def test_source(source_id: int):
    """Dry-run fetch: calls the adapter and returns a preview without persisting anything."""
    source = get_source(source_id)
    steps: list[str] = []
    try:
        if source["kind"] == "custom_board":
            postings, steps = custom_board.fetch_with_report(source)
        else:
            postings = ADAPTERS[source["kind"]](source)
    except Exception as exc:  # noqa: BLE001 - surface any adapter failure to the UI, not a 500
        raise HTTPException(502, f"test fetch failed: {exc}") from exc
    return {"count": len(postings), "preview": postings[:5], "steps": steps}
