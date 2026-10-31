import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.core.company_resolver import resolve_company
from jobhunt.core.llm import LLMError
from jobhunt.db.connection import get_db
from jobhunt.ingest.adapters import custom_board
from jobhunt.pipeline.daily_run import ADAPTERS, schedule_source_refresh

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


@router.get("")
def list_sources():
    """Each source with `job_count`, its active jobs (shown on Manage Sources)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT s.*, (SELECT COUNT(*) FROM jobs j WHERE j.source_id = s.id AND j.status = 'active')
                   AS job_count
               FROM sources s ORDER BY s.created_at DESC, s.id DESC"""
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


@router.post("")
def create_source(payload: SourceRequest):
    if payload.kind not in VALID_KINDS:
        raise HTTPException(400, f"kind must be one of: {sorted(VALID_KINDS)}")
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO sources
               (kind, name, company_name, config_json, enabled, poll_frequency_hours)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                payload.kind,
                payload.name,
                payload.company_name,
                json.dumps(payload.config),
                int(payload.enabled),
                payload.poll_frequency_hours,
            ),
        )
        source_id = cursor.lastrowid
    if payload.enabled:
        schedule_source_refresh(source_id)
    return get_source(source_id)


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

    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO sources (kind, name, company_name, config_json) VALUES (?, ?, ?, ?)",
            (
                resolved.kind,
                payload.company_name,
                payload.company_name,
                json.dumps(resolved.config),
            ),
        )
        source_id = cursor.lastrowid
    schedule_source_refresh(source_id)
    return get_source(source_id)


@router.get("/{source_id}")
def get_source(source_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "source not found")
    return _row_to_dict(row)


@router.patch("/{source_id}")
def update_source(source_id: int, payload: SourceUpdateRequest):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if "config" in updates:
        updates["config_json"] = json.dumps(updates.pop("config"))
    if "enabled" in updates:
        updates["enabled"] = int(updates["enabled"])
    if not updates:
        return get_source(source_id)
    with get_db() as conn:
        row = conn.execute("SELECT id, enabled FROM sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "source not found")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE sources SET {set_clause} WHERE id = ?", (*updates.values(), source_id))
    if updates.get("enabled") == 1 and not row["enabled"]:
        schedule_source_refresh(source_id)  # re-enabled: check it now, like a new source
    return get_source(source_id)


@router.delete("/{source_id}")
def delete_source(source_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT id FROM sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "source not found")
        conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
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
