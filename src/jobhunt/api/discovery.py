import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from jobhunt.core.discovery import suggest_companies, suggest_job_boards
from jobhunt.core.llm import LLMError
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.settings import get_settings
from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/discovery", tags=["discovery"])


def _already_monitored_company_names() -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT company_name FROM sources WHERE company_name IS NOT NULL"
        ).fetchall()
    return sorted(r["company_name"] for r in rows)


def _already_monitored_board_urls() -> list[str]:
    with get_db() as conn:
        rows = conn.execute("SELECT config_json FROM sources WHERE kind = 'custom_board'").fetchall()
    urls = []
    for r in rows:
        config = json.loads(r["config_json"] or "{}")
        if config.get("url"):
            urls.append(config["url"])
    return urls


class DiscoveryRequest(BaseModel):
    """Optional per-search steering. `guidance` is free text ("mission-driven tech startups");
    `like` describes a source or suggestion to find more of ("Nava PBC (https://...)")."""

    guidance: str | None = Field(default=None, max_length=500)
    like: str | None = Field(default=None, max_length=500)


@router.post("/suggest")
def suggest(payload: DiscoveryRequest | None = None):
    """Returns suggested companies for the user to review -- nothing is persisted here. Approving
    one calls POST /api/sources/resolve from the frontend. Companies already monitored are
    excluded so re-running discovery doesn't keep re-suggesting the same ones."""
    try:
        suggestions = suggest_companies(
            get_current_candidate_profile(),
            get_settings(),
            exclude_names=_already_monitored_company_names(),
            guidance=payload.guidance if payload else None,
            like=payload.like if payload else None,
        )
    except ValueError as exc:
        # _extract_json raises a plain ValueError if Claude's response has no parseable JSON
        # (e.g. cut off mid-output after a long web-search tool-use turn); pydantic.ValidationError
        # from a malformed suggestion also subclasses ValueError, so this catches both -- without
        # it, either case propagates as an unhandled exception and surfaces as a raw 500.
        raise HTTPException(
            502, f"Claude's response couldn't be parsed (likely cut off mid-output) — try again: {exc}"
        ) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return [s.model_dump() for s in suggestions]


@router.post("/suggest-boards")
def suggest_boards(payload: DiscoveryRequest | None = None):
    """Returns suggested multi-employer job boards for the user to review -- nothing is persisted
    here. Approving one calls POST /api/sources directly (kind=custom_board) from the frontend,
    since a board's URL is already known -- no resolution step needed. Boards already monitored
    are excluded by hostname."""
    try:
        suggestions = suggest_job_boards(
            get_current_candidate_profile(),
            get_settings(),
            exclude_urls=_already_monitored_board_urls(),
            guidance=payload.guidance if payload else None,
            like=payload.like if payload else None,
        )
    except ValueError as exc:
        raise HTTPException(
            502, f"Claude's response couldn't be parsed (likely cut off mid-output) — try again: {exc}"
        ) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return [s.model_dump() for s in suggestions]
