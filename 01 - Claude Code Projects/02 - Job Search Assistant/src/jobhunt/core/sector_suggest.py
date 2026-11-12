"""Sector suggestions from the resume (asked 2026-11-12: a tester didn't know what to put for
"sector" when thinking of an organization like a political-data firm).

"What are you looking for?" shows a few sector names drawn from the current resume as clickable
chips in its hint; a click adds one, and nothing is filled in for them. One small Claude Haiku call,
kept in the `sector_suggestions` setting against the resume it came from, so revisiting the form
costs nothing until the current resume changes.
"""

from pydantic import BaseModel

from jobhunt.core.llm import LLMError, get_client, llm_error_boundary, log_call
from jobhunt.core.settings import get_setting, get_settings, update_settings
from jobhunt.core.users import current_user_id
from jobhunt.db.connection import get_db

MODEL = "claude-haiku-5-5"
SETTING = "sector_suggestions"
RESUME_CHARS = 8_000

SYSTEM_PROMPT = """From a job seeker's resume, suggest 4 to 6 sectors they might want to work in: the \
kind of work or organization, in two or three plain words each, like "civic tech", "political data \
and campaigns", "public health", "education", "climate", "fintech" or "nonprofit". Base them on the \
fields their experience points to, including adjacent ones they could move into. Don't repeat any \
they already listed."""


class Suggestions(BaseModel):
    sectors: list[str]


def _current_resume() -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, raw_text FROM resumes WHERE user_id = ? AND kind = 'resume' "
            "ORDER BY is_template DESC, is_archived ASC, "
            "COALESCE(source_modified_at, uploaded_at) DESC, id DESC "
            "LIMIT 1",
            (current_user_id(),),
        ).fetchone()
    return dict(row) if row else None


def suggestions() -> list[str]:
    """Sector names for the current resume, minus any already listed. [] with no resume."""
    resume = _current_resume()
    if resume is None or not (resume["raw_text"] or "").strip():
        return []
    listed = {s.lower() for s in get_settings().get("prefilter_sectors") or []}
    saved = get_setting(SETTING) or {}
    if saved.get("resume_id") == resume["id"]:
        return [s for s in saved.get("sectors", []) if s.lower() not in listed]

    client = get_client()
    with llm_error_boundary("suggesting sectors"):
        response = client.messages.parse(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Already listed: {', '.join(sorted(listed)) or '(none)'}\n\n"
                    f"## Resume\n{resume['raw_text'][:RESUME_CHARS]}",
                }
            ],
            output_format=Suggestions,
            output_config={"effort": "low"},
        )
    log_call(
        call_type="parsing",
        model=MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note="sector suggestions",
    )
    if response.parsed_output is None:
        raise LLMError("no sector suggestions came back")
    sectors = [s.strip() for s in response.parsed_output.sectors if s.strip()][:6]
    update_settings({SETTING: {"resume_id": resume["id"], "sectors": sectors}})
    return [s for s in sectors if s.lower() not in listed]
