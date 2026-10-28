from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from jobhunt.core.generate import generate_starter_cover_letter
from jobhunt.core.llm import LLMError
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.settings import get_settings
from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])


@router.get("/status")
def onboarding_status():
    """Backs the frontend's first-run gate in one call. Onboarding is only forced on a user with no
    resumes/cover letters at all who hasn't finished (or skipped) it -- anyone who already has
    documents, e.g. from the bulk CLI import, never gets redirected."""
    settings = get_settings()
    with get_db() as conn:
        counts = {
            row["kind"]: row["n"]
            for row in conn.execute("SELECT kind, COUNT(*) AS n FROM resumes GROUP BY kind").fetchall()
        }
    has_any_docs = bool(counts)
    return {
        "needs_onboarding": not settings.get("onboarding_completed", False) and not has_any_docs,
        "step": settings.get("onboarding_step", "resume"),
        "has_resume": counts.get("resume", 0) > 0,
        "has_cover_letter": counts.get("cover_letter", 0) > 0,
        "has_profile": get_current_candidate_profile() is not None,
    }


@router.post("/generate-cover-letter")
def generate_cover_letter_for_onboarding():
    try:
        return generate_starter_cover_letter()
    except ValidationError as exc:
        # Subclasses ValueError -- must be caught first, same as api/generate.py.
        raise HTTPException(
            502, f"Claude's response couldn't be parsed (likely cut off mid-output) — try again: {exc}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
