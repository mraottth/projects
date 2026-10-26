from fastapi import APIRouter, HTTPException

from jobhunt.core.discovery import suggest_companies
from jobhunt.core.llm import LLMError
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.settings import get_settings

router = APIRouter(prefix="/api/discovery", tags=["discovery"])


@router.post("/suggest")
def suggest():
    """Returns suggested companies for the user to review -- nothing is persisted here. Approving
    one calls POST /api/sources/resolve (optionally with favorite=true, which routes through
    /api/favorites instead) from the frontend."""
    try:
        suggestions = suggest_companies(get_current_candidate_profile(), get_settings())
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return [s.model_dump() for s in suggestions]
