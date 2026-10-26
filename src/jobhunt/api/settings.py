from fastapi import APIRouter, HTTPException

from jobhunt.core.llm import LLMError
from jobhunt.core.preference import get_current_preference_profile, rebuild_preference_profile

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/preference-profile")
def get_preference_profile():
    text = get_current_preference_profile()
    if text is None:
        raise HTTPException(404, "no preference profile yet — mark some jobs or track an application")
    return {"text": text}


@router.post("/preference-profile/rebuild")
def rebuild_preference_profile_route():
    try:
        text = rebuild_preference_profile()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"text": text}
