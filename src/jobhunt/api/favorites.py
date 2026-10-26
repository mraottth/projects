from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.core import favorites

router = APIRouter(prefix="/api/favorites", tags=["favorites"])


class AddFavoriteRequest(BaseModel):
    company_name: str
    kind: str | None = None
    config: dict | None = None


@router.get("")
def list_favorites():
    return favorites.list_favorites()


@router.post("")
def add_favorite(payload: AddFavoriteRequest):
    # Given a kind+config (the user supplied a job board URL, detected client-side), skip
    # company_resolver entirely -- nothing to guess, nothing that can fail. Otherwise fall back to
    # name-only resolution (used by Discovery's "Monitor & favorite" approval, which has no URL).
    # add_favorite() absorbs resolution failures (including LLM-call errors) into a 'failed'
    # record rather than raising -- see its docstring -- so there's nothing to catch here either way.
    if payload.kind and payload.config is not None:
        return favorites.add_favorite_with_url(payload.company_name, payload.kind, payload.config)
    return favorites.add_favorite(payload.company_name)


@router.post("/{favorite_id}/retry")
def retry_favorite(favorite_id: int):
    try:
        return favorites.retry_favorite(favorite_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/{favorite_id}")
def remove_favorite(favorite_id: int):
    favorites.remove_favorite(favorite_id)
    return {"ok": True}
