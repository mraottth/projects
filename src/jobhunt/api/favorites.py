from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobhunt.core import favorites

router = APIRouter(prefix="/api/favorites", tags=["favorites"])


class AddFavoriteRequest(BaseModel):
    company_name: str


@router.get("")
def list_favorites():
    return favorites.list_favorites()


@router.post("")
def add_favorite(payload: AddFavoriteRequest):
    # add_favorite() absorbs resolution failures (including LLM-call errors) into a 'failed'
    # record rather than raising -- see its docstring -- so there's nothing to catch here.
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
