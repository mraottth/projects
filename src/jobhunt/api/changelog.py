"""The Timeline page's data (scripts/build_changelog.py). Served here rather than bundled into the
frontend: the frontend's files are readable by anyone who gets through IAP's sign-in, approved or
not, while every /api route needs an approved person (api/auth.py)."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["changelog"])

CHANGELOG_PATH = Path(__file__).resolve().parents[3] / "frontend" / "src" / "changelog.json"


@router.get("/api/changelog")
def changelog():
    if not CHANGELOG_PATH.exists():
        raise HTTPException(404, "the Timeline hasn't been built (scripts/build_changelog.py)")
    return FileResponse(CHANGELOG_PATH, media_type="application/json")
