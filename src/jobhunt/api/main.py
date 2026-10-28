from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from jobhunt.api import (
    applications,
    digest,
    discovery,
    favorites,
    generate,
    internal,
    jobs,
    onboarding,
    pipeline,
    resumes,
    settings,
    sources,
)
from jobhunt.db.connection import init_db

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Job Search Assistant", lifespan=lifespan)

app.include_router(resumes.router)
app.include_router(jobs.router)
app.include_router(digest.router)
app.include_router(applications.router)
app.include_router(settings.router)
app.include_router(sources.router)
app.include_router(internal.router)
app.include_router(generate.router)
app.include_router(favorites.router)
app.include_router(discovery.router)
app.include_router(onboarding.router)
app.include_router(pipeline.router)

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    # SPA fallback: react-router handles routing client-side, so a direct navigation or refresh
    # on e.g. /profile has to get index.html too, not a 404 -- StaticFiles(html=True) alone only
    # serves index.html for "/" and real directories, not arbitrary unmatched paths (caught live:
    # every page but Digest 404'd on direct navigation before this).
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith(("api/", "internal/")):
            raise HTTPException(404)
        return FileResponse(FRONTEND_DIST / "index.html")
