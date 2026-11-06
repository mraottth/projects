import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from jobhunt.api import (
    access,
    admin,
    applications,
    changelog,
    digest,
    discovery,
    generate,
    internal,
    jobby,
    jobs,
    onboarding,
    pipeline,
    rescore,
    resumes,
    settings,
    sources,
)
from jobhunt.api.auth import AuthMiddleware
from jobhunt.db.connection import init_db
from jobhunt.pipeline.daily_run import backfill_salaries

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


@asynccontextmanager
def _backfill_salaries_safely() -> None:
    try:
        backfill_salaries()
    except Exception:  # noqa: BLE001 - never let this take the app down
        logging.getLogger(__name__).exception("salary backfill at startup failed")


async def lifespan(app: FastAPI):
    init_db()
    # Fill in salaries readable from text already stored, so a deploy that improves pay reading
    # shows on existing job cards right away instead of after the next daily run. Regex only. It
    # runs in the background so a cold start serves the first page without waiting for it; shutdown
    # waits briefly for it to finish.
    backfill = threading.Thread(target=_backfill_salaries_safely, name="salary-backfill", daemon=True)
    backfill.start()
    yield
    backfill.join(timeout=30)


app = FastAPI(title="Job Search Assistant", lifespan=lifespan)
# Every /api request runs as its verified, approved signed-in person (D-021; api/auth.py).
app.add_middleware(AuthMiddleware)

app.include_router(access.router)
app.include_router(admin.router)
app.include_router(changelog.router)
app.include_router(resumes.router)
app.include_router(jobs.router)
app.include_router(digest.router)
app.include_router(applications.router)
app.include_router(settings.router)
app.include_router(sources.router)
app.include_router(internal.router)
app.include_router(generate.router)
app.include_router(discovery.router)
app.include_router(onboarding.router)
app.include_router(pipeline.router)
app.include_router(jobby.router)
app.include_router(rescore.router)

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
