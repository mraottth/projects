from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from jobhunt.api import digest, jobs, resumes
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

if FRONTEND_DIST.exists():
    app.mount(
        "/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend"
    )
