"""Your LinkedIn network (D-036): the Connections.csv upload on Manage Sources, who to reach out to,
and the people you know at a job's organization (core/network.py)."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from jobhunt.core import network
from jobhunt.core.llm import LLMError
from jobhunt.core.settings import update_settings
from jobhunt.core.users import current_user_id, job_is_visible
from jobhunt.db.connection import get_db

router = APIRouter(prefix="/api/network", tags=["network"])
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10_000_000


@router.get("")
def get_network():
    return network.overview()


@router.post("/connections")
async def upload_connections(file: UploadFile = File(...)):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"that file is over {MAX_UPLOAD_BYTES // 1_000_000} MB")
    try:
        rows = network.parse_csv(contents)
    except network.CsvError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not rows:
        raise HTTPException(400, "That file has no connections in it.")
    network.replace_connections(rows)
    # Organizations with jobs are judged by the free rules; Claude reads only the rest, by name.
    with get_db() as conn:
        with_jobs = set(network._jobs_by_org(conn, current_user_id()))
    try:
        network.pick_organizations(exclude=with_jobs)
        update_settings({"network_orgs_error": None})
    except LLMError as exc:
        logger.warning("network organization picks failed: %s", exc)
        update_settings({"network_orgs": {}, "network_orgs_error": str(exc)})
    return network.overview()


@router.delete("/connections")
def delete_connections():
    with get_db() as conn:
        conn.execute("DELETE FROM connections WHERE user_id = ?", (current_user_id(),))
    update_settings({"network_orgs": {}, "network_uploaded_at": None, "network_orgs_error": None})
    return network.overview()


@router.get("/job/{job_id}")
def people_for_job(job_id: int):
    if not job_is_visible(job_id):
        raise HTTPException(404, "job not found")
    with get_db() as conn:
        row = conn.execute("SELECT company FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return network.people_at(row["company"] if row else None)
