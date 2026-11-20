"""Your LinkedIn network (D-036): the Connections.csv upload on Manage Sources, who to reach out to,
and the people you know at a job's organization (core/network.py)."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from jobhunt.core import network
from jobhunt.core.llm import LLMError
from jobhunt.core.settings import get_settings, update_settings
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
    _rate()
    return network.overview()


@router.post("/rate")
def rate_organizations():
    """Rates the connections' organizations again (e.g. ratings saved before fit/interest existed)."""
    _rate()
    return network.overview()


def _rate() -> None:
    """Claude rates every connection's organization, by name, except your own. A failure keeps the
    free tiers and is shown on the card."""
    with get_db() as conn:
        uid = current_user_id()
        people = [dict(r) for r in conn.execute("SELECT company FROM connections WHERE user_id = ?", (uid,))]
        mine, _ = network.current_org(conn, uid, people, get_settings())
    try:
        network.pick_organizations(exclude={network.org_key(mine)} if mine else set())
        update_settings({"network_orgs_error": None})
    except LLMError as exc:
        logger.warning("rating network organizations failed: %s", exc)
        update_settings({"network_orgs": {}, "network_orgs_error": str(exc)})


class ReachedOutRequest(BaseModel):
    id: int
    reached: bool


@router.post("/reached-out")
def reached_out(body: ReachedOutRequest):
    """The "Reached out" checkmark on My Network's table (asked 2026-11-17)."""
    try:
        return {"reached_out": network.set_reached_out(body.id, body.reached)}
    except LookupError:
        raise HTTPException(404, "connection not found") from None


class ContactRequest(BaseModel):
    name: str = Field(max_length=200)
    url: str = Field(max_length=500)
    position: str | None = Field(default=None, max_length=300)
    company: str | None = Field(default=None, max_length=300)


@router.post("/contacts")
def add_contact(body: ContactRequest):
    """Adds one person by hand, without a new Connections.csv (asked 2026-11-20)."""
    try:
        added = network.add_contact(body.name, body.url, body.position, body.company)
    except network.ContactError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {**network.overview(), "added_id": added["id"]}


@router.delete("/contacts/{connection_id}")
def remove_contact(connection_id: int):
    try:
        network.remove_contact(connection_id)
    except LookupError:
        raise HTTPException(404, "contact not found") from None
    return network.overview()


class PriorityRequest(BaseModel):
    id: int
    # "High" / "Medium" / "Low" / "Not listed", or null to go back to the automatic tier.
    priority: str | None = None


@router.post("/priority")
def set_priority(body: PriorityRequest):
    """Your own priority for a person (asked 2026-11-20), over the automatic tier."""
    try:
        network.set_priority(body.id, body.priority)
    except network.ContactError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LookupError:
        raise HTTPException(404, "connection not found") from None
    return network.overview()


@router.delete("/connections")
def delete_connections():
    with get_db() as conn:
        # The uploaded ones; people added by hand stay until removed one by one.
        conn.execute("DELETE FROM connections WHERE user_id = ? AND manual = 0", (current_user_id(),))
    update_settings(
        {
            "network_orgs": {},
            "network_uploaded_at": None,
            "network_orgs_error": None,
            "network_current_org": None,  # guessed again from the next upload
            "network_current_org_guessed": False,
        }
    )
    return network.overview()


@router.get("/job/{job_id}")
def people_for_job(job_id: int):
    if not job_is_visible(job_id):
        raise HTTPException(404, "job not found")
    with get_db() as conn:
        row = conn.execute("SELECT company FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return network.people_at(row["company"] if row else None)
