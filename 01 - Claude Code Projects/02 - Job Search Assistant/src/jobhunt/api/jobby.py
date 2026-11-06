import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from jobhunt.core import jobby

router = APIRouter(prefix="/api/jobby", tags=["jobby"])

# Every route takes an optional ?job_id= -- omitted means the general (no specific job) conversation.


def _check_job(job_id: int | None) -> dict | None:
    if job_id is None:
        return None
    job = jobby.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@router.get("/thread")
def get_thread(job_id: int | None = None):
    job = _check_job(job_id)
    return {
        "job": (
            {"id": job["id"], "title": job["title"], "company": job["company"],
             "status": job.get("application_status")}
            if job
            else None
        ),
        "messages": jobby.list_messages(job_id),
    }


@router.delete("/thread")
def clear_thread(job_id: int | None = None):
    _check_job(job_id)
    jobby.clear_thread(job_id)
    return {"ok": True}


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)


@router.post("/thread/messages")
def send_message(payload: MessageRequest, job_id: int | None = None):
    """Streams Jobby's reply as server-sent events (one JSON object per `data:` line; see
    core/jobby.stream_reply for the event shapes). Errors after the stream has started arrive as
    an `error` event, not an HTTP status, since the 200 has already been sent by then."""
    _check_job(job_id)

    def events():
        for event in jobby.stream_reply(job_id, payload.content):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Don't let a proxy buffer the stream into one late chunk.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
