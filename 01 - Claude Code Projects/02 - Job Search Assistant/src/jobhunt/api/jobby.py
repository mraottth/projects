import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from jobhunt.core import jobby

router = APIRouter(prefix="/api/jobby", tags=["jobby"])

# A conversation is addressed by ?thread_id= (any chat, reopened from Jobby's home), or by ?job_id=
# (a job's one chat). With neither, it's the first general chat, or with ?new=true on a message, a
# fresh general one.


def _check_job(job_id: int | None) -> dict | None:
    if job_id is None:
        return None
    job = jobby.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


def _thread(thread_id: int) -> dict:
    thread = jobby.thread_by_id(thread_id)
    if thread is None:
        raise HTTPException(404, "chat not found")
    return thread


@router.get("/threads")
def list_threads():
    """Jobby's home: this person's past chats, most recent first."""
    return jobby.list_threads()


@router.delete("/threads/{thread_id}")
def delete_thread(thread_id: int):
    _thread(thread_id)
    jobby.delete_thread(thread_id)
    return {"ok": True}


def _job_summary(job: dict | None) -> dict | None:
    if not job:
        return None
    return {"id": job["id"], "title": job["title"], "company": job["company"],
            "status": job.get("application_status")}  # fmt: skip


@router.get("/thread")
def get_thread(job_id: int | None = None, thread_id: int | None = None):
    if thread_id is not None:
        thread = _thread(thread_id)
        job = jobby.get_job(thread["job_id"]) if thread["job_id"] is not None else None
        return {
            "thread_id": thread_id,
            "job": _job_summary(job),
            "messages": jobby.thread_messages(thread_id),
        }
    job = _check_job(job_id)
    existing = jobby.get_thread(job_id)
    return {
        "thread_id": existing["id"] if existing else None,
        "job": _job_summary(job),
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
def send_message(
    payload: MessageRequest, job_id: int | None = None, thread_id: int | None = None, new: bool = False
):
    """Streams Jobby's reply as server-sent events (one JSON object per `data:` line; see
    core/jobby.stream_reply for the event shapes). Errors after the stream has started arrive as
    an `error` event, not an HTTP status, since the 200 has already been sent by then. The `done`
    event says which chat it went to (`thread_id`), so a new general chat can be reopened."""
    if thread_id is not None:
        job_id = _thread(thread_id)["job_id"]
    _check_job(job_id)

    def events():
        for event in jobby.stream_reply(job_id, payload.content, thread_id=thread_id, new=new):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Don't let a proxy buffer the stream into one late chunk.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
