"""Jobby: the interview-prep chat assistant. One saved conversation per tracked job (plus one
general conversation, job_id NULL), with the job posting, the candidate profile, and the user's
documents as context.

The context goes in the system prompt behind a cache breakpoint, so every later message in a
conversation re-reads it from cache instead of paying full price for it again. Replies stream
(stream_reply yields events for api/jobby.py to forward as server-sent events) and can use web
search, e.g. for company research; a long server-tool turn that pauses (stop_reason
"pause_turn") is resumed, same as discovery's run_web_search."""

from collections.abc import Iterator

from jobhunt.core.llm import DEFAULT_MODEL, LLMError, cache_tokens, get_client, llm_error_boundary, log_call
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.settings import get_setting, get_settings
from jobhunt.db.connection import get_db

PERSONA = """You are Jobby, a friendly, sharp interview-prep coach inside a personal job-search \
app. You're talking with the job seeker directly -- write to them in the second person ("you", \
"your background"). You're warm and encouraging but practical and honest: concrete advice beats \
pep talk, and if an answer they draft is weak, say so kindly and show how to make it stronger.

You have their context below: the job they're preparing for (if any), their candidate profile, \
what they're looking for, and their resumes and cover letters. Ground everything in it -- tie \
advice to specific requirements in the posting and specific experience in their documents. Never \
invent experience, skills, employers, or results they don't have; if something the role wants \
isn't in their background, help them address the gap honestly instead.

Keep replies focused and skimmable: short paragraphs, bullets or numbered lists where they help, \
**bold** for the few things that matter most. Don't pad.

Running a mock interview: play the interviewer for this specific role. Ask ONE question at a \
time, then stop and wait for their answer. After each answer, give brief feedback (what landed, \
what to sharpen, one concrete improvement), then ask the next question. Mix behavioral and \
role-specific questions, about 6-8 in total unless they ask for more. When it's done, sum up \
their strengths and the 2-3 things to work on.

When answering behavioral questions, help them use the STAR structure (situation, task, action, \
result) with real stories from their documents.

You can search the web. Use it when current, specific facts would make your answer better -- \
the company's recent news, funding, leadership, products, or reported interview process -- and \
not for things their documents or general knowledge already cover. Mention what you found and \
where, briefly."""

# Older documents are trimmed so a big archive doesn't swamp the context (they matter far less
# than the current ones, which are kept whole up to a generous limit).
CURRENT_DOC_CHARS = 15_000
OLDER_DOC_CHARS = 4_000
OLDER_DOCS_TOTAL_CHARS = 20_000
TAILORED_DOC_CHARS = 8_000
HISTORY_MESSAGES = 40
MAX_WEB_SEARCHES = 3
MAX_CONTINUATIONS = 3
JOBBY_EFFORT = "medium"


def _clip(text: str | None, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n[... trimmed ...]"


def get_job(job_id: int) -> dict | None:
    """The job plus its tracking status and latest fit rationale -- what Jobby needs about it."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT j.*, a.status AS application_status, a.notes AS application_notes,
                      a.applied_at, js.interest_fit, js.candidate_fit, js.rationale_text
               FROM jobs j
               LEFT JOIN applications a ON a.job_id = j.id
               LEFT JOIN job_scores js ON js.id = (
                   SELECT id FROM job_scores WHERE job_id = j.id ORDER BY id DESC LIMIT 1
               )
               WHERE j.id = ?""",
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def _documents_text(job_id: int | None) -> str:
    with get_db() as conn:
        docs = [
            dict(r)
            for r in conn.execute(
                "SELECT kind, file_name, raw_text, is_template, "
                "COALESCE(source_modified_at, uploaded_at) AS effective_date FROM resumes "
                "WHERE raw_text IS NOT NULL ORDER BY is_template DESC, effective_date DESC"
            )
        ]
        tailored = []
        if job_id is not None:
            tailored = [
                dict(r)
                for r in conn.execute(
                    "SELECT kind, content_text FROM generated_documents WHERE job_id = ? "
                    "ORDER BY id DESC",
                    (job_id,),
                )
            ]

    parts = []
    for doc in (d for d in docs if d["is_template"]):
        label = "Current resume" if doc["kind"] == "resume" else "Current cover letter"
        parts.append(f"### {label} ({doc['file_name']})\n{_clip(doc['raw_text'], CURRENT_DOC_CHARS)}")

    seen_kinds = set()
    for doc in tailored:
        if doc["kind"] in seen_kinds:
            continue  # only the latest of each kind
        seen_kinds.add(doc["kind"])
        label = "resume" if doc["kind"] == "resume" else "cover letter"
        parts.append(
            f"### Tailored {label} written for this job\n{_clip(doc['content_text'], TAILORED_DOC_CHARS)}"
        )

    budget = OLDER_DOCS_TOTAL_CHARS
    for doc in (d for d in docs if not d["is_template"]):
        if budget <= 0:
            parts.append("### (more older documents not shown)")
            break
        text = _clip(doc["raw_text"], min(OLDER_DOC_CHARS, budget))
        budget -= len(text)
        kind = "resume" if doc["kind"] == "resume" else "cover letter"
        parts.append(f"### Older {kind} ({doc['file_name']}, {doc['effective_date'][:10]})\n{text}")

    return "\n\n".join(parts) or "(no documents uploaded)"


def build_context(job_id: int | None) -> str:
    settings = get_settings()
    sections = []

    if job_id is not None:
        job = get_job(job_id)
        if job is None:
            raise ValueError(f"job {job_id} not found")
        fit = ""
        if job.get("interest_fit"):
            fit = (
                f"\nFit rating: interest {job['interest_fit']}, candidate fit {job['candidate_fit']}"
                f"\nWhy: {job.get('rationale_text') or ''}"
            )
        tracking = ""
        if job.get("application_status"):
            tracking = f"\nApplication status: {job['application_status']}"
            if job.get("applied_at"):
                tracking += f" (applied {job['applied_at'][:10]})"
            if job.get("application_notes"):
                tracking += f"\nTheir notes: {job['application_notes']}"
        sections.append(
            "## The job they're preparing for\n"
            f"Title: {job['title']}\nCompany: {job['company']}\n"
            f"Location: {job.get('location') or 'unknown'}{tracking}{fit}\n\n"
            f"Posting:\n{_clip(job.get('description_text') or '(no description saved)', 20_000)}"
        )
    else:
        sections.append(
            "## The job they're preparing for\nNone selected -- this is general interview help. If "
            "they want help with a specific role, they can pick it in the job list on the left."
        )

    titles = ", ".join(settings.get("prefilter_titles_include") or []) or "(not specified)"
    sectors = ", ".join(settings.get("prefilter_sectors") or []) or "(not specified)"
    notes = (settings.get("looking_for_notes") or "").strip() or "(not specified)"
    sections.append(
        "## What they're looking for\n"
        f"Target roles: {titles}\nTarget sectors: {sectors}\nIn their own words: {notes}"
    )
    sections.append(
        "## Their candidate profile\n"
        + (get_current_candidate_profile() or "(not built yet -- rely on their documents)")
    )
    sections.append("## Their documents\n" + _documents_text(job_id))
    if job_id is None:
        # The read-out Jobby gave on My Jobs ("Ask Jobby what's working"), so the general
        # conversation can pick it up when they want to talk it over.
        insights = get_setting("application_insights")
        if insights:
            sections.append(
                f"## What you told them about their applications ({insights['generated_at'][:10]})\n"
                + insights["text"]
            )
    return "\n\n".join(sections)


# ---------- conversations


def get_thread(job_id: int | None) -> dict | None:
    with get_db() as conn:
        if job_id is None:
            row = conn.execute(
                "SELECT * FROM jobby_threads WHERE job_id IS NULL ORDER BY id LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM jobby_threads WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def _get_or_create_thread(job_id: int | None) -> int:
    thread = get_thread(job_id)
    if thread:
        return thread["id"]
    with get_db() as conn:
        return conn.execute("INSERT INTO jobby_threads (job_id) VALUES (?)", (job_id,)).lastrowid


def list_messages(job_id: int | None) -> list[dict]:
    thread = get_thread(job_id)
    if thread is None:
        return []
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM jobby_messages WHERE thread_id = ? ORDER BY id",
            (thread["id"],),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_thread(job_id: int | None) -> None:
    thread = get_thread(job_id)
    if thread is None:
        return
    with get_db() as conn:
        conn.execute("DELETE FROM jobby_messages WHERE thread_id = ?", (thread["id"],))
        conn.execute("DELETE FROM jobby_threads WHERE id = ?", (thread["id"],))


def _add_message(thread_id: int, role: str, content: str) -> dict:
    with get_db() as conn:
        message_id = conn.execute(
            "INSERT INTO jobby_messages (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content),
        ).lastrowid
        conn.execute(
            "UPDATE jobby_threads SET updated_at = datetime('now') WHERE id = ?", (thread_id,)
        )
        row = conn.execute(
            "SELECT id, role, content, created_at FROM jobby_messages WHERE id = ?", (message_id,)
        ).fetchone()
    return dict(row)


def _history(thread_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM jobby_messages WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
            (thread_id, HISTORY_MESSAGES),
        ).fetchall()
    messages = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    # The trimmed window must open with a user turn.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


# ---------- replies


def stream_reply(job_id: int | None, content: str) -> Iterator[dict]:
    """Saves the user's message, streams Jobby's reply, and saves that too. Yields events:
    {"type": "delta", "text"} as text arrives, {"type": "status", "text"} while searching the web,
    then {"type": "done", "user": message, "assistant": message} -- or {"type": "error",
    "message"}, in which case nothing from this turn is kept, so the user can just retry."""
    content = content.strip()
    if not content:
        yield {"type": "error", "message": "Type a message first."}
        return
    try:
        context = build_context(job_id)
    except ValueError as exc:
        yield {"type": "error", "message": str(exc)}
        return

    thread_id = _get_or_create_thread(job_id)
    user_message = _add_message(thread_id, "user", content)
    messages = _history(thread_id)
    system = [
        {"type": "text", "text": PERSONA},
        {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
    ]

    reply_parts: list[str] = []
    try:
        client = get_client()
        for _ in range(MAX_CONTINUATIONS + 1):
            with llm_error_boundary("Jobby's reply"):
                with client.messages.stream(
                    model=DEFAULT_MODEL,
                    max_tokens=8000,
                    system=system,
                    messages=messages,
                    tools=[
                        {"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_WEB_SEARCHES}
                    ],
                    output_config={"effort": JOBBY_EFFORT},
                    # Automatic caching for the growing conversation (the system block above has
                    # its own breakpoint): a reply within 5 minutes, as in a mock interview,
                    # re-reads the earlier turns from cache instead of at full price.
                    cache_control={"type": "ephemeral"},
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_start" and getattr(
                            event.content_block, "type", None
                        ) == "server_tool_use":
                            yield {"type": "status", "text": "Searching the web..."}
                        elif event.type == "content_block_delta" and getattr(
                            event.delta, "type", None
                        ) == "text_delta":
                            reply_parts.append(event.delta.text)
                            yield {"type": "delta", "text": event.delta.text}
                    final = stream.get_final_message()
            searches = sum(1 for b in final.content if b.type == "server_tool_use")
            log_call(
                call_type="generation",
                model=DEFAULT_MODEL,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                job_id=job_id,
                note=f"jobby chat ({searches} web search(es))",
                **cache_tokens(final.usage),
            )
            if final.stop_reason != "pause_turn":
                break
            messages = [*messages, {"role": "assistant", "content": final.content}]
    except LLMError as exc:
        _delete_message(user_message["id"])
        yield {"type": "error", "message": str(exc)}
        return

    reply = "".join(reply_parts).strip() or "(Jobby didn't have anything to say -- try rephrasing.)"
    assistant_message = _add_message(thread_id, "assistant", reply)
    yield {"type": "done", "user": user_message, "assistant": assistant_message}


def _delete_message(message_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM jobby_messages WHERE id = ?", (message_id,))
