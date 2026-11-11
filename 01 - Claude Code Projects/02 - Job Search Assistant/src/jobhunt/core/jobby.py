"""Jobby: the interview-prep chat assistant. One saved conversation per tracked job (plus one
general conversation, job_id NULL), with the job posting, the candidate profile, and the user's
documents as context.

The context goes in the system prompt behind a cache breakpoint, so every later message in a
conversation re-reads it from cache instead of paying full price for it again. Replies stream
(stream_reply yields events for api/jobby.py to forward as server-sent events) and can use web
search, e.g. for company research; a long server-tool turn that pauses (stop_reason
"pause_turn") is resumed, same as discovery's run_web_search."""

import logging
import re
from collections.abc import Iterator

from jobhunt.core.llm import (
    DEFAULT_MODEL,
    LLMError,
    cache_tokens,
    get_client,
    llm_error_boundary,
    log_call,
    response_text,
)
from jobhunt.core.preference import get_current_preference_profile
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.settings import get_setting, get_settings
from jobhunt.db.connection import get_db
from jobhunt.core.users import VISIBLE_JOB_SQL, current_user_id

PERSONA = """You are Jobby, a friendly, sharp job-search coach inside a personal job-search app. \
With a job selected, you help them prepare for it: interview practice, likely questions, framing \
their background, research. In a general chat, you help with the search itself: finding \
organizations and roles worth pursuing, clarifying their goals and strategy, feedback on their \
resume and cover letter, outreach to contacts, and planning their week. You're talking with the \
job seeker directly -- always write to them in the second person ("you", "your background"), \
never about them in the third person. You're warm and encouraging but practical and honest: \
concrete advice beats pep talk, and if something they draft is weak, say so kindly and show how \
to make it stronger.

You have their context below: the job they're preparing for (if any), what they're looking for, \
their candidate profile, what their marks and applications say they want, the jobs that score \
highest for them and why, and their resumes. Ground everything in it -- tie advice to specific \
requirements in the posting and specific experience in their resumes and profile. Their resumes \
and profile are the record of what they've done; their highest-scoring jobs and what makes them \
score well show what they're best suited to and most want. A cover letter, if one is included, \
is only a sample of how they write: not every application needed one and not all of theirs are \
here, so don't treat what it emphasizes as what matters most. Never invent experience, skills, \
employers, or results they don't have; if something the role wants isn't in their background, \
help them address the gap honestly instead.

Keep replies focused and skimmable: short paragraphs, bullets or numbered lists where they help, \
**bold** for the few things that matter most. Don't pad.

Running a mock interview: play the interviewer for this specific role. Ask ONE question at a \
time, then stop and wait for their answer. After each answer, give brief feedback (what landed, \
what to sharpen, one concrete improvement), then ask the next question. Mix behavioral and \
role-specific questions, about 6-8 in total unless they ask for more. When it's done, sum up \
their strengths and the 2-3 things to work on.

When answering behavioral questions, help them use the STAR structure (situation, task, action, \
result) with real stories from their resumes and profile.

You can search the web. Use it when current, specific facts would make your answer better -- \
the company's recent news, funding, leadership, products, or reported interview process -- and \
not for things their documents or general knowledge already cover. Mention what you found and \
where, briefly."""

# Older resumes are trimmed so a big archive doesn't swamp the context (they matter far less
# than the current ones, which are kept whole up to a generous limit).
CURRENT_DOC_CHARS = 15_000
OLDER_DOC_CHARS = 4_000
OLDER_DOCS_TOTAL_CHARS = 20_000
TAILORED_DOC_CHARS = 8_000
# Cover letters skewed Jobby (asked 2026-11-10): not every application needs one and only some are
# uploaded, so a partial set read as what matters most. Only the current one goes in, short, as a
# writing sample; older ones are left out.
COVER_LETTER_SAMPLE_CHARS = 5_000  # room for a whole letter, for "Cover letter feedback"
# The jobs that score highest for them, and why (asked 2026-11-10): what they're best suited to.
TOP_JOBS = 8
TOP_JOB_REASON_CHARS = 600
HIGHLIGHT = re.compile(r"==(?:[if]:)?(.*?)==", re.S)
HISTORY_MESSAGES = 40
MAX_WEB_SEARCHES = 3
MAX_CONTINUATIONS = 3
JOBBY_EFFORT = "medium"


def _clip(text: str | None, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n[... trimmed ...]"


def get_job(job_id: int) -> dict | None:
    """The job plus its tracking status and latest fit rationale -- what Jobby needs about it.
    None if there's no such job, or it isn't one this person can see."""
    with get_db() as conn:
        row = conn.execute(
            f"""SELECT j.*, a.status AS application_status, a.notes AS application_notes,
                      a.applied_at, js.interest_fit, js.candidate_fit, js.rationale_text
               FROM jobs j
               LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = :uid
               LEFT JOIN job_scores js ON js.id = (
                   SELECT id FROM job_scores WHERE user_id = :uid AND job_id = j.id ORDER BY id DESC LIMIT 1
               )
               WHERE j.id = :job AND {VISIBLE_JOB_SQL}""",
            {"uid": current_user_id(), "job": job_id},
        ).fetchone()
    return dict(row) if row else None


def _documents_text(job_id: int | None) -> str:
    with get_db() as conn:
        docs = [
            dict(r)
            for r in conn.execute(
                "SELECT kind, file_name, raw_text, is_template, "
                "COALESCE(source_modified_at, uploaded_at) AS effective_date FROM resumes "
                "WHERE user_id = ? AND raw_text IS NOT NULL ORDER BY is_template DESC, effective_date DESC",
                (current_user_id(),),
            )
        ]
        tailored = []
        if job_id is not None:
            tailored = [
                dict(r)
                for r in conn.execute(
                    "SELECT kind, content_text FROM generated_documents WHERE user_id = ? AND job_id = ? "
                    "ORDER BY id DESC",
                    (current_user_id(), job_id),
                )
            ]

    resumes = [d for d in docs if d["kind"] == "resume"]
    letters = [d for d in docs if d["kind"] != "resume"]
    parts = []
    for doc in (d for d in resumes if d["is_template"]):
        parts.append(f"### Current resume ({doc['file_name']})\n{_clip(doc['raw_text'], CURRENT_DOC_CHARS)}")

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
    for doc in (d for d in resumes if not d["is_template"]):
        if budget <= 0:
            parts.append("### (more older resumes not shown)")
            break
        text = _clip(doc["raw_text"], min(OLDER_DOC_CHARS, budget))
        budget -= len(text)
        parts.append(f"### Older resume ({doc['file_name']}, {doc['effective_date'][:10]})\n{text}")

    current_letter = next((d for d in letters if d["is_template"]), None)
    if current_letter is not None:
        parts.append(
            "### A sample of how they write: their current cover letter (only some applications needed "
            "one, and not all are uploaded, so it isn't a guide to what matters most)\n"
            + _clip(current_letter["raw_text"], COVER_LETTER_SAMPLE_CHARS)
        )
    older_letters = len(letters) - (current_letter is not None)
    if older_letters:
        parts.append(f"### ({older_letters} older cover letter{'s' if older_letters != 1 else ''} not shown)")

    return "\n\n".join(parts) or "(no documents uploaded)"


def _top_jobs_text(exclude_job_id: int | None) -> str | None:
    """The jobs that score highest for them (interest + fit, both High or Med), with the reasons,
    and where each stands on My Jobs. Jobs they marked Not interested are left out."""
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT j.id, j.title, j.company, j.location, js.interest_fit, js.candidate_fit,
                       js.rationale_text, a.status AS application_status
                FROM jobs j
                JOIN job_scores js ON js.id = (
                    SELECT id FROM job_scores WHERE user_id = :uid AND job_id = j.id ORDER BY id DESC LIMIT 1
                )
                LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = :uid
                WHERE {VISIBLE_JOB_SQL}
                  AND js.interest_fit IN ('High', 'Med') AND js.candidate_fit IN ('High', 'Med')
                  AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.user_id = :uid AND f.job_id = j.id
                                  AND f.interested = 'not_interested')
                  AND j.id != :exclude
                ORDER BY js.interest_score + js.candidate_fit_score DESC, j.id DESC
                LIMIT :limit""",
            {"uid": current_user_id(), "exclude": exclude_job_id or -1, "limit": TOP_JOBS},
        ).fetchall()
    if not rows:
        return None
    lines = []
    for r in rows:
        status = f"; on My Jobs: {r['application_status']}" if r["application_status"] else ""
        reason = _clip(HIGHLIGHT.sub(r"\1", r["rationale_text"] or ""), TOP_JOB_REASON_CHARS)
        lines.append(
            f"- **{r['title']}** at {r['company']} ({r['location'] or 'location unknown'}): interest "
            f"{r['interest_fit']}, fit {r['candidate_fit']}{status}\n  Why: {reason}"
        )
    return "\n".join(lines)


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
        + (get_current_candidate_profile() or "(not built yet -- rely on their resumes)")
    )
    preference = get_current_preference_profile()
    if preference:
        sections.append("## What their marks and applications say they want\n" + preference)
    top = _top_jobs_text(job_id)
    if top:
        sections.append(
            "## The jobs that score highest for them, and why\n"
            "What they're best suited to and most want: the qualities these share are worth drawing on.\n"
            + top
        )
    sections.append("## Their resumes (and a writing sample)\n" + _documents_text(job_id))
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
                "SELECT * FROM jobby_threads WHERE user_id = ? AND job_id IS NULL ORDER BY id LIMIT 1",
                (current_user_id(),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM jobby_threads WHERE user_id = ? AND job_id = ?", (current_user_id(), job_id)
            ).fetchone()
    return dict(row) if row else None


def thread_by_id(thread_id: int) -> dict | None:
    """One of this person's conversations, or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobby_threads WHERE id = ? AND user_id = ?", (thread_id, current_user_id())
        ).fetchone()
    return dict(row) if row else None


PREVIEW_CHARS = 160


def list_threads() -> list[dict]:
    """This person's conversations that have messages, most recently active first: the job (if
    any) and its My Jobs status, how many messages, and the start of the last one (asked
    2026-11-10: a way to see and reopen past chats). One per job; any number of general ones."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT t.id, t.job_id, t.title, t.updated_at, j.title AS job_title, j.company AS job_company,
                      a.status AS application_status,
                      (SELECT COUNT(*) FROM jobby_messages m WHERE m.thread_id = t.id) AS message_count,
                      (SELECT content FROM jobby_messages m WHERE m.thread_id = t.id
                       ORDER BY m.id DESC LIMIT 1) AS last_message
               FROM jobby_threads t
               LEFT JOIN jobs j ON j.id = t.job_id
               LEFT JOIN applications a ON a.job_id = t.job_id AND a.user_id = t.user_id
               WHERE t.user_id = ?
                 AND EXISTS (SELECT 1 FROM jobby_messages m WHERE m.thread_id = t.id)
               -- Newest message first (ids always increase; updated_at only has whole seconds).
               ORDER BY (SELECT MAX(m.id) FROM jobby_messages m WHERE m.thread_id = t.id) DESC""",
            (current_user_id(),),
        ).fetchall()
    threads = []
    for r in rows:
        preview = " ".join((r["last_message"] or "").split())
        threads.append(
            {
                **{k: r[k] for k in ("id", "job_id", "title", "updated_at", "job_title", "job_company",
                                       "application_status", "message_count")},
                "preview": preview[:PREVIEW_CHARS] + ("…" if len(preview) > PREVIEW_CHARS else ""),
            }
        )  # fmt: skip
    return threads


TITLE_PROMPT = (
    "Write a short title for this conversation between a job seeker and their job-search assistant: "
    '3 to 6 words, in title case, naming its topic (e.g. "Finding Civic Tech Organizations", '
    '"Resume Feedback for Data Roles"). Reply with the title only: no quotes, no period.'
)


def _fallback_title(first_message: str) -> str:
    words = " ".join(first_message.split()).split(" ")
    return " ".join(words[:7]) + ("…" if len(words) > 7 else "")


def make_title(first_message: str, reply: str) -> str:
    """A general chat's title from its first exchange (asked 2026-11-10). One short call at low
    effort; any failure (budget, API, an odd reply) falls back to the start of the first message."""
    try:
        client = get_client()
        with llm_error_boundary("titling the chat"):
            response = client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=200,
                system=TITLE_PROMPT,
                messages=[{"role": "user", "content": f"Job seeker: {_clip(first_message, 1500)}\n\n"
                           f"Assistant: {_clip(reply, 1500)}"}],
                output_config={"effort": "low"},
            )  # fmt: skip
        title = response_text(response, "titling the chat").strip().strip("\"'").rstrip(".")
        # llm_calls' CHECK constraint can't take a new call_type without a table rebuild.
        log_call(
            call_type="generation",
            model=DEFAULT_MODEL,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            note="jobby chat title",
        )
    except Exception:  # noqa: BLE001 - a title is a nicety: never let it break the reply
        logging.getLogger(__name__).exception("titling a Jobby chat failed")
        return _fallback_title(first_message)
    return title[:80] if title and "\n" not in title else _fallback_title(first_message)


def _set_title(thread_id: int, title: str) -> None:
    with get_db() as conn:
        conn.execute("UPDATE jobby_threads SET title = ? WHERE id = ?", (title, thread_id))


def thread_messages(thread_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM jobby_messages WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_thread(thread_id: int) -> bool:
    """Deletes one of this person's conversations and its messages. False if it isn't theirs."""
    thread = thread_by_id(thread_id)
    if thread is None:
        return False
    with get_db() as conn:
        conn.execute("DELETE FROM jobby_messages WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM jobby_threads WHERE id = ?", (thread_id,))
    return True


def _get_or_create_thread(job_id: int | None) -> int:
    thread = get_thread(job_id)
    return thread["id"] if thread else _create_thread(job_id)


def _create_thread(job_id: int | None) -> int:
    with get_db() as conn:
        return conn.execute(
            "INSERT INTO jobby_threads (user_id, job_id) VALUES (?, ?)", (current_user_id(), job_id)
        ).lastrowid


def list_messages(job_id: int | None) -> list[dict]:
    thread = get_thread(job_id)
    return thread_messages(thread["id"]) if thread else []


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
        conn.execute("UPDATE jobby_threads SET updated_at = datetime('now') WHERE id = ?", (thread_id,))
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


def stream_reply(
    job_id: int | None, content: str, thread_id: int | None = None, new: bool = False
) -> Iterator[dict]:
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

    # A chat reopened by id keeps going; a new general chat starts a fresh conversation (any
    # number of those); otherwise it's the job's one conversation (or the first general one).
    if thread_id is None:
        thread_id = _create_thread(None) if new and job_id is None else _get_or_create_thread(job_id)
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
                        if (
                            event.type == "content_block_start"
                            and getattr(event.content_block, "type", None) == "server_tool_use"
                        ):
                            yield {"type": "status", "text": "Searching the web..."}
                        elif (
                            event.type == "content_block_delta"
                            and getattr(event.delta, "type", None) == "text_delta"
                        ):
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
    yield {"type": "done", "user": user_message, "assistant": assistant_message, "thread_id": thread_id}

    # A general chat gets a title from its first exchange, sent as one more event after "done" (the
    # reply is already on screen by then). Job chats are titled by the job.
    if job_id is None:
        thread = thread_by_id(thread_id)
        if thread and not thread.get("title"):
            title = make_title(content, reply)
            _set_title(thread_id, title)
            yield {"type": "title", "thread_id": thread_id, "title": title}


def _delete_message(message_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM jobby_messages WHERE id = ?", (message_id,))
