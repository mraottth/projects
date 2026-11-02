from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call
from jobhunt.db.connection import get_db
from jobhunt.core.users import current_user_id

SYSTEM_PROMPT = """You are maintaining a running preference profile for a job seeker, used to improve \
future job-fit scoring. You'll be given the previous profile (may be empty), their application history \
(status progression per job -- this is the strongest signal: a job that reached interview or offer says \
far more about genuine fit than a bare "interested" mark. Statuses: saved (drafting), waiting (applied, \
no response yet), interviewing, offer, rejected_no_interview (a weaker match on paper), \
rejected_interviewed (got through screening, so not a fit problem on paper), withdrawn (they lost \
interest). Distinguish "wasn't a good candidate fit" from "wasn't actually interested"), and their \
raw interested/not_interested/maybe marks (weaker, faster signal).

Write an updated profile (200-400 words): what roles/companies/locations/comp ranges they actually \
respond to, patterns in what they reject even when "on paper" qualified, and anything the application \
outcomes reveal about where they're a strong vs. weak candidate. Merge with the previous profile rather \
than just appending -- update or drop points the new evidence contradicts."""


def get_current_preference_profile() -> str | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT text FROM preference_profile WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (current_user_id(),),
        ).fetchone()
    return row["text"] if row else None


def _signal_text() -> tuple[str, int, int]:
    user_id = current_user_id()
    with get_db() as conn:
        applications = conn.execute(
            """SELECT j.title, j.company, a.status, a.notes,
                      (SELECT GROUP_CONCAT(status || ' @ ' || changed_at, ' -> ')
                       FROM application_status_history WHERE application_id = a.id) AS history
               FROM applications a JOIN jobs j ON j.id = a.job_id
               WHERE a.user_id = ?
               ORDER BY a.updated_at DESC""",
            (user_id,),
        ).fetchall()
        feedback = conn.execute(
            """SELECT f.interested, f.note, j.title, j.company
               FROM feedback f JOIN jobs j ON j.id = f.job_id
               WHERE f.user_id = ?
               ORDER BY f.created_at DESC LIMIT 50""",
            (user_id,),
        ).fetchall()

    app_lines = [
        f"- {row['title']} at {row['company']}: {row['history'] or row['status']}"
        + (f" (notes: {row['notes']})" if row["notes"] else "")
        for row in applications
    ]
    feedback_lines = [
        f"- {row['interested']}: {row['title']} at {row['company']}"
        + (f" (note: {row['note']})" if row["note"] else "")
        for row in feedback
    ]
    text = (
        "## Application history (strongest signal)\n"
        + ("\n".join(app_lines) if app_lines else "(none yet)")
        + "\n\n## Raw feedback marks (weaker signal)\n"
        + ("\n".join(feedback_lines) if feedback_lines else "(none yet)")
    )
    return text, len(feedback), len(applications)


def rebuild_preference_profile() -> str:
    signal_text, feedback_count, application_count = _signal_text()
    if feedback_count == 0 and application_count == 0:
        raise ValueError("no feedback or application history yet to build a preference profile from")

    previous = get_current_preference_profile()
    user_content = f"## Previous profile\n{previous or '(none yet -- first build)'}\n\n{signal_text}"

    client = get_client()
    with llm_error_boundary("rebuilding the preference profile"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    text = next(b.text for b in response.content if b.type == "text")

    log_call(
        call_type="profile_rebuild",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    with get_db() as conn:
        conn.execute(
            """INSERT INTO preference_profile
               (user_id, text, model, based_on_feedback_count, based_on_application_events_count)
               VALUES (?, ?, ?, ?, ?)""",
            (current_user_id(), text, DEFAULT_MODEL, feedback_count, application_count),
        )

    return text
