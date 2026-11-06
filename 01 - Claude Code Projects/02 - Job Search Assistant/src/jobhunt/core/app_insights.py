"""My Jobs' "Ask Jobby what's working": one Claude call, in Jobby's voice, that reads every tracked
application (the job, its fit ratings, salary and how far it got) and says which kinds of jobs are
getting responses and which aren't. The counts on the page are computed in the browser; this is
the part that needs judgment. Run only when the user asks, and the last answer is kept in settings
(`application_insights`) so revisiting the page costs nothing."""

from datetime import date, datetime, timezone

from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call, response_text
from jobhunt.core.settings import get_setting, update_settings
from jobhunt.db.connection import get_db
from jobhunt.core.users import current_user_id

SETTING = "application_insights"

STATUS_WORDS = {
    "saved": "drafting (not applied yet)",
    "waiting": "applied, no response yet",
    "interviewing": "interviewing",
    "offer": "offer",
    "rejected_no_interview": "rejected without an interview",
    "rejected_interviewed": "rejected after interviewing",
    "withdrawn": "withdrew",
}

# Written as Jobby (core/jobby.py's interview-prep coach), whose face it wears on My Jobs; the
# general Jobby conversation also gets the latest answer in its context to talk it over.
SYSTEM_PROMPT = """You are Jobby, a friendly, sharp job-search coach. Help the job seeker learn from \
their own applications, in your warm but direct voice -- encouraging, never fluffy. You get every job \
they're tracking: title, employer, location, pay, the app's interest and fit ratings \
(High/Med/Low), the status it reached, and how long ago they applied.

Write a short, practical read-out in Markdown, addressed to them in the second person. Open with \
one plain sentence, before any heading, giving the single most useful takeaway (it's shown on its \
own as a preview). Then:
## What's working
## What isn't
## Try next

Under each heading, 2-4 bullets. Ground every point in their actual applications: name the \
patterns (kinds of roles, seniority, sectors or employer types, pay bands, locations, the fit \
ratings) and cite a couple of examples by employer. Getting an interview is the main signal of \
what's working, and an offer even more so. A rejection without an interview points to a weaker \
match on paper; a rejection after interviewing doesn't. Treat "applied, no response yet" as \
unknown unless it's been several weeks. Say plainly when there are too few applications to tell, \
rather than inventing a pattern. Keep it under 300 words."""


def get_saved() -> dict | None:
    return get_setting(SETTING)


def _application_lines() -> tuple[list[str], int]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT a.status, a.applied_at, a.notes, j.title, j.company, j.location,
                      j.salary_min, j.salary_max, j.salary_period, s.interest_fit, s.candidate_fit
               FROM applications a JOIN jobs j ON j.id = a.job_id
               LEFT JOIN job_scores s ON s.id = (
                   SELECT id FROM job_scores WHERE user_id = :uid AND job_id = j.id
                   ORDER BY scored_at DESC, id DESC LIMIT 1)
               WHERE a.user_id = :uid
               ORDER BY a.applied_at DESC""",
            {"uid": current_user_id()},
        ).fetchall()
    lines, applied = [], 0
    for r in rows:
        if r["status"] != "saved":
            applied += 1
        pay = ""
        if r["salary_min"] or r["salary_max"]:
            low, high = r["salary_min"] or r["salary_max"], r["salary_max"] or r["salary_min"]
            pay = f", pay {low:,.0f}-{high:,.0f}/{r['salary_period'] or 'year'}"
        when = ""
        if r["applied_at"]:
            days = (date.today() - date.fromisoformat(r["applied_at"][:10])).days
            when = f", applied {days} days ago"
        ratings = ""
        if r["interest_fit"]:
            ratings = f", interest {r['interest_fit']}, fit {r['candidate_fit']}"
        notes = f" (notes: {r['notes']})" if r["notes"] else ""
        lines.append(
            f"- {r['title']} at {r['company']} ({r['location'] or 'location n/a'}{pay}{ratings}{when}): "
            f"{STATUS_WORDS.get(r['status'], r['status'])}{notes}"
        )
    return lines, applied


def build() -> dict:
    lines, applied = _application_lines()
    if applied == 0:
        raise ValueError("No applications sent yet: mark a job Waiting or later on My Jobs first.")
    looking_for = get_setting("looking_for_notes") or ""
    content = (
        (f"## What they say they're looking for\n{looking_for}\n\n" if looking_for else "")
        + "## Their applications\n"
        + "\n".join(lines)
    )
    client = get_client()
    with llm_error_boundary("reading your applications"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": "medium"},
        )
    text = response_text(response, "reading your applications").strip()
    # llm_calls' CHECK constraint can't take a new call_type without a table rebuild (see Jobby).
    log_call(
        call_type="profile_rebuild",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note="application insights",
    )
    saved = {
        "text": text,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "applications": len(lines),
        "applied": applied,
    }
    update_settings({SETTING: saved})
    return saved
