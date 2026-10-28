import json
from typing import Literal

from pydantic import BaseModel

from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call
from jobhunt.db.connection import get_db

SYSTEM_PROMPT = """You are helping a job seeker triage incoming postings. You will be given: their \
candidate profile (career history/competencies/interests), their current preference profile (what \
they tend to respond to, learned from past feedback and application outcomes -- may be empty if they're \
just getting started), their desired sectors/industries, their desired job titles, their minimum \
salary, a free-text note in their own words about what they're looking for, a few of their most \
recent raw feedback marks, and one job posting.

The free-text note is the most direct signal you have -- it's the candidate describing what they \
want in their own words, not a structured field that forces them into a category. Weigh it heavily \
alongside the structured fields, and let it override a generic reading of the structured fields \
when the two seem to pull in different directions.

Desired titles and minimum salary are GUIDANCE, not a filter -- do not require a literal title match. \
If they said "Analytics Director," a posting titled "VP of Analytics," "Director of Data," or "Head \
of Analytics" can be just as strong a match if the role's actual scope and seniority line up; judge \
the substance of the role against their career level and the kind of work it involves, not the job \
title string. Likewise, a posting slightly below their stated minimum salary isn't disqualifying on \
its own -- note it in the rationale, but still rate the role on its actual merits.

Rate two independent things:
- interest_fit: how likely THIS SPECIFIC PERSON is to be genuinely interested in this role, based on \
their stated/inferred interests, desired sectors, desired titles, and what they've responded to \
before. A role in a sector they didn't ask for should usually score lower on interest even if it's \
otherwise a strong match -- not how good a job it is in the abstract.
- candidate_fit: how strong a candidate they'd be for this specific role, based on their competencies \
and experience vs. what the posting asks for.

Each gets a High/Med/Low label plus a 0-100 score (Low roughly 0-33, Med 33-67, High 67-100 -- but use \
the full range within each band, don't default to the midpoint). Also write a 1-2 sentence summary of \
the role itself (for a digest card) and a short rationale explaining both ratings (what drove them up \
or down) -- the rationale is for the user, so be specific about which parts of the posting or their \
background mattered.

The user reads the rationale directly, so write it to them in the second person: "you", "your \
background", "your stated minimum" -- never "the candidate", "they", or their name (e.g. "Your \
==8 years leading analytics teams== lines up well, but you said you want remote and this is \
on-site."). The candidate profile you're given is written in the third person; don't carry that \
voice over.

In the rationale, wrap the 1-3 most decision-critical phrases in ==double equals== (e.g. "==10+ years \
of Python==" or "==no remote option==") -- the specific skill match, dealbreaker, or standout detail \
that most explains the rating, not whole sentences. The UI renders these as highlighted text, so use \
it sparingly and only for words/short phrases that genuinely matter most, not as general emphasis."""


# Medium rather than the default high: a High/Med/Low rating plus a short rationale doesn't need
# the deepest reasoning, and scoring runs up to daily_llm_cap times per run, so per-call latency is
# most of how long a run (and onboarding's last step) takes. Raise it if ratings start missing nuance.
SCORING_EFFORT = "medium"

class JobScore(BaseModel):
    interest_fit: Literal["High", "Med", "Low"]
    interest_score: float
    candidate_fit: Literal["High", "Med", "Low"]
    candidate_fit_score: float
    summary_text: str
    rationale_text: str


def _recent_feedback_text(limit: int = 10) -> str:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT f.interested, f.note, j.title, j.company
               FROM feedback f JOIN jobs j ON j.id = f.job_id
               ORDER BY f.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    if not rows:
        return "(no feedback recorded yet)"
    return "\n".join(
        f"- {row['interested']}: {row['title']} at {row['company']}"
        + (f" (note: {row['note']})" if row["note"] else "")
        for row in rows
    )


def score_job(
    job: dict,
    candidate_profile: str | None,
    preference_profile: str | None,
    desired_sectors: list[str] | None = None,
    desired_titles: list[str] | None = None,
    salary_floor: float | None = None,
    looking_for_notes: str | None = None,
) -> JobScore:
    posting_text = (
        f"Title: {job['title']}\nCompany: {job['company']}\nLocation: {job.get('location') or 'unknown'}\n"
        f"Remote: {job.get('remote_type') or 'unknown'}\n"
        f"Salary: {job.get('salary_min') or '?'}-{job.get('salary_max') or '?'} "
        f"{job.get('salary_currency') or ''} {job.get('salary_period') or ''}\n\n"
        f"{job.get('description_text') or ''}"
    )
    sectors_text = ", ".join(desired_sectors) if desired_sectors else "(not specified)"
    titles_text = ", ".join(desired_titles) if desired_titles else "(not specified)"
    salary_text = f"{salary_floor:,.0f}" if salary_floor else "(not specified)"
    notes_text = looking_for_notes or "(not specified)"

    user_content = (
        f"## Candidate profile\n{candidate_profile or '(not yet built)'}\n\n"
        f"## Preference profile\n{preference_profile or '(not yet built -- no feedback history)'}\n\n"
        f"## Desired sectors/industries\n{sectors_text}\n\n"
        f"## Desired job titles (guidance, not a required exact match)\n{titles_text}\n\n"
        f"## Minimum salary (guidance, not a hard cutoff)\n{salary_text}\n\n"
        f"## In their own words, what they're looking for\n{notes_text}\n\n"
        f"## Recent feedback\n{_recent_feedback_text()}\n\n"
        f"## Job posting to score\n{posting_text}"
    )

    client = get_client()
    with llm_error_boundary("scoring the job"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_format=JobScore,
            output_config={"effort": SCORING_EFFORT},
        )
    log_call(
        call_type="scoring",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        job_id=job.get("id"),
    )
    return response.parsed_output


def store_score(job_id: int, score: JobScore) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO job_scores
               (job_id, model, interest_fit, interest_score, candidate_fit, candidate_fit_score,
                summary_text, rationale_text, raw_response_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                DEFAULT_MODEL,
                score.interest_fit,
                score.interest_score,
                score.candidate_fit,
                score.candidate_fit_score,
                score.summary_text,
                score.rationale_text,
                json.dumps(score.model_dump()),
            ),
        )
        return cursor.lastrowid
