import json
from typing import Literal

from pydantic import BaseModel

from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call
from jobhunt.db.connection import get_db

SYSTEM_PROMPT = """You are helping a job seeker triage incoming postings. You will be given: their \
candidate profile (career history/competencies/interests), their current preference profile (what \
they tend to respond to, learned from past feedback and application outcomes -- may be empty if they're \
just getting started), their desired sectors/industries, a few of their most recent raw feedback \
marks, and one job posting.

Rate two independent things:
- interest_fit: how likely THIS SPECIFIC PERSON is to be genuinely interested in this role, based on \
their stated/inferred interests, desired sectors, and what they've responded to before. A role in a \
sector they didn't ask for should usually score lower on interest even if it's otherwise a strong \
match -- not how good a job it is in the abstract.
- candidate_fit: how strong a candidate they'd be for this specific role, based on their competencies \
and experience vs. what the posting asks for.

Each gets a High/Med/Low label plus a 0-100 score (Low roughly 0-33, Med 33-67, High 67-100 -- but use \
the full range within each band, don't default to the midpoint). Also write a 1-2 sentence summary of \
the role itself (for a digest card) and a short rationale explaining both ratings (what drove them up \
or down) -- the rationale is for the user, so be specific about which parts of the posting or their \
background mattered."""


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
) -> JobScore:
    posting_text = (
        f"Title: {job['title']}\nCompany: {job['company']}\nLocation: {job.get('location') or 'unknown'}\n"
        f"Remote: {job.get('remote_type') or 'unknown'}\n"
        f"Salary: {job.get('salary_min') or '?'}-{job.get('salary_max') or '?'} "
        f"{job.get('salary_currency') or ''} {job.get('salary_period') or ''}\n\n"
        f"{job.get('description_text') or ''}"
    )
    sectors_text = ", ".join(desired_sectors) if desired_sectors else "(not specified)"

    user_content = (
        f"## Candidate profile\n{candidate_profile or '(not yet built)'}\n\n"
        f"## Preference profile\n{preference_profile or '(not yet built -- no feedback history)'}\n\n"
        f"## Desired sectors/industries\n{sectors_text}\n\n"
        f"## Recent feedback\n{_recent_feedback_text()}\n\n"
        f"## Job posting to score\n{posting_text}"
    )

    client = get_client()
    with llm_error_boundary("scoring the job"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_format=JobScore,
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
