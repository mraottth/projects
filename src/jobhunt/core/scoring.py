import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from jobhunt.core.llm import (
    DEFAULT_MODEL,
    cache_tokens,
    estimate_cost_usd,
    get_client,
    llm_error_boundary,
    log_call,
)
from jobhunt.db.connection import get_db
from jobhunt.core.users import current_user_id

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

In the rationale, mark the 2-4 most decision-critical phrases -- the specific skill match, \
dealbreaker, or standout detail that most explains a rating, not whole sentences -- and tag each with \
the rating it explains: ==i:phrase== when it drives interest_fit (e.g. "==i:mission-driven civic \
tech==", "==i:no remote option==") and ==f:phrase== when it drives candidate_fit (e.g. "==f:8 years \
leading analytics teams==", "==f:no Kubernetes experience=="). Give each rating at least one marked \
phrase when something clearly drives it. The UI highlights each in that rating's color, so use it \
sparingly and only for words/short phrases that genuinely matter most, not as general emphasis.

Finally, report the pay the posting itself states, if any: salary_min/salary_max as plain numbers \
(160000, not "160k"; the same number in both for a single figure), salary_currency as a code like \
USD, and salary_period "year" or "hour". Use null for all four if the posting doesn't state pay -- \
never estimate it, and don't use the candidate's own minimum."""


# Medium rather than the default high: a High/Med/Low rating plus a short rationale doesn't need
# the deepest reasoning, and scoring runs up to daily_llm_cap times per run, so per-call latency is
# most of how long a run (and onboarding's last step) takes. Raise it if ratings start missing nuance.
SCORING_EFFORT = "medium"


@dataclass(frozen=True)
class ScoringSettings:
    """How a scoring call is made. CURRENT is what live scoring uses; LEAN is the cheaper candidate
    being measured against it (core/scoring_experiment.py, asked 2026-11-03): most of a score's
    cost is output (reasoning plus a free-length rationale), then the whole posting as input."""

    name: str
    effort: str
    max_posting_chars: int | None = None
    short_rationale: bool = False


CURRENT = ScoringSettings("current", SCORING_EFFORT)
LEAN = ScoringSettings("lean", "low", max_posting_chars=6000, short_rationale=True)

SHORT_RATIONALE = """

Keep the rationale to 2-3 sentences. Still mark the 2-4 most decision-critical phrases as above."""


class JobScore(BaseModel):
    interest_fit: Literal["High", "Med", "Low"]
    interest_score: float
    candidate_fit: Literal["High", "Med", "Low"]
    candidate_fit_score: float
    summary_text: str
    rationale_text: str


class ScoredPosting(JobScore):
    """What the scoring call returns: the score plus the pay the posting states. Claude reads the
    whole description here anyway, so this catches pay that core/salary_extract.py's patterns
    miss, at a few output tokens' cost (salary showed "N/A" while Claude's own summary quoted
    "$160-210k", caught live). Nullable rather than defaulted, like manual_import's fields."""

    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    salary_period: Literal["year", "hour"] | None


def _recent_feedback_text(limit: int = 10) -> str:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT f.interested, f.note, j.title, j.company
               FROM feedback f JOIN jobs j ON j.id = f.job_id
               WHERE f.user_id = ?
               ORDER BY f.created_at DESC LIMIT ?""",
            (current_user_id(), limit),
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
    score, _cost, _output = score_job_with(
        job,
        candidate_profile,
        preference_profile,
        desired_sectors,
        desired_titles,
        salary_floor,
        looking_for_notes,
    )
    return score


def score_job_with(
    job: dict,
    candidate_profile: str | None,
    preference_profile: str | None,
    desired_sectors: list[str] | None = None,
    desired_titles: list[str] | None = None,
    salary_floor: float | None = None,
    looking_for_notes: str | None = None,
    settings: ScoringSettings = CURRENT,
    note: str | None = None,
) -> tuple[ScoredPosting, float | None, int]:
    """Scores one job with the given settings. Returns (score, estimated cost, output tokens)."""
    description = job.get("description_text") or ""
    if settings.max_posting_chars and len(description) > settings.max_posting_chars:
        description = description[: settings.max_posting_chars].rstrip() + "\n\n...(truncated)"
    posting_text = (
        f"Title: {job['title']}\nCompany: {job['company']}\nLocation: {job.get('location') or 'unknown'}\n"
        f"Remote: {job.get('remote_type') or 'unknown'}\n"
        f"Salary: {job.get('salary_min') or '?'}-{job.get('salary_max') or '?'} "
        f"{job.get('salary_currency') or ''} {job.get('salary_period') or ''}\n\n"
        f"{description}"
    )
    sectors_text = ", ".join(desired_sectors) if desired_sectors else "(not specified)"
    titles_text = ", ".join(desired_titles) if desired_titles else "(not specified)"
    salary_text = f"{salary_floor:,.0f}" if salary_floor else "(not specified)"
    notes_text = looking_for_notes or "(not specified)"

    # Everything about the candidate is the same for every job in a run, so it goes in the system
    # prompt behind a cache breakpoint: the first call of a run writes it (1.25x input price) and
    # the rest read it (0.1x). Only the posting varies. Nothing here may change between calls in a
    # run (no timestamps, stable ordering), or every call becomes a cache miss.
    candidate_context = (
        f"## Candidate profile\n{candidate_profile or '(not yet built)'}\n\n"
        f"## Preference profile\n{preference_profile or '(not yet built -- no feedback history)'}\n\n"
        f"## Desired sectors/industries\n{sectors_text}\n\n"
        f"## Desired job titles (guidance, not a required exact match)\n{titles_text}\n\n"
        f"## Minimum salary (guidance, not a hard cutoff)\n{salary_text}\n\n"
        f"## In their own words, what they're looking for\n{notes_text}\n\n"
        f"## Recent feedback\n{_recent_feedback_text()}"
    )
    system = [
        {"type": "text", "text": SYSTEM_PROMPT + (SHORT_RATIONALE if settings.short_rationale else "")},
        {"type": "text", "text": candidate_context, "cache_control": {"type": "ephemeral"}},
    ]

    client = get_client()
    with llm_error_boundary("scoring the job"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=3000,
            system=system,
            messages=[{"role": "user", "content": f"## Job posting to score\n{posting_text}"}],
            output_format=ScoredPosting,
            output_config={"effort": settings.effort},
        )
    usage = response.usage
    log_call(
        call_type="scoring",
        model=DEFAULT_MODEL,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        job_id=job.get("id"),
        note=note,
        **cache_tokens(usage),
    )
    cost = estimate_cost_usd(DEFAULT_MODEL, usage.input_tokens, usage.output_tokens, **cache_tokens(usage))
    return response.parsed_output, cost, usage.output_tokens


def store_score(job_id: int, score: JobScore) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO job_scores
               (user_id, job_id, model, interest_fit, interest_score, candidate_fit, candidate_fit_score,
                summary_text, rationale_text, raw_response_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                current_user_id(),
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
        score_id = cursor.lastrowid
        # The pay Claude read off the posting fills a salary the job doesn't have yet (never
        # overwriting one from the ATS or the description).
        salary_min = getattr(score, "salary_min", None)
        salary_max = getattr(score, "salary_max", None)
        if salary_min or salary_max:
            conn.execute(
                "UPDATE jobs SET salary_min = ?, salary_max = ?, salary_currency = ?, salary_period = ? "
                "WHERE id = ? AND salary_min IS NULL AND salary_max IS NULL",
                (
                    salary_min,
                    salary_max if salary_max != salary_min else None,
                    getattr(score, "salary_currency", None) or "USD",
                    getattr(score, "salary_period", None) or "year",
                    job_id,
                ),
            )
        return score_id
