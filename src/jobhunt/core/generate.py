"""Tailored resume/cover-letter generation. Called only from a user action on a specific job the
user has decided to pursue (JobDetailPage/tracker "generate for this application"), never
automatically for every high-ranking digest entry -- see api/generate.py."""

from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call

RESUME_SYSTEM_PROMPT = """You tailor a job seeker's resume for a specific posting. You'll be given \
their candidate profile (career-progression summary), their base resume text, and the job posting.

Rewrite the resume (markdown) to foreground the experience, skills, and language most relevant to \
this posting -- reorder/reweight bullet points, tighten or expand descriptions where it helps the \
match, and mirror the posting's own terminology where it's honestly accurate. Do not fabricate \
experience, skills, titles, dates, or employers that aren't in the base resume -- tailoring means \
emphasis and framing, not invention. Keep it to one page's worth of content."""

COVER_LETTER_SYSTEM_PROMPT = """You write a cover letter for a job seeker applying to a specific \
posting. You'll be given their candidate profile (career-progression summary) and the job posting.

Write a concise, specific cover letter (markdown, 250-400 words) that connects their actual \
background to what this posting is asking for -- avoid generic enthusiasm ("I'm excited about...") \
in favor of concrete reasons grounded in their experience and the posting's specifics. Do not \
fabricate experience, skills, or achievements not implied by their candidate profile."""


def _job_text(job: dict) -> str:
    return (
        f"Title: {job['title']}\nCompany: {job['company']}\nLocation: {job.get('location') or 'unknown'}\n\n"
        f"{job.get('description_text') or ''}"
    )


def generate_resume(job: dict, candidate_profile: str | None, base_resume_text: str) -> str:
    user_content = (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n"
        f"## Base resume\n{base_resume_text}\n\n"
        f"## Job posting\n{_job_text(job)}"
    )
    client = get_client()
    with llm_error_boundary("generating a tailored resume"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=3000,
            system=RESUME_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    text = next(b.text for b in response.content if b.type == "text")
    log_call(
        call_type="generation",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        job_id=job.get("id"),
        note="resume",
    )
    return text


def generate_cover_letter(job: dict, candidate_profile: str | None) -> str:
    user_content = (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n## Job posting\n{_job_text(job)}"
    )
    client = get_client()
    with llm_error_boundary("generating a cover letter"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2000,
            system=COVER_LETTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    text = next(b.text for b in response.content if b.type == "text")
    log_call(
        call_type="generation",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        job_id=job.get("id"),
        note="cover_letter",
    )
    return text
