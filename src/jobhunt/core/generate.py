"""Tailored resume/cover-letter generation. Called only from a user action on a specific job the
user has decided to pursue (JobDetailPage/tracker "generate for this application"), never
automatically for every high-ranking digest entry -- see api/generate.py.

Prefers cloning the user's own most relevant .docx as a style template (core/docx_writer.py) over
plain markdown text, since visually matching "the formatting I've used" requires an actual docx,
not prose describing formatting to an LLM. Falls back to plain-text generation when no usable docx
template is on file (e.g. only a PDF was ever uploaded), so the feature still works either way."""

import re
from datetime import date
from pathlib import Path

from jobhunt.config import RESUME_STORAGE_DIR
from jobhunt.core.docx_writer import (
    CoverLetterContent,
    ResumeContent,
    TemplateError,
    count_cover_letter_slots,
    write_cover_letter_docx,
    write_resume_docx,
)
from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call
from jobhunt.core.settings import get_setting
from jobhunt.db.connection import get_db

RESUME_STRUCTURED_SYSTEM_PROMPT = """You tailor a job seeker's resume for a specific posting. \
You'll be given their candidate profile (career-progression summary), the full text of their most \
recent resume (a structural and length reference -- match its section headings, roughly how many \
bullets per role, and overall density; write NEW tailored content grounded in the candidate \
profile, don't just copy the reference resume's wording), and the job posting.

Reorder/reweight content to foreground what's most relevant to this posting, and mirror the \
posting's own terminology where it's honestly accurate. Do not fabricate experience, skills, \
titles, dates, or employers that aren't implied by the candidate profile or reference resume -- \
tailoring means emphasis and framing, not invention.

Output structured content matching the reference resume's shape: a name, one contact line, and a \
list of sections (e.g. "WORK EXPERIENCE", "EDUCATION") each containing a list of lines. Each line \
is one of: "normal" (a plain line, or one with a bold_part prefix for job/school header lines like \
"ACME CORP | Senior Engineer" bold + " | Jan 2024 - Present" not bold), "bullet" (one bullet point), \
or "description" (a one-line company/role blurb, if the reference resume uses those)."""

RESUME_PLAIN_SYSTEM_PROMPT = """You tailor a job seeker's resume for a specific posting. You'll be \
given their candidate profile (career-progression summary), their base resume text, and the job \
posting.

Rewrite the resume (markdown) to foreground the experience, skills, and language most relevant to \
this posting -- reorder/reweight bullet points, tighten or expand descriptions where it helps the \
match, and mirror the posting's own terminology where it's honestly accurate. Do not fabricate \
experience, skills, titles, dates, or employers that aren't in the base resume -- tailoring means \
emphasis and framing, not invention. Keep it to one page's worth of content."""

COVER_LETTER_STRUCTURED_SYSTEM_PROMPT = """You write a cover letter for a job seeker applying to a \
specific posting. You'll be given their candidate profile (career-progression summary), the job \
posting, and a target paragraph count matching their own cover-letter template.

Write exactly {n} paragraphs (no greeting/closing lines beyond what's implied by that count -- \
match the template's own structure, e.g. if it opens "To Whom It May Concern:" as its own paragraph \
and closes with "Sincerely," and a signature line as separate paragraphs, do the same). Be concise \
and specific -- avoid generic enthusiasm ("I'm excited about...") in favor of concrete reasons \
grounded in their actual background and the posting's specifics. Do not fabricate experience, \
skills, or achievements not implied by their candidate profile."""

COVER_LETTER_PLAIN_SYSTEM_PROMPT = """You write a cover letter for a job seeker applying to a \
specific posting. You'll be given their candidate profile (career-progression summary) and the job \
posting.

Write a concise, specific cover letter (markdown, 250-400 words) that connects their actual \
background to what this posting is asking for -- avoid generic enthusiasm ("I'm excited about...") \
in favor of concrete reasons grounded in their experience and the posting's specifics. Do not \
fabricate experience, skills, or achievements not implied by their candidate profile."""


def _job_text(job: dict) -> str:
    return (
        f"Title: {job['title']}\nCompany: {job['company']}\nLocation: {job.get('location') or 'unknown'}\n\n"
        f"{job.get('description_text') or ''}"
    )


def _pick_resume(kind: str, resume_id: int | None) -> dict | None:
    with get_db() as conn:
        if resume_id is not None:
            row = conn.execute(
                "SELECT * FROM resumes WHERE id = ? AND kind = ?", (resume_id, kind)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM resumes WHERE kind = ? ORDER BY is_template DESC, is_archived ASC, "
                "COALESCE(source_modified_at, uploaded_at) DESC LIMIT 1",
                (kind,),
            ).fetchone()
    return dict(row) if row else None


def _usable_template_path(resume: dict) -> Path | None:
    source_path = resume.get("source_path")
    if not source_path or resume.get("file_type") != "docx":
        return None
    path = Path(source_path)
    return path if path.exists() else None


def _output_dir() -> Path:
    library_path = get_setting("resume_library_path")
    if library_path:
        return Path(library_path) / "AI Generated"
    return RESUME_STORAGE_DIR / "generated"


def _safe_filename(*parts: str) -> str:
    joined = "_".join(p for p in parts if p)
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", joined).strip("_")
    return cleaned[:120] or "document"


def _render_resume_markdown(content: ResumeContent) -> str:
    lines = [f"# {content.name}", content.contact_line, ""]
    for section in content.sections:
        lines.append(f"## {section.heading}")
        for line in section.lines:
            if line.kind == "bullet":
                lines.append(f"- {line.text}")
            elif line.kind == "description":
                lines.append(f"*{line.text}*")
            elif line.bold_part:
                lines.append(f"**{line.bold_part}**{line.text}")
            else:
                lines.append(line.text)
        lines.append("")
    return "\n".join(lines).strip()


def generate_resume(job: dict, candidate_profile: str | None, resume_id: int | None = None) -> dict:
    """Returns {"content_text": str, "file_path": str | None}."""
    resume = _pick_resume("resume", resume_id)
    if resume is None:
        raise ValueError("no base resume available")

    template_path = _usable_template_path(resume)
    if template_path is None:
        text = _generate_resume_plain(job, candidate_profile, resume.get("raw_text") or "")
        return {"content_text": text, "file_path": None, "based_on_resume_id": resume["id"]}

    reference_resume_text = resume.get("raw_text") or ""
    user_content = (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n"
        f"## Reference resume (structure/length only -- write new content)\n{reference_resume_text}\n\n"
        f"## Job posting\n{_job_text(job)}"
    )
    client = get_client()
    with llm_error_boundary("generating a tailored resume"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=8000,
            system=RESUME_STRUCTURED_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_format=ResumeContent,
        )
    log_call(
        call_type="generation",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        job_id=job.get("id"),
        note="resume (docx template)",
    )
    content = response.parsed_output

    output_path = _output_dir() / (
        _safe_filename(job.get("company"), job.get("title"), "resume", date.today().isoformat())
        + ".docx"
    )
    try:
        write_resume_docx(template_path, content, output_path)
        file_path = str(output_path)
    except TemplateError:
        file_path = None

    return {
        "content_text": _render_resume_markdown(content),
        "file_path": file_path,
        "based_on_resume_id": resume["id"],
    }


def _generate_resume_plain(job: dict, candidate_profile: str | None, base_resume_text: str) -> str:
    user_content = (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n"
        f"## Base resume\n{base_resume_text}\n\n"
        f"## Job posting\n{_job_text(job)}"
    )
    client = get_client()
    with llm_error_boundary("generating a tailored resume"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=4000,
            system=RESUME_PLAIN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    text = next(b.text for b in response.content if b.type == "text")
    log_call(
        call_type="generation",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        job_id=job.get("id"),
        note="resume (plain text -- no usable docx template)",
    )
    return text


def generate_cover_letter(job: dict, candidate_profile: str | None) -> dict:
    """Returns {"content_text": str, "file_path": str | None}."""
    cover_letter = _pick_resume("cover_letter", None)
    template_path = _usable_template_path(cover_letter) if cover_letter else None
    if template_path is None:
        text = _generate_cover_letter_plain(job, candidate_profile)
        return {"content_text": text, "file_path": None}

    slot_count = count_cover_letter_slots(template_path)
    user_content = (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n## Job posting\n{_job_text(job)}"
    )
    client = get_client()
    with llm_error_boundary("generating a cover letter"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=4000,
            system=COVER_LETTER_STRUCTURED_SYSTEM_PROMPT.format(n=slot_count),
            messages=[{"role": "user", "content": user_content}],
            output_format=CoverLetterContent,
        )
    log_call(
        call_type="generation",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        job_id=job.get("id"),
        note="cover_letter (docx template)",
    )
    content = response.parsed_output

    output_path = _output_dir() / (
        _safe_filename(job.get("company"), job.get("title"), "cover_letter", date.today().isoformat())
        + ".docx"
    )
    try:
        write_cover_letter_docx(template_path, content, output_path)
        file_path = str(output_path)
    except TemplateError:
        file_path = None

    return {"content_text": "\n\n".join(content.paragraphs), "file_path": file_path}


def _generate_cover_letter_plain(job: dict, candidate_profile: str | None) -> str:
    user_content = (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n## Job posting\n{_job_text(job)}"
    )
    client = get_client()
    with llm_error_boundary("generating a cover letter"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=3000,
            system=COVER_LETTER_PLAIN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    text = next(b.text for b in response.content if b.type == "text")
    log_call(
        call_type="generation",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        job_id=job.get("id"),
        note="cover_letter (plain text -- no usable docx template)",
    )
    return text
