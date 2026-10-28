"""Tailored resume/cover-letter generation. Called only from a user action on a specific job the
user has decided to pursue (JobDetailPage/tracker "generate for this application"), never
automatically for every high-ranking digest entry -- see api/generate.py.

Prefers cloning the user's own most relevant .docx as a style template (core/docx_writer.py) over
plain markdown text, since visually matching "the formatting I've used" requires an actual docx,
not prose describing formatting to an LLM. Falls back to plain-text generation when no usable docx
template is on file (e.g. only a PDF was ever uploaded), so the feature still works either way."""

import logging
import re
import uuid
from datetime import date
from pathlib import Path

from jobhunt.config import RESUME_STORAGE_DIR
from docx import Document

from jobhunt.core.docx_writer import (
    AmbiguousAnchorError,
    AnchorNotFoundError,
    CoverLetterContent,
    DisallowedAnchorError,
    ResumeEditOp,
    ResumeEditPlan,
    TemplateError,
    TemplateLine,
    count_cover_letter_slots,
    extract_addressable_lines,
    write_cover_letter_docx,
    write_resume_edits,
)
from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call
from jobhunt.core.settings import get_setting, get_settings
from jobhunt.db.connection import get_db

logger = logging.getLogger(__name__)

RESUME_STRUCTURED_SYSTEM_PROMPT = """You tailor a job seeker's resume for a specific posting by \
proposing a short list of targeted EDITS to their real resume -- not by writing a new resume from \
scratch. You'll be given their candidate profile (career-progression summary), the resume's own \
lines (each tagged with its structural role, and for bullets/headers, which section/job it falls \
under), and the job posting.

Each edit operation is anchored to one existing line's exact text:
- "replace": reword that line in place (e.g. tailor a bullet's wording, or more accurately \
represent an internal title). The anchor_text you give must be copied character-for-character \
from the line list above -- not retyped, not corrected for an apparent typo -- or it will fail to \
match and the edit will be rejected.
- "insert_after": add a brand-new line immediately after an existing one (it automatically \
inherits that line's exact formatting -- use this to add a new bullet under a specific job, \
anchored on any of that job's existing bullets).
- "delete": remove a line that isn't relevant to this posting.

Rules:
- Never propose an edit anchored on a "title" or "contact" line -- a name and contact info don't \
vary by job posting.
- Never change an employer name, a job title actually held, or a date inside a "job_header" line, \
even via "replace" -- if a job_header line needs rewording at all, keep those facts byte-identical.
- Keep "replace" edits roughly the same length as what they're replacing, and keep the total \
number of net new lines small (favor rewording over deleting a bullet and inserting a different \
one) -- the original resume already fits one page, and small, local edits are the only reliable \
way to protect that.
- If a line's exact text isn't unique across the resume (the same wording appears under two \
different jobs), don't anchor an edit on it -- pick a different, unique line instead; an ambiguous \
anchor will be rejected.
- Do not fabricate experience, skills, titles, dates, or employers that aren't implied by the \
candidate profile."""

RESUME_PLAIN_SYSTEM_PROMPT = """You tailor a job seeker's resume for a specific posting. You'll be \
given their candidate profile (career-progression summary), their base resume text, and the job \
posting.

Rewrite the resume (markdown) to foreground the experience, skills, and language most relevant to \
this posting -- reorder/reweight bullet points, tighten or expand descriptions where it helps the \
match, and mirror the posting's own terminology where it's honestly accurate. Do not fabricate \
experience, skills, titles, dates, or employers that aren't in the base resume -- tailoring means \
emphasis and framing, not invention. Keep it to one page's worth of content.

If you include the candidate's name, use their full name exactly as it appears in the base \
resume's own header (e.g. "Matt Roth") -- not a surname-only form like "Roth"."""

COVER_LETTER_STRUCTURED_SYSTEM_PROMPT = """You write a cover letter for a job seeker applying to a \
specific posting. You'll be given their candidate profile (career-progression summary), the job \
posting, and a target paragraph count matching their own cover-letter template.

Write exactly {n} paragraphs (no greeting/closing lines beyond what's implied by that count -- \
match the template's own structure, e.g. if it opens "To Whom It May Concern:" as its own paragraph \
and closes with "Sincerely," and a signature line as separate paragraphs, do the same). Be concise \
and specific -- avoid generic enthusiasm ("I'm excited about...") in favor of concrete reasons \
grounded in their actual background and the posting's specifics. Do not fabricate experience, \
skills, or achievements not implied by their candidate profile.

If a signature line needs their name, use their actual full/preferred name (e.g. "Matt Roth," \
using a parenthetical preferred first name if the profile gives one, like "Matt (Matthew) Roth" \
-> "Matt Roth") -- not a surname-only or first-name-only form like "Roth" or "Matt"."""

COVER_LETTER_PLAIN_SYSTEM_PROMPT = """You write a cover letter for a job seeker applying to a \
specific posting. You'll be given their candidate profile (career-progression summary) and the job \
posting.

Write a concise, specific cover letter (markdown, 250-400 words) that connects their actual \
background to what this posting is asking for -- avoid generic enthusiasm ("I'm excited about...") \
in favor of concrete reasons grounded in their experience and the posting's specifics. Do not \
fabricate experience, skills, or achievements not implied by their candidate profile.

If a signature line needs their name, use their actual full/preferred name (e.g. "Matt Roth," \
using a parenthetical preferred first name if the profile gives one, like "Matt (Matthew) Roth" \
-> "Matt Roth") -- not a surname-only or first-name-only form like "Roth" or "Matt"."""


STARTER_COVER_LETTER_SYSTEM_PROMPT = """You write a reusable, general-purpose cover letter for a \
job seeker who doesn't have a recent one on file. You'll be given their current resume and what \
they're looking for (target roles, sectors, and a description in their own words). There is no \
specific job posting -- write a letter aimed at the kind of role they want, which they'll later \
tailor per posting, and which will also be used as the paragraph-structure template for tailored \
cover letters.

Return one paragraph per element: a greeting line ("Dear Hiring Manager,") as its own paragraph, \
3-4 body paragraphs, a closing line ("Sincerely,") as its own paragraph, and a signature line as its \
own paragraph. Be concise and specific -- avoid generic enthusiasm ("I'm excited about...") in favor \
of concrete strengths grounded in their actual resume, connected to the roles/sectors they want. Do \
not fabricate experience, skills, or achievements not in their resume, and don't name a specific \
company.

For the signature line, use their actual full/preferred name as it appears at the top of their \
resume (e.g. "Matt Roth"), not a surname-only form."""


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
    """`stored_path` is NOT NULL for every resume regardless of how it got here (bulk-ingested or
    uploaded through the web UI); `source_path` is populated only for bulk-ingested ones (the
    original file's location, before `core/ingest_resumes.py` copied it into storage). Checking
    `source_path` alone used to mean every web-uploaded "current" resume/cover letter silently
    fell back to plain-text generation -- a perfectly good .docx sitting right there in
    `stored_path` just wasn't being looked at."""
    if resume.get("file_type") != "docx":
        return None
    for key in ("stored_path", "source_path"):
        candidate = resume.get(key)
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _output_dir() -> Path:
    library_path = get_setting("resume_library_path")
    if library_path:
        return Path(library_path) / "AI Generated"
    return RESUME_STORAGE_DIR / "generated"


def _safe_filename(*parts: str) -> str:
    joined = "_".join(p for p in parts if p)
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", joined).strip("_")
    return cleaned[:120] or "document"


def _serialize_lines_for_prompt(lines: list[TemplateLine]) -> str:
    rows = []
    for line in lines:
        meta = []
        if line.section:
            meta.append(f'section="{line.section}"')
        if line.under:
            meta.append(f'under="{line.under}"')
        meta_str = (" " + " ".join(meta)) if meta else ""
        rows.append(f'- role={line.role}{meta_str} text="{line.text}"')
    return "\n".join(rows)


def _apply_ops_to_lines(lines: list[TemplateLine], ops: list[ResumeEditOp]) -> list[TemplateLine]:
    """A pure-Python preview of the edited resume's final line sequence, used only to render
    content_text -- safe to trust blindly here because write_resume_edits has already validated
    every op's anchor resolves to exactly one non-title/contact line before this is ever called."""
    result = list(lines)
    for op in ops:
        idx = next((i for i, line in enumerate(result) if line.text == op.anchor_text), None)
        if idx is None:
            continue
        new_text = f"{op.bold_part}{op.new_text}" if op.bold_part else op.new_text
        if op.op == "replace":
            result[idx] = result[idx].model_copy(update={"text": new_text})
        elif op.op == "insert_after":
            result.insert(idx + 1, result[idx].model_copy(update={"text": new_text}))
        else:
            del result[idx]
    return result


def _render_lines_markdown(lines: list[TemplateLine]) -> str:
    rendered = []
    for line in lines:
        if line.role == "title":
            rendered.append(f"# {line.text}")
        elif line.role == "heading":
            rendered.append(f"\n## {line.text}")
        elif line.role == "bullet":
            rendered.append(f"- {line.text}")
        elif line.role == "description":
            rendered.append(f"*{line.text}*")
        else:
            rendered.append(line.text)
    return "\n".join(rendered).strip()


def generate_resume(job: dict, candidate_profile: str | None, resume_id: int | None = None) -> dict:
    """Returns {"content_text": str, "file_path": str | None}."""
    resume = _pick_resume("resume", resume_id)
    if resume is None:
        raise ValueError("no base resume available")

    template_path = _usable_template_path(resume)
    if template_path is None:
        text = _generate_resume_plain(job, candidate_profile, resume.get("raw_text") or "")
        return {"content_text": text, "file_path": None, "based_on_resume_id": resume["id"]}

    lines = extract_addressable_lines(template_path)
    user_content = (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n"
        f"## Resume's own lines (anchor edits to these -- copy text character-for-character)\n"
        f"{_serialize_lines_for_prompt(lines)}\n\n"
        f"## Job posting\n{_job_text(job)}"
    )
    client = get_client()
    with llm_error_boundary("generating tailored resume edits"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=4000,
            system=RESUME_STRUCTURED_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_format=ResumeEditPlan,
        )
    log_call(
        call_type="generation",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        job_id=job.get("id"),
        note="resume (docx template, edit plan)",
    )
    plan = response.parsed_output

    output_path = _output_dir() / (
        _safe_filename(job.get("company"), job.get("title"), "resume", date.today().isoformat())
        + ".docx"
    )
    try:
        write_resume_edits(template_path, plan.ops, output_path)
        file_path = str(output_path)
    except TemplateError:
        file_path = None
    except (AnchorNotFoundError, AmbiguousAnchorError, DisallowedAnchorError):
        # Claude's proposed edits didn't line up with this template -- a failed generation
        # attempt, not a template limitation. Let it propagate as a real error (api/generate.py
        # turns it into a clean 502) rather than silently degrading to a plain-text result that
        # would look like success.
        raise
    except Exception:
        # The docx-writing path manipulates real-world XML in ways that can hit an edge case we
        # haven't seen yet (a template quirk, an unusual content shape) -- that shouldn't cost the
        # user the tailored content itself, which is already generated and the far more expensive
        # part. Fall back to markdown-only, same as a recognized TemplateError, but log it so a
        # recurring cause is actually diagnosable instead of silently eaten.
        logger.exception("write_resume_edits failed for an unexpected reason; falling back to text-only")
        file_path = None

    return {
        "content_text": _render_lines_markdown(_apply_ops_to_lines(lines, plan.ops)),
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
    except Exception:
        logger.exception(
            "write_cover_letter_docx failed for an unexpected reason; falling back to text-only"
        )
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


def generate_starter_cover_letter() -> dict:
    """Onboarding's "write one for me" option: a general-purpose cover letter generated from the
    current resume plus the "What are you looking for?" settings -- there's no candidate profile
    yet at that point in onboarding, and no job, so generate_cover_letter() doesn't apply. Saved as
    a plain .docx and inserted as the new *current* cover letter (demoting any previous current
    one, same as an upload), so later per-job generation uses it as its template.
    Returns {"id": int, "content_text": str}. Raises ValueError if no resume is on file."""
    resume = _pick_resume("resume", None)
    if resume is None or not (resume.get("raw_text") or "").strip():
        raise ValueError("no resume on file to write a cover letter from")

    settings = get_settings()
    titles = ", ".join(settings.get("prefilter_titles_include") or []) or "(not specified)"
    sectors = ", ".join(settings.get("prefilter_sectors") or []) or "(not specified)"
    notes = (settings.get("looking_for_notes") or "").strip() or "(not specified)"
    user_content = (
        f"## Current resume\n{resume['raw_text']}\n\n"
        f"## What they're looking for\nTarget roles: {titles}\nTarget sectors: {sectors}\n"
        f"In their own words: {notes}"
    )

    client = get_client()
    with llm_error_boundary("writing a starter cover letter"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=4000,
            system=STARTER_COVER_LETTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_format=CoverLetterContent,
        )
    log_call(
        call_type="generation",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note="starter cover letter (onboarding)",
    )
    paragraphs = [p.strip() for p in response.parsed_output.paragraphs if p.strip()]
    if not paragraphs:
        raise ValueError("Claude returned an empty cover letter -- try again")

    RESUME_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = RESUME_STORAGE_DIR / f"{uuid.uuid4().hex}.docx"
    doc = Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    doc.save(str(stored_path))

    content_text = "\n\n".join(paragraphs)
    with get_db() as conn:
        conn.execute(
            "UPDATE resumes SET is_template = 0, is_archived = 1 "
            "WHERE kind = 'cover_letter' AND is_template = 1"
        )
        cursor = conn.execute(
            """INSERT INTO resumes
               (kind, label, file_name, file_type, stored_path, raw_text, is_archived, is_template)
               VALUES ('cover_letter', 'AI-generated starter', 'starter_cover_letter.docx', 'docx',
                       ?, ?, 0, 1)""",
            (str(stored_path), content_text),
        )
        doc_id = cursor.lastrowid
    return {"id": doc_id, "content_text": content_text}
