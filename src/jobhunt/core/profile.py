import json

from jobhunt.core.llm import DEFAULT_MODEL, get_client, log_call
from jobhunt.db.connection import get_db

SYSTEM_PROMPT = """You are helping a job seeker understand their own career history so a job-matching \
system can score new postings against it. You will be given the full text of one or more resumes and \
cover letters, oldest to newest.

Write a concise narrative profile (think "briefing note", 300-500 words) covering:
- Career progression: roles, seniority trajectory, and how responsibilities have evolved over time.
- Core competencies: skills, tools, and domains they have real depth in (vs. just mentioned once).
- Apparent interests: the kinds of problems, industries, or company stages they gravitate toward, \
based on patterns across documents (not a single job).
- Anything distinctive worth a scoring model knowing (career changes, specializations, gaps explained \
in a cover letter, etc).

Write in third person, as a reference document, not addressed to the candidate. Be specific and concrete; \
avoid generic resume-speak."""


def rebuild_candidate_profile() -> str:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, kind, label, raw_text, uploaded_at FROM resumes WHERE raw_text IS NOT NULL ORDER BY uploaded_at"
        ).fetchall()

    if not rows:
        raise ValueError("no resumes/cover letters with extracted text to build a profile from")

    documents = "\n\n---\n\n".join(
        f"[{row['kind']} — {row['label'] or 'untitled'} — uploaded {row['uploaded_at']}]\n{row['raw_text']}"
        for row in rows
    )

    client = get_client()
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": documents}],
    )
    text = next(b.text for b in response.content if b.type == "text")

    log_call(
        call_type="career_profile",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    with get_db() as conn:
        conn.execute(
            "INSERT INTO candidate_profile (text, model, source_resume_ids_json) VALUES (?, ?, ?)",
            (text, DEFAULT_MODEL, json.dumps([row["id"] for row in rows])),
        )

    return text


def get_current_candidate_profile() -> str | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT text FROM candidate_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["text"] if row else None
