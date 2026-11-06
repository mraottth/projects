import json

from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call, response_text
from jobhunt.db.connection import get_db
from jobhunt.core.users import current_user_id

SYSTEM_PROMPT = """You are helping a job seeker understand their own career history so a job-matching \
system can score new postings against it. You will be given the full text of one or more resumes and \
cover letters, ordered oldest to newest -- each one tagged [archived] if it came from an old-files \
folder (not necessarily the author's own archive naming, just a signal it's not current).

Weight recent documents far more heavily than old ones: the most recent 1-2 non-archived resumes and \
cover letters should dominate your read of their CURRENT seniority, skills, and role/industry \
preferences. Older and [archived] documents are mainly useful for career-progression context (what \
path got them here) and for competencies that recur across many jobs (real depth), not for describing \
their present-day target role -- don't let a 2021 cover letter's stated preferences override what a \
2026 one says.

Write a concise narrative profile (think "briefing note", 300-500 words) covering:
- Career progression: roles, seniority trajectory, and how responsibilities have evolved over time.
- Core competencies: skills, tools, and domains they have real depth in (vs. just mentioned once).
- Apparent interests: the kinds of problems, industries, or company stages they gravitate toward, \
based on patterns across documents (weighted toward the most recent ones, not a single old job).
- Anything distinctive worth a scoring model knowing (career changes, specializations, gaps explained \
in a cover letter, etc).

Write in third person, as a reference document, not addressed to the candidate. Refer to the \
candidate by their first name -- the preferred one if the documents give one (e.g. "Matt" for \
"Matt (Matthew) Roth") -- never by surname alone ("Roth"), which reads as cold and isn't how they \
want to be referred to. Be specific and concrete; avoid generic resume-speak."""


REVISE_SYSTEM_PROMPT = """You revise a job seeker's candidate profile -- a third-person briefing note \
a job-matching system scores new postings against -- according to their own instructions. Apply the \
requested changes faithfully (they know their own career and goals better than the documents it \
was synthesized from), and leave everything they didn't ask to change as-is. Keep the same \
third-person reference-document style and roughly the same length, referring to the candidate by \
their first name, never by surname alone (rewrite any surname-only references you find). Don't \
invent experience, skills, or achievements beyond what the profile or their instructions state. \
Return only the revised profile text, with no preamble."""


def rebuild_candidate_profile() -> str:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, kind, label, raw_text, is_archived, "
            "COALESCE(source_modified_at, uploaded_at) AS effective_date "
            "FROM resumes WHERE user_id = ? AND raw_text IS NOT NULL "
            "ORDER BY effective_date ASC",
            (current_user_id(),),
        ).fetchall()

    if not rows:
        raise ValueError("no resumes/cover letters with extracted text to build a profile from")

    documents = "\n\n---\n\n".join(
        f"[{row['kind']} — {row['label'] or 'untitled'} — {row['effective_date']}"
        f"{' — archived' if row['is_archived'] else ''}]\n{row['raw_text']}"
        for row in rows
    )

    client = get_client()
    with llm_error_boundary("rebuilding the candidate profile"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=4000,
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": documents}],
        )
    text = response_text(response, "writing your profile")

    log_call(
        call_type="career_profile",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    with get_db() as conn:
        conn.execute(
            "INSERT INTO candidate_profile (user_id, text, model, source_resume_ids_json) "
            "VALUES (?, ?, ?, ?)",
            (current_user_id(), text, DEFAULT_MODEL, json.dumps([row["id"] for row in rows])),
        )

    return text


def get_current_candidate_profile() -> str | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT text FROM candidate_profile WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (current_user_id(),),
        ).fetchone()
    return row["text"] if row else None


def set_candidate_profile(text: str) -> str:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO candidate_profile (user_id, text, model, source_resume_ids_json) "
            "VALUES (?, ?, ?, ?)",
            (current_user_id(), text, "manual_edit", json.dumps([])),
        )
    return text


def revise_candidate_profile(instructions: str) -> str:
    """Applies the user's plain-language change request ("emphasize X, drop Y") to the current
    profile in one Claude call, inserting the result as a new row like every other rebuild/edit.
    Raises ValueError if there's no profile yet or the instructions are blank."""
    if not instructions.strip():
        raise ValueError("describe what you'd like changed")
    current = get_current_candidate_profile()
    if current is None:
        raise ValueError("no candidate profile yet — build one first")

    client = get_client()
    with llm_error_boundary("revising the candidate profile"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=4000,
            output_config={"effort": "medium"},
            system=REVISE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"## Current profile\n{current}\n\n## Requested changes\n{instructions}",
                }
            ],
        )
    text = response_text(response, "writing your profile")

    log_call(
        call_type="career_profile",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note="revision from user instructions",
    )

    with get_db() as conn:
        conn.execute(
            "INSERT INTO candidate_profile (user_id, text, model, source_resume_ids_json) "
            "VALUES (?, ?, ?, ?)",
            (current_user_id(), text, DEFAULT_MODEL, json.dumps([])),
        )
    return text
