"""The title screen (asked 2026-11-12, to cut the cost of scoring).

A full score costs about 0.9 cents and reads the whole posting. Many waiting jobs are plainly not
for the person from the title alone (a legal-engineering or sales role for a data leader), and
scoring them in full spent the daily cap on a foregone conclusion. Before a run scores, the waiting
jobs' title, organization and location go to Claude Haiku in a few cheap calls (about 150 jobs per
call), which sets aside only clear mismatches and keeps anything it isn't sure of. A set-aside job
is never scored (queue.select_batch skips it) and is hidden on All Jobs unless asked for; it's not
deleted, so the person can still find and track it.

`scoring_queue.screen` is NULL (not screened yet), 'keep' or 'drop'.
"""

import logging

from pydantic import BaseModel

from jobhunt.core import dealbreakers
from jobhunt.core.llm import LLMError, cache_tokens, get_client, llm_error_boundary, log_call
from jobhunt.core.preference import get_current_preference_profile
from jobhunt.core.profile import get_current_candidate_profile
from jobhunt.core.settings import get_settings
from jobhunt.core.users import current_user_id
from jobhunt.db.connection import get_db

logger = logging.getLogger(__name__)

SCREEN_MODEL = "claude-haiku-5-5"
CHUNK = 150
MAX_PER_RUN = 1500

SYSTEM_PROMPT = """You screen job postings for a job seeker before a closer read. You'll get who they \
are and what they want, then a numbered list of postings (title, organization, location).

List the numbers of postings that are clearly not a fit from the title alone: a different line of \
work (for example legal, sales, nursing or software engineering for someone in data and analytics), \
or far too junior or too senior. Keep anything plausible, adjacent, or that you're unsure about: a \
posting you set aside is never read closely, so a wrong "set aside" costs them a job they wanted, \
while a wrong "keep" only costs a closer read.

Don't set anything aside because of its location, pay or organization: those are handled separately."""


class ScreenResult(BaseModel):
    set_aside: list[int]


def screen_pending(source_ids: list[int]) -> dict:
    """Screens the current person's waiting, unscreened jobs from `source_ids`. Returns
    {"screened", "set_aside", "errors"}. A failed call leaves its jobs unscreened (so they're
    scored as before), never failing the run."""
    settings = get_settings()
    only = ",".join(str(int(i)) for i in source_ids) or "NULL"
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT q.id AS queue_id, j.title, j.company, j.location, j.remote_type,
                       j.salary_min, j.salary_max, j.salary_currency, j.salary_period
                  FROM scoring_queue q JOIN jobs j ON j.id = q.job_id
                 WHERE q.user_id = ? AND q.status = 'pending' AND q.screen IS NULL
                   AND j.source_id IN ({only})
                 ORDER BY q.priority DESC, q.id""",
            (current_user_id(),),
        ).fetchall()
    # A job a dealbreaker hides is never scored anyway: no need to screen it.
    waiting = [dict(r) for r in rows if not dealbreakers.blocks(dict(r), settings)][:MAX_PER_RUN]
    outcome = {"screened": 0, "set_aside": 0, "errors": []}
    if not waiting:
        return outcome
    context = _context(settings)
    for start in range(0, len(waiting), CHUNK):
        chunk = waiting[start : start + CHUNK]
        try:
            dropped = _screen_chunk(context, chunk)
        except LLMError as exc:
            outcome["errors"].append(f"title screen: {exc}")
            break
        with get_db() as conn:
            conn.executemany(
                "UPDATE scoring_queue SET screen = ?, screened_at = datetime('now') WHERE id = ?",
                [("drop" if n in dropped else "keep", row["queue_id"]) for n, row in enumerate(chunk, 1)],
            )
        outcome["screened"] += len(chunk)
        outcome["set_aside"] += len(dropped)
    return outcome


RECENT = 15


def insight() -> dict:
    """For Settings' Title screen card (asked 2026-11-12): what it has screened, set aside and kept,
    what's still waiting, when it last ran, what it cost this month, and its latest set-asides."""
    uid = current_user_id()
    with get_db() as conn:
        counts = conn.execute(
            """SELECT
                   SUM(screen IS NOT NULL) AS screened,
                   SUM(screen = 'drop' AND status = 'pending') AS set_aside,
                   SUM(screen = 'keep') AS kept,
                   SUM(screen IS NULL AND status = 'pending') AS waiting,
                   MAX(screened_at) AS last_screened_at
                 FROM scoring_queue WHERE user_id = ?""",
            (uid,),
        ).fetchone()
        spend = conn.execute(
            """SELECT COALESCE(SUM(estimated_cost_usd), 0) AS cost, COUNT(*) AS calls
                 FROM llm_calls WHERE user_id = ? AND note LIKE 'title screen%'
                  AND created_at >= date('now', 'start of month')""",
            (uid,),
        ).fetchone()
        recent = conn.execute(
            """SELECT j.id, j.title, j.company, j.location, q.screened_at
                 FROM scoring_queue q JOIN jobs j ON j.id = q.job_id
                WHERE q.user_id = ? AND q.screen = 'drop' AND q.status = 'pending' AND j.status = 'active'
                ORDER BY q.screened_at DESC, q.id DESC LIMIT ?""",
            (uid, RECENT),
        ).fetchall()
    return {
        "screened": counts["screened"] or 0,
        "set_aside": counts["set_aside"] or 0,
        "kept": counts["kept"] or 0,
        "waiting": counts["waiting"] or 0,
        "last_screened_at": counts["last_screened_at"],
        "cost_this_month": round(spend["cost"], 4),
        "calls_this_month": spend["calls"],
        "recent": [dict(r) for r in recent],
    }


def put_back(job_id: int) -> bool:
    """A set-aside job back in line to be scored, when the screen got it wrong."""
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE scoring_queue SET screen = 'keep' WHERE user_id = ? AND job_id = ? AND screen = 'drop'",
            (current_user_id(), job_id),
        )
    return cursor.rowcount > 0


def _context(settings: dict) -> str:
    titles = ", ".join(settings.get("prefilter_titles_include") or []) or "(not specified)"
    sectors = ", ".join(settings.get("prefilter_sectors") or []) or "(not specified)"
    wants = get_current_preference_profile() or "(none yet)"
    return (
        f"## Candidate profile\n{get_current_candidate_profile() or '(not yet built)'}\n\n"
        f"## What they tend to want (from their feedback)\n{wants}\n\n"
        f"## Titles they're after\n{titles}\n\n## Sectors\n{sectors}\n\n"
        f"## In their own words\n{settings.get('looking_for_notes') or '(not specified)'}"
    )


def _screen_chunk(context: str, chunk: list[dict]) -> set[int]:
    listing = "\n".join(
        f"{n}. {row['title']} | {row['company']} | {row['location'] or 'location not given'}"
        for n, row in enumerate(chunk, 1)
    )
    client = get_client()
    with llm_error_boundary("screening job titles"):
        response = client.messages.parse(
            model=SCREEN_MODEL,
            max_tokens=4000,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT},
                # The same for every chunk of a run, so later chunks read it from cache.
                {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": f"## Postings\n{listing}"}],
            output_format=ScreenResult,
            output_config={"effort": "low"},
        )
    usage = response.usage
    log_call(
        call_type="parsing",
        model=SCREEN_MODEL,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        note=f"title screen ({len(chunk)} jobs)",
        **cache_tokens(usage),
    )
    if response.parsed_output is None:  # e.g. a refusal: keep everything rather than guess
        raise LLMError(f"the title screen gave no answer (stop reason: {response.stop_reason})")
    return {n for n in response.parsed_output.set_aside if 1 <= n <= len(chunk)}
