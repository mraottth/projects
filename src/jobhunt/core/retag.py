"""One-off: colors the highlights in job summaries scored before highlights were tagged.

Scoring now marks each key phrase in a rationale as explaining the interest rating (==i:...==) or
the fit rating (==f:...==), and the job cards color it like that rating's chip. Summaries scored
earlier have plain ==...== marks, which all show amber (caught live: a refresh left every card
amber, since only newly scored jobs get tags). This asks Claude only which rating each existing
highlight explains -- a list of "i"/"f" -- and splices the tags in, so the wording, ratings and
everything else stay exactly as they were. An answer that doesn't line up with the highlights
leaves that summary untouched. Started from Settings; only each job's latest score is touched."""

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel

from jobhunt.core.llm import DEFAULT_MODEL, estimate_cost_usd, get_client, llm_error_boundary, log_call
from jobhunt.db.connection import get_db

logger = logging.getLogger(__name__)

UNTAGGED = re.compile(r"==(?![if]:)([^=]+)==")
CONCURRENCY = 4
# Rough per-summary usage for the cost estimate shown before running: a short prompt plus the
# summary in, a few letters out.
EST_INPUT_TOKENS = 700
EST_OUTPUT_TOKENS = 80

SYSTEM_PROMPT = """A job-matching app rated a job posting for a candidate on two things:
- interest: how much this candidate would want this job (their interests, preferred sector,
  location, pay, mission);
- fit: how strong a candidate they'd be (their skills, experience and seniority versus what the
  posting asks for).
Below is the app's explanation of both ratings. Some phrases in it are highlighted and numbered
like [[1: phrase]]. For each highlighted phrase, in order, answer "i" if it mainly explains the
interest rating or "f" if it mainly explains the fit rating. Return exactly one answer per
highlighted phrase."""


class HighlightTags(BaseModel):
    tags: list[Literal["i", "f"]]


def _latest_untagged() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT js.id, js.interest_fit, js.candidate_fit, js.rationale_text
               FROM job_scores js
               WHERE js.id = (SELECT MAX(id) FROM job_scores WHERE job_id = js.job_id)
                 AND js.rationale_text LIKE '%==%'"""
        ).fetchall()
    return [dict(r) for r in rows if UNTAGGED.search(r["rationale_text"] or "")]


def pending_summary() -> dict:
    count = len(_latest_untagged())
    per_call = estimate_cost_usd(DEFAULT_MODEL, EST_INPUT_TOKENS, EST_OUTPUT_TOKENS) or 0.0
    return {"pending": count, "estimated_cost_usd": round(count * per_call, 2)}


def retag_rationale(text: str, interest: str | None, fit: str | None, client=None) -> str | None:
    """The rationale with each untagged highlight tagged, or None if Claude's answer didn't line up."""
    phrases = UNTAGGED.findall(text)
    if not phrases:
        return None
    counter = iter(range(1, len(phrases) + 1))
    numbered = UNTAGGED.sub(lambda m: f"[[{next(counter)}: {m.group(1)}]]", text)
    client = client or get_client()
    with llm_error_boundary("tagging highlights"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Interest rating: {interest}\nFit rating: {fit}\n\n{numbered}",
                }
            ],
            output_format=HighlightTags,
            output_config={"effort": "low"},
        )
    log_call(
        call_type="parsing",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note="retag highlights",
    )
    tags = response.parsed_output.tags
    if len(tags) != len(phrases):
        return None
    it = iter(tags)
    return UNTAGGED.sub(lambda m: f"=={next(it)}:{m.group(1)}==", text)


# One pass at a time; progress for the streaming route (api/settings.py) to report.
_lock = threading.Lock()
_state: dict = {"running": False, "done": 0, "total": 0, "retagged": 0, "skipped": 0, "error": None}


def get_state() -> dict:
    with _lock:
        return dict(_state)


def start() -> bool:
    """Starts the pass in a background thread; False if one is already running (it's followed)."""
    with _lock:
        if _state["running"]:
            return False
        _state.update(running=True, done=0, total=0, retagged=0, skipped=0, error=None)
    threading.Thread(target=_run, daemon=True).start()
    return True


def _retag_one(row: dict) -> bool:
    try:
        new_text = retag_rationale(row["rationale_text"], row["interest_fit"], row["candidate_fit"])
    except Exception:  # noqa: BLE001 - one failure skips that summary, it doesn't end the pass
        logger.exception("retagging score %s failed", row["id"])
        new_text = None
    if new_text:
        with get_db() as conn:
            conn.execute("UPDATE job_scores SET rationale_text = ? WHERE id = ?", (new_text, row["id"]))
    with _lock:
        _state["done"] += 1
        _state["retagged" if new_text else "skipped"] += 1
    return bool(new_text)


def _run() -> None:
    try:
        rows = _latest_untagged()
        with _lock:
            _state["total"] = len(rows)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            list(pool.map(_retag_one, rows))
    except Exception as exc:  # noqa: BLE001 - reported to the page
        logger.exception("retag pass failed")
        with _lock:
            _state["error"] = str(exc)
    finally:
        with _lock:
            _state["running"] = False
