"""Shared Claude client + usage logging.

Model choice: Sonnet 5 is the default for scoring/parsing/generation. This is a deliberate
cost/quality tradeoff for a cost-capped personal tool (mirrors Shelf Life's chat.py tiering,
which also reaches for the cheaper tier for routine calls) -- override via JOBHUNT_LLM_MODEL
if quality ever needs to beat cost here.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

from jobhunt.core import budget
from jobhunt.db.connection import get_db
from jobhunt.core.users import optional_user_id

if TYPE_CHECKING:
    import anthropic

# The SDK is imported where it's first used, not at startup: it's about half of the app's import
# time, which a cold start on Cloud Run waits for before serving the first page.


class LLMError(Exception):
    """Raised for any Claude-call failure (missing/invalid credentials, rate limit, API error)."""


class BudgetExceeded(LLMError):
    """The person this call is for has used up this month's budget (core/budget.py)."""


DEFAULT_MODEL = os.environ.get("JOBHUNT_LLM_MODEL", "claude-sonnet-5")

# $ per 1M tokens, (input, output). Update when pricing changes; unknown models log with null cost.
PRICING_PER_MTOK = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    """The shared client. Every paid call site gets it right before calling Claude, so this is
    also where a person's monthly budget is enforced (core/budget.py); shared work, with nobody
    set, is never blocked."""
    global _client
    user_id = optional_user_id()
    if user_id is not None:
        budget.check(user_id)
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


OUT_OF_CREDITS_MESSAGE = (
    "your Anthropic API account is out of credits. API usage is billed separately from a "
    "claude.ai plan — add credits at platform.claude.com (Plans & Billing) for the organization "
    "that owns this app's API key, then try again."
)


@contextmanager
def llm_error_boundary(action: str):
    """Wrap a Claude call site so missing credentials / API errors surface as a clean LLMError
    instead of an unhandled 500 -- `anthropic`'s missing-auth check raises a bare TypeError, not
    one of its typed API errors, so that's caught here too."""
    import anthropic

    try:
        yield
    except anthropic.BadRequestError as exc:
        # Caught live: the raw "credit balance is too low" 400 read as an app bug. The usual cause
        # is paying for a claude.ai plan, which doesn't fund the API -- say so plainly, since this
        # hits every Claude-backed feature at once.
        if "credit balance is too low" in str(exc).lower():
            raise LLMError(f"{action} failed: {OUT_OF_CREDITS_MESSAGE}") from exc
        raise LLMError(f"{action} failed: {exc}") from exc
    except anthropic.APIError as exc:
        raise LLMError(f"{action} failed: {exc}") from exc
    except TypeError as exc:
        if "authentication" in str(exc).lower():
            raise LLMError(f"{action} failed: Claude API credentials not configured ({exc})") from exc
        raise


# Prompt-cache pricing, as multiples of the input rate: a read is 0.1x, a write with the default
# 5-minute lifetime 1.25x (the only lifetime this app uses). Cached tokens aren't part of
# usage.input_tokens, so they're priced separately here.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    rates = PRICING_PER_MTOK.get(model)
    if rates is None:
        return None
    in_rate, out_rate = rates
    billed_input = (
        input_tokens + cache_read_tokens * CACHE_READ_MULTIPLIER + cache_write_tokens * CACHE_WRITE_MULTIPLIER
    )
    return (billed_input / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def response_text(response, action: str) -> str:
    """The text of a Claude reply. A reply can come back with no text block at all, when the
    model's thinking uses up max_tokens before it writes anything (stop_reason "max_tokens"). Taking
    the first text block with a bare next() then raised StopIteration, which nothing caught: it
    killed every daily run once a preference-profile rebuild was due (2026-11-03, Admin page:
    "StopIteration:"). This raises an LLMError instead, which callers already handle."""
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    if not text.strip():
        stop = getattr(response, "stop_reason", None)
        raise LLMError(f"Claude didn't return any text while {action} (stop reason: {stop}) -- try again")
    return text


def cache_tokens(usage) -> dict:
    """A response's cache reads and writes, as log_call keyword arguments (0 when absent)."""
    return {
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None) or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", None) or 0,
    }


def log_call(
    call_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    job_id: int | None = None,
    note: str | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> None:
    cost = estimate_cost_usd(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO llm_calls
               (user_id, call_type, job_id, model, input_tokens, output_tokens, estimated_cost_usd, note,
                cache_read_tokens, cache_write_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                # Whoever the call was for; None for shared work done outside anyone's request
                # (the daily run reading a web-page board), which no one's budget pays for.
                optional_user_id(),
                call_type,
                job_id,
                model,
                input_tokens,
                output_tokens,
                cost,
                note,
                cache_read_tokens,
                cache_write_tokens,
            ),
        )
    user_id = optional_user_id()
    if user_id is not None:
        budget.after_call(user_id)
