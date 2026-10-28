"""Suggests companies worth monitoring, given the user's candidate profile and desired-role
settings -- via Claude + web search. Deliberately returns suggestions for the user to review and
approve (api/discovery.py), not auto-added sources; resolving an approved suggestion into an
actual monitored source reuses core/company_resolver.resolve_company() (POST /api/sources/resolve)."""

import json
import re
from urllib.parse import urlparse

from pydantic import BaseModel

from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call

SYSTEM_PROMPT = """You suggest companies a job seeker should consider monitoring for openings, \
based on their candidate profile and the roles/sectors/locations they're targeting -- desired \
sector is a real constraint, not a tiebreaker: don't suggest a well-known company outside it just \
because it's a strong brand name. Use web search to find real companies that are actually hiring \
for roles like this right now, not just companies that seem like a generic fit -- prioritize \
current, concrete evidence (an open posting, a recent hiring announcement) over general reputation.

Suggest 5-8 companies. For each, give a one-sentence reason tied to their specific profile/targets \
(not generic "great culture" reasoning). Prefer a mix of company sizes/stages over obvious mega-cap \
names only.

Respond with ONLY a JSON object on the last line of your response, no other text after it:
{"suggestions": [{"company_name": "...", "reason": "..."}, ...]}"""


class CompanySuggestion(BaseModel):
    company_name: str
    reason: str


BOARD_SYSTEM_PROMPT = """You suggest job boards worth monitoring for a job seeker, based on their \
candidate profile and the roles/sectors/locations they're targeting. "Job board" here means a \
multi-employer board, not a single company's careers page: a niche/industry board (e.g. a board \
specific to their sector), a regional or government jobs board, a remote-work board, or a general \
aggregator that lets you filter by role/keyword -- somewhere that lists openings across many \
employers in one place, growing the number of postings the user sees without them having to name \
every company by hand. Desired sector is a real constraint, not a tiebreaker.

Use web search to confirm each board is real, currently active, and actually relevant to the kind \
of role they want -- don't suggest a well-known generic board (e.g. LinkedIn, Indeed) that offers \
no narrower reach than what the user can already search themselves.

Suggest 3-6 boards. For each, give its direct URL and a one-sentence reason tied to their specific \
profile/targets (not generic reasoning).

Respond with ONLY a JSON object on the last line of your response, no other text after it:
{"suggestions": [{"name": "...", "url": "...", "reason": "..."}, ...]}"""


class JobBoardSuggestion(BaseModel):
    name: str
    url: str
    reason: str


# Low effort + a small search budget keeps a discovery call to roughly a minute instead of several
# (high, the default, ran up to five filtered web searches with long reasoning between them); the
# task is "find a handful of real, relevant names," which doesn't need deep deliberation.
DISCOVERY_EFFORT = "low"
MAX_WEB_SEARCHES = 3
# A long server-tool turn can stop with stop_reason "pause_turn" before the final JSON is written;
# re-sending the paused turn lets the server pick up where it left off. Capped so a misbehaving
# turn can't loop forever.
MAX_CONTINUATIONS = 3


def _extract_json(text: str) -> dict:
    """Parses the last {"suggestions": ...} object in the response. Scanning for that specific
    object (rather than greedily matching from the first "{" to the last "}") keeps stray braces in
    the prose before it -- e.g. a quoted snippet from a search result -- from breaking the parse."""
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r'\{\s*"suggestions"', text))):
        try:
            data, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"no suggestions JSON found in response: {text[:200]!r}")


def _run_discovery(system: str, user_content: str, note: str) -> str:
    """One discovery call with web search, resuming on pause_turn. Returns all text output."""
    client = get_client()
    messages = [{"role": "user", "content": user_content}]
    texts = []
    for _ in range(MAX_CONTINUATIONS + 1):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=6000,
            system=system,
            messages=messages,
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_WEB_SEARCHES}],
            output_config={"effort": DISCOVERY_EFFORT},
        )
        texts.extend(b.text for b in response.content if b.type == "text")
        search_uses = sum(1 for b in response.content if b.type == "server_tool_use")
        log_call(
            call_type="parsing",
            model=DEFAULT_MODEL,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            note=f"{note} ({search_uses} web search(es))",
        )
        if getattr(response, "stop_reason", None) != "pause_turn":
            break
        messages = [*messages, {"role": "assistant", "content": response.content}]
    return "\n".join(texts)


def _profile_text(candidate_profile: str | None, settings: dict, exclude: list[str] | None = None) -> str:
    titles = ", ".join(settings.get("prefilter_titles_include", [])) or "(not specified)"
    sectors = ", ".join(settings.get("prefilter_sectors", [])) or "(not specified)"
    locations = ", ".join(settings.get("prefilter_locations", [])) or "(not specified)"
    remote_only = settings.get("prefilter_remote_only", False)
    salary_floor = settings.get("salary_floor")
    notes = settings.get("looking_for_notes") or "(not specified)"
    exclude_text = ", ".join(exclude) if exclude else "(none yet)"
    return (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n"
        f"## Desired titles\n{titles}\n\n"
        f"## Desired sectors/industries\n{sectors}\n\n"
        f"## Desired locations\n{locations}{' (remote only)' if remote_only else ''}\n\n"
        f"## Minimum salary\n{salary_floor if salary_floor else '(not specified)'}\n\n"
        f"## In their own words, what they're looking for\n{notes}\n\n"
        f"## Already being monitored -- do not suggest these again\n{exclude_text}"
    )


def _hostname(url: str) -> str:
    host = urlparse(url if "://" in url else f"//{url}").hostname or url
    return host.lower().removeprefix("www.")


def suggest_companies(
    candidate_profile: str | None, settings: dict, exclude_names: list[str] | None = None
) -> list[CompanySuggestion]:
    with llm_error_boundary("suggesting companies to monitor"):
        text = _run_discovery(
            SYSTEM_PROMPT,
            _profile_text(candidate_profile, settings, exclude_names),
            "discovery:suggest_companies",
        )

    data = _extract_json(text)
    suggestions = [CompanySuggestion(**s) for s in data.get("suggestions", [])]
    if exclude_names:
        excluded = {n.strip().lower() for n in exclude_names}
        suggestions = [s for s in suggestions if s.company_name.strip().lower() not in excluded]
    return suggestions


def suggest_job_boards(
    candidate_profile: str | None, settings: dict, exclude_urls: list[str] | None = None
) -> list[JobBoardSuggestion]:
    exclude_hosts = [_hostname(u) for u in exclude_urls] if exclude_urls else None
    with llm_error_boundary("suggesting job boards to monitor"):
        text = _run_discovery(
            BOARD_SYSTEM_PROMPT,
            _profile_text(candidate_profile, settings, exclude_urls),
            "discovery:suggest_job_boards",
        )

    data = _extract_json(text)
    suggestions = [JobBoardSuggestion(**s) for s in data.get("suggestions", [])]
    if exclude_hosts:
        suggestions = [s for s in suggestions if _hostname(s.url) not in exclude_hosts]
    return suggestions
