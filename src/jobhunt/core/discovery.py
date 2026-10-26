"""Suggests companies worth monitoring, given the user's candidate profile and desired-role
settings -- via Claude + web search. Deliberately returns suggestions for the user to review and
approve (api/discovery.py), not auto-added sources; resolving an approved suggestion into an
actual monitored source reuses core/company_resolver.resolve_company(), same as Favorites."""

import json
import re

from pydantic import BaseModel

from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call

SYSTEM_PROMPT = """You suggest companies a job seeker should consider monitoring for openings, \
based on their candidate profile and the roles/sectors/locations they're targeting -- desired \
sector is a real constraint, not a tiebreaker: don't suggest a well-known company outside it just \
because it's a strong brand name. Use web search to find real companies that are actually hiring \
for roles like this right now, not just companies that seem like a generic fit -- prioritize \
current, concrete evidence (an open posting, a recent hiring announcement) over general reputation.

Suggest 6-10 companies. For each, give a one-sentence reason tied to their specific profile/targets \
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

Suggest 4-8 boards. For each, give its direct URL and a one-sentence reason tied to their specific \
profile/targets (not generic reasoning).

Respond with ONLY a JSON object on the last line of your response, no other text after it:
{"suggestions": [{"name": "...", "url": "...", "reason": "..."}, ...]}"""


class JobBoardSuggestion(BaseModel):
    name: str
    url: str
    reason: str


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def _profile_text(candidate_profile: str | None, settings: dict) -> str:
    titles = ", ".join(settings.get("prefilter_titles_include", [])) or "(not specified)"
    sectors = ", ".join(settings.get("prefilter_sectors", [])) or "(not specified)"
    locations = ", ".join(settings.get("prefilter_locations", [])) or "(not specified)"
    remote_only = settings.get("prefilter_remote_only", False)
    salary_floor = settings.get("salary_floor")
    notes = settings.get("looking_for_notes") or "(not specified)"
    return (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n"
        f"## Desired titles\n{titles}\n\n"
        f"## Desired sectors/industries\n{sectors}\n\n"
        f"## Desired locations\n{locations}{' (remote only)' if remote_only else ''}\n\n"
        f"## Minimum salary\n{salary_floor if salary_floor else '(not specified)'}\n\n"
        f"## In their own words, what they're looking for\n{notes}"
    )


def suggest_companies(candidate_profile: str | None, settings: dict) -> list[CompanySuggestion]:
    client = get_client()
    with llm_error_boundary("suggesting companies to monitor"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _profile_text(candidate_profile, settings)}],
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
        )
    text = "\n".join(b.text for b in response.content if b.type == "text")
    search_uses = sum(1 for b in response.content if b.type == "server_tool_use")
    log_call(
        call_type="parsing",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note=f"discovery:suggest_companies ({search_uses} web search(es))",
    )

    data = _extract_json(text)
    return [CompanySuggestion(**s) for s in data.get("suggestions", [])]


def suggest_job_boards(candidate_profile: str | None, settings: dict) -> list[JobBoardSuggestion]:
    client = get_client()
    with llm_error_boundary("suggesting job boards to monitor"):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=3000,
            system=BOARD_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _profile_text(candidate_profile, settings)}],
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
        )
    text = "\n".join(b.text for b in response.content if b.type == "text")
    search_uses = sum(1 for b in response.content if b.type == "server_tool_use")
    log_call(
        call_type="parsing",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note=f"discovery:suggest_job_boards ({search_uses} web search(es))",
    )

    data = _extract_json(text)
    return [JobBoardSuggestion(**s) for s in data.get("suggestions", [])]
