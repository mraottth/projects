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
    return (
        f"## Candidate profile\n{candidate_profile or '(not available)'}\n\n"
        f"## Desired titles\n{titles}\n\n"
        f"## Desired sectors/industries\n{sectors}\n\n"
        f"## Desired locations\n{locations}{' (remote only)' if remote_only else ''}\n\n"
        f"## Minimum salary\n{salary_floor if salary_floor else '(not specified)'}"
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
