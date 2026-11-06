"""Resolve a company name (typed by the user, or suggested by core/discovery.py) into an actual
monitorable source: try cheap direct probes against the three ATS APIs first (no LLM call --
confirmed live during development that this works directly for a good fraction of real companies,
e.g. "anthropic" and "figma" resolve immediately on Greenhouse), and only fall back to a Claude +
web-search call when none of the guesses pan out."""

import json
import logging
import re
from dataclasses import dataclass

from jobhunt.core.discovery import run_web_search
from jobhunt.core.llm import get_client, llm_error_boundary
from jobhunt.ingest.adapters import ashby, greenhouse, lever

logger = logging.getLogger(__name__)

ATS_PROBES = [
    ("ats_greenhouse", "board_token", greenhouse.fetch),
    ("ats_lever", "company_slug", lever.fetch),
    ("ats_ashby", "board_name", ashby.fetch),
]

WEB_SEARCH_SYSTEM_PROMPT = """You find which job board a company uses, if any. Search the web for \
the company's careers/jobs page. If it's hosted on Greenhouse (boards.greenhouse.io), Lever \
(jobs.lever.co), or Ashby (jobs.ashbyhq.com), give the board token/slug from that URL. Otherwise, \
if they have their own careers page, give its direct URL. If you can't find a careers page at all, \
say so.

Respond with ONLY a JSON object on the last line of your response, no other text after it:
{"kind": "ats_greenhouse" | "ats_lever" | "ats_ashby" | "custom_board" | "not_found", \
"value": "<board token/slug, or the careers page URL for custom_board, or null for not_found>"}"""


@dataclass
class ResolvedSource:
    kind: str
    config: dict
    note: str


def _slug_candidates(company_name: str) -> list[str]:
    lowered = company_name.strip().lower()
    no_punct = re.sub(r"[^a-z0-9\s-]", "", lowered)
    candidates = [
        no_punct.replace(" ", ""),
        no_punct.replace(" ", "-"),
    ]
    # de-dupe while preserving order
    seen = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]


def _probe_ats(company_name: str) -> ResolvedSource | None:
    for kind, config_key, fetch in ATS_PROBES:
        for slug in _slug_candidates(company_name):
            try:
                postings = fetch({"config": {config_key: slug}, "company_name": company_name})
            except Exception:  # noqa: BLE001 - a guess that errors (404, non-JSON page, odd
                # payload) just means "not this one"; only HTTP errors used to be caught, so any
                # other failure crashed the whole request instead of moving on to the next guess.
                continue
            if postings:
                return ResolvedSource(
                    kind=kind, config={config_key: slug}, note=f"found via slug guess '{slug}'"
                )
    return None


def _extract_json(text: str) -> dict:
    match = re.search(r"\{[^{}]*\}", text.strip().splitlines()[-1]) or re.search(
        r"\{[^{}]*\}", text, re.DOTALL
    )
    if not match:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def _resolve_via_web_search(company_name: str) -> ResolvedSource | None:
    with llm_error_boundary(f"resolving {company_name}'s job board via web search"):
        text = run_web_search(
            WEB_SEARCH_SYSTEM_PROMPT,
            f"Company: {company_name}",
            f"resolve_company:{company_name}",
            client=get_client(),
        )
    try:
        result = _extract_json(text)
    except ValueError:
        # An unparseable answer is a "couldn't find it", not a crash (it used to surface as a raw 500).
        logger.warning("couldn't parse job-board lookup for %s: %r", company_name, text[:200])
        return None
    kind, value = result.get("kind"), result.get("value")
    if kind == "not_found" or not value:
        return None
    if kind == "custom_board":
        note = "found via web search (custom page)"
        return ResolvedSource(kind=kind, config={"url": value}, note=note)
    config_key = {
        "ats_greenhouse": "board_token",
        "ats_lever": "company_slug",
        "ats_ashby": "board_name",
    }[kind]
    return ResolvedSource(kind=kind, config={config_key: value}, note=f"found via web search ({kind})")


def resolve_company(company_name: str) -> ResolvedSource | None:
    return _probe_ats(company_name) or _resolve_via_web_search(company_name)
