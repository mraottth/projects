from typing import Literal

import httpx
from pydantic import BaseModel

from jobhunt.core.dedupe import compute_dedupe_key
from jobhunt.core.html_extract import extract_readable_text
from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

PARSE_SYSTEM_PROMPT = """You extract structured job-posting fields from raw page text or a pasted job \
description. The text may be messy (nav/footer boilerplate, cookie banners) if it came from a web page \
-- ignore anything that isn't actually part of the job posting.

If a field genuinely cannot be determined, use null rather than guessing. For salary, only fill in \
numbers if the posting states them explicitly -- do not estimate a market rate."""


class ParsedJob(BaseModel):
    title: str
    company: str
    location: str | None
    remote_type: Literal["remote", "hybrid", "onsite"] | None
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    salary_period: Literal["year", "hour"] | None
    description_text: str
    posted_at: str | None  # ISO 8601 date, if stated or clearly inferable


class ManualImportError(Exception):
    pass


def fetch_url_text(url: str) -> str:
    try:
        resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=15.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ManualImportError(
            f"could not fetch {url} ({exc}) -- paste the job text directly instead"
        ) from exc
    text = extract_readable_text(resp.text)
    if len(text) < 200:
        raise ManualImportError(
            "fetched page had too little text to parse (likely a login wall or JS-rendered page) "
            "-- paste the job text directly instead"
        )
    return text


def parse_job_posting(*, url: str | None = None, raw_text: str | None = None) -> dict:
    if not url and not raw_text:
        raise ManualImportError("provide either a url or raw_text")

    source_text = raw_text if raw_text else fetch_url_text(url)  # type: ignore[arg-type]

    client = get_client()
    with llm_error_boundary("parsing the job posting"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=2000,
            system=PARSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": source_text}],
            output_format=ParsedJob,
        )
    log_call(
        call_type="parsing",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    parsed = response.parsed_output
    data = parsed.model_dump()
    data["url"] = url
    data["dedupe_key"] = compute_dedupe_key(
        url=url, company=parsed.company, title=parsed.title, location=parsed.location
    )
    return data
