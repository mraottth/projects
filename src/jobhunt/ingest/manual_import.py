from typing import Literal

from pydantic import BaseModel

from jobhunt.core.dedupe import compute_dedupe_key
from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call
from jobhunt.core.page_fetch import PageFetchError, fetch_page_text

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
        return fetch_page_text(url)
    except PageFetchError as exc:
        raise ManualImportError(f"{exc} -- paste the job text directly instead") from exc


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
            output_config={"effort": "low"},  # field extraction, like custom_board's
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
