"""Generic "any URL" adapter: fetches a listing page (career page or multi-employer board) and has
Claude extract postings from it. Covers both a single company's /careers page and niche
multi-employer boards (e.g. techjobsforgov.com) the user specifically wants tracked -- both are
"fetch a listing page, extract postings from it" in shape.

Full job descriptions are NOT fetched here (a board can list hundreds of postings; fetching every
one's page would be expensive). Only lightweight listing-page fields are extracted; the daily
pipeline backfills description_text by fetching a job's own URL only for postings selected for
scoring, keeping cost proportional to what actually gets scored, not what's merely discovered."""

from datetime import date
from typing import Literal

from pydantic import BaseModel

from jobhunt.core.ats_detect import detect_embedded_ats
from jobhunt.core.llm import DEFAULT_MODEL, get_client, llm_error_boundary, log_call
from jobhunt.core.page_fetch import PageFetchError, fetch_page
from jobhunt.ingest.adapters import ashby, greenhouse, lever

MAX_POSTINGS_PER_FETCH = 30
# Low, not the default (high): copying postings off a page is mechanical, and at high effort each
# board spent a minute or more reasoning first -- "Check every source now" looked stuck on a few
# boards (caught live). Lower effort is also fewer billed output tokens.
EXTRACTION_EFFORT = "low"
PAGE_MAX_CHARS = 40_000

ATS_ADAPTERS = {"ats_greenhouse": greenhouse.fetch, "ats_lever": lever.fetch, "ats_ashby": ashby.fetch}


def _system_prompt() -> str:
    # Built fresh per call (not a module-level constant) so "today" stays correct in a
    # long-running process (e.g. an always-warm Cloud Run instance).
    return (
        "You extract job postings from the text of a job-board listing page. The page may list "
        "postings from one company or many; use whatever company name is shown per posting (if "
        "the page doesn't name one, it's a single-company board -- leave company null and the "
        "caller will fill in the known company). Links appear inline as `text [https://...]` -- "
        "use the URL immediately following a posting's title as that posting's url.\n\n"
        f'Dates are often relative ("Posted 3 days ago", "Closes in 10d") -- today\'s date is '
        f"{date.today().isoformat()}; convert a relative posted-date to an absolute ISO date, and "
        f"use null if you can't determine one. Extract at most {MAX_POSTINGS_PER_FETCH} postings; "
        f"if there are more, take the first {MAX_POSTINGS_PER_FETCH} as they appear on the page "
        "(ordinarily newest-first)."
    )


class JobListing(BaseModel):
    title: str
    company: str | None
    location: str | None
    url: str | None
    posted_at: str | None
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    salary_period: Literal["year", "hour"] | None


class JobListingsOutput(BaseModel):
    postings: list[JobListing]


def fetch(source: dict) -> list[dict]:
    """Reads the page and asks Claude for its postings, with two ways around a careers page that
    lists nothing in its own text (the usual case for an organization's /careers page, caught
    live when "Test fetch" came back empty on several): an embedded Greenhouse/Lever/Ashby board
    is read through that ATS's API instead, and if Claude finds nothing in the static page, it's
    re-read in the headless browser (listings loaded by JS, or in an iframe) and tried again."""
    url = source["config"]["url"]
    try:
        page = fetch_page(url, preserve_links=True, max_chars=PAGE_MAX_CHARS)
    except PageFetchError as exc:
        raise RuntimeError(f"custom board {url} could not be fetched: {exc}") from exc

    embedded = _fetch_embedded_ats(page.html, source)
    if embedded is not None:
        return embedded
    postings = _extract_postings(page.text, source)
    if postings or page.rendered:
        return postings

    try:
        rendered = fetch_page(url, preserve_links=True, max_chars=PAGE_MAX_CHARS, force_render=True)
    except PageFetchError:
        return postings
    embedded = _fetch_embedded_ats(rendered.html, source)
    if embedded is not None:
        return embedded
    if rendered.text == page.text:
        return postings  # nothing new to read; don't pay for the same answer twice
    return _extract_postings(rendered.text, source)


def _fetch_embedded_ats(html: str, source: dict) -> list[dict] | None:
    """Postings from the ATS board embedded in this page, or None if there isn't exactly one or
    reading it fails (then the page itself is read as usual)."""
    found = detect_embedded_ats(html, single_organization=bool(source.get("company_name")))
    if found is None:
        return None
    kind, config = found
    try:
        return ATS_ADAPTERS[kind]({**source, "kind": kind, "config": config})
    except Exception:  # noqa: BLE001 - a wrong guess at the board's name is just "not this way"
        return None


def _extract_postings(text: str, source: dict) -> list[dict]:
    url = source["config"]["url"]
    client = get_client()
    with llm_error_boundary(f"extracting postings from {url}"):
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=12000,
            system=_system_prompt(),
            messages=[{"role": "user", "content": text}],
            output_format=JobListingsOutput,
            output_config={"effort": EXTRACTION_EFFORT},
        )
    log_call(
        call_type="parsing",
        model=DEFAULT_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note=f"custom_board:{url}",
    )

    default_company = source.get("company_name")
    postings = []
    for listing in response.parsed_output.postings[:MAX_POSTINGS_PER_FETCH]:
        company = listing.company or default_company or "unknown"
        postings.append(
            {
                "external_id": None,
                "title": listing.title,
                "company": company,
                "location": listing.location,
                "remote_type": None,
                "salary_min": listing.salary_min,
                "salary_max": listing.salary_max,
                "salary_currency": listing.salary_currency,
                "salary_period": listing.salary_period,
                "description_text": None,  # backfilled on demand by the pipeline before scoring
                "url": listing.url,
                "posted_at": listing.posted_at,
                "raw_json": listing.model_dump(),
            }
        )
    return postings
