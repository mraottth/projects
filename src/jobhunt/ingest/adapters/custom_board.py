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

from jobhunt.core.ats_detect import detect_embedded_ats, detect_hosted_board
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


# Fewer postings than this from a page's static HTML and it's read again in the browser, then
# through the hosted job board it links to, if any. A careers page often shows one featured role
# (or a general application) in its HTML and loads the rest with JS, in an iframe, or on an ATS
# page behind "See all openings" (caught live: Test fetch found 1 posting on one careers page and
# 0 on another, both with many openings).
FEW_POSTINGS = 3


def fetch(source: dict) -> list[dict]:
    return fetch_with_report(source)[0]


def fetch_with_report(source: dict) -> tuple[list[dict], list[str]]:
    """The postings, plus a few words per step on how they were found (shown by Test fetch).

    1. An embedded Greenhouse/Lever/Ashby board is read through that ATS's API, no Claude call.
    2. Otherwise Claude reads the page's text.
    3. Too few postings: the page is re-rendered in the headless browser, which also catches an
       ATS board that only its scripts request, and read again if it now shows more.
    4. Still too few: the one hosted job board it links to (Workable, BambooHR, JazzHR, Workday...)
       is read instead. The biggest result wins."""
    url = source["config"]["url"]
    steps: list[str] = []
    try:
        page = fetch_page(url, preserve_links=True, max_chars=PAGE_MAX_CHARS)
    except PageFetchError as exc:
        raise RuntimeError(f"custom board {url} could not be fetched: {exc}") from exc
    how = "in the browser" if page.rendered else "as plain HTML"

    embedded = _fetch_embedded_ats(page.html, source, steps)
    if embedded is not None:
        return embedded, steps
    postings = _extract_postings(page.text, source)
    steps.append(f"read the page {how}: {len(postings)}")
    html = page.html

    if len(postings) < FEW_POSTINGS and not page.rendered:
        try:
            rendered = fetch_page(url, preserve_links=True, max_chars=PAGE_MAX_CHARS, force_render=True)
        except PageFetchError as exc:
            steps.append(f"browser re-render failed ({exc})")
        else:
            html = rendered.html
            embedded = _fetch_embedded_ats(rendered.html, source, steps)
            if embedded is not None:
                return embedded, steps
            if _worth_rereading(page.text, rendered.text, had_postings=bool(postings)):
                again = _extract_postings(rendered.text, source)
                steps.append(f"read the page in the browser: {len(again)}")
                postings = max(postings, again, key=len)
            else:
                steps.append("browser showed nothing new")

    if len(postings) < FEW_POSTINGS:
        board = detect_hosted_board(html)
        if board and board.rstrip("/") != url.rstrip("/"):
            try:
                board_page = fetch_page(board, preserve_links=True, max_chars=PAGE_MAX_CHARS)
            except PageFetchError as exc:
                steps.append(f"couldn't open its job board {board} ({exc})")
            else:
                from_board = _extract_postings(board_page.text, source)
                steps.append(f"read its job board {board}: {len(from_board)}")
                postings = max(postings, from_board, key=len)
    return postings, steps


def _worth_rereading(static_text: str, rendered_text: str, *, had_postings: bool) -> bool:
    """Whether the rendered page is worth a second paid read. With nothing found yet, any change
    is. With a posting or two already, only a page with more links than before, since dynamic
    bits (dates, counters) change the text of most pages without adding any listings."""
    if rendered_text == static_text:
        return False
    return not had_postings or rendered_text.count("[http") > static_text.count("[http")


def _fetch_embedded_ats(html: str, source: dict, steps: list[str]) -> list[dict] | None:
    """Postings from the ATS board embedded in this page, or None if there isn't exactly one or
    reading it fails (then the page itself is read as usual)."""
    found = detect_embedded_ats(html, single_organization=bool(source.get("company_name")))
    if found is None:
        return None
    kind, config = found
    board = next(iter(config.values()))
    try:
        postings = ATS_ADAPTERS[kind]({**source, "kind": kind, "config": config})
    except Exception:  # noqa: BLE001 - a wrong guess at the board's name is just "not this way"
        steps.append(f"found a {kind.removeprefix('ats_').title()} board '{board}' but couldn't read it")
        return None
    steps.append(f"read its {kind.removeprefix('ats_').title()} board '{board}': {len(postings)}")
    return postings


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
