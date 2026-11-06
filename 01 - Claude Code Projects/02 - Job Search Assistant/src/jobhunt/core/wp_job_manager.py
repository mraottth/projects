"""Job boards built on WP Job Manager, a common WordPress job-board plugin (e.g. Civic Work Studio,
formerly techjobsforgov.com), read through the plugin's own listings endpoint instead of the page.

Such a page arrives with an empty `<ul class="job_listings">`; the plugin's script then asks
`/jm-ajax/get_listings/` for the jobs, a few at a time. So reading the page, even in the browser,
found at best the first handful (caught live: Civic Work Studio has 362 jobs, and the owner said
boards "rarely seem to work"). The endpoint returns every job as clean HTML cards, up to 100 per
request, with no Claude call. The source URL's query string (the board's own filters, e.g. remote
only and job types) is sent the way the plugin's script sends it (`form_data`), plus the standard
fields directly, so a board's filters keep working. Verified live against civicworkstudio.com
(2026-11-06): all jobs at 100 per page, and the owner's filtered URL gave its 30 remote jobs.

Postings carry no external_id, like the page reader's, so they keep the URL-based dedupe keys of
jobs already stored from these boards. Descriptions are fetched later, before scoring, as for any
web-page board."""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from jobhunt.core.page_fetch import USER_AGENT
from jobhunt.core.salary_extract import extract_salary

PER_PAGE = 100
MAX_PAGES = 5  # up to 500 jobs from one board per read
TIMEOUT = 20.0
PROBE_TIMEOUT = 10.0  # when the page already failed: keep the whole Test fetch short

# The plugin's script settings on the page: "ajax_url":"\/jm-ajax\/%%endpoint%%\/" (slashes escaped).
AJAX_URL = re.compile(r'"ajax_url"\s*:\s*"([^"]*%%endpoint%%[^"]*)"')
MARKERS = ("wp-job-manager-ajax-filters", "jm-ajax")
# Fields the plugin reads straight from the request, sent alongside form_data for boards that
# don't pass the whole form through.
STANDARD_FIELDS = ("search_keywords", "search_location", "search_categories[]", "filter_job_type[]")

log = logging.getLogger(__name__)


def listings_endpoint(page_url: str, html: str) -> str | None:
    """The board's get_listings URL, if the page runs WP Job Manager's listing script."""
    found = AJAX_URL.search(html)
    if found:
        template = found.group(1).replace("\\/", "/")
        return urljoin(page_url, template.replace("%%endpoint%%", "get_listings"))
    if any(marker in html for marker in MARKERS):
        return urljoin(page_url, "/jm-ajax/get_listings/")
    return None


def fetch(page_url: str, html: str | None, source: dict, steps: list[str]) -> list[dict] | None:
    """Every listing (up to MAX_PAGES pages), or None when this isn't a WP Job Manager board or
    its endpoint can't be read, so the page is read the usual way instead.

    With `html` None (the page itself couldn't be loaded), the site's standard endpoint is tried
    anyway, and only an answer shaped like WP Job Manager's counts. Live, Civic Work Studio's page
    failed to load from the deployed app while its listings endpoint answered from Google Cloud."""
    probing = html is None
    endpoint = urljoin(page_url, "/jm-ajax/get_listings/") if probing else listings_endpoint(page_url, html)
    if endpoint is None:
        return None
    query = urlsplit(page_url).query
    params = parse_qs(query, keep_blank_values=True)
    base: dict[str, str | list[str]] = {
        k: [v for v in params[k] if v] for k in STANDARD_FIELDS if k in params
    }
    base = {k: v for k, v in base.items() if v}
    base.update(form_data=query, per_page=str(PER_PAGE))

    postings: list[dict] = []
    total = None
    for page in range(1, MAX_PAGES + 1):
        try:
            resp = httpx.post(
                endpoint,
                data={**base, "page": str(page)},
                timeout=PROBE_TIMEOUT if probing else TIMEOUT,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict) or "found_jobs" not in data:
                raise ValueError("not a WP Job Manager listings answer")
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            log.info("WP Job Manager listings at %s unreadable: %s", endpoint, exc)
            if postings:
                break
            if not probing:  # a guess that didn't pan out isn't worth a step
                steps.append(f"found a WP Job Manager board but couldn't read its listings ({exc})")
            return None
        total = data.get("total_jobs", total)
        found = parse_cards(data.get("html") or "", source)
        if not found and page == 1 and data.get("found_jobs"):
            steps.append("found a WP Job Manager board but couldn't make out its listings")
            return None  # listings we can't parse: let the page reader try
        postings += found
        if not found or page >= int(data.get("max_num_pages") or 1):
            break

    more = f" of {total}" if isinstance(total, int) and total > len(postings) else ""
    how = (
        "read its listings directly (WP Job Manager)"
        if probing
        else "read the board's listings (WP Job Manager)"
    )
    steps.append(f"{how}: {len(postings)}{more}")
    return postings


def parse_cards(html: str, source: dict) -> list[dict]:
    """One posting per `li.job_listing` card: link, title, organization, location, posted date,
    and pay when the card shows it."""
    soup = BeautifulSoup(html, "html.parser")
    default_company = source.get("company_name") or "unknown"
    postings = []
    for card in soup.select("li.job_listing"):
        link = card.find("a", href=True)
        title_el = card.select_one(".position h3") or card.find(["h3", "h2"])
        if link is None or title_el is None:
            continue
        company_el = card.select_one(".company strong") or card.select_one(".company")
        location_el = card.select_one(".location")
        posted_el = card.find("time")
        card_text = card.get_text(" ", strip=True)
        location = _clean(location_el.get_text(" ", strip=True)) if location_el else None
        salary = extract_salary(card_text) or {}
        posted = posted_el.get("datetime") if posted_el else None
        postings.append(
            {
                "external_id": None,
                "title": _clean(title_el.get_text(" ", strip=True)),
                "company": (_clean(company_el.get_text(" ", strip=True)) if company_el else None)
                or default_company,
                "location": location or None,
                "remote_type": "remote" if re.search(r"\bremote\b", card_text, re.I) else None,
                "salary_min": salary.get("salary_min"),
                "salary_max": salary.get("salary_max"),
                "salary_currency": salary.get("salary_currency"),
                "salary_period": salary.get("salary_period"),
                "description_text": None,  # backfilled before scoring, as for any web-page board
                "url": link["href"],
                "posted_at": posted[:10] if posted and re.match(r"\d{4}-\d{2}-\d{2}", posted) else None,
                "raw_json": {"card": card_text[:1000], "via": "wp_job_manager"},
            }
        )
    return postings


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
