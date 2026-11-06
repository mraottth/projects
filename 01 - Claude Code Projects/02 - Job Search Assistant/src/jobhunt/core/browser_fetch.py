"""Headless-browser fallback for pages whose content loads via JS (confirmed needed for at least
techjobsforgov.com, a WP Job Manager site with no static HTML or feed fallback for its listings).
Only used when a plain httpx GET yields too little text -- a static fetch is always tried first
since it's far cheaper."""

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


class BrowserFetchError(Exception):
    pass


# Hosts whose requests are worth keeping (see render_page_html): job-board APIs a careers page's
# own scripts may call to fill in its listings.
ATS_REQUEST_HINTS = (
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "workable.com",
    "bamboohr.com",
    "applytojob.com",
    "breezy.hr",
    "recruitee.com",
    "rippling.com",
    "paylocity.com",
    "myworkdayjobs.com",
    "jobvite.com",
    "smartrecruiters.com",
    "dover.com",
    "pinpointhq.com",
)


def render_page_html(url: str, timeout_ms: int = 20_000) -> str:
    """The rendered page's HTML, with each embedded frame's HTML appended: an ATS job board is
    often an <iframe>, and page.content() alone returns only the outer page. Requests the page made
    to a job-board host are appended too, in an HTML comment (invisible to text extraction): a
    page whose script fetches its listings from, say, the Greenhouse API names the board only in
    that request, never in its HTML."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                requested: list[str] = []
                page.on(
                    "request",
                    lambda r: requested.append(r.url) if any(h in r.url for h in ATS_REQUEST_HINTS) else None,
                )
                try:
                    page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                except PlaywrightTimeoutError:
                    # Analytics or chat widgets can keep the network busy forever; whatever has
                    # rendered by now is still worth reading.
                    pass
                parts = [page.content()]
                for frame in page.frames[1:]:
                    try:
                        parts.append(f'<div data-frame-url="{frame.url}">{frame.content()}</div>')
                    except Exception:  # noqa: BLE001 - a detached/cross-origin frame just gets skipped
                        continue
                if requested:
                    parts.append("<!-- requests:\n" + "\n".join(dict.fromkeys(requested)) + "\n-->")
                return "\n".join(parts)
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 - surface any Playwright failure as our own error type
        raise BrowserFetchError(f"could not render {url}: {exc}") from exc
