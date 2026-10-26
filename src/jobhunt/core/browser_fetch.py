"""Headless-browser fallback for pages whose content loads via JS (confirmed needed for at least
techjobsforgov.com, a WP Job Manager site with no static HTML or feed fallback for its listings).
Only used when a plain httpx GET yields too little text -- a static fetch is always tried first
since it's far cheaper."""

from playwright.sync_api import sync_playwright


class BrowserFetchError(Exception):
    pass


def render_page_html(url: str, timeout_ms: int = 20_000) -> str:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                return page.content()
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 - surface any Playwright failure as our own error type
        raise BrowserFetchError(f"could not render {url}: {exc}") from exc
