import pytest

from jobhunt.core import page_fetch
from jobhunt.core.browser_fetch import BrowserFetchError
from jobhunt.core.page_fetch import PageFetchError, fetch_page_text


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


def test_static_fetch_used_when_content_is_substantial(monkeypatch):
    html = f"<html><body>{'Job description text. ' * 30}</body></html>"
    def fail(*a, **k):
        raise AssertionError("should not render when static content is already substantial")

    monkeypatch.setattr(page_fetch.httpx, "get", lambda *a, **k: FakeResponse(html))
    monkeypatch.setattr(page_fetch, "render_page_html", fail)

    text = fetch_page_text("https://example.com/job/1")
    assert "Job description text." in text


def test_falls_back_to_browser_when_static_content_thin(monkeypatch):
    thin_html = "<html><body>Please sign in</body></html>"
    rendered_html = f"<html><body>{'Real job posting content. ' * 30}</body></html>"

    monkeypatch.setattr(page_fetch.httpx, "get", lambda *a, **k: FakeResponse(thin_html))
    monkeypatch.setattr(page_fetch, "render_page_html", lambda *a, **k: rendered_html)

    text = fetch_page_text("https://example.com/job/1")
    assert "Real job posting content." in text


def test_raises_when_both_static_and_browser_fail(monkeypatch):
    monkeypatch.setattr(page_fetch.httpx, "get", lambda *a, **k: FakeResponse("<html></html>"))

    def fail(*a, **k):
        raise BrowserFetchError("boom")

    monkeypatch.setattr(page_fetch, "render_page_html", fail)

    with pytest.raises(PageFetchError):
        fetch_page_text("https://example.com/job/1")


def test_preserve_links_keeps_hrefs_inline(monkeypatch):
    html = (
        "<html><body>"
        + "<a href='/jobs/1'>Senior Engineer</a> " * 20
        + "</body></html>"
    )
    monkeypatch.setattr(page_fetch.httpx, "get", lambda *a, **k: FakeResponse(html))

    text = fetch_page_text("https://example.com/board", preserve_links=True)
    assert "[https://example.com/jobs/1]" in text


def test_force_render_skips_the_static_fetch(monkeypatch):
    def no_static(*a, **k):
        raise AssertionError("force_render should go straight to the browser")

    rendered_html = f"<html><body>{'Rendered listing. ' * 30}</body></html>"
    monkeypatch.setattr(page_fetch.httpx, "get", no_static)
    monkeypatch.setattr(page_fetch, "render_page_html", lambda *a, **k: rendered_html)

    page = page_fetch.fetch_page("https://example.com/jobs", force_render=True)
    assert page.rendered
    assert "Rendered listing." in page.text
    assert page.html == rendered_html
