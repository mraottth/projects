import httpx

from jobhunt.core import company_resolver
from tests.fakes import FakeAnthropicClient, FakeCreateResponse


class Resp404:
    def raise_for_status(self):
        raise httpx.HTTPStatusError("404", request=None, response=httpx.Response(404))


class RespJobs:
    def __init__(self, jobs):
        self._jobs = jobs

    def raise_for_status(self):
        pass

    def json(self):
        return {"jobs": self._jobs}


def test_probe_ats_resolves_on_first_matching_ats(monkeypatch):
    # greenhouse.httpx, lever.httpx, and ashby.httpx are all the *same* httpx module -- a single
    # patch has to branch on the URL, not three independent per-adapter patches.
    def fake_get(url, **kwargs):
        if "ashbyhq.com" in url:
            return RespJobs([{"id": "1", "title": "Engineer", "descriptionPlain": "x"}])
        return Resp404()

    monkeypatch.setattr(httpx, "get", fake_get)

    resolved = company_resolver._probe_ats("Notion")
    assert resolved is not None
    assert resolved.kind == "ats_ashby"
    assert resolved.config == {"board_name": "notion"}


def test_probe_ats_returns_none_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp404())
    assert company_resolver._probe_ats("TotallyFakeCo") is None


def test_resolve_company_falls_back_to_web_search_when_probe_fails(monkeypatch):
    monkeypatch.setattr(company_resolver, "_probe_ats", lambda name: None)
    monkeypatch.setattr(
        company_resolver,
        "get_client",
        lambda: FakeAnthropicClient(
            create_result=FakeCreateResponse('{"kind": "ats_lever", "value": "acme"}')
        ),
    )

    resolved = company_resolver.resolve_company("Acme")
    assert resolved.kind == "ats_lever"
    assert resolved.config == {"company_slug": "acme"}


def test_resolve_via_web_search_returns_none_for_not_found(monkeypatch):
    monkeypatch.setattr(
        company_resolver,
        "get_client",
        lambda: FakeAnthropicClient(
            create_result=FakeCreateResponse('{"kind": "not_found", "value": null}')
        ),
    )
    assert company_resolver._resolve_via_web_search("NoSuchCo") is None


def test_resolve_via_web_search_handles_custom_board(monkeypatch):
    monkeypatch.setattr(
        company_resolver,
        "get_client",
        lambda: FakeAnthropicClient(
            create_result=FakeCreateResponse(
                '{"kind": "custom_board", "value": "https://acme.com/careers"}'
            )
        ),
    )
    resolved = company_resolver._resolve_via_web_search("Acme")
    assert resolved.kind == "custom_board"
    assert resolved.config == {"url": "https://acme.com/careers"}


def test_extract_json_handles_surrounding_prose():
    text = 'I searched and found it.\n{"kind": "ats_greenhouse", "value": "acme"}'
    assert company_resolver._extract_json(text) == {"kind": "ats_greenhouse", "value": "acme"}


def test_slug_candidates_handles_spaces_and_punctuation():
    candidates = company_resolver._slug_candidates("Acme, Inc.")
    assert "acme inc" not in candidates  # spaces must be resolved one way or the other
    assert any("acme" in c for c in candidates)
