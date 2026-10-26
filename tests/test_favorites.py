from jobhunt.core import company_resolver, favorites
from jobhunt.core.llm import LLMError
from jobhunt.db.connection import get_db


def test_add_favorite_success_links_a_source(monkeypatch):
    monkeypatch.setattr(
        favorites,
        "resolve_company",
        lambda name: company_resolver.ResolvedSource(
            kind="ats_greenhouse", config={"board_token": "acme"}, note="found via slug guess"
        ),
    )

    result = favorites.add_favorite("Acme Corp")
    assert result["resolution_status"] == "resolved"
    assert result["source_id"] is not None

    with get_db() as conn:
        source = conn.execute(
            "SELECT * FROM sources WHERE id = ?", (result["source_id"],)
        ).fetchone()
    assert source["is_favorite"] == 1
    assert source["kind"] == "ats_greenhouse"


def test_add_favorite_failure_keeps_record_without_a_source(monkeypatch):
    monkeypatch.setattr(favorites, "resolve_company", lambda name: None)

    result = favorites.add_favorite("Nonexistent Co")
    assert result["resolution_status"] == "failed"
    assert result["source_id"] is None


def test_add_favorite_is_idempotent_by_name(monkeypatch):
    monkeypatch.setattr(favorites, "resolve_company", lambda name: None)
    favorites.add_favorite("Acme")
    favorites.add_favorite("Acme")
    assert len(favorites.list_favorites()) == 1


def test_retry_favorite_reattempts_resolution(monkeypatch):
    monkeypatch.setattr(favorites, "resolve_company", lambda name: None)
    added = favorites.add_favorite("Acme")
    assert added["resolution_status"] == "failed"

    monkeypatch.setattr(
        favorites,
        "resolve_company",
        lambda name: company_resolver.ResolvedSource(
            kind="ats_lever", config={"company_slug": "acme"}, note="found later"
        ),
    )
    retried = favorites.retry_favorite(added["id"])
    assert retried["resolution_status"] == "resolved"


def test_remove_favorite_unflags_source_but_keeps_it(monkeypatch):
    monkeypatch.setattr(
        favorites,
        "resolve_company",
        lambda name: company_resolver.ResolvedSource(
            kind="ats_greenhouse", config={"board_token": "acme"}, note="x"
        ),
    )
    added = favorites.add_favorite("Acme")
    source_id = added["source_id"]

    favorites.remove_favorite(added["id"])

    assert favorites.list_favorites() == []
    with get_db() as conn:
        source = conn.execute("SELECT is_favorite FROM sources WHERE id = ?", (source_id,)).fetchone()
    assert source["is_favorite"] == 0


def test_is_favorite_company_name_case_insensitive(monkeypatch):
    monkeypatch.setattr(
        favorites,
        "resolve_company",
        lambda name: company_resolver.ResolvedSource(kind="ats_greenhouse", config={}, note="x"),
    )
    favorites.add_favorite("Acme Corp")
    assert favorites.is_favorite_company_name("acme corp") is True
    assert favorites.is_favorite_company_name("ACME CORP") is True
    assert favorites.is_favorite_company_name("Someone Else") is False
    assert favorites.is_favorite_company_name(None) is False


def test_is_favorite_company_name_false_when_resolution_failed(monkeypatch):
    monkeypatch.setattr(favorites, "resolve_company", lambda name: None)
    favorites.add_favorite("Acme")
    assert favorites.is_favorite_company_name("Acme") is False


def test_add_favorite_with_url_skips_resolver_entirely(monkeypatch):
    def fail_if_called(name):
        raise AssertionError("resolve_company should not be called for a direct-URL favorite")

    monkeypatch.setattr(favorites, "resolve_company", fail_if_called)

    result = favorites.add_favorite_with_url(
        "Blue Rose Research", "custom_board", {"url": "https://blueroseresearch.org/jobs/"}
    )
    assert result["resolution_status"] == "resolved"
    assert result["source_id"] is not None

    with get_db() as conn:
        source = conn.execute("SELECT * FROM sources WHERE id = ?", (result["source_id"],)).fetchone()
    assert source["is_favorite"] == 1
    assert source["kind"] == "custom_board"
    assert source["company_name"] == "Blue Rose Research"


def test_add_favorite_with_url_is_idempotent_by_name(monkeypatch):
    favorites.add_favorite_with_url("Acme", "ats_lever", {"company_slug": "acme"})
    favorites.add_favorite_with_url("Acme", "ats_lever", {"company_slug": "acme"})
    assert len(favorites.list_favorites()) == 1


def test_add_favorite_records_failure_instead_of_sticking_at_pending(monkeypatch):
    # Regression: an LLMError from the web-search fallback (e.g. missing credentials) used to
    # propagate out of add_favorite(), leaving the row at resolution_status='pending' forever
    # with no error recorded and no way for the UI to offer a retry.
    def raise_llm_error(name):
        raise LLMError("credentials not configured")

    monkeypatch.setattr(favorites, "resolve_company", raise_llm_error)

    result = favorites.add_favorite("Acme")
    assert result["resolution_status"] == "failed"
    assert "credentials not configured" in result["resolution_note"]
    assert result["source_id"] is None
