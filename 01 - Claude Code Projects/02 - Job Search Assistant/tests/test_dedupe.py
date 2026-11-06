from jobhunt.core.dedupe import compute_dedupe_key, normalize_url


def test_normalize_url_strips_query_and_trailing_slash():
    a = normalize_url("https://example.com/jobs/123/?utm=foo")
    b = normalize_url("https://EXAMPLE.com/jobs/123")
    assert a == b


def test_dedupe_key_same_url_matches():
    k1 = compute_dedupe_key(url="https://x.com/jobs/1?a=1", company="Acme", title="Eng", location="NYC")
    k2 = compute_dedupe_key(url="https://x.com/jobs/1?a=2", company="Acme", title="Eng", location="NYC")
    assert k1 == k2


def test_dedupe_key_falls_back_to_fields_without_url():
    k1 = compute_dedupe_key(url=None, company="Acme", title="Software Engineer", location="NYC")
    k2 = compute_dedupe_key(url=None, company="acme", title="  software engineer  ", location="nyc")
    assert k1 == k2


def test_dedupe_key_differs_for_different_jobs():
    k1 = compute_dedupe_key(url=None, company="Acme", title="Engineer", location="NYC")
    k2 = compute_dedupe_key(url=None, company="Acme", title="Designer", location="NYC")
    assert k1 != k2


def test_dedupe_key_prefers_external_id_over_url():
    # Regression: Greenhouse-shaped URLs put the real job id in the query string
    # ("?gh_jid=123"), which normalize_url strips -- without preferring external_id, every
    # posting on a board would collapse onto the same key.
    shared_url = "https://stripe.com/jobs/search"
    k1 = compute_dedupe_key(
        url=f"{shared_url}?gh_jid=1", company="Stripe", title="A", location=None,
        source_id=1, external_id="1",
    )
    k2 = compute_dedupe_key(
        url=f"{shared_url}?gh_jid=2", company="Stripe", title="B", location=None,
        source_id=1, external_id="2",
    )
    assert k1 != k2


def test_dedupe_key_same_external_id_matches_even_with_different_url_noise():
    k1 = compute_dedupe_key(
        url="https://stripe.com/jobs/search?gh_jid=1", company="Stripe", title="A",
        location=None, source_id=1, external_id="1",
    )
    k2 = compute_dedupe_key(
        url="https://stripe.com/jobs/search?gh_jid=1&utm=x", company="Stripe", title="A",
        location=None, source_id=1, external_id="1",
    )
    assert k1 == k2
