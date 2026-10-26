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
