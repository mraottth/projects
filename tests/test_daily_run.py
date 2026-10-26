from jobhunt.core import company_resolver, favorites, queue, settings
from jobhunt.core.scoring import JobScore
from jobhunt.db.connection import get_db
from jobhunt.pipeline import daily_run
from tests.fakes import FakeAnthropicClient, FakeCreateResponse, FakeParseResponse


def _make_posting(title: str, company: str = "Acme") -> dict:
    return {
        "external_id": title,
        "title": title,
        "company": company,
        "location": "Remote",
        "remote_type": "remote",
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
        "description_text": "Great role.",
        "url": f"https://example.com/{title}",
        "posted_at": "2026-10-20",
        "raw_json": {},
    }


def _add_source(kind="ats_greenhouse", config=None) -> int:
    import json

    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO sources (kind, name, config_json) VALUES (?, ?, ?)",
            (kind, "Test Source", json.dumps(config or {})),
        )
        return cursor.lastrowid


def _fake_score_client(interest="High", candidate="Med"):
    score = JobScore(
        interest_fit=interest,
        interest_score=80,
        candidate_fit=candidate,
        candidate_fit_score=60,
        summary_text="summary",
        rationale_text="rationale",
    )
    return FakeAnthropicClient(parse_result=FakeParseResponse(score))


def test_run_daily_full_cycle(monkeypatch):
    _add_source()
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"ats_greenhouse": lambda source: [_make_posting("Engineer A"), _make_posting("Engineer B")]},
    )
    from jobhunt.core import scoring

    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    summary = daily_run.run_daily()

    assert summary["sources_polled"] == 1
    assert summary["jobs_fetched"] == 2
    assert summary["jobs_upserted"] == 2
    assert summary["prefiltered_count"] == 2
    assert summary["llm_scored_count"] == 2
    assert summary["carried_over_count"] == 0

    with get_db() as conn:
        jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
        scores = conn.execute("SELECT COUNT(*) AS n FROM job_scores").fetchone()["n"]
        digest = conn.execute("SELECT COUNT(*) AS n FROM digest_entries").fetchone()["n"]
        runs = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
    assert jobs == 2
    assert scores == 2
    assert digest == 2
    assert runs == 1


def test_run_daily_respects_daily_cap_and_carries_over(monkeypatch):
    from jobhunt.core import scoring

    _add_source()
    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {"ats_greenhouse": lambda source: [_make_posting(f"Job {i}") for i in range(5)]},
    )
    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())
    settings.update_settings({"daily_llm_cap": 2})

    summary = daily_run.run_daily()

    assert summary["llm_scored_count"] == 2
    assert summary["carried_over_count"] == 3
    assert queue.pending_count() == 3


def test_run_daily_dedupes_already_seen_jobs(monkeypatch):
    _add_source()
    monkeypatch.setattr(
        daily_run, "ADAPTERS", {"ats_greenhouse": lambda source: [_make_posting("Same Job")]}
    )
    from jobhunt.core import scoring

    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    daily_run.run_daily()
    daily_run.run_daily()  # same posting seen again on a second run

    with get_db() as conn:
        jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
        scores = conn.execute("SELECT COUNT(*) AS n FROM job_scores").fetchone()["n"]
    assert jobs == 1
    assert scores == 1  # not re-scored once it already has a score


def test_run_daily_one_bad_source_does_not_abort_run(monkeypatch):
    _add_source()
    _add_source()

    def failing_adapter(source):
        raise RuntimeError("boom")

    def working_adapter(source):
        return [_make_posting("Survivor")]

    monkeypatch.setattr(
        daily_run, "ADAPTERS", {"ats_greenhouse": failing_adapter}
    )
    # Second source shares the same kind in this fixture, so patch per-call via a counter instead.
    calls = {"n": 0}

    def dispatch(source):
        calls["n"] += 1
        if calls["n"] == 1:
            return failing_adapter(source)
        return working_adapter(source)

    monkeypatch.setattr(daily_run, "ADAPTERS", {"ats_greenhouse": dispatch})
    from jobhunt.core import scoring

    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    summary = daily_run.run_daily()
    assert summary["jobs_fetched"] == 1
    assert len(summary["errors"]) == 1


def test_run_daily_triggers_preference_rebuild_when_signal_present(monkeypatch):
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (dedupe_key, title, company) VALUES ('k1', 'Engineer', 'Acme')"
        )
        job_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO feedback (job_id, interested) VALUES (?, 'interested')", (job_id,)
        )

    from jobhunt.core import preference

    monkeypatch.setattr(daily_run, "ADAPTERS", {})
    monkeypatch.setattr(
        preference,
        "get_client",
        lambda: FakeAnthropicClient(create_result=FakeCreateResponse("profile text")),
    )

    summary = daily_run.run_daily()
    assert summary["preference_profile_rebuilt"] is True


def test_favorite_company_jobs_bypass_prefilter_excludes_and_cap(monkeypatch):
    from jobhunt.core import scoring

    monkeypatch.setattr(
        favorites,
        "resolve_company",
        lambda name: company_resolver.ResolvedSource(kind="ats_greenhouse", config={}, note="x"),
    )
    # add_favorite resolves to its own ats_greenhouse source -- that's the only source this test
    # needs; adding a second one would double-poll the same fake adapter below.
    favorites.add_favorite("Acme")

    settings.update_settings({"daily_llm_cap": 1, "prefilter_keywords_exclude": ["great role"]})

    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {
            "ats_greenhouse": lambda source: [
                _make_posting("Favorite Role", company="Acme"),  # excluded keyword, but favorite
                _make_posting("Other Role", company="Someone Else"),  # not excluded, not favorite
            ]
        },
    )
    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    summary = daily_run.run_daily()

    # both get scored: the favorite bypasses its exclude-keyword match, and the cap of 1 is
    # still fully available to the non-favorite job since favorites don't consume it.
    assert summary["llm_scored_count"] == 2
    with get_db() as conn:
        scored_titles = {
            row["title"]
            for row in conn.execute(
                "SELECT j.title FROM job_scores js JOIN jobs j ON j.id = js.job_id"
            )
        }
    assert scored_titles == {"Favorite Role", "Other Role"}


def test_favorite_job_board_bypasses_cap_by_source_not_company(monkeypatch):
    # A favorite job board lists postings from many different companies -- none of them are in
    # favorite_companies, so only the source-level is_favorite flag can catch this case.
    from jobhunt.core import scoring

    import json

    with get_db() as conn:
        conn.execute(
            "INSERT INTO sources (kind, name, config_json, is_favorite) VALUES (?, ?, ?, 1)",
            ("custom_board", "Favorite Board", json.dumps({"url": "https://board.example.com"})),
        )

    settings.update_settings({"daily_llm_cap": 1, "prefilter_keywords_exclude": ["great role"]})

    monkeypatch.setattr(
        daily_run,
        "ADAPTERS",
        {
            "custom_board": lambda source: [
                _make_posting("Board Role", company="Unrelated Co"),  # excluded kw, but favorite board
            ]
        },
    )
    monkeypatch.setattr(scoring, "get_client", lambda: _fake_score_client())

    summary = daily_run.run_daily()

    assert summary["llm_scored_count"] == 1
    with get_db() as conn:
        scored_titles = {
            row["title"]
            for row in conn.execute(
                "SELECT j.title FROM job_scores js JOIN jobs j ON j.id = js.job_id"
            )
        }
    assert scored_titles == {"Board Role"}
