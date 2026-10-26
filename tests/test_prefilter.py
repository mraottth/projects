from jobhunt.core.prefilter import score_job


def test_hard_excludes_on_excluded_keyword():
    job = {"title": "Senior Recruiter", "description_text": "staffing agency role"}
    settings = {"prefilter_keywords_exclude": ["recruiter"]}
    score, reasons = score_job(job, settings)
    assert score == 0.0
    assert "excluded_keywords" in reasons


def test_boosts_on_included_keyword():
    job = {"title": "Backend Engineer", "description_text": "we use Python and Postgres"}
    settings = {"prefilter_keywords_include": ["python"]}
    score, reasons = score_job(job, settings)
    assert score > 50.0
    assert reasons["matched_keywords"] == ["python"]


def test_penalizes_missing_included_keyword():
    job = {"title": "Backend Engineer", "description_text": "we use Ruby"}
    settings = {"prefilter_keywords_include": ["python"]}
    score, reasons = score_job(job, settings)
    assert score < 50.0
    assert reasons["no_keyword_match"] is True


def test_remote_only_penalizes_non_remote():
    job = {"title": "Engineer", "remote_type": "onsite"}
    settings = {"prefilter_remote_only": True}
    score, reasons = score_job(job, settings)
    assert score < 50.0
    assert reasons["not_remote"] is True


def test_salary_floor_penalizes_below_floor():
    job = {"title": "Engineer", "salary_max": 90000}
    settings = {"salary_floor": 120000}
    score, reasons = score_job(job, settings)
    assert score < 50.0
    assert reasons["below_salary_floor"] is True


def test_no_settings_gives_baseline():
    job = {"title": "Engineer"}
    score, reasons = score_job(job, {})
    assert score == 50.0
    assert reasons == {}


def test_score_clamped_to_0_100():
    job = {"title": "x", "salary_max": 1}
    settings = {
        "prefilter_remote_only": True,
        "salary_floor": 999999,
        "prefilter_titles_include": ["engineer"],
    }
    score, _ = score_job(job, settings)
    assert 0.0 <= score <= 100.0
