"""Cheap rule-based scoring run on every newly-seen job before any LLM call, so the daily LLM cap
is spent on the most promising candidates first (see core/queue.py for the priority queue this
feeds). Scores are heuristic, not predictive -- they only decide scoring order/inclusion, not what
gets shown to the user (that's job_scores, from the LLM)."""

BASELINE = 50.0


def score_job(job: dict, settings: dict) -> tuple[float, dict]:
    reasons: dict = {}
    score = BASELINE

    text = f"{job.get('title') or ''} {job.get('description_text') or ''}".lower()

    exclude_kw = [k.lower() for k in settings.get("prefilter_keywords_exclude", [])]
    matched_exclude = [k for k in exclude_kw if k in text]
    if matched_exclude:
        reasons["excluded_keywords"] = matched_exclude
        return 0.0, reasons  # hard exclude -- still enqueued (status pending) but sinks to the bottom

    include_kw = [k.lower() for k in settings.get("prefilter_keywords_include", [])]
    if include_kw:
        matched_include = [k for k in include_kw if k in text]
        if matched_include:
            score += 20 * len(matched_include)
            reasons["matched_keywords"] = matched_include
        else:
            score -= 20
            reasons["no_keyword_match"] = True

    titles = [t.lower() for t in settings.get("prefilter_titles_include", [])]
    if titles:
        title_lower = (job.get("title") or "").lower()
        if any(t in title_lower for t in titles):
            score += 15
            reasons["title_match"] = True
        else:
            score -= 15
            reasons["title_no_match"] = True

    locations = [loc.lower() for loc in settings.get("prefilter_locations", [])]
    remote_only = settings.get("prefilter_remote_only", False)
    job_location = (job.get("location") or "").lower()
    remote_type = job.get("remote_type")

    if remote_only and remote_type != "remote":
        score -= 25
        reasons["not_remote"] = True
    elif locations:
        if remote_type == "remote" or any(loc in job_location for loc in locations):
            score += 10
            reasons["location_match"] = True
        else:
            score -= 10
            reasons["location_no_match"] = True

    salary_floor = settings.get("salary_floor")
    salary_max = job.get("salary_max")
    if salary_floor and salary_max is not None and salary_max < salary_floor:
        score -= 30
        reasons["below_salary_floor"] = True

    return max(0.0, min(100.0, score)), reasons
