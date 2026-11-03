"""The scoring cost experiment (asked 2026-11-03), run from the Admin page.

Scoring costs about 2-3 cents a job, mostly output: reasoning at effort "medium" plus a free-length
rationale, then the whole posting as input. scoring.LEAN (effort "low", a 2-3 sentence rationale,
the posting capped at 6,000 characters) should cost about half that. Before switching live scoring,
this rescores up to SAMPLE_SIZE jobs the person has already judged, once under each setting, and
compares the cost and the ratings, against each other and against the person's own marks.

Both arms are scored now, rather than comparing LEAN with the stored scores, since the profile has
changed since most of those were made. Nothing is written to job_scores, the queue or the digest;
the calls are logged (note "scoring experiment: <arm>") and the result is kept in the
`scoring_experiment` setting so a revisit costs nothing."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from jobhunt.core.generic_postings import is_generic_posting
from jobhunt.core.scoring import CURRENT, LEAN, score_job_with
from jobhunt.core.settings import get_setting, update_settings
from jobhunt.core.users import VISIBLE_JOB_SQL, carry, current_user_id
from jobhunt.db.connection import get_db

logger = logging.getLogger(__name__)

SETTING = "scoring_experiment"
SAMPLE_SIZE = 20
CONCURRENCY = 4
ARMS = (CURRENT, LEAN)
# Per job, both arms together: about 2.5 cents current plus about 1.2 lean.
EST_COST_PER_JOB = 0.037
REACHED_INTERVIEW = ("interviewing", "offer", "rejected_interviewed")

_lock = threading.Lock()
_state: dict = {"running": False, "done": 0, "total": 0, "error": None}


def sample_jobs(limit: int = SAMPLE_SIZE) -> list[dict]:
    """Jobs the current person has judged, newest judgment first: half they wanted (an Interested
    mark, or any tracked application) and half they didn't (a Not interested mark), where there
    are enough of each. `interviewed` marks an application that reached an interview."""
    uid = current_user_id()
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT j.*, f.interested AS mark, f.created_at AS marked_at,
                       a.status AS app_status, a.updated_at AS app_at
                FROM jobs j
                LEFT JOIN (SELECT job_id, interested, created_at,
                                  ROW_NUMBER() OVER (PARTITION BY job_id ORDER BY id DESC) AS rn
                           FROM feedback WHERE user_id = :uid) f ON f.job_id = j.id AND f.rn = 1
                LEFT JOIN applications a ON a.job_id = j.id AND a.user_id = :uid
                WHERE {VISIBLE_JOB_SQL}
                  AND (a.id IS NOT NULL OR f.interested IN ('interested', 'not_interested'))""",
            {"uid": uid},
        ).fetchall()
    positive, negative = [], []
    for row in rows:
        job = dict(row)
        if is_generic_posting(job["title"]):
            continue
        job["judgment"] = (
            "interested" if job["app_status"] or job["mark"] == "interested" else "not_interested"
        )
        job["interviewed"] = job["app_status"] in REACHED_INTERVIEW
        job["judged_at"] = max(filter(None, [job["marked_at"], job["app_at"]]), default="")
        (positive if job["judgment"] == "interested" else negative).append(job)
    positive.sort(key=lambda j: j["judged_at"], reverse=True)
    negative.sort(key=lambda j: j["judged_at"], reverse=True)
    half = limit // 2
    take_pos = min(len(positive), max(half, limit - len(negative)))
    take_neg = min(len(negative), limit - take_pos)
    return positive[:take_pos] + negative[:take_neg]


def status() -> dict:
    """What the Admin card shows before (and after) a run."""
    n = len(sample_jobs())
    return {
        "sample_size": n,
        "estimate_usd": round(n * EST_COST_PER_JOB, 2),
        "result": get_setting(SETTING),
        "running": get_state()["running"],
    }


def get_state() -> dict:
    with _lock:
        return dict(_state)


def start() -> bool:
    """Starts a run in a background thread; False if one is already going (it's followed)."""
    with _lock:
        if _state["running"]:
            return False
        _state.update(running=True, done=0, total=0, error=None)
    threading.Thread(target=carry(_run), daemon=True).start()
    return True


def _run() -> None:
    try:
        update_settings({SETTING: run()})
    except Exception as exc:  # noqa: BLE001 - reported to the page
        logger.exception("scoring experiment failed")
        with _lock:
            _state["error"] = str(exc) or type(exc).__name__
    finally:
        with _lock:
            _state["running"] = False


def run() -> dict:
    """Scores the sample under each arm in turn (each arm's candidate context stays cached within
    its own batch) and returns the comparison."""
    from jobhunt.pipeline.daily_run import _backfill_description, _score_args

    jobs = [_backfill_description(job) for job in sample_jobs()]
    with _lock:
        _state["total"] = len(jobs) * len(ARMS)
    args = _score_args()
    results: dict[str, list[dict | None]] = {}
    for arm in ARMS:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            results[arm.name] = list(pool.map(carry(lambda job, arm=arm: _score(job, args, arm)), jobs))

    rows = []
    for i, job in enumerate(jobs):
        rows.append(
            {
                "job_id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "judgment": job["judgment"],
                "interviewed": job["interviewed"],
                **{arm.name: results[arm.name][i] for arm in ARMS},
            }
        )
    return {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jobs": rows,
        "summary": summarize(rows),
    }


def _score(job: dict, args: tuple, arm) -> dict | None:
    try:
        score, cost, output_tokens = score_job_with(
            job, *args, settings=arm, note=f"scoring experiment: {arm.name}"
        )
    except Exception:  # noqa: BLE001 - one failure leaves that cell empty, it doesn't end the run
        logger.exception("experiment scoring of job %s (%s) failed", job["id"], arm.name)
        score = None
    with _lock:
        _state["done"] += 1
    if score is None:
        return None
    return {
        "interest_fit": score.interest_fit,
        "interest_score": score.interest_score,
        "candidate_fit": score.candidate_fit,
        "candidate_fit_score": score.candidate_fit_score,
        "rationale_text": score.rationale_text,
        "cost_usd": cost,
        "output_tokens": output_tokens,
    }


def _share(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None


def summarize(rows: list[dict]) -> dict:
    """Per arm: cost, and agreement with the person's marks (Interested rated High/Med interest,
    Not interested rated Low, and an interview rated High/Med fit). Between arms: how often the
    lean labels match the current ones, and the average score difference."""
    arms = {}
    for arm in ARMS:
        scored = [(r, r[arm.name]) for r in rows if r[arm.name]]
        costs = [s["cost_usd"] for _, s in scored if s["cost_usd"] is not None]
        wanted = [s for r, s in scored if r["judgment"] == "interested"]
        unwanted = [s for r, s in scored if r["judgment"] == "not_interested"]
        interviewed = [s for r, s in scored if r["interviewed"]]
        arms[arm.name] = {
            "scored": len(scored),
            "failed": len(rows) - len(scored),
            "total_cost_usd": round(sum(costs), 4),
            "cost_per_job_usd": round(sum(costs) / len(costs), 4) if costs else None,
            "output_tokens_per_job": round(sum(s["output_tokens"] for _, s in scored) / len(scored))
            if scored
            else None,
            "interested_rated_high_or_med": _share(
                sum(s["interest_fit"] != "Low" for s in wanted), len(wanted)
            ),
            "not_interested_rated_low": _share(
                sum(s["interest_fit"] == "Low" for s in unwanted), len(unwanted)
            ),
            "interviewed_fit_high_or_med": _share(
                sum(s["candidate_fit"] != "Low" for s in interviewed), len(interviewed)
            ),
        }
    both = [(r[CURRENT.name], r[LEAN.name]) for r in rows if r[CURRENT.name] and r[LEAN.name]]
    agreement = {
        "compared": len(both),
        "interest_same": _share(sum(c["interest_fit"] == n["interest_fit"] for c, n in both), len(both)),
        "fit_same": _share(sum(c["candidate_fit"] == n["candidate_fit"] for c, n in both), len(both)),
        "interest_score_diff": round(
            sum(abs(c["interest_score"] - n["interest_score"]) for c, n in both) / len(both), 1
        )
        if both
        else None,
        "fit_score_diff": round(
            sum(abs(c["candidate_fit_score"] - n["candidate_fit_score"]) for c, n in both) / len(both), 1
        )
        if both
        else None,
    }
    return {
        "arms": arms,
        "agreement": agreement,
        "judged": {
            "interested": sum(r["judgment"] == "interested" for r in rows),
            "not_interested": sum(r["judgment"] == "not_interested" for r in rows),
            "interviewed": sum(r["interviewed"] for r in rows),
        },
    }
