"""Phase-1 profile of goodreads_interactions_dedup.json.gz before any rebuild (the data-swap decision gate).

Streams the interactions file once and reports, without building anything:
- rows by kind: rated (1-5), read but unrated, to-read (shelved, not read), and rows whose book isn't in the
  editions table;
- readers, and ratings / reads per reader;
- catalog size at several rater thresholds, with s04's language rule (an English or unlabeled edition);
- how many ratings today's catalog would get, and how many readers could join the readers-like-you pool;
- date coverage (read_at, date_added) on rated rows;
- whether today's 10,000 held-out test readers are present, and how long their histories become.

Raters per work count rated rows, so a reader who rated two editions of one work counts twice (rare; the
real pipeline collapses editions). Run with `make exp-profile`; writes eval_interactions/profile.md and .json.
Production files are only read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from goodrec.config import EVAL_DIR, RAW_DIR, ROOT, load_config  # noqa: E402
from goodrec.pipeline.io import iter_jsonl_gz  # noqa: E402

PROD_INTERIM = ROOT / "data" / "interim"          # today's build, read only
THRESHOLDS = (20, 30, 50, 75, 100, 150, 200)


def pct(a: np.ndarray, q) -> list[float]:
    return [float(x) for x in np.percentile(a, q)] if len(a) else []


def main(limit: int | None = None) -> None:
    cfg = load_config()
    src = RAW_DIR / cfg["data"]["files"]["ratings"]
    if "interactions" not in src.name:
        sys.exit(f"profile: expected the interactions file, config points at {src.name} (run via make exp-profile)")
    t0 = time.time()

    ed = pl.read_parquet(PROD_INTERIM / "editions.parquet", columns=["book_id", "work_id", "language_code"])
    works = ed["work_id"].unique().sort().to_numpy()
    widx = {int(w): i for i, w in enumerate(works)}
    book_to_w = {int(b): widx[int(w)] for b, w in zip(ed["book_id"].to_list(), ed["work_id"].to_list())}
    lang_ok = np.zeros(len(works), bool)
    ok_works = ed.filter(pl.col("language_code").is_in(cfg["catalog"]["languages"]))["work_id"].unique().to_list()
    lang_ok[[widx[int(w)] for w in ok_works]] = True
    cat_now = pl.read_parquet(PROD_INTERIM / "catalog.parquet", columns=["work_id"])["work_id"].to_list()
    in_cat_now = np.zeros(len(works), bool)
    in_cat_now[[widx[int(w)] for w in cat_now]] = True
    del ed

    raters = np.zeros(len(works), np.int32)
    users: dict[str, int] = {}
    cap = 1_200_000
    u_rated, u_read, u_toread, u_rated_cat = (np.zeros(cap, np.int32) for _ in range(4))
    kinds = dict(rated=0, read_unrated=0, to_read=0, unmapped_book=0)
    has_read_at = has_added = rated_rows = 0
    n = 0
    for r in iter_jsonl_gz(src):
        n += 1
        if limit and n > limit:
            n -= 1
            break
        if n % 10_000_000 == 0:
            print(f"  {n / 1e6:.0f}M rows, {len(users):,} readers ({time.time() - t0:.0f}s)", flush=True)
        w = book_to_w.get(int(r["book_id"]))
        if w is None:
            kinds["unmapped_book"] += 1
            continue
        u = users.setdefault(r["user_id"], len(users))
        rating = int(r.get("rating") or 0)
        if rating > 0:
            kinds["rated"] += 1
            raters[w] += 1
            u_rated[u] += 1
            if in_cat_now[w]:
                u_rated_cat[u] += 1
            rated_rows += 1
            has_read_at += bool(r.get("read_at"))
            has_added += bool(r.get("date_added"))
        elif r.get("is_read"):
            kinds["read_unrated"] += 1
            u_read[u] += 1
        else:
            kinds["to_read"] += 1
            u_toread[u] += 1
    nu = len(users)
    u_rated, u_read, u_toread, u_rated_cat = u_rated[:nu], u_read[:nu], u_toread[:nu], u_rated_cat[:nu]

    # Today's held-out test readers, by Goodreads user id.
    prod_users = pl.read_parquet(PROD_INTERIM / "users.parquet")
    test_idx = np.load(PROD_INTERIM / "test_users.npy")
    test_ids = prod_users.filter(pl.col("user_idx").is_in(test_idx.tolist()))["user_id"].to_list()
    found = [users[t] for t in test_ids if t in users]
    old_counts = (pl.scan_parquet(PROD_INTERIM / "ratings.parquet").filter(pl.col("rating") > 0)
                    .filter(pl.col("user_idx").is_in(test_idx.tolist())).group_by("user_idx").len().collect()["len"]
                    .to_numpy())

    catalog = {t: {"works": int(((raters >= t) & lang_ok).sum()),
                   "ratings_in_catalog": int(raters[(raters >= t) & lang_ok].sum()),
                   "todays_books_kept": int(((raters >= t) & lang_ok & in_cat_now).sum())} for t in THRESHOLDS}
    rep = {
        "file": src.name, "rows": n, "minutes": round((time.time() - t0) / 60, 1), "kinds": kinds,
        "readers": nu,
        "ratings_per_reader": {"mean": float(u_rated.mean()), "p50_p90_p99": pct(u_rated, [50, 90, 99])},
        "readers_with_ratings": int((u_rated > 0).sum()),
        "readers_pool_ge10_ratings_in_current_catalog": int((u_rated_cat >= 10).sum()),
        "ratings_on_current_catalog": int(raters[in_cat_now].sum()),
        "current_catalog_works": int(in_cat_now.sum()),
        "catalog_by_threshold": catalog,
        "date_coverage_on_rated": {"read_at": has_read_at / max(rated_rows, 1), "date_added": has_added / max(rated_rows, 1)},
        "test_readers": {"today": len(test_ids), "present": len(found),
                         "ratings_now_p50_p90": pct(old_counts, [50, 90]),
                         "ratings_after_p50_p90": pct(u_rated[found], [50, 90]) if found else []},
    }
    out = EVAL_DIR / ("profile" if not limit else "profile_smoke")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    out.with_suffix(".md").write_text(render(rep))
    print(render(rep))


def render(r: dict) -> str:
    k = r["kinds"]
    lines = [f"# Interactions profile ({r['file']})", "",
             f"{r['rows']:,} rows in {r['minutes']} min. Rated {k['rated']:,}; read but unrated {k['read_unrated']:,}; "
             f"to-read {k['to_read']:,}; book not in the editions table {k['unmapped_book']:,}.", "",
             f"- Readers: {r['readers']:,} ({r['readers_with_ratings']:,} with ratings). Ratings per reader: mean "
             f"{r['ratings_per_reader']['mean']:.0f}, median / p90 / p99 "
             + " / ".join(f"{x:.0f}" for x in r['ratings_per_reader']['p50_p90_p99']) + ".",
             f"- Today's catalog ({r['current_catalog_works']:,} works) would get {r['ratings_on_current_catalog']:,} "
             f"ratings; {r['readers_pool_ge10_ratings_in_current_catalog']:,} readers have 10+ of them (the "
             "readers-like-you pool rule).",
             f"- Dates on rated rows: read_at {100 * r['date_coverage_on_rated']['read_at']:.0f}%, date_added "
             f"{100 * r['date_coverage_on_rated']['date_added']:.0f}%.",
             f"- Test readers: {r['test_readers']['present']:,} of {r['test_readers']['today']:,} present. Ratings per "
             f"test reader, median / p90: now " + " / ".join(f"{x:.0f}" for x in r['test_readers']['ratings_now_p50_p90'])
             + ", after " + " / ".join(f"{x:.0f}" for x in r['test_readers']['ratings_after_p50_p90']) + ".", "",
             f"| Rater threshold | Catalog works | Ratings on those works | Today's {r['current_catalog_works']:,} books kept |",
             "|---|---|---|---|"]
    for t, c in r["catalog_by_threshold"].items():
        kept = c.get("todays_books_kept")
        lines.append(f"| {t} | {c['works']:,} | {c['ratings_in_catalog']:,} | "
                     + (f"{kept:,} ({100 * kept / r['current_catalog_works']:.1f}%)" if kept is not None else "–") + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after this many rows (a smoke test)")
    main(**vars(ap.parse_args()))
