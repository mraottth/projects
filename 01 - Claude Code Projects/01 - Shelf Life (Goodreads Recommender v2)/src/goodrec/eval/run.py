"""Offline evaluation: per-user temporal hide-and-predict on held-out users (goodrec.eval.split).

For each evaluated user, the model sees their visible history (cut to the n most recent ratings for each
n in eval.n_buckets). Two tracks, on the same users, histories and hidden books:
- Track 1, recommendation quality: rank unread books; a hit is a hidden book they rated >= 4.
- Track 2, rating prediction: predict the rating of every hidden book (goodrec.eval.rating).
Shelf Life (the For you ranking and the card's predicted rating, as served) is compared with random, popular
and genre+popularity baselines (ranking), book-average, user-offset and user-mean baselines (rating), the
2023 recommender's similar-readers and SVD methods (both tracks, goodrec.eval.legacy2023), the previous best
model (eval/champion.json) and, with --ablations, its own components. Writes
eval/reports/<stamp>_<model>.md/.json and per-user results to eval/runs/.

  uv run python -m goodrec.eval.run                          # test set, all baselines
  uv run python -m goodrec.eval.run --set validation --users 300 --models shelf_life,popular   # quick check
  uv run python -m goodrec.eval.run --set validation --models shelf_life --grid "k_a=20,50;a_max=0.5,1.0"
  uv run python -m goodrec.eval.run --promote                # make this model the champion if it beats it
  uv run python -m goodrec.eval.run --params "delta_pred=0" --name "..."   # re-score other settings (not promotable)
  uv run python -m goodrec.eval.run --rating "item_means=global;calibration=none"   # an earlier rating model
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker process

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import orjson  # noqa: E402

from goodrec.config import ARTIFACTS_DIR, EVAL_DIR, ROOT, load_config  # noqa: E402  (EVAL_DIR: eval/, or GOODREC_EVAL)
from goodrec.core.artifacts import load_artifacts  # noqa: E402
from goodrec.core.scoring import Params  # noqa: E402
from goodrec.eval import report  # noqa: E402
from goodrec.eval.metrics import (METRICS, POOL_LEN, RATING_COLS, at_k, paired, pooled_summary,  # noqa: E402
                                  rating_pool, rating_user)
from goodrec.eval.models import (Recommender, ShelfLife, rating_baselines, shelf_life_models,  # noqa: E402
                                 simple_baselines)
from goodrec.eval.rating import (STYLES, RatingSettings, population_sd, ranking_artifacts,  # noqa: E402
                                 rating_artifacts, rating_style, user_sigma)
from goodrec.eval.split import load_split  # noqa: E402

CHAMPION = EVAL_DIR / "champion.json"
KMAX = 20
COLS = [f"{m}@{k}" for k in (10, 20) for m in METRICS]

_STATE: dict = {}      # set before forking the worker pool


def _work(job: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score one model on a chunk of cases, on each track it takes part in (runs in a worker process)."""
    mi, idx = job
    rec: Recommender = _STATE["models"][mi]
    cases, buckets, rel_min, seed = _STATE["cases"], _STATE["buckets"], _STATE["rel_min"], _STATE["seed"]
    fallback, pop_sd = _STATE["fallback"], _STATE["pop_sd"]
    met = np.full((len(idx), len(buckets), len(COLS)), np.nan, np.float32)
    tops = np.full((len(idx), len(buckets), KMAX), -1, np.int32)
    rmet = np.full((len(idx), len(buckets), len(RATING_COLS)), np.nan, np.float32)
    pool = np.zeros((len(idx), len(buckets), POOL_LEN), np.float32)
    for r, ci in enumerate(idx):
        c = cases[ci]
        rel = c.relevant(rel_min)
        items, truth = c.hidden.astype(np.int64), c.hidden_r.astype(np.float64)
        for b, n in enumerate(buckets):
            user = c.user_input(n)
            top, pred = rec.run(user, KMAX, np.random.default_rng([seed, c.user, b]), items)
            if top is not None:
                top = np.asarray(top, dtype=np.int64)
                tops[r, b, : len(top)] = top
                met[r, b] = [*at_k(top, rel, 10), *at_k(top, rel, 20)]
            if pred is not None:
                # Only visible ratings and training-only artifacts reach `pred` and σ_u; `truth` is only compared.
                pred = np.asarray(pred, dtype=np.float64)
                miss = np.isnan(pred)
                pred = np.clip(np.where(miss, fallback[items], pred), 1, 5)
                sigma = user_sigma(np.fromiter(user.ratings.values(), dtype=np.float64), pop_sd)
                rmet[r, b] = rating_user(pred, truth, sigma)
                pool[r, b] = rating_pool(pred, truth, int(miss.sum()))
    return mi, met, tops, rmet, pool


def _subsample(n_cases: int, size: int | None, seed: int) -> np.ndarray:
    if not size or size >= n_cases:
        return np.arange(n_cases)
    return np.sort(np.random.default_rng(seed).choice(n_cases, size, replace=False))


def evaluate(models: list[Recommender], cases, buckets, cfg, workers: int) -> dict:
    """Run every model; returns key -> {"idx", "met", "tops", "rmet", "pool"} (NaN / zeros off its tracks)."""
    _STATE.update(models=models, cases=cases, buckets=buckets, rel_min=cfg["relevant_min_rating"], seed=cfg["seed"])
    jobs, out = [], {}
    for mi, rec in enumerate(models):
        idx = _subsample(len(cases), rec.subsample, cfg["seed"])
        out[rec.key] = {"idx": idx, "met": [], "tops": [], "rmet": [], "pool": []}
        for chunk in np.array_split(idx, max(1, min(len(idx), workers * 6))):
            jobs.append((mi, chunk))
    t0 = time.time()
    done = 0
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for res in pool.imap(_work, jobs):
                done = _collect(res, models, out, done, len(jobs), t0)
    else:
        for job in jobs:
            done = _collect(_work(job), models, out, done, len(jobs), t0)
    for v in out.values():
        for f in FIELDS:
            v[f] = np.concatenate(v[f])
    return out


def _collect(res, models, out, done, total, t0) -> int:
    mi, *arrays = res
    o = out[models[mi].key]
    for f, a in zip(FIELDS, arrays):
        o[f].append(a)
    done += 1
    if done % max(1, total // 20) == 0 or done == total:
        print(f"  {done}/{total} chunks ({time.time() - t0:.0f}s)", flush=True)
    return done


FIELDS = ("met", "tops", "rmet", "pool")
FILL = {"met": np.nan, "tops": -1, "rmet": np.nan, "pool": 0}


def per_user(res: dict, n_cases: int, f: str = "met") -> np.ndarray:
    """(n_cases, buckets, ...) for one result field, NaN (-1, 0) for users a subsampled model didn't run on."""
    full = np.full((n_cases, *res[f].shape[1:]), FILL[f], res[f].dtype)
    full[res["idx"]] = res[f]
    return full


def git_info() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "config"], capture_output=True,
                                    text=True, cwd=ROOT).stdout.strip())
        return sha, dirty
    except OSError:
        return "unknown", False


def params_json(p: Params) -> dict:
    return dataclasses.asdict(p)


def params_from_json(d: dict) -> Params:
    d = dict(d)
    d["confidence"] = {int(k): float(v) for k, v in d["confidence"].items()}
    return Params(**d)


def grid_points(spec: str) -> list[dict]:
    """"k_a=20,50;a_max=0.5,1.0" -> every combination, as Params overrides. "null"/"none" mean None."""
    def value(v: str):
        v = v.strip()
        if v.lower() in ("null", "none"):
            return None
        return int(v) if v.lstrip("-").isdigit() else float(v)
    axes = []
    for part in filter(None, (x.strip() for x in spec.split(";"))):
        name, _, vals = part.partition("=")
        if not vals or name.strip() not in {f.name for f in dataclasses.fields(Params)}:
            raise SystemExit(f"--grid: bad axis {part!r} (expected param=v1,v2,... with a Params field)")
        axes.append([(name.strip(), value(v)) for v in vals.split(",")])
    return [dict(combo) for combo in itertools.product(*axes)]


def cache_path(rec: Recommender, split_hash: str, set_: str, users: int | None, cfg: dict) -> Path:
    """Where a baseline's per-user results are cached. Baselines don't change between runs unless their
    code, the split, the evaluation settings or the artifacts do, and all of those are in the key."""
    h = hashlib.sha256(repr((rec.key, rec.subsample, rec.description, split_hash, set_, users, cfg["n_buckets"],
                             cfg["relevant_min_rating"], cfg["seed"], KMAX)).encode())
    h.update((ARTIFACTS_DIR / "manifest.json").read_bytes())
    for f in ("models.py", "legacy2023.py", "metrics.py", "rating.py", "split.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    return EVAL_DIR / "runs" / f"cache_{rec.key}_{h.hexdigest()[:12]}.npz"


def load_champion() -> dict | None:
    return orjson.loads(CHAMPION.read_bytes()) if CHAMPION.exists() else None


def main(set_: str = "test", users: int | None = None, models: str | None = None, ablations: bool = False,
         grid: str | None = None, workers: int | None = None, promote: bool = False, name: str | None = None,
         fresh: bool = False, params: str | None = None, rating: str | None = None) -> None:
    if grid and set_ != "validation":
        raise SystemExit("--grid tunes parameters, so it only runs on --set validation.")
    if promote and (params or rating):
        raise SystemExit("--params and --rating re-score other settings; they can't be promoted (change the config instead).")
    rsettings = RatingSettings.parse(rating)
    rm = load_config().get("rating_model", {}) or {}
    if not rating and rm.get("mode", "knn") != "knn":          # production's configured rating model
        rsettings = RatingSettings(predictor=f"{rm['mode']}:production", calibration=rm.get("calibration", "evidence"))
    overrides = {}
    if params:
        points = grid_points(params)
        if len(points) != 1:
            raise SystemExit("--params takes one value per setting, e.g. \"delta_pred=0;pred_floor_offset=null\".")
        overrides = points[0]
    if promote and set_ != "test":
        raise SystemExit("--promote compares on the test set; drop --set validation.")
    full_cfg = load_config()
    cfg = full_cfg["eval"]
    workers = workers or cfg["workers"]
    split = load_split(cfg=cfg)
    cases = split.select(set_, users, seed=cfg["seed"])
    buckets = cfg["n_buckets"]
    # Start from the item-kNN predictor; the rating settings (run's and champion's) add any other model.
    art = dataclasses.replace(load_artifacts(with_readers=False), rating_mode="knn", rating_mf=None,
                              rating_calibration="evidence")
    prior = np.asarray(orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"])
    bayes_m = full_cfg["blend"]["bayes_m"]
    _STATE.update(fallback=np.clip(art.meta.bayes, 1, 5).astype(np.float64), pop_sd=population_sd(prior))

    rart = rating_artifacts(art, rsettings, bayes_m)
    recs = shelf_life_models(ranking_artifacts(art, rsettings, rart), prior, ablations=ablations,
                             name=name or "Shelf Life (current config)",
                             params=Params.from_config(**overrides), rating=rsettings, rating_art=rart)
    recs += simple_baselines(art) + rating_baselines(art)
    wanted = set(models.split(",")) if models else None
    if wanted is None or wanted & {"similar_readers_2023", "svd_2023"}:
        from goodrec.eval.legacy2023 import legacy_baselines
        recs += legacy_baselines(art, cfg["subsample"])
    if wanted:
        recs = [r for r in recs if r.key in wanted or r.key in ("shelf_life", "shelf_life_display", "shelf_life_raw")]
    for point in grid_points(grid) if grid else []:
        label = ", ".join(f"{k}={v}" for k, v in point.items())
        recs.append(ShelfLife(key="grid_" + "_".join(f"{k}{v}" for k, v in point.items()), name=f"grid: {label}",
                              kind="ablation", art=ranking_artifacts(art, rsettings, rart), prior=prior,
                              description=f"Grid point: {label}.",
                              params=Params.from_config(**point), display="author_penalty" in point))

    notes = []
    if overrides:
        notes.append("Model settings overridden for this run: " + ", ".join(f"{k}={v}" for k, v in overrides.items()) + ".")
    if not rsettings.is_default:
        notes.append(f"Rating model reconstructed for this run ({rsettings.text()}); see goodrec/eval/rating.py. "
                     "It runs on today's data and artifacts.")
    champ = load_champion()
    champ_res = None
    if champ and (wanted is None or "champion" in wanted):
        same_split = champ.get("split_hash") == split.hash and champ.get("set") == set_ and not users
        runs = EVAL_DIR / champ["runs"] if champ.get("runs") else None
        crs = RatingSettings(**champ.get("rating", {}))
        crart = rating_artifacts(art, crs, bayes_m) if not champ.get("artifacts") else art
        live = dict(key="champion", name=f"Previous best: {champ['name']}", kind="baseline",
                    art=ranking_artifacts(art, crs, crart), prior=prior,
                    params=params_from_json(champ["params"]), rating=crs, rating_art=crart,
                    description=f"Champion from {champ.get('commit', '?')} ({champ.get('report', '')}), "
                                "re-run on the current artifacts.")
        if champ.get("artifacts"):           # a model from another build (a data swap's bridge, D-055)
            from goodrec.eval.models import ForeignShelfLife
            other = load_artifacts(ROOT / champ["artifacts"], with_readers=False)
            oprior = np.asarray(orjson.loads((other.root / "population_stats.json").read_bytes())["rating_dist"])
            recs.append(ForeignShelfLife.build(
                art, other, key="champion", name=f"Previous best: {champ['name']}", prior=oprior,
                params=params_from_json(champ["params"]),
                description=f"{champ['name']} on its own build ({champ['artifacts']}, {champ.get('report', '')}): "
                            "readers' books translated to its catalog by Goodreads work id; it can't recommend or "
                            "rate books outside that catalog."))
            notes.append(f"Previous best ({champ['name']}) runs on its own build, {champ['artifacts']}, "
                         "against this build's readers and hidden books.")
        elif same_split and runs and runs.exists():
            z = np.load(runs)
            champ_res = {"idx": np.arange(len(cases)), "met": z["met"], "tops": z["tops"]}
            if "rmet" in z.files:
                champ_res.update(rmet=z["rmet"], pool=z["pool"])
                notes.append(f"Previous best ({champ['name']}) reused from {champ['runs']} (same split).")
            else:    # saved before the rating track: reuse its rankings, predict its ratings now
                recs.append(ShelfLife(**live, tracks=("rating",)))
                notes.append(f"Previous best ({champ['name']}): rankings reused from {champ['runs']} (same split), "
                             "ratings predicted in this run.")
        else:
            recs.append(ShelfLife(**live, tracks=("ranking", "rating")))
    elif not champ:
        notes.append("No previous best yet (eval/champion.json is created by --promote).")

    # Baselines are cached by everything that can change their results (see cache_path).
    cacheable = [r for r in recs if r.kind == "baseline" and r.key != "champion"]
    cached = {}
    if not fresh:
        for r in cacheable:
            path = cache_path(r, split.hash, set_, users, cfg)
            if path.exists():
                z = np.load(path)
                cached[r.key] = {f: z[f] for f in ("idx", *FIELDS)}
    if cached:
        notes.append("Baseline results reused from the cache (same code, split, settings and artifacts): "
                     + ", ".join(r.name for r in cacheable if r.key in cached) + ".")
    to_run = [r for r in recs if r.key not in cached]
    print(f"  eval: {len(cases):,} {set_} users, split {split.hash}, buckets={buckets}, "
          f"models={[r.key for r in to_run]}, cached={list(cached)}, workers={workers}")
    t = time.time()
    res = evaluate(to_run, cases, buckets, cfg, workers)
    (EVAL_DIR / "runs").mkdir(parents=True, exist_ok=True)
    for r in cacheable:
        if r.key in res:
            np.savez_compressed(cache_path(r, split.hash, set_, users, cfg), idx=res[r.key]["idx"],
                                **{f: res[r.key][f] for f in FIELDS})
    res.update(cached)
    if champ_res is not None:
        if "rmet" not in champ_res:
            champ_res.update(rmet=res["champion"]["rmet"], pool=res["champion"]["pool"])
        recs = [r for r in recs if r.key != "champion"]
        recs.append(Recommender(key="champion", name=f"Previous best: {champ['name']}", kind="baseline",
                                tracks=("ranking", "rating"),
                                description=f"Champion from {champ.get('commit', '?')} ({champ.get('report', '')})."))
        res["champion"] = champ_res
    elapsed = time.time() - t

    summary = summarize(recs, res, cases, buckets, art, cfg, population_sd(prior))
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    model = next(r for r in recs if r.key == "shelf_life")
    sha, dirty = git_info()
    summary.update(
        stamp=stamp, commit=sha, dirty=dirty, set=set_, users_arg=users, n_users=len(cases), split_hash=split.hash,
        split=split.params, runtime_s=elapsed, notes=notes, buckets=buckets,
        model={"key": model.key, "name": model.name, "description": model.description},
        params=params_json(model.params),
        params_text=", ".join(f"{k}={v}" for k, v in params_json(model.params).items()),
        rating=dataclasses.asdict(rsettings), rating_text=rsettings.text(),
        artifacts=str(ARTIFACTS_DIR.relative_to(ROOT)),
        artifacts_built=dt.datetime.fromtimestamp((ARTIFACTS_DIR / "item_meta.npz").stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
    )

    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for r in recs:
        if r.key in res and r.key != "champion":
            np.savez_compressed(runs_dir / f"{stamp}_{set_}_{r.key}.npz", users=np.array([c.user for c in cases]),
                                **{f: per_user(res[r.key], len(cases), f) for f in FIELDS})
    out_dir = EVAL_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "validation" if set_ == "validation" else model.key
    md, js = out_dir / f"{stamp}_{tag}.md", out_dir / f"{stamp}_{tag}.json"
    js.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS))
    md.write_text(report.render(summary))
    print(f"  report: {md.relative_to(ROOT)}")
    for row in summary["rows"]:
        print(f"    {row['name']:<40} NDCG@10 all={row['by_n']['-1']['ndcg@10']:.4f}")
    for row in summary["rating_rows"]:
        print(f"    {row['name']:<40} MAE all={row['by_n']['-1']['mae']:.4f}")

    if promote:
        promote_if_better(summary, champ, stamp, set_, md, runs_dir)


def summarize(recs, res, cases, buckets, art, cfg, pop_sd: float) -> dict:
    order = {"model": 0, "display": 1, "baseline": 2, "ablation": 3}
    recs_sorted = sorted(recs, key=lambda r: (order[r.kind], r.key != "champion"))
    rows = []
    for r in recs_sorted:
        if r.key not in res or "ranking" not in r.tracks:
            continue
        full = per_user(res[r.key], len(cases))
        tops = per_user(res[r.key], len(cases), "tops")
        by_n = {}
        for b, n in enumerate(buckets):
            ok = ~np.isnan(full[:, b, 0])
            cell = {c: float(np.nanmean(full[:, b, j])) for j, c in enumerate(COLS)}
            t20 = tops[ok, b]
            t10 = tops[ok, b, :10]
            cell["coverage"] = len(np.unique(t20[t20 >= 0])) / art.meta.n
            cell["popularity"] = float(art.meta.log_pop[t10[t10 >= 0]].mean()) if (t10 >= 0).any() else None
            # Author variety of each user's top 10: distinct (main) authors, and the most books by one author.
            distinct, most = [], []
            for row_items in t10:
                a = art.meta.author_id[row_items[row_items >= 0]]
                a = a[a >= 0]
                if len(a):
                    _, c = np.unique(a, return_counts=True)
                    distinct.append(len(c))
                    most.append(int(c.max()))
            cell["authors10"] = float(np.mean(distinct)) if distinct else None
            cell["top_author10"] = float(np.mean(most)) if most else None
            by_n[str(n)] = cell
        rows.append({"key": r.key, "name": r.name, "kind": r.kind, "description": r.description,
                     "n_users": int(len(res[r.key]["idx"])), "by_n": by_n})

    model = per_user(res["shelf_life"], len(cases))
    j = COLS.index("ndcg@10")
    h2h = []
    legacy = [r for r in rows if r["key"].endswith("_2023")]
    best_legacy = max(legacy, key=lambda r: r["by_n"]["-1"]["ndcg@10"])["key"] if legacy else None
    for r in rows:
        if r["kind"] != "baseline":
            continue
        base = per_user(res[r["key"]], len(cases))
        name = r["name"] + (" (2023 best)" if r["key"] == best_legacy else "")
        h2h.append({"key": r["key"], "name": name, "by_n": {
            str(n): paired(model[:, b, j], base[:, b, j], cfg["bootstrap"], cfg["seed"]) for b, n in enumerate(buckets)}})

    out = {"rows": rows, "head_to_head": h2h, "best_2023": best_legacy}
    out.update(summarize_rating(recs_sorted, res, cases, buckets, cfg, pop_sd))
    return out


def summarize_rating(recs_sorted, res, cases, buckets, cfg, pop_sd: float) -> dict:
    """Track 2: per-user rating metrics averaged over users, pooled ones from summed counts, by rating style
    (σ_u of the full visible history against the validation-set cut points) and per-user MAE head-to-heads."""
    sigma = np.array([user_sigma(c.visible_r, pop_sd) for c in cases])
    style = rating_style(sigma, cfg["rating_style_cuts"])
    col = {c: i for i, c in enumerate(RATING_COLS)}
    full_b = buckets.index(-1) if -1 in buckets else len(buckets) - 1
    rows = []
    for r in recs_sorted:
        if r.key not in res or "rating" not in r.tracks:
            continue
        rm = per_user(res[r.key], len(cases), "rmet")
        pool = per_user(res[r.key], len(cases), "pool").astype(np.float64)
        ran = ~np.isnan(rm[:, 0, 0])
        by_n = {}
        for b, n in enumerate(buckets):
            x = rm[:, b]
            p = pooled_summary(pool[ran, b].sum(axis=0))
            by_n[str(n)] = {
                "mae": float(np.nanmean(x[:, col["mae"]])), "rmse": float(np.sqrt(np.nanmean(x[:, col["mse"]]))),
                "pearson": p["pearson"], "spearman": float(np.nanmean(x[:, col["spearman"]])),
                "spearman_users": float((~np.isnan(x[ran, col["spearman"]])).mean()),
                "within1": float(np.nanmean(x[:, col["within1"]])), "within05": float(np.nanmean(x[:, col["within05"]])),
                "bias": float(np.nanmean(x[:, col["bias"]])), "nmae": float(np.nanmean(x[:, col["nmae"]])),
                "sd_pred": p["sd_pred"], "sd_actual": p["sd_actual"], "fallback": p["fallback"],
            }
        p = pooled_summary(pool[ran, full_b].sum(axis=0))
        x = rm[:, full_b]
        by_style = []
        for g, label in enumerate(STYLES):
            m = ran & (style == g)
            by_style.append({"style": label, "users": int(m.sum()),
                             **{k: float(np.nanmean(x[m, col[k]])) if m.any() else None
                                for k in ("mae", "nmae", "spearman", "bias")}})
        rows.append({"key": r.key, "name": r.name, "kind": r.kind, "description": r.description,
                     "n_users": int(ran.sum()), "by_n": by_n, "calibration": p["calibration"],
                     "by_star": p["by_star"], "by_style": by_style})

    model = per_user(res["shelf_life"], len(cases), "rmet")[..., col["mae"]]
    legacy = [r for r in rows if r["key"].endswith("_2023")]
    best_legacy = min(legacy, key=lambda r: r["by_n"]["-1"]["mae"])["key"] if legacy else None
    h2h = []
    for r in rows:
        if r["kind"] != "baseline":
            continue
        base = per_user(res[r["key"]], len(cases), "rmet")[..., col["mae"]]
        name = r["name"] + (" (2023 best)" if r["key"] == best_legacy else "")
        h2h.append({"key": r["key"], "name": name, "by_n": {
            str(n): paired(model[:, b], base[:, b], cfg["bootstrap"], cfg["seed"], lower_is_better=True)
            for b, n in enumerate(buckets)}})
    cuts = cfg["rating_style_cuts"]
    styles = [{"style": s, "users": int((style == g).sum())} for g, s in enumerate(STYLES)]
    return {"rating_rows": rows, "rating_head_to_head": h2h, "rating_best_2023": best_legacy,
            "rating_styles": {"cuts": cuts, "pop_sd": pop_sd, "groups": styles}}


def promote_if_better(summary: dict, champ: dict | None, stamp: str, set_: str, md, runs_dir) -> None:
    """Promote when (D-050, D-053), on full history against the champion:
    - NDCG@10 is higher and per-user MAE isn't significantly worse (CI of the MAE difference not entirely > 0); or
    - per-user MAE is significantly better (CI entirely < 0) and NDCG@10 isn't significantly worse (its CI not
      entirely < 0): a rating-model improvement that leaves the ranking at least as good."""
    score = next(r for r in summary["rows"] if r["key"] == "shelf_life")["by_n"]["-1"]["ndcg@10"]
    mae = next(r for r in summary["rating_rows"] if r["key"] == "shelf_life")["by_n"]["-1"]["mae"]
    prev = next((r for r in summary["rows"] if r["key"] == "champion"), None)
    guard = next((h["by_n"]["-1"] for h in summary["rating_head_to_head"] if h["key"] == "champion"), None)
    rank = next((h["by_n"]["-1"] for h in summary["head_to_head"] if h["key"] == "champion"), None)
    ok, why = promotion_decision(score, prev["by_n"]["-1"]["ndcg@10"] if prev else None, guard, rank)
    if not ok:
        print(f"  not promoted: {why}")
        return
    src = runs_dir / f"{stamp}_{set_}_shelf_life.npz"
    keep = runs_dir / f"champion_{stamp}.npz"
    keep.write_bytes(src.read_bytes())
    rec = {"key": "shelf_life", "name": summary["model"]["name"], "params": summary["params"],
           "rating": summary["rating"], "commit": summary["commit"], "split_hash": summary["split_hash"], "set": set_,
           "ndcg@10": score, "mae": mae,
           "report": str(md.relative_to(EVAL_DIR)), "runs": str(keep.relative_to(EVAL_DIR)), "promoted": stamp}
    CHAMPION.write_bytes(orjson.dumps(rec, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    print(f"  promoted to champion (NDCG@10 {score:.4f}" + (f", previous {prev['by_n']['-1']['ndcg@10']:.4f})" if prev else ")"))


def promotion_decision(ndcg: float, prev_ndcg: float | None, mae_vs: dict | None, ndcg_vs: dict | None) -> tuple[bool, str]:
    """(promote?, reason). mae_vs / ndcg_vs: paired() results of the model against the champion (full history)."""
    if prev_ndcg is None:
        return True, "no previous best"
    sig = lambda c: bool(c and c.get("n"))  # noqa: E731
    if ndcg > prev_ndcg:
        if rating_guardrail_ok(mae_vs):
            return True, f"NDCG@10 {ndcg:.4f} > {prev_ndcg:.4f}"
        return False, (f"rating MAE significantly worse than the previous best ({mae_vs['mean_diff']:+.4f}, "
                       f"95% CI [{mae_vs['ci95'][0]:+.4f}, {mae_vs['ci95'][1]:+.4f}])")
    if sig(mae_vs) and mae_vs["ci95"][1] < 0:
        if sig(ndcg_vs) and ndcg_vs["ci95"][1] < 0:
            return False, "MAE significantly better, but NDCG@10 significantly worse"
        return True, f"MAE significantly better ({mae_vs['mean_diff']:+.4f}) and NDCG@10 not significantly worse"
    return False, f"NDCG@10 {ndcg:.4f} <= previous best {prev_ndcg:.4f} and MAE not significantly better"


def rating_guardrail_ok(vs_champion: dict | None) -> bool:
    """False when the model's per-user MAE is significantly worse than the champion's (CI of the difference > 0)."""
    return not (vs_champion and vs_champion.get("n") and vs_champion["ci95"][0] > 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_", choices=["test", "validation"], default="test")
    ap.add_argument("--users", type=int, default=None, help="evaluate a fixed random subsample of this many users")
    ap.add_argument("--models", default=None, help="comma-separated model keys (shelf_life is always included)")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--grid", default=None, metavar="SPEC",
                    help='sweep Params, e.g. "k_a=20,50;a_max=0.5,1.0" (validation only)')
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--promote", action="store_true", help="record this model as the champion if it beats it")
    ap.add_argument("--name", default=None, help="label for the model under test")
    ap.add_argument("--fresh", action="store_true", help="recompute baselines instead of using cached results")
    ap.add_argument("--params", default=None, metavar="SPEC",
                    help='score the model with these settings instead of the config, e.g. "delta_pred=0;a_max=1"')
    ap.add_argument("--rating", default=None, metavar="SPEC",
                    help='reconstruct an earlier rating model, e.g. "item_means=global;calibration=none"')
    main(**vars(ap.parse_args()))
