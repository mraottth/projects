"""Offline evaluation: per-user temporal hide-and-predict on held-out users (goodrec.eval.split).

For each evaluated user, the model sees their visible history (cut to the n most recent ratings for each
n in eval.n_buckets) and ranks unread books; a hit is a hidden book they rated >= 4. Shelf Life (the For
you ranking as served) is compared with random, popular and genre+popularity baselines, the three 2023
recommenders (goodrec.eval.legacy2023), the previous best model (eval/champion.json) and, with
--ablations, its own components. Writes eval/reports/<stamp>_<model>.md/.json and per-user results to
eval/runs/.

  uv run python -m goodrec.eval.run                          # test set, all baselines
  uv run python -m goodrec.eval.run --set validation --users 300 --models shelf_life,popular   # quick check
  uv run python -m goodrec.eval.run --set validation --grid  # tune blend params (validation only)
  uv run python -m goodrec.eval.run --promote                # make this model the champion if it beats it
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

from goodrec.config import ARTIFACTS_DIR, ROOT, load_config  # noqa: E402
from goodrec.core.artifacts import load_artifacts  # noqa: E402
from goodrec.core.scoring import Params, predict_ratings  # noqa: E402
from goodrec.eval import report  # noqa: E402
from goodrec.eval.metrics import METRICS, at_k, paired  # noqa: E402
from goodrec.eval.models import Recommender, ShelfLife, shelf_life_models, simple_baselines  # noqa: E402
from goodrec.eval.split import load_split  # noqa: E402

EVAL_DIR = ROOT / "eval"
CHAMPION = EVAL_DIR / "champion.json"
KMAX = 20
COLS = [f"{m}@{k}" for k in (10, 20) for m in METRICS]

_STATE: dict = {}      # set before forking the worker pool


def _work(job: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score one model on a chunk of cases (runs in a worker process)."""
    mi, idx = job
    rec: Recommender = _STATE["models"][mi]
    cases, buckets, rel_min, seed = _STATE["cases"], _STATE["buckets"], _STATE["rel_min"], _STATE["seed"]
    art = _STATE.get("art")
    met = np.full((len(idx), len(buckets), len(COLS)), np.nan, np.float32)
    tops = np.full((len(idx), len(buckets), KMAX), -1, np.int32)
    sse = np.zeros((len(idx), len(buckets), 3))          # rating error: predicted, item mean, Goodreads avg
    cnt = np.zeros(len(idx))
    for r, ci in enumerate(idx):
        c = cases[ci]
        rel = c.relevant(rel_min)
        for b, n in enumerate(buckets):
            user = c.user_input(n)
            top = np.asarray(rec.recommend(user, KMAX, np.random.default_rng([seed, c.user, b])), dtype=np.int64)
            tops[r, b, : len(top)] = top
            met[r, b] = [*at_k(top, rel, 10), *at_k(top, rel, 20)]
            if rec.key == "shelf_life":
                pred = predict_ratings(art, user, c.hidden.astype(np.int64))
                truth = c.hidden_r.astype(np.float64)
                sse[r, b] = [((pred - truth) ** 2).sum(), ((np.clip(art.meta.bayes[c.hidden], 1, 5) - truth) ** 2).sum(),
                             ((art.meta.avg_rating[c.hidden] - truth) ** 2).sum()]
        cnt[r] = len(c.hidden)
    return mi, met, tops, sse, cnt


def _subsample(n_cases: int, size: int | None, seed: int) -> np.ndarray:
    if not size or size >= n_cases:
        return np.arange(n_cases)
    return np.sort(np.random.default_rng(seed).choice(n_cases, size, replace=False))


def evaluate(models: list[Recommender], cases, buckets, cfg, workers: int) -> dict:
    """Run every model; returns key -> {"idx", "met", "tops"} (+ "sse"/"cnt" for shelf_life)."""
    _STATE.update(models=models, cases=cases, buckets=buckets, rel_min=cfg["relevant_min_rating"], seed=cfg["seed"])
    jobs, out = [], {}
    for mi, rec in enumerate(models):
        idx = _subsample(len(cases), rec.subsample, cfg["seed"])
        out[rec.key] = {"idx": idx, "met": [], "tops": [], "sse": [], "cnt": [], "t": 0.0}
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
        for f in ("met", "tops", "sse"):
            v[f] = np.concatenate(v[f]) if v[f] else None
        v["cnt"] = np.concatenate(v["cnt"]) if v["cnt"] else None
    return out


def _collect(res, models, out, done, total, t0) -> int:
    mi, met, tops, sse, cnt = res
    o = out[models[mi].key]
    o["met"].append(met)
    o["tops"].append(tops)
    o["sse"].append(sse)
    o["cnt"].append(cnt)
    done += 1
    if done % max(1, total // 20) == 0 or done == total:
        print(f"  {done}/{total} chunks ({time.time() - t0:.0f}s)", flush=True)
    return done


def per_user(res: dict, n_cases: int) -> np.ndarray:
    """(n_cases, buckets, cols) with NaN for users a subsampled model didn't run on."""
    full = np.full((n_cases, *res["met"].shape[1:]), np.nan, np.float32)
    full[res["idx"]] = res["met"]
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


def cache_path(rec: Recommender, split_hash: str, set_: str, users: int | None, cfg: dict) -> Path:
    """Where a baseline's per-user results are cached. Baselines don't change between runs unless their
    code, the split, the evaluation settings or the artifacts do, and all of those are in the key."""
    h = hashlib.sha256(repr((rec.key, rec.subsample, rec.description, split_hash, set_, users, cfg["n_buckets"],
                             cfg["relevant_min_rating"], cfg["seed"], KMAX)).encode())
    h.update((ARTIFACTS_DIR / "manifest.json").read_bytes())
    for f in ("models.py", "legacy2023.py", "metrics.py", "split.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    return EVAL_DIR / "runs" / f"cache_{rec.key}_{h.hexdigest()[:12]}.npz"


def load_champion() -> dict | None:
    return orjson.loads(CHAMPION.read_bytes()) if CHAMPION.exists() else None


def main(set_: str = "test", users: int | None = None, models: str | None = None, ablations: bool = False,
         grid: bool = False, workers: int | None = None, promote: bool = False, name: str | None = None,
         fresh: bool = False) -> None:
    if grid and set_ != "validation":
        raise SystemExit("--grid tunes parameters, so it only runs on --set validation.")
    if promote and set_ != "test":
        raise SystemExit("--promote compares on the test set; drop --set validation.")
    full_cfg = load_config()
    cfg = full_cfg["eval"]
    workers = workers or cfg["workers"]
    split = load_split(cfg=cfg)
    cases = split.select(set_, users, seed=cfg["seed"])
    buckets = cfg["n_buckets"]
    art = load_artifacts(with_readers=False)
    prior = np.asarray(orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"])
    _STATE["art"] = art

    recs = shelf_life_models(art, prior, ablations=ablations, name=name or "Shelf Life (current config)")
    recs += simple_baselines(art)
    wanted = set(models.split(",")) if models else None
    if wanted is None or wanted & {"similar_readers_2023", "svd_2023", "gd_mf_2023"}:
        from goodrec.eval.legacy2023 import legacy_baselines
        recs += legacy_baselines(art, cfg["subsample"])
    if grid:
        for k_a, beta, gamma in itertools.product([2, 8, 20], [0.0, 0.1, 0.3], [0.0, 0.1, 0.3]):
            recs.append(ShelfLife(key=f"grid_ka{k_a}_b{beta}_g{gamma}", name=f"grid k_a={k_a} β={beta} γ={gamma}",
                                  kind="ablation", art=art, prior=prior, description="Blend grid point.",
                                  params=Params.from_config(k_a=k_a, beta_pop=beta, gamma_quality=gamma)))
    if wanted:
        recs = [r for r in recs if r.key in wanted or r.key == "shelf_life"]

    notes = []
    champ = load_champion()
    champ_res = None
    if champ and (wanted is None or "champion" in wanted):
        same_split = champ.get("split_hash") == split.hash and champ.get("set") == set_ and not users
        runs = EVAL_DIR / champ["runs"] if champ.get("runs") else None
        if same_split and runs and runs.exists():
            z = np.load(runs)
            champ_res = {"idx": np.arange(len(cases)), "met": z["met"], "tops": z["tops"]}
            notes.append(f"Previous best ({champ['name']}) reused from {champ['runs']} (same split).")
        else:
            recs.append(ShelfLife(key="champion", name=f"Previous best: {champ['name']}", kind="baseline",
                                  art=art, prior=prior,
                                  params=params_from_json(champ["params"]),
                                  description=f"Champion from {champ.get('commit', '?')} ({champ.get('report', '')}), "
                                              "re-run on the current artifacts."))
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
                cached[r.key] = {"idx": z["idx"], "met": z["met"], "tops": z["tops"]}
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
                                met=res[r.key]["met"], tops=res[r.key]["tops"])
    res.update(cached)
    if champ_res is not None:
        recs.append(Recommender(key="champion", name=f"Previous best: {champ['name']}", kind="baseline",
                                description=f"Champion from {champ.get('commit', '?')} ({champ.get('report', '')})."))
        res["champion"] = champ_res
    elapsed = time.time() - t

    summary = summarize(recs, res, cases, buckets, art, cfg)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    model = next(r for r in recs if r.key == "shelf_life")
    sha, dirty = git_info()
    summary.update(
        stamp=stamp, commit=sha, dirty=dirty, set=set_, users_arg=users, n_users=len(cases), split_hash=split.hash,
        split=split.params, runtime_s=elapsed, notes=notes, buckets=buckets,
        model={"key": model.key, "name": model.name, "description": model.description},
        params=params_json(model.params),
        params_text=", ".join(f"{k}={v}" for k, v in params_json(model.params).items()),
        artifacts=str(ARTIFACTS_DIR.relative_to(ROOT)),
        artifacts_built=dt.datetime.fromtimestamp((ARTIFACTS_DIR / "item_meta.npz").stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
    )

    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for r in recs:
        if r.key in res and r.key != "champion":
            np.savez_compressed(runs_dir / f"{stamp}_{set_}_{r.key}.npz", met=per_user(res[r.key], len(cases)),
                                tops=_full_tops(res[r.key], len(cases)), users=np.array([c.user for c in cases]))
    out_dir = EVAL_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "validation" if set_ == "validation" else model.key
    md, js = out_dir / f"{stamp}_{tag}.md", out_dir / f"{stamp}_{tag}.json"
    js.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS))
    md.write_text(report.render(summary))
    print(f"  report: {md.relative_to(ROOT)}")
    for row in summary["rows"]:
        print(f"    {row['name']:<40} NDCG@10 all={row['by_n']['-1']['ndcg@10']:.4f}")

    if promote:
        promote_if_better(summary, champ, stamp, set_, md, runs_dir)


def _full_tops(res: dict, n_cases: int) -> np.ndarray:
    full = np.full((n_cases, *res["tops"].shape[1:]), -1, np.int32)
    full[res["idx"]] = res["tops"]
    return full


def summarize(recs, res, cases, buckets, art, cfg) -> dict:
    order = {"model": 0, "baseline": 1, "ablation": 2}
    recs_sorted = sorted(recs, key=lambda r: (order[r.kind], r.key != "champion"))
    rows = []
    for r in recs_sorted:
        if r.key not in res:
            continue
        full = per_user(res[r.key], len(cases))
        tops = _full_tops(res[r.key], len(cases))
        by_n = {}
        for b, n in enumerate(buckets):
            ok = ~np.isnan(full[:, b, 0])
            cell = {c: float(np.nanmean(full[:, b, j])) for j, c in enumerate(COLS)}
            t20 = tops[ok, b]
            t10 = tops[ok, b, :10]
            cell["coverage"] = len(np.unique(t20[t20 >= 0])) / art.meta.n
            cell["popularity"] = float(art.meta.log_pop[t10[t10 >= 0]].mean()) if (t10 >= 0).any() else None
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
    sl = res["shelf_life"]
    if sl.get("sse") is not None:
        tot = sl["cnt"].sum()
        out["rating_rmse"] = {nm: {str(n): float(np.sqrt(sl["sse"][:, b, i].sum() / tot)) for b, n in enumerate(buckets)}
                              for i, nm in enumerate(("Shelf Life predicted rating", "item mean", "Goodreads average"))}
    return out


def promote_if_better(summary: dict, champ: dict | None, stamp: str, set_: str, md, runs_dir) -> None:
    score = next(r for r in summary["rows"] if r["key"] == "shelf_life")["by_n"]["-1"]["ndcg@10"]
    prev = next((r for r in summary["rows"] if r["key"] == "champion"), None)
    if prev and score <= prev["by_n"]["-1"]["ndcg@10"]:
        print(f"  not promoted: NDCG@10 {score:.4f} <= previous best {prev['by_n']['-1']['ndcg@10']:.4f}")
        return
    src = runs_dir / f"{stamp}_{set_}_shelf_life.npz"
    keep = runs_dir / f"champion_{stamp}.npz"
    keep.write_bytes(src.read_bytes())
    rec = {"key": "shelf_life", "name": summary["model"]["name"], "params": summary["params"],
           "commit": summary["commit"], "split_hash": summary["split_hash"], "set": set_, "ndcg@10": score,
           "report": str(md.relative_to(EVAL_DIR)), "runs": str(keep.relative_to(EVAL_DIR)), "promoted": stamp}
    CHAMPION.write_bytes(orjson.dumps(rec, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    print(f"  promoted to champion (NDCG@10 {score:.4f}" + (f", previous {prev['by_n']['-1']['ndcg@10']:.4f})" if prev else ")"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_", choices=["test", "validation"], default="test")
    ap.add_argument("--users", type=int, default=None, help="evaluate a fixed random subsample of this many users")
    ap.add_argument("--models", default=None, help="comma-separated model keys (shelf_life is always included)")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--grid", action="store_true", help="also sweep blend params (validation only)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--promote", action="store_true", help="record this model as the champion if it beats it")
    ap.add_argument("--name", default=None, help="label for the model under test")
    ap.add_argument("--fresh", action="store_true", help="recompute baselines instead of using cached results")
    main(**vars(ap.parse_args()))
