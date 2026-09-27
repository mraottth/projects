"""Offline evaluation on held-out test users (excluded from all model training in s05).

For each test user: hide `holdout_frac` of their ratings (relevant = hidden ratings >= 4), fold in
the visible ratings truncated to n in `eval.n_buckets`, and measure the top-k list produced by the
serving code (`core.scoring`) with content filters off.

  uv run python -m goodrec.eval.run                 # baselines + current config blend
  uv run python -m goodrec.eval.run --grid          # also sweep blend params (k_a, beta, gamma)
  uv run python -m goodrec.eval.run --users 2000    # faster subsample
"""

import argparse
import datetime as dt
import itertools
import time

import numpy as np
import orjson
from scipy import sparse

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR, ROOT, load_config
from goodrec.core.artifacts import load_artifacts
from goodrec.core.scoring import Filters, Params, UserInput, blend, calibration, filter_mask, prediction_floor, \
    predict_ratings, raw_scores


def split_users(R: sparse.csr_matrix, frac: float, seed: int, users: np.ndarray):
    rng = np.random.default_rng(seed)
    out = []
    for u in users:
        s, e = R.indptr[u], R.indptr[u + 1]
        items, ratings = R.indices[s:e], R.data[s:e]
        order = rng.permutation(len(items))
        n_hide = max(1, int(round(frac * len(items))))
        hidden, visible = order[:n_hide], order[n_hide:]
        relevant = set(items[hidden][ratings[hidden] >= 4].tolist())
        if relevant and len(visible):
            out.append(([(int(items[i]), int(ratings[i])) for i in visible], relevant,
                        items[hidden].astype(np.int64), ratings[hidden].astype(np.float32)))
    return out


def metrics(top: np.ndarray, relevant: set[int], k: int) -> tuple[float, float]:
    hits = np.array([i in relevant for i in top[:k]], dtype=float)
    recall = hits.sum() / min(k, len(relevant))
    dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
    idcg = (1 / np.log2(np.arange(2, min(k, len(relevant)) + 2))).sum()
    return recall, dcg / idcg


def _floored(art, user, mask, p: Params, prior) -> np.ndarray:
    """Apply the serving prediction floor (scoring.prediction_floor) to an eval mask."""
    if prior is None or p.pred_floor_offset is None:
        return mask
    cal = calibration(art, user, prior)
    if cal is None:
        return mask
    idx = np.flatnonzero(mask)
    out = np.zeros_like(mask)
    raw_pred, ev = predict_ratings(art, user, idx, return_evidence=True)
    out[idx[prediction_floor(user, raw_pred, cal, p.pred_floor_offset, ev)]] = True
    return out


def evaluate(art, cases, buckets, k, methods: dict[str, Params], log_every=500, prior=None):
    m = art.meta
    res = {name: {n: {"recall": [], "ndcg": [], "items": set(), "pop": []} for n in buckets} for name in methods}
    res.update({b: {n: {"recall": [], "ndcg": [], "items": set(), "pop": []} for n in buckets}
                for b in ("popularity", "top_rated")})
    no_filters = Filters.none()
    base = next(iter(methods.values()))
    t0 = time.time()
    for ci, (visible, relevant, *_) in enumerate(cases):
        for n in buckets:
            vis = visible if n < 0 else visible[:n]
            user = UserInput(ratings=dict(vis))
            mask = filter_mask(art, user, no_filters)
            raw = raw_scores(art, user, base)
            idx = np.flatnonzero(mask)
            ranked = {
                "popularity": idx[np.argsort(-m.log_pop[idx])[:k]],
                "top_rated": idx[np.argsort(-m.bayes[idx])[:k]],
            }
            for name, p in methods.items():
                ranked[name] = blend(art, raw, _floored(art, user, mask, p, prior), p, user)[0][:k]
            for name, top in ranked.items():
                r, nd = metrics(top, relevant, k)
                cell = res[name][n]
                cell["recall"].append(r)
                cell["ndcg"].append(nd)
                cell["items"].update(top.tolist())
                cell["pop"].append(float(m.log_pop[top].mean()) if len(top) else 0.0)
        if (ci + 1) % log_every == 0:
            print(f"  {ci + 1}/{len(cases)} users ({time.time() - t0:.0f}s)")
    summary = {}
    for name, by_n in res.items():
        summary[name] = {str(n): {"recall@k": float(np.mean(c["recall"])), "ndcg@k": float(np.mean(c["ndcg"])),
                                  "coverage": len(c["items"]) / m.n, "mean_log_pop": float(np.mean(c["pop"]))}
                         for n, c in by_n.items()}
    return summary


def rating_metrics(art, cases, buckets) -> dict:
    """RMSE/MAE of predicted star ratings on each user's hidden ratings (all hidden, not just >= 4)."""
    m = art.meta
    out = {}
    for n in buckets:
        errs = {"item mean": [], "Goodreads avg": [], "predicted": []}
        for visible, _, hidden_items, hidden_ratings in cases:
            user = UserInput(ratings=dict(visible if n < 0 else visible[:n]))
            errs["item mean"].append(np.clip(m.bayes[hidden_items], 1, 5) - hidden_ratings)
            errs["Goodreads avg"].append(m.avg_rating[hidden_items] - hidden_ratings)
            errs["predicted"].append(predict_ratings(art, user, hidden_items) - hidden_ratings)
        for name, e in errs.items():
            e = np.concatenate(e)
            out.setdefault(name, {})[str(n)] = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean())}
    return out


def format_table(summary: dict, buckets, metric: str) -> str:
    head = "| method | " + " | ".join("all" if n < 0 else f"n={n}" for n in buckets) + " |"
    sep = "|---|" + "---|" * len(buckets)
    rows = [f"| {name} | " + " | ".join(f"{s[str(n)][metric]:.4f}" for n in buckets) + " |"
            for name, s in summary.items()]
    return "\n".join([head, sep, *rows])


def main(users: int | None = None, grid: bool = False) -> None:
    cfg = load_config()["eval"]
    buckets, k = cfg["n_buckets"], cfg["k"]
    art = load_artifacts(with_readers=False)
    R = sparse.load_npz(INTERIM_DIR / "R_test.npz").tocsr()
    rng = np.random.default_rng(cfg["seed"])
    rows = np.arange(R.shape[0])
    if users and users < len(rows):
        rows = np.sort(rng.choice(rows, users, replace=False))
    cases = split_users(R, cfg["holdout_frac"], cfg["seed"], rows)
    print(f"  eval: {len(cases):,} test users, buckets={buckets}, k={k}")

    p = Params.from_config()
    methods = {
        "blend (config)": p,
        "item-item only": Params.from_config(a_override=0.0, beta_pop=0.0, gamma_quality=0.0, pred_floor_offset=None),
        "ALS only": Params.from_config(a_override=1.0, beta_pop=0.0, gamma_quality=0.0, pred_floor_offset=None),
        "blend, no prediction floor": Params.from_config(pred_floor_offset=None),
        "blend, no prediction boost": Params.from_config(delta_pred=0.0),
        "blend, floor at avg - 0.25": Params.from_config(pred_floor_offset=0.25),
    }
    if grid:
        for k_a, beta, gamma in itertools.product([2, 8, 20], [0.0, 0.1, 0.3], [0.0, 0.1, 0.3]):
            methods[f"blend k_a={k_a} b={beta} g={gamma}"] = Params.from_config(
                k_a=k_a, beta_pop=beta, gamma_quality=gamma)

    t = time.time()
    prior = orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"]
    summary = evaluate(art, cases, buckets, k, methods, prior=prior)
    ratings = rating_metrics(art, cases, buckets)
    elapsed = time.time() - t

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stamp}.json").write_bytes(orjson.dumps(
        {"config": load_config(), "n_users": len(cases), "summary": summary, "ratings": ratings}, option=orjson.OPT_NON_STR_KEYS))
    report = [f"# Eval {stamp}", "",
              f"{len(cases):,} held-out users; hidden {cfg['holdout_frac']:.0%} of ratings (relevant = hidden >= 4 stars); "
              f"top-{k}; content filters off; {elapsed:.0f}s.", "",
              f"## NDCG@{k}", format_table(summary, buckets, "ndcg@k"), "",
              f"## Recall@{k}", format_table(summary, buckets, "recall@k"), "",
              f"## Catalog coverage@{k}", format_table(summary, buckets, "coverage"), "",
              "## Mean log-popularity of recommendations (lower = less popularity bias)",
              format_table(summary, buckets, "mean_log_pop"), "",
              "## Predicted-rating error on hidden ratings (RMSE, stars; lower is better)",
              format_table(ratings, buckets, "rmse")]
    (out_dir / f"{stamp}.md").write_text("\n".join(report) + "\n")
    if grid:
        best = max((name for name in summary if name.startswith("blend k_a")),
                   key=lambda nm: np.mean([summary[nm][str(n)]["ndcg@k"] for n in buckets]))
        print(f"  best grid setting by mean NDCG across buckets: {best}")
    print(format_table({n: s for n, s in summary.items() if not n.startswith("blend k_a")}, buckets, "ndcg@k"))
    print(f"  report: {out_dir / (stamp + '.md')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=None)
    ap.add_argument("--grid", action="store_true")
    main(**vars(ap.parse_args()))
