"""Sweep matrix-factorization rating predictors (biased MF, SVD++) on the validation readers (D-052).

Each configuration is trained on the training readers (goodrec.pipeline.train_rating_mf), checked every 5
epochs on the validation readers (the best epoch is kept), then its fold-in penalties are tuned. The score is
per-reader MAE averaged over the six history sizes (1, 3, 5, 10, 25, all), so short histories count; the
rating track's own metric code (metrics.rating_user) computes it. The test set is never touched here.

  uv run python -m goodrec.eval.tune_rating                    # full staged sweep (~2-3 h)
  uv run python -m goodrec.eval.tune_rating --users 200 --quick   # smoke run

Writes eval/reports/rating_sweep_<stamp>.md/.json and the best model of each kind to
data/interim/rating_mf/best_<kind>.npz (use with run.py --rating "predictor=mf:best_mf").
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker process

import argparse
import dataclasses
import datetime as dt
import itertools
import multiprocessing as mp
import time

import numpy as np
import orjson

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR, ROOT, load_config
from goodrec.core.rating_mf import MFModel
from goodrec.eval.metrics import RATING_COLS, rating_user
from goodrec.eval.rating import population_sd, user_sigma
from goodrec.eval.split import load_split
from goodrec.pipeline.train_rating_mf import TrainConfig, load_training, train

MODELS_DIR = INTERIM_DIR / "rating_mf"
BUCKETS = [1, 3, 5, 10, 25, -1]
FOLD_GRID = [(lb, lp) for lp in (1.0, 5.0, 20.0, 50.0, 150.0, 500.0) for lb in (0.5, 2.0, 5.0, 15.0)]
_S: dict = {}


def _chunk(idx: np.ndarray) -> np.ndarray:
    model, cases, pop_sd = _S["model"], _S["cases"], _S["pop_sd"]
    out = np.full((len(idx), len(BUCKETS), len(RATING_COLS)), np.nan)
    for r, ci in enumerate(idx):
        c = cases[ci]
        truth = c.hidden_r.astype(np.float64)
        for b, n in enumerate(BUCKETS):
            u = c.user_input(n)
            pred = model.predict(u.ratings, c.hidden, u.read)
            out[r, b] = rating_user(pred, truth, user_sigma(np.fromiter(u.ratings.values(), float), pop_sd))
    return out


def evaluate(model: MFModel, cases, pop_sd: float, workers: int) -> np.ndarray:
    """(readers, buckets, RATING_COLS) per-reader rating metrics for an MF model on these readers."""
    _S.update(model=model, cases=cases, pop_sd=pop_sd)
    chunks = np.array_split(np.arange(len(cases)), max(1, workers * 4))
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_chunk, chunks)
    else:
        parts = [_chunk(c) for c in chunks]
    return np.concatenate(parts)


class Reference:
    """Today's predictors behind the same predict() interface, so the sweep table compares like with like."""

    def __init__(self, art, prior, mode: str):
        self.art, self.prior, self.mode = art, prior, mode

    def predict(self, ratings, items, read=None):
        from goodrec.core.scoring import UserInput, calibration, predict_ratings, shown_predictions
        user = UserInput(ratings=ratings, read=read or set())
        items = np.asarray(items, dtype=np.int64)
        if self.mode == "shown":
            return shown_predictions(self.art, user, items, calibration(self.art, user, self.prior))
        if self.mode == "raw":
            return predict_ratings(self.art, user, items)
        m = self.art.meta                                              # book average + user offset
        ri = np.fromiter(ratings, dtype=np.int64)
        rv = np.fromiter(ratings.values(), dtype=np.float64)
        b_u = float((rv - m.bayes[ri]).sum() / (len(ri) + 5.0)) if len(ri) else 0.0
        return np.clip(m.bayes[items] + b_u, 1, 5)


def score(rmet: np.ndarray) -> float:
    """Mean per-reader MAE, averaged over history sizes."""
    return float(np.nanmean(np.nanmean(rmet[:, :, 0], axis=0)))


def summary(rmet: np.ndarray) -> dict:
    col = {c: i for i, c in enumerate(RATING_COLS)}
    mae = np.nanmean(rmet[:, :, col["mae"]], axis=0)
    return {"score": float(mae.mean()), "mae": {str(n): float(m) for n, m in zip(BUCKETS, mae)},
            "rmse_all": float(np.sqrt(np.nanmean(rmet[:, -1, col["mse"]]))),
            "spearman_all": float(np.nanmean(rmet[:, -1, col["spearman"]]))}


def fit(cfg: TrainConfig, cases, pop_sd, workers, data) -> dict:
    t = time.time()
    model, curve = train(cfg, *data, checkpoint=lambda m, ep: score(evaluate(m, cases, pop_sd, workers)))
    best_ep = min((c for c in curve if not np.isnan(c[2])), key=lambda c: c[2])[0]
    # Fold-in penalties: tuned on the best snapshot without retraining.
    folds = []
    for lb, lp in FOLD_GRID:
        model.lam_b, model.lam_p = lb, lp
        folds.append(((lb, lp), score(evaluate(model, cases, pop_sd, workers))))
    (lb, lp), _ = min(folds, key=lambda f: f[1])
    model.lam_b, model.lam_p = lb, lp
    s = summary(evaluate(model, cases, pop_sd, workers))
    res = {"config": dataclasses.asdict(cfg), "label": cfg.label(), "best_epoch": best_ep, "lam_b": lb, "lam_p": lp,
           "curve": curve, "folds": [[a, b, v] for (a, b), v in folds], "minutes": (time.time() - t) / 60, **s}
    print(f"  {cfg.label()}: epoch {best_ep}, λ_b={lb}, λ_p={lp} -> score {s['score']:.4f} "
          f"(all {s['mae']['-1']:.4f}, n=1 {s['mae']['1']:.4f}) [{res['minutes']:.1f} min]", flush=True)
    return res | {"_model": model}


def refine(best: dict, kind: str) -> list[TrainConfig]:
    """Stage 2: the neighbours of the stage-1 winner."""
    c = TrainConfig(**best["config"])
    ks = sorted({c.k, max(8, c.k // 2), c.k * 2} - {c.k})
    out = [dataclasses.replace(c, lr=lr) for lr in (0.003, 0.01)]
    out += [dataclasses.replace(c, reg_b=rb) for rb in (0.005, 0.02) if rb != c.reg]
    out += [dataclasses.replace(c, reg=round(c.reg * f, 4)) for f in (0.6, 1.5)]
    out += [dataclasses.replace(c, k=k) for k in ks]
    if kind == "svdpp":
        out = [dataclasses.replace(c, lr=0.003), dataclasses.replace(c, reg_y=round(c.reg * 3, 4)),
               dataclasses.replace(c, reg=round(c.reg * 1.5, 4)), dataclasses.replace(c, k=ks[-1])]
    return out


def parse_configs(spec: str) -> list[TrainConfig]:
    """"kind=mf,k=16,reg_b=0.001 | kind=mf,k=4" -> TrainConfigs (unset fields keep their defaults)."""
    out = []
    for part in filter(None, (x.strip() for x in spec.split("|"))):
        kw = {}
        for kv in part.split(","):
            k, _, v = kv.strip().partition("=")
            kw[k] = v if k == "kind" else (int(v) if k in ("k", "epochs", "seed") else float(v))
        out.append(TrainConfig(**kw))
    return out


def extend(report_path, configs: list[TrainConfig], workers: int | None = None) -> None:
    """Stage 3: run extra configurations (e.g. past a grid edge) on the same readers and fold them into an
    existing sweep report, rewriting it and the best_<kind> models if they improve."""
    cfg = load_config()["eval"]
    workers = workers or cfg["workers"]
    rep = orjson.loads(report_path.with_suffix(".json").read_bytes())
    cases = load_split(cfg=cfg).select("validation", rep["n_users"] if rep["n_users"] < 2527 else None, seed=cfg["seed"])
    prior = np.asarray(orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"])
    pop_sd, data = population_sd(prior), load_training()
    best = {k: next(x for x in rep["results"] if x["label"] == v) for k, v in rep["best"].items()}
    for c in configs:
        r = fit(c, cases, pop_sd, workers, data)
        r["stage"] = "3"
        model = r.pop("_model")
        rep["results"].append(r)
        if c.kind not in best or r["score"] < best[c.kind]["score"]:
            best[c.kind] = r
            rep["best"][c.kind] = r["label"]
            model.name = f"best_{c.kind}: {r['label']}"
            model.save(MODELS_DIR / f"best_{c.kind}.npz")
    report_path.with_suffix(".json").write_bytes(orjson.dumps(rep, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))
    report_path.with_suffix(".md").write_text(render(rep))
    print(f"  report updated: {report_path}; best: {rep['best']}")


def combine(report_path, workers: int | None = None) -> None:
    """Stage 4: each kind's best model on its own (mf) and under the item-item residual (hybrid), with each
    calibration mode, as the app would show it (eval harness, rating track only), next to today's item-kNN.
    Adds a table to the sweep report."""
    import dataclasses as dc

    from goodrec.core.artifacts import load_artifacts
    from goodrec.core.scoring import Params
    from goodrec.eval import run
    from goodrec.eval.models import ShelfLife
    from goodrec.eval.rating import RatingSettings, rating_artifacts

    cfg = load_config()
    ev = cfg["eval"]
    workers = workers or ev["workers"]
    rep = orjson.loads(report_path.with_suffix(".json").read_bytes())
    cases = load_split(cfg=ev).select("validation", rep["n_users"] if rep["n_users"] < 2527 else None, seed=ev["seed"])
    prior = np.asarray(orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"])
    art = dc.replace(load_artifacts(with_readers=False), rating_mode="knn", rating_mf=None, rating_calibration="evidence")
    run._STATE.update(fallback=np.clip(art.meta.bayes, 1, 5).astype(np.float64), pop_sd=population_sd(prior))
    recs = []
    predictors = ["knn"] + [f"{mode}:best_{k}" for k in rep["best"] for mode in ("mf", "hybrid")]
    for pred in predictors:
        for cal in ("evidence", "full", "none"):
            s = RatingSettings(calibration=cal, predictor=pred)
            recs.append(ShelfLife(key=f"{pred}|{cal}", name=f"{pred} · calibration {cal}", description="", art=art,
                                  prior=prior, params=Params.from_config(), tracks=("rating",), rating=s,
                                  rating_art=rating_artifacts(art, s, cfg["blend"]["bayes_m"])))
    res = run.evaluate(recs, cases, BUCKETS, ev, workers)
    rows = []
    for r in recs:
        rows.append({"predictor": r.rating.predictor, "calibration": r.rating.calibration,
                     **summary(res[r.key]["rmet"].astype(np.float64))})
        print(f"  {r.name}: score {rows[-1]['score']:.4f} (all {rows[-1]['mae']['-1']:.4f})", flush=True)
    rep["combined"] = rows
    report_path.with_suffix(".json").write_bytes(orjson.dumps(rep, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))
    report_path.with_suffix(".md").write_text(render(rep))
    print(f"  report updated: {report_path}")


def main(users: int | None = None, workers: int | None = None, quick: bool = False, kinds: str = "mf,svdpp") -> None:
    cfg = load_config()["eval"]
    workers = workers or cfg["workers"]
    cases = load_split(cfg=cfg).select("validation", users, seed=cfg["seed"])
    prior = np.asarray(orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"])
    pop_sd = population_sd(prior)
    data = load_training()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  rating sweep: {len(cases):,} validation readers, buckets {BUCKETS}, workers {workers}")
    from goodrec.core.artifacts import load_artifacts
    art = load_artifacts(with_readers=False)
    refs = []
    for mode, label in (("shown", "Champion v5: displayed rating (item-kNN, calibrated)"),
                        ("raw", "Champion v5: uncalibrated item-kNN"), ("bias", "Book average + user offset")):
        refs.append({"label": label, **summary(evaluate(Reference(art, prior, mode), cases, pop_sd, workers))})
        print(f"  reference {label}: score {refs[-1]['score']:.4f} (all {refs[-1]['mae']['-1']:.4f})", flush=True)

    stage1 = {
        "mf": [TrainConfig(kind="mf", k=k, reg=reg) for k, reg in itertools.product((16, 32, 64, 128), (0.02, 0.05, 0.1))],
        "svdpp": [TrainConfig(kind="svdpp", k=k, reg=reg) for k, reg in itertools.product((16, 32, 64), (0.02, 0.05, 0.1))],
    }
    if quick:
        stage1 = {"mf": [TrainConfig(kind="mf", k=16, reg=0.05, epochs=4)],
                  "svdpp": [TrainConfig(kind="svdpp", k=16, reg=0.05, epochs=4)]}
    results, best = [], {}
    for kind in kinds.split(","):
        for stage, configs in (("1", stage1[kind]), ("2", None)):
            if configs is None:
                if quick:
                    continue
                configs = refine(best[kind], kind)
            for c in configs:
                r = fit(c, cases, pop_sd, workers, data)
                r["stage"] = stage
                model = r.pop("_model")
                results.append(r)
                if kind not in best or r["score"] < best[kind]["score"]:
                    best[kind] = r
                    model.name = f"best_{kind}: {r['label']}"
                    model.save(MODELS_DIR / f"best_{kind}.npz")

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    base = orjson.loads((ROOT / "eval" / "champion.json").read_bytes())
    report = {"stamp": stamp, "n_users": len(cases), "set": "validation", "buckets": BUCKETS,
              "fold_grid": FOLD_GRID, "base_report": base.get("report"), "references": refs, "results": results,
              "best": {k: v["label"] for k, v in best.items()}}
    out = ROOT / "eval" / "reports" / f"rating_sweep_{stamp}"
    out.with_suffix(".json").write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))
    out.with_suffix(".md").write_text(render(report))
    print(f"  report: {out.relative_to(ROOT)}.md; best: {report['best']}")


def render(r: dict) -> str:
    lines = [f"# Rating model sweep {r['stamp']}", "",
             f"Matrix-factorization rating predictors trained on the training readers and scored on **{r['n_users']:,} "
             "validation readers** (the test set isn't used). Score: per-reader MAE averaged over the visible-history "
             "sizes 1, 3, 5, 10, 25 and all (lower is better). Each configuration keeps its best epoch (checked every "
             "5) and its best fold-in penalties (λ_b on the reader's offset, λ_p on their taste vector).", ""]
    lines += ["## References (today's predictors, same readers and score)", "",
              "| | score | MAE n=1 | MAE n=5 | MAE all | RMSE all | Spearman all |", "|---|---|---|---|---|---|---|"]
    for x in r.get("references", []):
        lines.append(f"| {x['label']} | {x['score']:.4f} | {x['mae']['1']:.4f} | {x['mae']['5']:.4f} | {x['mae']['-1']:.4f} | "
                     f"{x['rmse_all']:.4f} | {x['spearman_all']:.3f} |")
    lines.append("")
    for kind, title in (("mf", "Biased MF (SVD)"), ("svdpp", "SVD++")):
        rows = [x for x in r["results"] if x["config"]["kind"] == kind]
        if not rows:
            continue
        lines += [f"## {title}", "",
                  "| Stage | Configuration | epoch | λ_b | λ_p | score | MAE n=1 | MAE n=5 | MAE all | RMSE all | Spearman all | min |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in sorted(rows, key=lambda x: x["score"]):
            star = "**" if x["label"] == r["best"].get(kind) else ""
            lines.append(f"| {x['stage']} | {star}{x['label']}{star} | {x['best_epoch']} | {x['lam_b']:g} | {x['lam_p']:g} | "
                         f"{x['score']:.4f} | {x['mae']['1']:.4f} | {x['mae']['5']:.4f} | {x['mae']['-1']:.4f} | "
                         f"{x['rmse_all']:.4f} | {x['spearman_all']:.3f} | {x['minutes']:.1f} |")
        lines.append("")
    if r.get("combined"):
        lines += ["## As displayed: hybrid and calibration", "",
                  "Each kind's best model on its own (mf) or under today's item-item residual correction (hybrid), "
                  "with each calibration mode (evidence = today's evidence-weighted stretch to the reader's own "
                  "rating histogram; full = the whole stretch for every book; none = uncalibrated), scored through "
                  "the evaluation harness exactly as the app would show it. knn is today's predictor.", "",
                  "| Predictor | calibration | score | MAE n=1 | MAE n=5 | MAE all | RMSE all | Spearman all |",
                  "|---|---|---|---|---|---|---|---|"]
        best_c = min(r["combined"], key=lambda x: x["score"])
        for x in sorted(r["combined"], key=lambda x: x["score"]):
            star = "**" if x is best_c else ""
            lines.append(f"| {star}{x['predictor']}{star} | {x['calibration']} | {x['score']:.4f} | {x['mae']['1']:.4f} | "
                         f"{x['mae']['5']:.4f} | {x['mae']['-1']:.4f} | {x['rmse_all']:.4f} | {x['spearman_all']:.3f} |")
        lines.append("")
    lines += [f"Best: " + "; ".join(f"{k}: {v}" for k, v in r["best"].items()) + ".", "",
              f"Tuned against the champion of {r.get('base_report')}."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kinds", default="mf,svdpp")
    ap.add_argument("--extend", default=None, metavar="REPORT", help="add configurations to an existing sweep report")
    ap.add_argument("--configs", default="", help='with --extend: "kind=mf,k=16,reg_b=0.001 | kind=mf,k=4"')
    ap.add_argument("--combine", default=None, metavar="REPORT", help="hybrid / calibration comparison for a sweep")
    a = ap.parse_args()
    if a.combine:
        from pathlib import Path
        combine(Path(a.combine), a.workers)
    elif a.extend:
        from pathlib import Path
        extend(Path(a.extend), parse_configs(a.configs), a.workers)
    else:
        main(a.users, a.workers, a.quick, a.kinds)
