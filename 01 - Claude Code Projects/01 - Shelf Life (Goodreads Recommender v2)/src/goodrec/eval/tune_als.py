"""Sweep ALS hyperparameters: retrain into a scratch artifacts dir and evaluate ALS-only + blend.

Tuning uses the validation users of the temporal split (goodrec.eval.split) only; the test set is kept
for reported results.

  uv run python -m goodrec.eval.tune_als --users 800
Writes eval/reports/als_sweep_<stamp>.md. Apply the winner to config/pipeline.yaml (als:) and
re-run `make artifacts` (s08 onward).
"""

import argparse
import datetime as dt
import itertools
import shutil
import tempfile
from pathlib import Path

import numpy as np
import orjson

from goodrec.config import ARTIFACTS_DIR, EVAL_DIR, ROOT, load_config
from goodrec.core.artifacts import load_artifacts
from goodrec.core.scoring import Params
from goodrec.eval.models import ShelfLife
from goodrec.eval.run import COLS, evaluate
from goodrec.eval.split import load_split
from goodrec.pipeline import s08_als

SHARED = ["catalog.db", "item_meta.npz", "item_meta_names.json", "item_nbrs_idx.npy", "item_nbrs_sim.npy",
          "population_stats.json"]


def main(users: int = 800, workers: int | None = None) -> None:
    cfg = load_config()
    ev = cfg["eval"]
    cases = load_split(cfg=ev).select("validation", users, seed=ev["seed"])
    buckets = [3, 10, -1]
    prior = np.asarray(orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"])
    j = COLS.index("ndcg@10")

    grid = list(itertools.product([64, 128], [0.05, 1.0], [3.0, 10.0, 30.0]))
    lines = ["| factors | reg | alpha | ALS n=3 | ALS n=10 | ALS all | blend n=3 | blend n=10 | blend all |",
             "|---|---|---|---|---|---|---|---|---|"]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in SHARED:
            (tmp / f).symlink_to(ARTIFACTS_DIR / f)
        for factors, reg, alpha in grid:
            for f in ("item_factors.npy", "yty.npy", "user_factors.npy", "user_rows.npy"):
                (tmp / f).unlink(missing_ok=True)
            s08_als.main(force=True, out_dir=tmp, factors=factors, regularization=reg, alpha=alpha)
            art = load_artifacts(tmp, with_readers=False)
            models = [
                ShelfLife(key="als", name="ALS only", description="", art=art, prior=prior,
                          params=Params.from_config(a_override=1.0, beta_pop=0.0, gamma_quality=0.0,
                                                    pred_floor_offset=None, alpha=alpha, regularization=reg)),
                ShelfLife(key="blend", name="blend", description="", art=art, prior=prior,
                          params=Params.from_config(alpha=alpha, regularization=reg)),
            ]
            res = evaluate(models, cases, buckets, ev, workers or ev["workers"])
            vals = [float(np.nanmean(res[m]["met"][:, b, j])) for m in ("als", "blend") for b in range(len(buckets))]
            results.append(((factors, reg, alpha), vals))
            lines.append(f"| {factors} | {reg} | {alpha} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
            print(lines[-1], flush=True)

    best = max(results, key=lambda r: np.mean(r[1][3:]))
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    out = EVAL_DIR / "reports" / f"als_sweep_{stamp}.md"
    out.write_text(f"# ALS sweep {stamp}\n\n{len(cases)} validation users (temporal split); NDCG@10.\n\n"
                   + "\n".join(lines)
                   + f"\n\nBest by mean blend NDCG: factors={best[0][0]} reg={best[0][1]} alpha={best[0][2]}\n")
    print(f"best: {best[0]}  report: {out}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=800)
    ap.add_argument("--workers", type=int, default=None)
    main(**vars(ap.parse_args()))
