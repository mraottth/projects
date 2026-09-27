"""Sweep ALS hyperparameters: retrain into a scratch artifacts dir and evaluate ALS-only + blend.

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
from scipy import sparse

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR, ROOT, load_config
from goodrec.core.artifacts import load_artifacts
from goodrec.core.scoring import Params
from goodrec.eval.run import evaluate, split_users
from goodrec.pipeline import s08_als

SHARED = ["catalog.db", "item_meta.npz", "item_meta_names.json", "item_nbrs_idx.npy", "item_nbrs_sim.npy"]


def main(users: int = 800) -> None:
    cfg = load_config()
    ev = cfg["eval"]
    R = sparse.load_npz(INTERIM_DIR / "R_test.npz").tocsr()
    rng = np.random.default_rng(ev["seed"])
    rows = np.sort(rng.choice(R.shape[0], min(users, R.shape[0]), replace=False))
    cases = split_users(R, ev["holdout_frac"], ev["seed"], rows)
    buckets = [3, 10, -1]

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
            methods = {"als": Params.from_config(a_override=1.0, beta_pop=0.0, gamma_quality=0.0,
                                                 alpha=alpha, regularization=reg),
                       "blend": Params.from_config(alpha=alpha, regularization=reg)}
            s = evaluate(art, cases, buckets, ev["k"], methods, log_every=10**9)
            vals = [s[m][str(n)]["ndcg@k"] for m in ("als", "blend") for n in buckets]
            results.append(((factors, reg, alpha), vals))
            lines.append(f"| {factors} | {reg} | {alpha} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
            print(lines[-1], flush=True)

    best = max(results, key=lambda r: np.mean(r[1][3:]))
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    out = ROOT / "eval" / "reports" / f"als_sweep_{stamp}.md"
    out.write_text(f"# ALS sweep {stamp}\n\n{len(cases)} held-out users; NDCG@{ev['k']}.\n\n" + "\n".join(lines)
                   + f"\n\nBest by mean blend NDCG: factors={best[0][0]} reg={best[0][1]} alpha={best[0][2]}\n")
    print(f"best: {best[0]}  report: {out}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=800)
    main(**vars(ap.parse_args()))
