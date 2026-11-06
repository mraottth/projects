"""s11: train the factorization rating model chosen in config rating_model (D-052) -> artifacts/rating_mf.npz.

Skipped when rating_model.mode is knn. Trains rating_model.train (kind, k, lr, reg, ..., epochs = the best
epoch found by goodrec.eval.tune_rating, same seed, so the sweep's model is reproduced exactly) on the
training readers, and stores the fold-in penalties with it. Runs before s10, which hashes it into the manifest.
"""

import argparse

from goodrec.config import ARTIFACTS_DIR, load_config
from goodrec.pipeline.io import skip_if_done
from goodrec.pipeline.train_rating_mf import TrainConfig, train

OUT = ARTIFACTS_DIR / "rating_mf.npz"


def main(force: bool = False) -> None:
    rm = load_config().get("rating_model", {}) or {}
    if rm.get("mode", "knn") == "knn":
        OUT.unlink(missing_ok=True)
        print("  s11: rating_model.mode is knn; no factorization model")
        return
    if skip_if_done(OUT, force=force):
        return
    t = dict(rm["train"])
    lam_b, lam_p = t.pop("lam_b"), t.pop("lam_p")
    cfg = TrainConfig(**t)
    model, curve = train(cfg)
    model.lam_b, model.lam_p, model.name = lam_b, lam_p, cfg.label()
    model.save(OUT)
    print(f"  s11: {cfg.label()}, {cfg.epochs} epochs (train RMSE {curve[-1][1]:.4f}), "
          f"λ_b={lam_b}, λ_p={lam_p} -> {OUT.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
