"""Rating prediction for the evaluation's second track: how close is the predicted star rating shown on a book
card to the rating the user went on to give? (DECISIONS D-050, D-051.)

Every prediction uses only what the app would have at the user's split date: their visible ratings (the same
truncated histories as the ranking track) and training-only artifacts (item means and the calibration prior
exclude the held-out users, D-040). Hidden ratings are only ever compared against.

`RatingSettings` reconstructs how earlier versions predicted ratings, for evaluation only:
- item_means "goodreads" (today, D-026): the dataset average shrunk toward the book's Goodreads average;
  "global": shrunk toward the global mean instead, as before D-026 (recomputed here from R_train).
- calibration "evidence" (today, D-027): quantile calibration to the user's own rating histogram, applied in
  proportion to each book's evidence; "full": the whole stretch for every book (the first calibration);
  "none": the raw baseline + item-item residual prediction.
- predictor "knn" (today) or a factorization model from goodrec.eval.tune_rating (D-052): "mf:<name>" or
  "hybrid:<name>" (the item-item residual on top of it), <name> a model in data/interim/rating_mf/
  ("production" = artifacts/rating_mf.npz). A non-knn predictor also drives the ranking's prediction floor
  and boost, as it would in production, so both tracks measure it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR
from goodrec.core.artifacts import Artifacts
from goodrec.core.scoring import UserInput, apply_calibration, calibration, predict_ratings

ITEM_MEANS = ("goodreads", "global")
CALIBRATION = ("evidence", "full", "none")
SIGMA_K = 5.0          # σ_u: the user's rating SD shrunk toward the population SD by this many pseudo-ratings
SIGMA_FLOOR = 0.5


@dataclass(frozen=True)
class RatingSettings:
    item_means: str = "goodreads"
    calibration: str = "evidence"
    predictor: str = "knn"

    def __post_init__(self):
        if self.item_means not in ITEM_MEANS or self.calibration not in CALIBRATION:
            raise ValueError(f"rating settings: item_means in {ITEM_MEANS}, calibration in {CALIBRATION}")
        if self.predictor != "knn" and self.predictor.partition(":")[0] not in ("mf", "hybrid"):
            raise ValueError("rating settings: predictor is knn, mf:<model> or hybrid:<model>")

    @classmethod
    def parse(cls, spec: str | None) -> "RatingSettings":
        """"item_means=global;calibration=full" -> RatingSettings. Unset fields keep today's values."""
        kw = {}
        for part in filter(None, (x.strip() for x in (spec or "").split(";"))):
            k, _, v = part.partition("=")
            if k.strip() not in {f.name for f in dataclasses.fields(cls)}:
                raise SystemExit(f"--rating: unknown setting {k!r} (item_means, calibration, predictor)")
            kw[k.strip()] = v.strip()
        try:
            return cls(**kw)
        except ValueError as e:
            raise SystemExit(f"--rating: {e}") from None

    @property
    def is_default(self) -> bool:
        return self == RatingSettings()

    def text(self) -> str:
        return f"item_means={self.item_means}, calibration={self.calibration}" + (
            f", predictor={self.predictor}" if self.predictor != "knn" else "")


def global_item_means(R: sparse.spmatrix, bayes_m: float) -> np.ndarray:
    """Item means as before D-026: each book's training average shrunk toward the global mean by bayes_m."""
    Rc = sparse.csc_matrix(R)
    n = np.diff(Rc.indptr).astype(np.float64)
    sums = np.asarray(Rc.sum(axis=0)).ravel().astype(np.float64)
    mu = float(Rc.data.mean())
    return ((sums + bayes_m * mu) / (n + bayes_m)).astype(np.float32)


def rating_artifacts(art: Artifacts, s: RatingSettings, bayes_m: float, R: sparse.spmatrix | None = None) -> Artifacts:
    """The artifacts a rating model predicts with: production's, or a shallow copy with other item means and/or
    a factorization predictor."""
    if s.item_means == "global":
        R = R if R is not None else sparse.load_npz(INTERIM_DIR / "R_train.npz")
        art = dataclasses.replace(art, meta=dataclasses.replace(art.meta, bayes=global_item_means(R, bayes_m)))
    if s.predictor != "knn":
        from goodrec.core.rating_mf import MFModel
        mode, _, name = s.predictor.partition(":")
        path = ARTIFACTS_DIR / "rating_mf.npz" if name == "production" else INTERIM_DIR / "rating_mf" / f"{name}.npz"
        art = dataclasses.replace(art, rating_mode=mode, rating_mf=MFModel.load(path), rating_calibration=s.calibration)
    return art


def ranking_artifacts(art: Artifacts, s: RatingSettings, rating_art: Artifacts) -> Artifacts:
    """What the ranking uses: a factorization predictor replaces the knn one everywhere (floor and boost too);
    reconstructed item means / calibration (D-051) only ever applied to the rating track."""
    return rating_art if s.predictor != "knn" else art


def model_ratings(art: Artifacts, user: UserInput, items: np.ndarray, s: RatingSettings, prior: np.ndarray) -> np.ndarray:
    """Shelf Life's predicted ratings under settings `s` (`art` from rating_artifacts). With today's settings this
    is exactly what a book card shows (core.scoring.shown_predictions)."""
    if s.calibration == "none":
        return predict_ratings(art, user, items)
    raw, ev = predict_ratings(art, user, items, return_evidence=True)
    return apply_calibration(raw, calibration(art, user, prior), ev if s.calibration == "evidence" else None)


def population_sd(prior: np.ndarray) -> float:
    """SD of the training ratings' 1-5 star distribution (population_stats rating_dist)."""
    p = np.asarray(prior, dtype=np.float64) / np.sum(prior)
    stars = np.arange(1, 6)
    return float(np.sqrt(p @ (stars - p @ stars) ** 2))


def user_sigma(ratings: np.ndarray, pop_sd: float) -> float:
    """How widely a user spreads their ratings, from their visible ratings only: the sample SD shrunk toward
    the population SD by SIGMA_K pseudo-ratings, floored at SIGMA_FLOOR."""
    r = np.asarray(ratings, dtype=np.float64)
    ss = float(((r - r.mean()) ** 2).sum()) if len(r) else 0.0
    dof = max(len(r) - 1, 0)
    return max(float(np.sqrt((ss + SIGMA_K * pop_sd ** 2) / (dof + SIGMA_K))), SIGMA_FLOOR)


STYLES = ("narrow", "typical", "wide")


def rating_style(sigma: np.ndarray, cuts: list[float]) -> np.ndarray:
    """0 / 1 / 2 = narrow / typical / wide raters, by σ_u against the validation-set tercile cut points."""
    return np.searchsorted(np.asarray(cuts, dtype=np.float64), np.asarray(sigma, dtype=np.float64), side="right")


if __name__ == "__main__":
    # One-off: the rating-style cut points (eval.rating_style_cuts) are the validation users' σ_u terciles,
    # so test-set ratings never define the groups.
    import orjson

    from goodrec.config import ARTIFACTS_DIR, load_config
    from goodrec.eval.split import load_split

    cfg = load_config()["eval"]
    prior = np.asarray(orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"])
    sd = population_sd(prior)
    sig = np.array([user_sigma(c.visible_r, sd) for c in load_split(cfg=cfg).select("validation", None, seed=cfg["seed"])])
    print(f"population SD {sd:.3f}; validation σ_u terciles: {np.round(np.quantile(sig, [1 / 3, 2 / 3]), 3).tolist()}")
