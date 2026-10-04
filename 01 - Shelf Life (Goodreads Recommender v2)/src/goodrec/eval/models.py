"""Recommenders compared by the evaluation harness, on two tracks (`Recommender.tracks`):
- ranking: a ranked top-k of unread catalog books (`work_idx`) for a user's visible history;
- rating: a predicted star rating for each of the user's held-out books, from the same visible history.

Shelf Life is measured exactly as served: the For you ranking (core.scoring.ranking) with the prediction
floor and boost on, young adult books included and no other content filters. The simple baselines rank
the same universe the For you list draws from (no box sets, children's books, comics or later volumes in
a series) so they compete on equal terms. Shelf Life's rating is the one a book card shows (calibrated
predict_ratings, see goodrec.eval.rating); the rating baselines use the user's visible ratings and training-only
item means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from goodrec.core.artifacts import Artifacts
from goodrec.core.scoring import Filters, Params, UserInput, diversify_authors, raw_scores, ranking
from goodrec.eval.rating import RatingSettings, model_ratings


@dataclass
class Recommender:
    key: str                      # short id, used in file names and the champion record
    name: str
    description: str
    subsample: int | None = None  # evaluate on a fixed subsample of this many users (slow baselines)
    kind: str = "baseline"        # "model" | "display" | "baseline" | "ablation"
    tracks: tuple = ("ranking",)  # "ranking" and/or "rating"

    def recommend(self, user: UserInput, k: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def rate(self, user: UserInput, items: np.ndarray) -> np.ndarray:
        """Predicted 1-5 rating per item; NaN where the method has no estimate (the harness then uses the book
        average and counts a fallback)."""
        raise NotImplementedError

    def run(self, user: UserInput, k: int, rng: np.random.Generator, items: np.ndarray):
        """(top-k or None, ratings for `items` or None), for the tracks this model takes part in."""
        top = self.recommend(user, k, rng) if "ranking" in self.tracks else None
        return top, (self.rate(user, items) if "rating" in self.tracks else None)


def _eligible(art: Artifacts) -> np.ndarray:
    """The For you universe before removing a user's books: no box sets, children's books, comics or
    later series volumes; young adult included."""
    return _static_mask(art, Filters(include_ya=True))


def _static_mask(art: Artifacts, f: Filters) -> np.ndarray:
    from goodrec.core.scoring import filter_mask
    return filter_mask(art, UserInput(ratings={}), f)


def _drop_seen(order: np.ndarray, user: UserInput, k: int) -> np.ndarray:
    seen = user.seen
    head = order[: k + len(seen)]
    if seen:
        head = head[~np.isin(head, np.fromiter(seen, dtype=np.int64))]
    return head[:k]


@dataclass
class ShelfLife(Recommender):
    art: Artifacts = None
    params: Params = None
    prior: np.ndarray | None = None
    kind: str = "model"
    display: bool = False         # score Best match as displayed (author variety), not the model's own ranking
    rating: RatingSettings = field(default_factory=RatingSettings)
    rating_art: Artifacts = field(default=None, repr=False)   # rating.rating_artifacts(art, rating); None = art

    def rate(self, user, items):
        return model_ratings(self.rating_art or self.art, user, items, self.rating, self.prior)

    def recommend(self, user, k, rng):
        raw = raw_scores(self.art, user, self.params)
        rk = ranking(self.art, raw, user, self.params, "match", self.prior, include_ya=True)
        if self.display and self.params.author_penalty:
            return diversify_authors(rk["order"], rk["score"], self.art.meta.author_id, self.params.author_penalty)[:k]
        return rk["order"][:k]


@dataclass
class RandomBaseline(Recommender):
    art: Artifacts = None
    pool: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.pool = np.flatnonzero(_eligible(self.art))

    def recommend(self, user, k, rng):
        pick = rng.choice(self.pool, size=min(len(self.pool), k + len(user.seen)), replace=False)
        return _drop_seen(pick, user, k)


@dataclass
class PopularBaseline(Recommender):
    art: Artifacts = None
    order: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        idx = np.flatnonzero(_eligible(self.art))
        self.order = idx[np.argsort(-self.art.meta.log_pop[idx], kind="stable")]

    def recommend(self, user, k, rng):
        return _drop_seen(self.order, user, k)


@dataclass
class GenrePopularBaseline(Recommender):
    """Score = the user's rating-weighted share of visible books in a book's parent genre x the book's
    popularity relative to the most-read book in that genre. Falls back to popularity for ties."""
    art: Artifacts = None
    pop_in_genre: np.ndarray = field(default=None, repr=False)
    ok: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        m = self.art.meta
        self.ok = _eligible(self.art) & (m.parent_genre >= 0)
        rate = m.reader_rate.astype(np.float64)
        g = np.where(m.parent_genre >= 0, m.parent_genre, 0)
        top = np.zeros(int(g.max()) + 1)
        np.maximum.at(top, g, rate)
        self.pop_in_genre = np.where(m.parent_genre >= 0, rate / np.maximum(top[g], 1e-12), 0.0)
        self._popular = PopularBaseline(key="", name="", description="", art=self.art)

    def recommend(self, user, k, rng):
        m = self.art.meta
        items = np.fromiter(user.ratings, dtype=np.int64)
        stars = np.fromiter(user.ratings.values(), dtype=np.float64)
        g = m.parent_genre[items]
        w = np.bincount(g[g >= 0], weights=stars[g >= 0], minlength=len(m.genre_names))
        if w.sum() == 0:
            return self._popular.recommend(user, k, rng)
        w /= w.sum()
        score = np.where(self.ok, w[np.maximum(m.parent_genre, 0)] * self.pop_in_genre, -1.0)
        score += 1e-9 * m.reader_rate        # popularity breaks ties (e.g. genres with no weight)
        idx = np.argpartition(-score, k + len(user.seen))[: k + len(user.seen)]
        return _drop_seen(idx[np.argsort(-score[idx], kind="stable")], user, k)


def shelf_life_models(art: Artifacts, prior: np.ndarray, ablations: bool = False,
                      name: str = "Shelf Life (current config)", params: Params | None = None,
                      rating: RatingSettings | None = None, rating_art: Artifacts | None = None) -> list[Recommender]:
    p = params or Params.from_config()
    rs = rating or RatingSettings()
    out = [ShelfLife(key="shelf_life", name=name, art=art, params=p, prior=prior, tracks=("ranking", "rating"),
                     rating=rs, rating_art=rating_art,
                     description="For you as served: item-kNN + ALS fold-in blend, fame-gated prediction boost, "
                                 "prediction floor; young adult included, no other content filters. Rating: the "
                                 "predicted rating a book card shows"
                                 + ("." if rs.is_default else f", reconstructed with {rs.text()}.")),
           ]
    if rs.calibration != "none":
        out.append(ShelfLife(key="shelf_life_raw", name="Shelf Life, uncalibrated rating", kind="ablation",
                             tracks=("rating",), art=art, params=p, prior=prior, rating_art=rating_art,
                             rating=RatingSettings(item_means=rs.item_means, calibration="none"),
                             description="Ablation: the raw predicted rating (item mean + the user's offset + "
                                         "item-item residual), before calibration to the user's rating scale."))
    if p.author_penalty:
        out.append(ShelfLife(key="shelf_life_display", name="Best match as displayed (author variety)", kind="display",
                             art=art, params=p, prior=prior, display=True,
                             description=f"The same model with the Recommendations page's display step: repeat authors "
                                         f"nudged down (author_penalty={p.author_penalty}). Diagnostic; not the model's score."))
    if ablations:
        for key, nm, kw in (
            ("item_knn_only", "item-kNN only", dict(a_override=0.0, beta_pop=0.0, gamma_quality=0.0, pred_floor_offset=None)),
            ("als_only", "ALS only", dict(a_override=1.0, beta_pop=0.0, gamma_quality=0.0, pred_floor_offset=None)),
            ("no_floor", "Shelf Life, no prediction floor", dict(pred_floor_offset=None)),
            ("no_boost", "Shelf Life, no prediction boost", dict(delta_pred=0.0)),
        ):
            out.append(ShelfLife(key=key, name=nm, kind="ablation", art=art, prior=prior,
                                 params=Params.from_config(**kw), description=f"Ablation: {nm}."))
    return out


@dataclass
class BookAverage(Recommender):
    """The book's average rating among training readers, shrunk toward its Goodreads average (meta.bayes)."""
    art: Artifacts = None
    tracks: tuple = ("rating",)

    def rate(self, user, items):
        return np.clip(self.art.meta.bayes[items], 1, 5).astype(np.float64)


@dataclass
class GoodreadsAverage(Recommender):
    art: Artifacts = None
    tracks: tuple = ("rating",)

    def rate(self, user, items):
        avg = self.art.meta.avg_rating[items].astype(np.float64)
        return np.where(avg > 0, avg, np.nan)


@dataclass
class UserMean(Recommender):
    tracks: tuple = ("rating",)

    def rate(self, user, items):
        return np.full(len(items), float(np.mean(list(user.ratings.values()))))


@dataclass
class BiasBaseline(Recommender):
    """Book average + the user's offset b_u = sum(r - book average) / (n + 5), as in predict_ratings' baseline."""
    art: Artifacts = None
    user_shrink: float = 5.0
    tracks: tuple = ("rating",)

    def rate(self, user, items):
        m = self.art.meta
        ri = np.fromiter(user.ratings, dtype=np.int64)
        rv = np.fromiter(user.ratings.values(), dtype=np.float64)
        b_u = float((rv - m.bayes[ri]).sum() / (len(ri) + self.user_shrink))
        return np.clip(m.bayes[items] + b_u, 1, 5).astype(np.float64)


def rating_baselines(art: Artifacts) -> list[Recommender]:
    return [
        BiasBaseline(key="bias", name="Book average + user offset", art=art,
                     description="The book's average plus how far above or below book averages the user rates "
                                 "(b_u, shrunk by 5 pseudo-ratings). Knows harsh from generous raters."),
        BookAverage(key="book_avg", name="Book average", art=art,
                    description="The book's average rating among training readers, shrunk toward its Goodreads "
                                "average (the item mean every Shelf Life prediction starts from)."),
        UserMean(key="user_mean", name="User's average rating",
                 description="The average of the user's visible ratings, for every book."),
        GoodreadsAverage(key="goodreads_avg", name="Goodreads average", art=art,
                         description="The book's Goodreads-wide average rating (2017 snapshot over all readers, "
                                     "including ratings after a user's split date; reference only)."),
    ]


def simple_baselines(art: Artifacts) -> list[Recommender]:
    return [
        RandomBaseline(key="random", name="Random", art=art,
                       description="Uniformly random unread books from the For you universe (seeded per user)."),
        PopularBaseline(key="popular", name="Popular",
                        art=art, description="Most-read books among training users."),
        GenrePopularBaseline(key="genre_popular", name="Genre + popularity", art=art,
                             description="Popular books within the parent genres the user rates most "
                                         "(rating-weighted genre share x within-genre popularity)."),
    ]
