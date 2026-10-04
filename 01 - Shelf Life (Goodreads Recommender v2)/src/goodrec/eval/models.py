"""Recommenders compared by the evaluation harness. Each returns a ranked top-k of unread catalog books
(`work_idx`) for a user's visible history.

Shelf Life is measured exactly as served: the For you ranking (core.scoring.ranking) with the prediction
floor and boost on, young adult books included and no other content filters. The simple baselines rank
the same universe the For you list draws from (no box sets, children's books, comics or later volumes in
a series) so they compete on equal terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from goodrec.core.artifacts import Artifacts
from goodrec.core.scoring import Filters, Params, UserInput, diversify_authors, raw_scores, ranking


@dataclass
class Recommender:
    key: str                      # short id, used in file names and the champion record
    name: str
    description: str
    subsample: int | None = None  # evaluate on a fixed subsample of this many users (slow baselines)
    kind: str = "baseline"        # "model" | "baseline" | "ablation"

    def recommend(self, user: UserInput, k: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


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
                      name: str = "Shelf Life (current config)", params: Params | None = None) -> list[Recommender]:
    p = params or Params.from_config()
    out = [ShelfLife(key="shelf_life", name=name, art=art, params=p, prior=prior,
                     description="For you as served: item-kNN + ALS fold-in blend, fame-gated prediction boost, "
                                 "prediction floor; young adult included, no other content filters.")]
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
