"""Recommendation scoring: ALS fold-in + item-item neighbors, blended by rating count.

All functions are numpy-only and operate on dense catalog indices (work_idx).
    final = a(n)*z(s_als) + (1-a(n))*z(s_ii) + beta*z(log_pop) + gamma*z(bayes),  a(n) = n/(n+k_a)
z-scores are computed within the candidate set (top-C of each model after filtering).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from goodrec.config import load_config
from goodrec.core.artifacts import Artifacts
from goodrec.core.textnorm import ascii_fold


@dataclass
class UserInput:
    ratings: dict[int, int]                       # work_idx -> 1..5
    read: set[int] = field(default_factory=set)   # read but unrated
    dismissed: set[int] = field(default_factory=set)
    recency: dict[int, float] | None = None       # work_idx -> 0..1 weight of each rating (None: all 1)

    @property
    def seen(self) -> set[int]:
        return set(self.ratings) | self.read | self.dismissed


@dataclass
class Filters:
    genres: list[int] = field(default_factory=list)          # parent genre ids (any-of)
    tags: list[int] = field(default_factory=list)            # subgenre tag ids (all-of: each click narrows)
    authors_include: list[int] = field(default_factory=list)
    authors_exclude: list[int] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    min_avg_rating: float | None = None
    min_ratings_count: int | None = None
    max_ratings_count: int | None = None
    text: str = ""
    include_children: bool = False
    include_comics: bool = False
    include_ya: bool = False
    include_boxsets: bool = False
    include_series_continuations: bool = False

    @classmethod
    def none(cls) -> "Filters":
        """No content filters at all (used by eval)."""
        return cls(include_children=True, include_comics=True, include_boxsets=True,
                   include_series_continuations=True, include_ya=True)


@dataclass
class Params:
    k_a: float
    beta_pop: float                   # popularity weight for a brand-new user
    gamma_quality: float
    candidates: int
    alpha: float
    regularization: float
    confidence: dict[int, float]
    ii_prior_mean: float = 3.0
    ii_prior_weight: float = 5.0
    beta_pop_many: float = 0.0        # popularity weight for a user with many ratings (interpolated by a(n))
    a_override: float | None = None   # force the ALS weight (eval: 0 = item-item only, 1 = ALS only)
    a_max: float = 1.0                # cap on the ALS weight a(n), however many ratings
    recency_half_life: float | None = None  # eval: weight the k-th most recent rating by 0.5 ** (k / half-life)
    pred_floor_offset: float | None = None  # For you: drop books predicted below (user average - offset)
    delta_pred: float = 0.0           # Best match: weight on z(predicted rating), scaled by a(n) and fame (config: 0.75)
    pred_fame_lo: float = 4.0         # log10 Goodreads ratings count where fame weight starts (10k)...
    pred_fame_hi: float = 6.0         # ...and where it reaches 1 (1M)
    pred_pool: int = 50               # famous high-prediction books added to the candidate pool

    @classmethod
    def from_config(cls, **overrides) -> "Params":
        cfg = load_config()
        b, a = cfg["blend"], cfg["als"]
        p = dict(k_a=b["k_a"], beta_pop=b["beta_pop"], beta_pop_many=b.get("beta_pop_many", b["beta_pop"]), gamma_quality=b["gamma_quality"],
                 pred_floor_offset=b.get("pred_floor_offset"), delta_pred=b.get("delta_pred", 0.0),
                 pred_fame_lo=b.get("pred_fame_lo", 4.0), pred_fame_hi=b.get("pred_fame_hi", 6.0),
                 pred_pool=b.get("pred_pool", 50), a_max=b.get("a_max", 1.0),
                 candidates=b["candidates"], alpha=a["alpha"], regularization=a["regularization"],
                 confidence={int(k): float(v) for k, v in a["confidence"].items()})
        p.update(overrides)
        return cls(**p)


@dataclass
class RawScores:
    """Per-user model outputs, independent of filters (cached by the API per ratings hash)."""
    n: int
    u: np.ndarray | None       # ALS user vector
    s_als: np.ndarray          # (N,)
    s_ii: np.ndarray           # (N,)
    weights: dict[int, float]  # item-item weight per rated item
    cache: dict = field(default_factory=dict)   # per-user derived rankings (see recommend())


def recency_weights(oldest_first: list[int], half_life: float) -> dict[int, float]:
    """0.5 ** (k / half_life) for the k-th most recent item (the last one in the list has weight 1)."""
    n = len(oldest_first)
    return {i: 0.5 ** ((n - 1 - k) / half_life) for k, i in enumerate(oldest_first)}


def fold_in(art: Artifacts, user: UserInput, p: Params) -> np.ndarray | None:
    """Solve the ALS user vector for a new user given fixed item factors (implicit's recalculate_user).
    With `user.recency`, each rating's confidence boost is scaled by its recency weight."""
    items, conf = [], []
    rec = user.recency or {}
    for i, r in user.ratings.items():
        g = p.confidence.get(int(r), 0.0) * rec.get(i, 1.0)
        if g > 0:
            items.append(i)
            conf.append(1 + p.alpha * g)
    g_read = p.confidence.get(0, 0.0)
    if g_read > 0:
        for i in user.read - set(user.ratings):
            items.append(i)
            conf.append(1 + p.alpha * g_read)
    if not items:
        return None
    Yi = art.Y[np.asarray(items)]
    c = np.asarray(conf, dtype=np.float32)
    A = art.YtY + (Yi.T * (c - 1)) @ Yi + p.regularization * np.eye(art.Y.shape[1], dtype=np.float32)
    b = (Yi.T * c).sum(axis=1)
    return np.linalg.solve(A, b).astype(np.float32)


def item_item_weights(user: UserInput, p: Params) -> dict[int, float]:
    """w_i = r_i - b_u, with b_u a user mean shrunk toward the prior (so one 5-star rating counts), scaled by
    each rating's recency weight when `user.recency` is set."""
    if not user.ratings:
        return {}
    r = np.asarray(list(user.ratings.values()), dtype=np.float32)
    b_u = (r.sum() + p.ii_prior_mean * p.ii_prior_weight) / (len(r) + p.ii_prior_weight)
    rec = user.recency or {}
    return {i: float(v - b_u) * rec.get(i, 1.0) for i, v in user.ratings.items()}


def item_item_scores(art: Artifacts, weights: dict[int, float]) -> np.ndarray:
    s = np.zeros(art.meta.n, dtype=np.float32)
    if not weights:
        return s
    items = np.fromiter(weights, dtype=np.int64)
    w = np.fromiter(weights.values(), dtype=np.float32)
    nb, sim = art.nbr_idx[items], art.nbr_sim[items]
    valid = nb >= 0
    np.add.at(s, nb[valid], (w[:, None] * sim)[valid])
    return s


def raw_scores(art: Artifacts, user: UserInput, p: Params) -> RawScores:
    u = fold_in(art, user, p)
    s_als = art.Y @ u if u is not None else np.zeros(art.meta.n, dtype=np.float32)
    weights = item_item_weights(user, p)
    return RawScores(n=len(user.ratings), u=u, s_als=s_als, s_ii=item_item_scores(art, weights), weights=weights)


def next_in_series(art: Artifacts, user: UserInput) -> set[int]:
    """For each series the user has rated/read, the lowest unread volume after their furthest one."""
    m = art.meta
    engaged = [i for i in (set(user.ratings) | user.read) if m.series_id[i] >= 0]
    if not engaged:
        return set()
    out = set()
    seen = user.seen
    furthest: dict[int, float] = {}
    for i in engaged:
        sid, pos = int(m.series_id[i]), m.series_pos[i]
        if not np.isnan(pos):
            furthest[sid] = max(furthest.get(sid, -1.0), float(pos))
    for sid, pos in furthest.items():
        members = np.flatnonzero(m.series_id == sid)
        cand = [j for j in members if j not in seen and m.series_pos[j] > pos and not m.is_boxset[j]]
        if cand:
            out.add(min(cand, key=lambda j: m.series_pos[j]))
    return out


def _asks_for_ya(m, f: "Filters") -> bool:
    """Explicitly filtering to the Young Adult genre implies including YA books."""
    return "Young Adult" in m.genre_names and m.genre_names.index("Young Adult") in f.genres


def filter_mask(art: Artifacts, user: UserInput, f: Filters, allow: set[int] = frozenset()) -> np.ndarray:
    """Boolean mask of recommendable items. `allow` bypasses the series-continuation rule."""
    m = art.meta
    mask = np.ones(m.n, dtype=bool)
    if not f.include_boxsets:
        mask &= ~m.is_boxset
    if not f.include_children:
        mask &= ~m.is_children
    if not f.include_comics:
        mask &= ~m.is_comic
    if not f.include_ya and not _asks_for_ya(m, f):
        mask &= ~m.is_ya
    if not f.include_series_continuations:
        cont = (m.series_id >= 0) & (m.series_pos > 1)  # nan compares False -> kept
        if allow:
            cont[np.fromiter(allow, dtype=np.int64)] = False
        mask &= ~cont
    if f.genres:
        mask &= np.isin(m.parent_genre, np.asarray(f.genres))
    if f.authors_include:
        mask &= np.isin(m.author_id, np.asarray(f.authors_include))
    if f.authors_exclude:
        mask &= ~np.isin(m.author_id, np.asarray(f.authors_exclude))
    if f.year_min is not None:
        mask &= m.year >= f.year_min
    if f.year_max is not None:
        mask &= (m.year <= f.year_max) & (m.year > 0)
    if f.min_avg_rating is not None:
        mask &= m.avg_rating >= f.min_avg_rating
    if f.min_ratings_count is not None:
        mask &= m.ratings_count >= f.min_ratings_count
    if f.max_ratings_count is not None:
        mask &= m.ratings_count <= f.max_ratings_count
    for t in f.tags:
        has = np.zeros(m.n, dtype=bool)
        has[m.tag_owner[m.tag_ids == t]] = True
        mask &= has
    if f.text.strip():
        terms = ascii_fold(f.text).lower().split()
        idx = np.flatnonzero(mask)
        keep = [i for i in idx if all(t in m.search_text[i] for t in terms)]
        mask[:] = False
        mask[keep] = True
    seen = user.seen
    if seen:
        mask[np.fromiter(seen, dtype=np.int64)] = False
    return mask


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def _top(scores: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    if len(idx) <= k:
        return idx
    part = np.argpartition(-scores[idx], k)[:k]
    return idx[part]


def _pool(art: Artifacts, raw: RawScores, mask: np.ndarray, p: Params):
    """Candidate pool and blend weights: union of each model's top-C within `mask`."""
    m = art.meta
    idx = np.flatnonzero(mask)
    empty = np.zeros(0, np.int64)
    if raw.n == 0:
        return _top(m.log_pop + m.bayes, idx, p.candidates), empty, empty, 0.0, 0.0
    top_als = _top(raw.s_als, idx, p.candidates) if raw.u is not None else empty
    ii_pos = idx[raw.s_ii[idx] > 0]
    top_ii = _top(raw.s_ii, ii_pos, p.candidates)
    cand = np.union1d(top_als, top_ii)
    if len(cand) == 0:
        cand = _top(m.log_pop, idx, p.candidates)
    a = min(raw.n / (raw.n + p.k_a), p.a_max) if raw.u is not None else 0.0
    if p.a_override is not None:
        a = p.a_override
        if a >= 1.0 and raw.u is not None:
            cand = top_als
        elif a <= 0.0 and len(top_ii):
            cand = top_ii
    # Popularity helps cold-start users but hurts heavy readers (eval): interpolate with a(n).
    beta = p.beta_pop if p.a_override is not None else (1 - a) * p.beta_pop + a * p.beta_pop_many
    return cand, top_ii, top_als, a, beta


def _z_with(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """z-score x using the mean/std of the reference values (the candidate pool)."""
    sd = ref.std()
    return (x - ref.mean()) / sd if sd > 1e-9 else np.zeros_like(x, dtype=np.float64)


def fame_weight(art: Artifacts, p: Params) -> np.ndarray:
    """0..1 by Goodreads ratings count (log scale): how well known a book is. Gates the prediction boost so
    obscure books with high predictions (driven mostly by their average) don't jump up the ranking."""
    lc = np.log10(np.maximum(art.meta.ratings_count, 1).astype(np.float64))
    return np.clip((lc - p.pred_fame_lo) / (p.pred_fame_hi - p.pred_fame_lo), 0.0, 1.0)


def add_pred_candidates(art: Artifacts, cand: np.ndarray, mask: np.ndarray, preds: np.ndarray, p: Params) -> np.ndarray:
    """Add the best fame-weighted predictions to the pool, so well-known books the user would likely rate
    highly can be ranked even when neither model put them in its top-C."""
    if not p.delta_pred or not p.pred_pool:
        return cand
    idx = np.flatnonzero(mask & ~np.isnan(preds))
    if not len(idx):
        return cand
    return np.union1d(cand, _top(preds * fame_weight(art, p), idx, p.pred_pool))


def score_all(art: Artifacts, raw: RawScores, cand: np.ndarray, a: float, beta: float, p: Params,
              preds: np.ndarray | None = None) -> np.ndarray:
    """Blend score for every book, normalized with the candidate pool's statistics.

    Within the pool this orders books exactly as blend() does; outside it the same formula extends
    the ranking to the rest of the catalog (used to give every book a stable rank). With `preds`
    (raw predicted ratings, NaN where unknown) and p.delta_pred, adds a(n) * delta * fame * z(pred).
    """
    m = art.meta
    if raw.n == 0:
        return _z_with(m.log_pop, m.log_pop[cand]) + _z_with(m.bayes, m.bayes[cand])
    s = (a * _z_with(raw.s_als, raw.s_als[cand]) + (1 - a) * _z_with(raw.s_ii, raw.s_ii[cand])
         + beta * _z_with(m.log_pop, m.log_pop[cand]) + p.gamma_quality * _z_with(m.bayes, m.bayes[cand]))
    if p.delta_pred and preds is not None:
        zp = np.nan_to_num(_z_with(preds, preds[cand]), nan=0.0)
        s = s + a * p.delta_pred * fame_weight(art, p) * zp
    return s


def blend(art: Artifacts, raw: RawScores, mask: np.ndarray, p: Params,
          user: UserInput | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (items sorted best-first, their scores, source codes: 1=ii, 2=als, 3=both).
    Pass `user` to apply the predicted-rating boost (p.delta_pred), as ranking() does."""
    if not mask.any():
        return np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.int8)
    cand, top_ii, top_als, a, beta = _pool(art, raw, mask, p)
    preds = None
    if p.delta_pred and user is not None and raw.n:
        # Score the fame-weighted pool additions (the most popular books) plus the model pool.
        famous = np.flatnonzero(mask & (fame_weight(art, p) > 0))
        scope = np.union1d(cand, famous)
        preds = np.full(art.meta.n, np.nan, np.float32)
        preds[scope] = predict_ratings(art, user, scope)
        cand = add_pred_candidates(art, cand, mask, preds, p)
    s = score_all(art, raw, cand, a, beta, p, preds)[cand]
    source = np.isin(cand, top_ii).astype(np.int8) + 2 * np.isin(cand, top_als).astype(np.int8)
    order = np.argsort(-s)
    return cand[order], s[order].astype(np.float32), source[order]


def explain(art: Artifacts, raw: RawScores, items: np.ndarray, max_because: int = 2) -> list[list[int]]:
    """'Because you liked X': rated items with the largest positive w_i * sim(i, j) for each j."""
    liked = {i: w for i, w in raw.weights.items() if w > 0}
    out: list[list[int]] = [[] for _ in items]
    if not liked or len(items) == 0:
        return out
    li = np.fromiter(liked, dtype=np.int64)
    w = np.fromiter(liked.values(), dtype=np.float32)
    nb, sim = art.nbr_idx[li], art.nbr_sim[li] * w[:, None]
    pos = {int(j): k for k, j in enumerate(items)}
    contrib: dict[int, list[tuple[float, int]]] = {}
    rows, cols = np.nonzero(np.isin(nb, items))
    for r, c in zip(rows, cols):
        contrib.setdefault(int(nb[r, c]), []).append((float(sim[r, c]), int(li[r])))
    Yl = art.Y[li]
    Yl_n = Yl / np.maximum(np.linalg.norm(Yl, axis=1, keepdims=True), 1e-8)
    for j, k in pos.items():
        # Prefer liked books from j's own neighbor list ("the books you liked that are most like j");
        # popular books sit in many lists, so the reverse direction alone gives odd pairings.
        own = [(float(s) * liked[int(i)], int(i)) for i, s in zip(art.nbr_idx[j], art.nbr_sim[j])
               if i >= 0 and int(i) in liked]
        if own:
            out[k] = [i for _, i in sorted(own, reverse=True)[:max_because]]
        elif j in contrib:
            out[k] = [i for _, i in sorted(contrib[j], reverse=True)[:max_because]]
        else:  # ALS-only recommendation: nearest liked book in embedding space
            yj = art.Y[j] / max(np.linalg.norm(art.Y[j]), 1e-8)
            out[k] = [int(li[np.argmax(Yl_n @ yj)])]
    return out


def predict_ratings(art: Artifacts, user: UserInput, items, shrink: float = 0.5,
                    user_shrink: float = 5.0, return_evidence: bool = False):
    """Predicted star rating (1-5) for each item: baseline + item-item residual.

        baseline(u, j) = item_mean(j) + b_u,  b_u = sum(r_i - item_mean(i)) / (n + user_shrink)
        pred(u, j) = baseline(u, j) + sum_i sim(i,j) * (r_i - baseline(u, i)) / (sum_i |sim(i,j)| + shrink)
    over rated books i linked to j in either direction of the neighbor lists (the larger similarity if
    both). item_mean is the Bayesian-shrunk mean from the training data; `shrink` pulls sparse-evidence
    residuals to 0. Vectorized as a sparse (items x rated) similarity matrix, so it scales to the catalog.

    With return_evidence, also returns each item's evidence weight den / (den + CALIB_EVIDENCE_K) in [0, 1):
    how much its prediction rests on the user's own ratings of similar books (den = summed similarity to
    them). apply_calibration stretches each prediction in proportion to it.
    """
    m = art.meta
    items = np.asarray(items, dtype=np.int64)
    if len(items) == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32)) if return_evidence else np.zeros(0, np.float32)
    if not user.ratings:
        base = np.clip(m.bayes[items], 1, 5).astype(np.float32)
        return (base, np.zeros(len(items), np.float32)) if return_evidence else base
    uniq, inv = np.unique(items, return_inverse=True)
    ri = np.fromiter(user.ratings, dtype=np.int64)
    rv = np.fromiter(user.ratings.values(), dtype=np.float64)
    b_u = float((rv - m.bayes[ri]).sum() / (len(ri) + user_shrink))
    resid = rv - (m.bayes[ri] + b_u)

    col_of = np.full(m.n, -1, np.int64)
    col_of[ri] = np.arange(len(ri))
    row_of = np.full(m.n, -1, np.int64)
    row_of[uniq] = np.arange(len(uniq))
    shape = (len(uniq), len(ri))

    nb, sim = art.nbr_idx[uniq], art.nbr_sim[uniq]                # j's own neighbors that are rated
    cols = np.where(nb >= 0, col_of[np.maximum(nb, 0)], -1)
    r1, k1 = np.nonzero(cols >= 0)
    own = sparse.csr_matrix((sim[r1, k1].astype(np.float64), (r1, cols[r1, k1])), shape=shape)
    nb, sim = art.nbr_idx[ri], art.nbr_sim[ri]                    # rated books' lists that contain j
    rows = np.where(nb >= 0, row_of[np.maximum(nb, 0)], -1)
    c2, k2 = np.nonzero(rows >= 0)
    rev = sparse.csr_matrix((sim[c2, k2].astype(np.float64), (rows[c2, k2], c2)), shape=shape)
    P = own.maximum(rev)                                         # similarities are stored positive

    num = P @ resid
    den = np.asarray(P.sum(axis=1)).ravel()
    pred = m.bayes[uniq] + b_u + num / (den + shrink)
    out = np.clip(pred, 1, 5).astype(np.float32)[inv]
    if return_evidence:
        return out, (den / (den + CALIB_EVIDENCE_K)).astype(np.float32)[inv]
    return out


def prediction_floor(user: UserInput, raw_pred: np.ndarray, cal, offset: float | None,
                     evidence: np.ndarray | None = None) -> np.ndarray:
    """Keep-mask: books whose calibrated predicted rating is >= the user's average rating - offset.

    Off (keep everything) when offset is None or there's no calibration yet (< CALIB_MIN_RATINGS).
    """
    if offset is None or cal is None or not user.ratings:
        return np.ones(len(raw_pred), dtype=bool)
    avg = float(np.mean(list(user.ratings.values())))
    return apply_calibration(raw_pred, cal, evidence) >= avg - offset


def ranking(art: Artifacts, raw: RawScores, user: UserInput, p: Params, sort: str = "match",
            prior: np.ndarray | None = None, include_ya: bool = False) -> dict:
    """The user's full, unfiltered For-you ranking for one sort (cached on `raw`).

    Universe: every book the default view could show (unread, not a box set/children's/comic/later
    series volume), minus books predicted below the user's average (prediction_floor, when `prior` is
    given for calibration). The candidate pool comes first in exactly the order blend() gives it (or by
    predicted rating, ties by blend score); the rest follows by the same key, so every eligible book has
    a rank that filters don't change.
    """
    key = ("ranking", sort, prior is not None and p.pred_floor_offset is not None, include_ya)
    if key in raw.cache:
        return raw.cache[key]
    base_user = UserInput(ratings=user.ratings, read=user.read)          # dismissals are removed per request
    base_default = filter_mask(art, base_user, Filters(include_ya=include_ya), allow=next_in_series(art, base_user))
    # Predictions for every unread book (vectorized, ~60 ms), shared by both sorts.
    if "preds_unread" not in raw.cache:
        unread = np.flatnonzero(filter_mask(art, base_user, Filters.none()))
        pa = np.full(art.meta.n, np.nan, np.float32)
        ev = np.zeros(art.meta.n, np.float32)
        pa[unread], ev[unread] = predict_ratings(art, user, unread, return_evidence=True)
        raw.cache["preds_unread"], raw.cache["evidence_unread"] = pa, ev
    preds_all, ev_all = raw.cache["preds_unread"], raw.cache["evidence_unread"]
    cal = None
    if prior is not None:
        if "calib" not in raw.cache:
            raw.cache["calib"] = calibration(art, user, prior)
        cal = raw.cache["calib"]
    # What the user sees (calibrated in proportion to evidence); used for the floor and the predicted sort.
    shown_all = apply_calibration(preds_all, cal, ev_all) if cal is not None else preds_all
    keep = np.ones(art.meta.n, dtype=bool)
    if cal is not None and p.pred_floor_offset is not None:
        known = ~np.isnan(preds_all)
        keep[known] = prediction_floor(user, preds_all[known], cal, p.pred_floor_offset, ev_all[known])
    base = base_default & keep
    cand, top_ii, top_als, a, beta = _pool(art, raw, base, p)
    boost = raw.n > 0 and p.delta_pred
    if boost:
        cand = add_pred_candidates(art, cand, base, preds_all, p)
    s_all = score_all(art, raw, cand, a, beta, p, preds_all if boost else None).astype(np.float32)
    in_pool = np.zeros(art.meta.n, dtype=bool)
    in_pool[cand] = True
    rest = np.flatnonzero(base & ~in_pool)
    if sort == "predicted":
        key_fn = lambda ix: np.lexsort((-s_all[ix], -shown_all[ix]))  # noqa: E731
    else:
        key_fn = lambda ix: np.argsort(-s_all[ix], kind="stable")  # noqa: E731
    order = np.concatenate([cand[key_fn(cand)], rest[key_fn(rest)]])
    source = np.zeros(art.meta.n, np.int8)
    source[top_ii] += 1
    source[top_als] += 2
    out = {"order": order, "score": s_all, "source": source, "preds": preds_all, "shown": shown_all, "alpha": a,
           "base": base, "base_default": base_default, "keep": keep, "sort": sort}
    raw.cache[key] = out
    return out


def apply_ranking(order: np.ndarray, mask: np.ndarray, extra_key: np.ndarray, dismissed=()) -> tuple[np.ndarray, np.ndarray]:
    """Filter a full ranking without renumbering it.

    Returns (items in rank order that pass `mask`, their 1-based ranks). Books that pass the mask but
    aren't in the ranking (e.g. later series volumes shown by a filter toggle) follow, rank 0.
    """
    if len(dismissed):
        order = order[~np.isin(order, np.fromiter(dismissed, dtype=np.int64))]
    rank_of = np.full(len(mask), -1, np.int64)
    rank_of[order] = np.arange(len(order))
    ranked = order[mask[order]]
    extra = np.flatnonzero(mask & (rank_of < 0))
    extra = extra[np.argsort(-extra_key[extra], kind="stable")]
    items = np.concatenate([ranked, extra])
    return items, np.where(rank_of[items] >= 0, rank_of[items] + 1, 0)


# ---- calibration: match predicted ratings to the user's own rating distribution ---------------------

# Each star level as a continuous band, so e.g. 5-star ratings spread over 4.5-5.0 instead of piling up at 5.
_STAR_BANDS = [(1.0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.0)]
CALIB_PRIOR_WEIGHT = 10     # the dataset-wide distribution counts as this many ratings
CALIB_MIN_RATINGS = 5       # below this, show the raw prediction
CALIB_TAIL = 0.35           # raw-star scale of the asymptotic approach to 1 / 5 outside the reference range
# Evidence gate: a book gets den / (den + K) of the calibration stretch, den = its summed similarity to the
# user's rated books. Eval 2026-09-27 (1,000 held-out users, displayed-rating RMSE): full stretch 0.998,
# K=0.1 0.879, K=0.5 0.866 (but top picks then never reach 4.5). 0.1 keeps most of the gain and the spread.
CALIB_EVIDENCE_K = 0.1
_CALIB_Q = np.linspace(0.02, 0.98, 25)


def _target_quantiles(counts: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Quantile function of a 1-5 star histogram with each star spread uniformly over its band."""
    cum = np.concatenate([[0.0], np.cumsum(counts / counts.sum())])
    out = np.empty(len(q))
    for k, qq in enumerate(q):
        s = int(np.clip(np.searchsorted(cum, qq, side="right") - 1, 0, 4))
        frac = (qq - cum[s]) / max(cum[s + 1] - cum[s], 1e-12)
        lo, hi = _STAR_BANDS[s]
        out[k] = lo + np.clip(frac, 0, 1) * (hi - lo)
    return out


def calibration(art: Artifacts, user: UserInput, prior: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Monotone map from raw predictions to the user's rating scale (quantile matching).

    Reference: raw predictions for the books the user rated (their own rating never enters its own
    residual, since neighbor lists exclude self). Target: the user's rating histogram, shrunk toward the
    dataset-wide histogram with weight CALIB_PRIOR_WEIGHT, with stars as continuous bands. Returns
    (knots_x, knots_y, weight) where weight = n / (n + CALIB_PRIOR_WEIGHT) blends mapped with raw.
    """
    n = len(user.ratings)
    if n < CALIB_MIN_RATINGS:
        return None
    ri = np.fromiter(user.ratings, dtype=np.int64)
    rv = np.fromiter(user.ratings.values(), dtype=np.int64)
    ref = predict_ratings(art, user, ri).astype(np.float64)
    counts = np.bincount(rv, minlength=6)[1:].astype(np.float64) + CALIB_PRIOR_WEIGHT * np.asarray(prior, dtype=np.float64)
    x = np.maximum.accumulate(np.quantile(ref, _CALIB_Q)) + np.arange(len(_CALIB_Q)) * 1e-6   # strictly increasing
    y = _target_quantiles(counts, _CALIB_Q)
    return x, y, n / (n + CALIB_PRIOR_WEIGHT)


def apply_calibration(raw_pred: np.ndarray, cal: tuple[np.ndarray, np.ndarray, float] | None,
                      evidence: np.ndarray | None = None) -> np.ndarray:
    """Map raw predictions onto the user's rating scale. With `evidence` (predict_ratings' evidence weights),
    each book moves only that fraction of the way: a book linked to many of the user's ratings gets the full
    stretch, while one with no link stays at its raw value (book average adjusted for the user's harshness),
    so noise in book averages isn't amplified. Without `evidence` the map is monotone in raw."""
    if cal is None:
        return raw_pred
    x, y, w = cal
    r = raw_pred.astype(np.float64)
    mapped = np.interp(r, x, y)
    hi, lo = r > x[-1], r < x[0]
    # Outside the reference range, approach 5 (or 1) asymptotically: no pile-up at the ceiling.
    mapped[hi] = y[-1] + (5.0 - y[-1]) * (1 - np.exp(-(r[hi] - x[-1]) / CALIB_TAIL))
    mapped[lo] = y[0] - (y[0] - 1.0) * (1 - np.exp(-(x[0] - r[lo]) / CALIB_TAIL))
    out = w * mapped + (1 - w) * r
    if evidence is not None:
        out = r + np.asarray(evidence, dtype=np.float64) * (out - r)
    return np.clip(out, 1, 5).astype(np.float32)


def shown_predictions(art: Artifacts, user: UserInput, items, cal) -> np.ndarray:
    """Predicted ratings as displayed: raw prediction, calibrated in proportion to the personal evidence."""
    raw_pred, ev = predict_ratings(art, user, items, return_evidence=True)
    return apply_calibration(raw_pred, cal, ev)


def display_ratings(art: Artifacts, user: UserInput, items, prior: np.ndarray | None,
                    cal: tuple | None | str = "compute") -> np.ndarray:
    """Predicted ratings as shown to the user: raw predictions calibrated to their rating distribution."""
    if prior is None:
        return predict_ratings(art, user, items)
    if isinstance(cal, str):
        cal = calibration(art, user, prior)
    return shown_predictions(art, user, items, cal)


def recommend(art: Artifacts, user: UserInput, f: Filters, p: Params, limit: int = 40, offset: int = 0,
              raw: RawScores | None = None, sort: str = "match", prior: np.ndarray | None = None) -> dict:
    """Recommendations for one page. Ranks come from the unfiltered ranking for `sort`, so applying
    filters narrows the list without renumbering it. Pass `raw` (from a cache) to skip model scoring,
    and `prior` (dataset rating distribution) to enable calibration and the prediction floor.
    `predicted` holds displayed predictions (evidence-gated calibration; raw without `prior`)."""
    raw = raw or raw_scores(art, user, p)
    rk = ranking(art, raw, user, p, sort, prior, include_ya=f.include_ya or _asks_for_ya(art.meta, f))
    nxt = next_in_series(art, user)
    mask = filter_mask(art, user, f, allow=nxt) & rk["keep"]      # the floor also applies to filter extras
    extra_key = (rk["score"] if sort != "predicted"
                 else np.nan_to_num(rk["shown"], nan=0.0) * 1e3 + rk["score"])
    items, ranks = apply_ranking(rk["order"], mask, extra_key, user.dismissed)
    page = slice(offset, offset + limit)
    items, ranks = items[page], ranks[page]
    preds = rk["shown"][items]                                   # displayed (calibrated when prior is given)
    if np.isnan(preds).any():
        preds = shown_predictions(art, user, items, raw.cache.get("calib"))
    return {
        "items": items, "ranks": ranks, "scores": rk["score"][items], "source": rk["source"][items],
        "predicted": preds, "because": explain(art, raw, items), "next_in_series": [int(i) in nxt for i in items],
        "total": int(mask.sum()), "alpha": min(raw.n / (raw.n + p.k_a), p.a_max) if raw.n else 0.0,
    }


def similar_books(art: Artifacts, j: int, k: int = 20) -> list[int]:
    """Item-item neighbors (same series collapsed), padded with ALS cosine neighbors."""
    m = art.meta
    out: list[int] = []
    series_seen = {int(m.series_id[j])} if m.series_id[j] >= 0 else set()
    def take(i: int) -> None:
        sid = int(m.series_id[i])
        if i == j or i in out or m.is_boxset[i] or (sid >= 0 and sid in series_seen):
            return
        if sid >= 0:
            series_seen.add(sid)
        out.append(i)
    for i in art.nbr_idx[j]:
        if i >= 0:
            take(int(i))
    if len(out) < k:
        Yn = art.Y / np.maximum(np.linalg.norm(art.Y, axis=1, keepdims=True), 1e-8)
        for i in np.argsort(-(Yn @ Yn[j]))[: 5 * k]:
            take(int(i))
            if len(out) >= k:
                break
    return out[:k]
