"""Recommendation scoring: ALS fold-in + item-item neighbors, blended by rating count.

All functions are numpy-only and operate on dense catalog indices (work_idx).
    final = a(n)*z(s_als) + (1-a(n))*z(s_ii) + beta*z(log_pop) + gamma*z(bayes),  a(n) = n/(n+k_a)
z-scores are computed within the candidate set (top-C of each model after filtering).
"""

from dataclasses import dataclass, field

import numpy as np

from goodrec.config import load_config
from goodrec.core.artifacts import Artifacts


@dataclass
class UserInput:
    ratings: dict[int, int]                       # work_idx -> 1..5
    read: set[int] = field(default_factory=set)   # read but unrated
    dismissed: set[int] = field(default_factory=set)

    @property
    def seen(self) -> set[int]:
        return set(self.ratings) | self.read | self.dismissed


@dataclass
class Filters:
    genres: list[int] = field(default_factory=list)          # parent genre ids (any-of)
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
    include_boxsets: bool = False
    include_series_continuations: bool = False

    @classmethod
    def none(cls) -> "Filters":
        """No content filters at all (used by eval)."""
        return cls(include_children=True, include_comics=True, include_boxsets=True,
                   include_series_continuations=True)


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

    @classmethod
    def from_config(cls, **overrides) -> "Params":
        cfg = load_config()
        b, a = cfg["blend"], cfg["als"]
        p = dict(k_a=b["k_a"], beta_pop=b["beta_pop"], beta_pop_many=b.get("beta_pop_many", b["beta_pop"]), gamma_quality=b["gamma_quality"],
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


def fold_in(art: Artifacts, user: UserInput, p: Params) -> np.ndarray | None:
    """Solve the ALS user vector for a new user given fixed item factors (implicit's recalculate_user)."""
    items, conf = [], []
    for i, r in user.ratings.items():
        g = p.confidence.get(int(r), 0.0)
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
    """w_i = r_i - b_u, with b_u a user mean shrunk toward the prior (so one 5-star rating counts)."""
    if not user.ratings:
        return {}
    r = np.asarray(list(user.ratings.values()), dtype=np.float32)
    b_u = (r.sum() + p.ii_prior_mean * p.ii_prior_weight) / (len(r) + p.ii_prior_weight)
    return {i: float(v - b_u) for i, v in user.ratings.items()}


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
    if f.text.strip():
        terms = f.text.lower().split()
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


def blend(art: Artifacts, raw: RawScores, mask: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (items sorted best-first, their scores, source codes: 1=ii, 2=als, 3=both)."""
    m = art.meta
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return idx, np.zeros(0, np.float32), np.zeros(0, np.int8)
    if raw.n == 0:
        cand = _top(m.log_pop + m.bayes, idx, p.candidates)
        s = zscore(m.log_pop[cand]) + zscore(m.bayes[cand])
        order = np.argsort(-s)
        return cand[order], s[order], np.zeros(len(cand), np.int8)

    top_als = _top(raw.s_als, idx, p.candidates) if raw.u is not None else np.zeros(0, np.int64)
    ii_pos = idx[raw.s_ii[idx] > 0]
    top_ii = _top(raw.s_ii, ii_pos, p.candidates)
    cand = np.union1d(top_als, top_ii)
    if len(cand) == 0:
        cand = _top(m.log_pop, idx, p.candidates)
    a = raw.n / (raw.n + p.k_a) if raw.u is not None else 0.0
    if p.a_override is not None:
        a = p.a_override
        if a >= 1.0 and raw.u is not None:
            cand = top_als
        elif a <= 0.0 and len(top_ii):
            cand = top_ii
    # Popularity helps cold-start users but hurts heavy readers (eval): interpolate with a(n).
    beta = p.beta_pop if p.a_override is not None else (1 - a) * p.beta_pop + a * p.beta_pop_many
    s = (a * zscore(raw.s_als[cand]) + (1 - a) * zscore(raw.s_ii[cand])
         + beta * zscore(m.log_pop[cand]) + p.gamma_quality * zscore(m.bayes[cand]))
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
                    user_shrink: float = 5.0) -> np.ndarray:
    """Predicted star rating (1-5) for each item: baseline + item-item residual.

        baseline(u, j) = item_mean(j) + b_u,  b_u = sum(r_i - item_mean(i)) / (n + user_shrink)
        pred(u, j) = baseline(u, j) + sum_i sim(i,j) * (r_i - baseline(u, i)) / (sum_i |sim(i,j)| + shrink)
    over rated books i linked to j in either direction of the neighbor lists. item_mean is the
    Bayesian-shrunk mean from the training data; `shrink` pulls sparse-evidence residuals to 0.
    """
    m = art.meta
    items = np.asarray(items, dtype=np.int64)
    if len(items) == 0:
        return np.zeros(0, np.float32)
    if not user.ratings:
        return np.clip(m.bayes[items], 1, 5).astype(np.float32)
    ri = np.fromiter(user.ratings, dtype=np.int64)
    rv = np.fromiter(user.ratings.values(), dtype=np.float32)
    b_u = float((rv - m.bayes[ri]).sum() / (len(ri) + user_shrink))
    resid = dict(zip(ri.tolist(), (rv - (m.bayes[ri] + b_u)).tolist()))

    pos = {int(j): k for k, j in enumerate(items)}
    pairs: dict[tuple[int, int], float] = {}
    nb, sim = art.nbr_idx[items], art.nbr_sim[items]            # j's own neighbors
    for r, c in zip(*np.nonzero(np.isin(nb, ri))):
        key = (int(r), int(nb[r, c]))
        pairs[key] = max(pairs.get(key, 0.0), float(sim[r, c]))
    nb, sim = art.nbr_idx[ri], art.nbr_sim[ri]                  # rated books' neighbor lists
    for r, c in zip(*np.nonzero(np.isin(nb, items))):
        key = (pos[int(nb[r, c])], int(ri[r]))
        pairs[key] = max(pairs.get(key, 0.0), float(sim[r, c]))
    num = np.zeros(len(items), np.float64)
    den = np.zeros(len(items), np.float64)
    for (k, i), s in pairs.items():
        num[k] += s * resid[i]
        den[k] += abs(s)
    pred = m.bayes[items] + b_u + num / (den + shrink)
    return np.clip(pred, 1, 5).astype(np.float32)


def recommend(art: Artifacts, user: UserInput, f: Filters, p: Params, limit: int = 40, offset: int = 0,
              raw: RawScores | None = None) -> dict:
    """Full ranking for one user. Pass `raw` (from a cache) to skip model scoring."""
    raw = raw or raw_scores(art, user, p)
    nxt = next_in_series(art, user)
    mask = filter_mask(art, user, f, allow=nxt)
    items, scores, source = blend(art, raw, mask, p)
    page = slice(offset, offset + limit)
    items, scores, source = items[page], scores[page], source[page]
    return {
        "items": items, "scores": scores, "source": source,
        "because": explain(art, raw, items), "next_in_series": [int(i) in nxt for i in items],
        "total": int(mask.sum()), "alpha": raw.n / (raw.n + p.k_a) if raw.n else 0.0,
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
