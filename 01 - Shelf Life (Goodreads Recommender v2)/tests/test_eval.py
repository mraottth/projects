"""Evaluation harness: metrics, the temporal split, leakage guards and the baselines.

The first group uses synthetic data; tests marked `needs_data` use the built artifacts and split inputs
(skipped if `make data artifacts` hasn't run)."""

import numpy as np
import pytest
from scipy import sparse

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR, load_config
from goodrec.core.scoring import UserInput
from goodrec.eval.metrics import at_k, paired
from goodrec.eval.split import choose_split, order_dates, split_user

needs_data = pytest.mark.skipif(
    not ((ARTIFACTS_DIR / "manifest.json").exists() and (INTERIM_DIR / "R_train.npz").exists()),
    reason="artifacts / interim data not built")
CFG = {"holdout_frac": 0.3, "min_ratings": 10, "min_visible": 7, "min_hidden_relevant": 3, "relevant_min_rating": 4}


# ---------------------------------------------------------------- metrics

def test_at_k_hand_computed():
    top = np.array([5, 1, 7, 2])
    p, r, nd = at_k(top, {1, 2, 9}, 4)
    assert p == pytest.approx(2 / 4) and r == pytest.approx(2 / 3)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)                 # hits at ranks 2 and 4
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)            # three relevant, all could fit in the top 4
    assert nd == pytest.approx(dcg / idcg)


def test_at_k_fewer_relevant_than_k_and_short_lists():
    p, r, nd = at_k(np.array([3]), {3}, 10)               # one slot filled, one relevant book: perfect NDCG
    assert (p, r, nd) == (pytest.approx(0.1), 1.0, pytest.approx(1.0))
    assert at_k(np.array([1, 2]), set(), 10) == (0.0, 0.0, 0.0)


def test_paired_counts_wins_ties_losses():
    a = np.array([0.5, 0.0, 0.2, 0.0, np.nan])
    b = np.array([0.1, 0.0, 0.4, 0.0, 0.3])
    c = paired(a, b, n_boot=200)
    assert c["n"] == 4                                     # the NaN user is dropped
    assert (c["win"], c["tie"], c["loss"]) == (0.25, 0.5, 0.25)
    assert c["mean_diff"] == pytest.approx((0.4 - 0.2) / 4)
    assert c["ci95"][0] <= c["mean_diff"] <= c["ci95"][1]


# ---------------------------------------------------------------- split

def test_choose_split_moves_to_date_boundary():
    dates = np.array([1, 2, 3, 4, 5, 6, 7, 7, 7, 7])      # 30% of 10 = 3 hidden, but the last day has 4 books
    cut = choose_split(dates, 0.3)
    assert dates[cut] != dates[cut - 1]                    # never splits inside a day
    assert cut == 6                                        # the closest boundary hides all 4 same-day books
    assert choose_split(np.array([5, 5, 5]), 0.3) is None  # one date: no order to split on


def test_order_dates_prefer_plausible_read_dates():
    read = np.array([20150301, 0, 18000101, 20991231])
    added = np.array([20170101, 20160505, 20160606, 20160707])
    assert order_dates(read, added).tolist() == [20150301, 20160505, 20160606, 20160707]


def _user(n_days: int, per_day: int = 1, hidden_rating: int = 5):
    dates = np.repeat(np.arange(20170101, 20170101 + n_days), per_day)
    items = np.arange(len(dates), dtype=np.int32)
    ratings = np.full(len(dates), 3, np.int8)
    ratings[-(len(dates) * 3 // 10 + 1):] = hidden_rating
    return dates, items, ratings


def test_split_user_hidden_is_later_and_disjoint():
    dates, items, ratings = _user(20)
    vis, vis_r, read, hid, hid_r, split_date = split_user(dates, items, ratings, CFG, np.random.default_rng(0))
    assert not set(vis.tolist()) & set(hid.tolist())
    by_item = dict(zip(items.tolist(), dates.tolist()))
    assert max(by_item[i] for i in vis.tolist()) < split_date <= min(by_item[i] for i in hid.tolist())
    assert len(vis) == 14 and len(hid) == 6
    assert list(vis) == sorted(vis, key=lambda i: by_item[i])   # oldest first


def test_split_user_excludes_reads_after_the_split():
    dates, items, ratings = _user(20)
    ratings[2] = 0                                         # read (unrated) early: visible
    ratings[-1] = 0                                        # read (unrated) after the split: excluded everywhere
    vis, _, read, hid, _, _ = split_user(dates, items, ratings, CFG, np.random.default_rng(0))
    assert read.tolist() == [2]
    assert 19 not in vis.tolist() + hid.tolist()


def test_split_user_eligibility():
    rng = np.random.default_rng(0)
    assert split_user(*_user(9), CFG, rng) is None                     # fewer than 10 ratings
    assert split_user(*_user(20, hidden_rating=3), CFG, rng) is None   # no hidden book rated 4+
    assert split_user(*_user(1, per_day=12), CFG, rng) is None         # all on one date


# ---------------------------------------------------------------- grid spec

def test_grid_points_parse_every_combination():
    from goodrec.eval.run import grid_points
    pts = grid_points("k_a=20,50; a_max=0.5,1.0; pred_floor_offset=null,0.25")
    assert len(pts) == 8
    assert {"k_a": 50, "a_max": 0.5, "pred_floor_offset": None} in pts
    with pytest.raises(SystemExit):
        grid_points("not_a_param=1")


# ---------------------------------------------------------------- built data: leakage and baselines

@needs_data
def test_eval_users_are_not_in_training():
    train = np.load(INTERIM_DIR / "train_users.npy")
    test = np.load(INTERIM_DIR / "test_users.npy")
    assert not np.intersect1d(train, test).size
    R = sparse.load_npz(INTERIM_DIR / "R_train.npz")
    assert R.shape[0] == len(train)
    readers = sparse.load_npz(ARTIFACTS_DIR / "readers_csr.npz")
    user_rows = np.load(ARTIFACTS_DIR / "user_rows.npy")
    assert readers.shape[0] == len(user_rows) and user_rows.max() < len(train)   # rows of R_train only


@needs_data
def test_item_means_use_training_users_only():
    """bayes (item mean) must be recomputable from R_train alone, so held-out ratings can't leak into it."""
    from goodrec.core.artifacts import load_meta
    m = load_meta(ARTIFACTS_DIR)
    R = sparse.load_npz(INTERIM_DIR / "R_train.npz").tocsc()
    n = np.diff(R.indptr).astype(np.float64)
    s = np.asarray(R.sum(axis=0)).ravel().astype(np.float64)
    mu = R.data.mean()
    data_avg = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    prior = np.where(m.avg_rating > 0, m.avg_rating, mu)
    bayes_m = load_config()["blend"]["bayes_m"]
    expect = (data_avg * n + bayes_m * prior) / (n + bayes_m)
    np.testing.assert_allclose(m.bayes, expect, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(m.n_raters, n.astype(np.int32))


@needs_data
def test_split_sets_are_disjoint_and_stable():
    from goodrec.eval.split import load_split
    s = load_split()
    val = {c.user for c in s.cases if not c.is_test}
    test = {c.user for c in s.cases if c.is_test}
    assert val and test and not val & test
    assert set(np.load(INTERIM_DIR / "test_users.npy").tolist()) >= val | test
    assert load_split().hash == s.hash


@needs_data
def test_baselines_return_k_unseen_books():
    from goodrec.core.artifacts import load_artifacts
    from goodrec.eval.models import simple_baselines
    art = load_artifacts(with_readers=False)
    user = UserInput(ratings={0: 5, 1: 4, 2: 2, 50: 5}, read={3})
    for rec in simple_baselines(art):
        top = rec.recommend(user, 20, np.random.default_rng(0))
        assert len(top) == 20 and len(set(top.tolist())) == 20, rec.key
        assert not set(top.tolist()) & user.seen, rec.key


@needs_data
def test_a_max_caps_the_taste_model_share():
    from goodrec.core.artifacts import load_artifacts
    from goodrec.core.scoring import Filters, Params, recommend
    art = load_artifacts(with_readers=False)
    user = UserInput(ratings={i: 4 + i % 2 for i in range(0, 400, 4)})      # 100 ratings: a(n) = 100/120
    assert recommend(art, user, Filters(), Params.from_config(), limit=5)["alpha"] == pytest.approx(100 / 120)
    assert recommend(art, user, Filters(), Params.from_config(a_max=0.5), limit=5)["alpha"] == pytest.approx(0.5)


def test_recency_weights_and_neutral_default():
    from goodrec.core.scoring import Params, item_item_weights, recency_weights
    w = recency_weights([7, 8, 9], half_life=1.0)
    assert w == {7: 0.25, 8: 0.5, 9: 1.0}
    p = Params.from_config()
    user = UserInput(ratings={7: 5, 8: 3, 9: 4})
    assert item_item_weights(user, p) == item_item_weights(UserInput(ratings=user.ratings, recency={7: 1.0, 8: 1.0, 9: 1.0}), p)


@needs_data
def test_full_recency_reproduces_scores():
    from goodrec.core.artifacts import load_artifacts
    from goodrec.core.scoring import Params, raw_scores
    art = load_artifacts(with_readers=False)
    p = Params.from_config()
    user = UserInput(ratings={0: 5, 4: 4, 9: 2, 50: 5})
    a = raw_scores(art, user, p)
    b = raw_scores(art, UserInput(ratings=user.ratings, recency={i: 1.0 for i in user.ratings}), p)
    np.testing.assert_array_equal(a.s_als, b.s_als)
    np.testing.assert_array_equal(a.s_ii, b.s_ii)
