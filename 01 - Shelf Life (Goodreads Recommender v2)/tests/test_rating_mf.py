"""Factorization rating predictors (core.rating_mf, pipeline.train_rating_mf) and their wiring into
predict_ratings, the evaluation settings and the promotion rule (DECISIONS D-052, D-053)."""

import dataclasses

import numpy as np
import pytest
from scipy import sparse

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR
from goodrec.core.rating_mf import MFModel
from goodrec.core.scoring import UserInput

needs_data = pytest.mark.skipif(
    not ((ARTIFACTS_DIR / "manifest.json").exists() and (INTERIM_DIR / "R_train.npz").exists()),
    reason="artifacts / interim data not built")


def toy(kind="mf", k=3, n=40, seed=0, lam_b=2.0, lam_p=3.0):
    rng = np.random.default_rng(seed)
    return MFModel(kind=kind, mu=3.8, bi=rng.normal(0, 0.3, n).astype(np.float32),
                   Q=rng.normal(0, 0.5, (n, k)).astype(np.float32),
                   Y=rng.normal(0, 0.3, (n, k)).astype(np.float32) if kind == "svdpp" else None,
                   lam_b=lam_b, lam_p=lam_p)


def brute(m: MFModel, ratings: dict, z: np.ndarray):
    """Ridge solution by stacking the penalty as extra rows (an independent least-squares formulation)."""
    ri = np.array(list(ratings)); rv = np.array(list(ratings.values()), float)
    X = np.hstack([np.ones((len(ri), 1)), m.Q[ri].astype(float)])
    y = rv - m.mu - m.bi[ri] - m.Q[ri].astype(float) @ z
    pen = np.diag(np.sqrt(np.r_[m.lam_b, np.full(m.k, m.lam_p)]))
    w, *_ = np.linalg.lstsq(np.vstack([X, pen]), np.r_[y, np.zeros(m.k + 1)], rcond=None)
    return w


@pytest.mark.parametrize("kind", ["mf", "svdpp"])
def test_fold_in_matches_ridge_least_squares(kind):
    m = toy(kind)
    ratings = {1: 5, 4: 3, 7: 4, 9: 2, 12: 5, 20: 4}
    b_u, p_u, z, *_ = m.fold_in(ratings, read={30, 31})
    w = brute(m, ratings, z)
    assert b_u == pytest.approx(w[0]) and np.allclose(p_u, w[1:])


def test_fold_in_without_factors_is_a_shrunk_offset():
    m = toy()
    m.Q[:] = 0
    ratings = {1: 5, 4: 3, 7: 4}
    b_u, p_u, *_ = m.fold_in(ratings)
    expect = sum(r - m.mu - m.bi[i] for i, r in ratings.items()) / (len(ratings) + m.lam_b)
    assert b_u == pytest.approx(expect, rel=1e-5) and np.allclose(p_u, 0)


def test_own_books_are_predicted_leave_one_out():
    m = toy()
    ratings = {1: 5, 4: 3, 7: 4, 9: 2, 12: 5}
    got = m.predict(ratings, [4, 25], clip=False)
    rest = {i: r for i, r in ratings.items() if i != 4}
    assert got[0] == pytest.approx(m.predict(rest, [4], clip=False)[0], abs=1e-6)    # refit without book 4
    assert got[1] == pytest.approx(m.predict(ratings, [25], clip=False, loo=False)[0])
    assert m.predict(ratings, [4], clip=False, loo=False)[0] != pytest.approx(got[0])


def test_svdpp_implicit_term_and_prediction_by_hand():
    m = toy("svdpp", k=2, n=10)
    m.Q[:] = 0
    m.Q[3] = [1.0, 2.0]
    z = m.implicit({1: 4}, read={2})
    np.testing.assert_allclose(z, (m.Y[1].astype(float) + m.Y[2]) / np.sqrt(2), rtol=1e-6)
    pred = m.predict({}, [3], read={1, 2}, clip=False)[0]
    assert pred == pytest.approx(m.mu + m.bi[3] + float(np.array([1.0, 2.0]) @ z), rel=1e-6)
    assert np.allclose(toy("mf").implicit({1: 4}, {2}), 0)


def test_save_and_load_round_trip(tmp_path):
    m = toy("svdpp")
    m.save(tmp_path / "m.npz")
    m2 = MFModel.load(tmp_path / "m.npz")
    assert (m2.kind, m2.mu, m2.lam_b, m2.lam_p) == (m.kind, pytest.approx(m.mu), m.lam_b, m.lam_p)
    np.testing.assert_array_equal(m2.Y, m.Y)


@pytest.mark.parametrize("kind", ["mf", "svdpp"])
def test_training_learns_a_low_rank_matrix(kind):
    from goodrec.pipeline.train_rating_mf import TrainConfig, train
    rng = np.random.default_rng(1)
    U, V = rng.normal(0, 0.7, (400, 3)), rng.normal(0, 0.7, (150, 3))
    full = np.clip(np.rint(3.6 + U @ V.T + rng.normal(0, 0.3, (400, 150))), 1, 5)
    mask = rng.random(full.shape) < 0.3
    R = sparse.csr_matrix(np.where(mask, full, 0).astype(np.int8))
    model, curve = train(TrainConfig(kind=kind, k=3, lr=0.01, reg=0.02, epochs=30), R=R,
                         Read=sparse.csr_matrix(R.shape, dtype=np.int8), verbose=False)
    assert curve[-1][1] < curve[0][1] - 0.15                       # training error falls
    # A new reader (row 0's held-out cells) folded in from their observed ratings beats the item means alone.
    errs, base = [], []
    for u in range(50):
        obs = {int(j): float(full[u, j]) for j in np.flatnonzero(mask[u])}
        hid = np.flatnonzero(~mask[u])
        errs.append(np.abs(model.predict(obs, hid) - full[u, hid]).mean())
        base.append(np.abs(np.clip(model.mu + model.bi[hid], 1, 5) - full[u, hid]).mean())
    assert np.mean(errs) < np.mean(base) - 0.05


def test_promotion_decision():
    from goodrec.eval.run import promotion_decision
    sig_better, sig_worse, flat = ({"n": 9, "ci95": [-0.02, -0.01], "mean_diff": -0.015},
                                   {"n": 9, "ci95": [0.01, 0.02], "mean_diff": 0.015},
                                   {"n": 9, "ci95": [-0.01, 0.01], "mean_diff": 0.0})
    assert promotion_decision(0.09, None, None, None)[0]
    assert promotion_decision(0.09, 0.08, flat, sig_better)[0]                 # NDCG up, MAE not worse
    assert not promotion_decision(0.09, 0.08, sig_worse, sig_better)[0]        # NDCG up, MAE significantly worse
    assert promotion_decision(0.08, 0.08, sig_better, flat)[0]                 # MAE better, ranking unchanged
    ndcg_worse = {"n": 9, "ci95": [-0.004, -0.001]}
    assert not promotion_decision(0.079, 0.08, sig_better, ndcg_worse)[0]      # MAE better, NDCG significantly worse
    assert not promotion_decision(0.08, 0.08, flat, flat)[0]                   # nothing better


def test_predictor_setting_parses():
    from goodrec.eval.rating import RatingSettings
    s = RatingSettings.parse("predictor=hybrid:best_svdpp;calibration=full")
    assert (s.predictor, s.calibration) == ("hybrid:best_svdpp", "full") and not s.is_default
    with pytest.raises(SystemExit):
        RatingSettings.parse("predictor=magic")


# ---------------------------------------------------------------- built data

@pytest.fixture(scope="module")
def built():
    import orjson

    from goodrec.core.artifacts import load_artifacts
    art = load_artifacts(with_readers=False)
    prior = np.asarray(orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"])
    rng = np.random.default_rng(0)
    m = MFModel(kind="svdpp", mu=3.9, bi=rng.normal(0, 0.3, art.meta.n).astype(np.float32),
                Q=rng.normal(0, 0.1, (art.meta.n, 8)).astype(np.float32),
                Y=rng.normal(0, 0.1, (art.meta.n, 8)).astype(np.float32))
    return art, prior, m


@needs_data
def test_predict_ratings_modes(built):
    from goodrec.core.scoring import calibration, predict_ratings
    art, prior, m = built
    assert art.rating_mode == "knn"
    user = UserInput(ratings={0: 5, 1: 4, 2: 2, 50: 5, 70: 3, 90: 4, 120: 1}, read={300})
    items = np.r_[np.arange(200, 260), 0, 1]
    mf = dataclasses.replace(art, rating_mode="mf", rating_mf=m)
    np.testing.assert_allclose(predict_ratings(mf, user, items), m.predict(user.ratings, items, user.read), rtol=1e-5)
    hy = dataclasses.replace(art, rating_mode="hybrid", rating_mf=m)
    assert not np.allclose(predict_ratings(hy, user, items), predict_ratings(mf, user, items))
    _, ev_full = predict_ratings(dataclasses.replace(mf, rating_calibration="full"), user, items, return_evidence=True)
    assert np.all(ev_full == 1)
    assert calibration(dataclasses.replace(art, rating_calibration="none"), user, prior) is None
    assert calibration(art, user, prior) is not None


@needs_data
def test_mf_predictor_never_sees_hidden_ratings(built, monkeypatch):
    """The harness's worker on a case and on a copy with different hidden ratings: identical predictions."""
    from goodrec.config import load_config
    from goodrec.eval import run
    from goodrec.eval.models import shelf_life_models
    from goodrec.eval.rating import RatingSettings, population_sd
    from goodrec.eval.split import load_split
    art, prior, m = built
    cfg = load_config()["eval"]
    case = load_split(cfg=cfg).select("validation", 5, seed=cfg["seed"])[1]
    flipped = dataclasses.replace(case, hidden_r=(6 - case.hidden_r).astype(case.hidden_r.dtype))
    rart = dataclasses.replace(art, rating_mode="hybrid", rating_mf=m)
    recs = [r for r in shelf_life_models(rart, prior, rating=RatingSettings(predictor="hybrid:x"), rating_art=rart)
            if r.key == "shelf_life_raw"]
    run._STATE.update(models=recs, cases=[case, flipped], buckets=[3, -1], rel_min=4, seed=0,
                      fallback=np.clip(art.meta.bayes, 1, 5), pop_sd=population_sd(prior))
    seen = []
    orig = recs[0].rate
    monkeypatch.setattr(recs[0], "rate", lambda u, i: seen.append(orig(u, i).tolist()) or np.array(seen[-1]))
    run._work((0, np.array([0, 1])))
    assert seen[:2] == seen[2:]


def test_sweep_config_parsing():
    from goodrec.eval.tune_rating import parse_configs
    a, b = parse_configs("kind=mf,k=16,reg=0.02,reg_b=0 | kind=svdpp,k=8,reg_y=0.4,epochs=3")
    assert (a.kind, a.k, a.reg, a.reg_b) == ("mf", 16, 0.02, 0.0)
    assert (b.kind, b.k, b.reg_y, b.epochs) == ("svdpp", 8, 0.4, 3)


@needs_data
def test_sweep_scores_match_the_harness(built):
    """The sweep's fast evaluator and the evaluation harness give the same per-reader MAE for the same model."""
    from goodrec.config import load_config
    from goodrec.core.scoring import Params
    from goodrec.eval import run
    from goodrec.eval.models import ShelfLife
    from goodrec.eval.rating import RatingSettings, population_sd
    from goodrec.eval.split import load_split
    from goodrec.eval.tune_rating import BUCKETS, evaluate
    art, prior, m = built
    cfg = load_config()["eval"]
    cases = load_split(cfg=cfg).select("validation", 12, seed=cfg["seed"])
    fast = evaluate(m, cases, population_sd(prior), workers=1)
    rart = dataclasses.replace(art, rating_mode="mf", rating_mf=m, rating_calibration="none")
    rec = ShelfLife(key="m", name="m", description="", art=art, prior=prior, params=Params.from_config(), tracks=("rating",),
                    rating=RatingSettings(calibration="none", predictor="mf:x"), rating_art=rart)
    run._STATE.update(fallback=np.clip(art.meta.bayes, 1, 5), pop_sd=population_sd(prior))
    harness = run.evaluate([rec], cases, BUCKETS, cfg, 1)["m"]["rmet"]
    np.testing.assert_allclose(fast[:, :, 0], harness[:, :, 0], rtol=1e-4)
