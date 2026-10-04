"""Rating-prediction track of the evaluation (goodrec.eval.rating, metrics, run): metrics on hand-computed
examples, the reconstruction switches, the promotion guardrail and leakage checks."""

import dataclasses

import numpy as np
import pytest
from scipy import sparse

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR, load_config
from goodrec.core.scoring import UserInput
from goodrec.eval.metrics import (CAL_EDGES, RATING_COLS, paired, pooled_summary, rating_pool, rating_user,
                                  spearman)
from goodrec.eval.rating import (RatingSettings, global_item_means, population_sd, rating_style, user_sigma)

needs_data = pytest.mark.skipif(
    not ((ARTIFACTS_DIR / "manifest.json").exists() and (INTERIM_DIR / "R_train.npz").exists()),
    reason="artifacts / interim data not built")


# ---------------------------------------------------------------- metrics

def test_rating_user_hand_computed():
    pred, actual = np.array([4.0, 3.5, 2.0, 5.0]), np.array([5, 3, 4, 5])
    got = dict(zip(RATING_COLS, rating_user(pred, actual, sigma=2.0)))
    # errors: -1, +0.5, -2, 0
    assert got["mae"] == pytest.approx(3.5 / 4)
    assert got["mse"] == pytest.approx((1 + 0.25 + 4 + 0) / 4)
    assert got["within1"] == pytest.approx(3 / 4)          # |-1| counts as within one star
    assert got["within05"] == pytest.approx(2 / 4)
    assert got["bias"] == pytest.approx(-2.5 / 4)
    assert got["nmae"] == pytest.approx(3.5 / 4 / 2.0)
    # ranks: pred (3, 2, 1, 4), actual (3.5, 1, 2, 3.5)
    assert got["spearman"] == pytest.approx(np.corrcoef([3, 2, 1, 4], [3.5, 1, 2, 3.5])[0, 1])


def test_spearman_needs_three_varied_ratings_and_scores_constant_predictions_zero():
    assert np.isnan(spearman(np.array([1.0, 2.0]), np.array([1, 5])))           # fewer than 3
    assert np.isnan(spearman(np.array([1.0, 2.0, 3.0]), np.array([4, 4, 4])))   # actual ratings all equal
    assert spearman(np.array([3.0, 3.0, 3.0]), np.array([1, 3, 5])) == 0.0      # no ordering information
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([2, 3, 5])) == pytest.approx(1.0)
    assert spearman(np.array([3.0, 2.0, 1.0]), np.array([2, 3, 5])) == pytest.approx(-1.0)


def test_pooled_summary_matches_direct_computation():
    rng = np.random.default_rng(0)
    p1, a1 = rng.uniform(1, 5, 30), rng.integers(1, 6, 30)
    p2, a2 = rng.uniform(1, 5, 12), rng.integers(1, 6, 12)
    s = pooled_summary(rating_pool(p1, a1, 3) + rating_pool(p2, a2, 1))     # two users' sums add up
    p, a = np.concatenate([p1, p2]), np.concatenate([a1, a2]).astype(float)
    assert s["n"] == 42 and s["fallback"] == pytest.approx(4 / 42)
    assert s["pearson"] == pytest.approx(np.corrcoef(p, a)[0, 1])
    assert s["sd_pred"] == pytest.approx(p.std()) and s["sd_actual"] == pytest.approx(a.std())
    b = np.clip(np.searchsorted(CAL_EDGES, p, side="right") - 1, 0, len(CAL_EDGES) - 2)
    for i in range(len(CAL_EDGES) - 1):
        assert s["calibration"]["n"][i] == (b == i).sum()
        if (b == i).any():
            assert s["calibration"]["mean_pred"][i] == pytest.approx(p[b == i].mean())
            assert s["calibration"]["mean_actual"][i] == pytest.approx(a[b == i].mean())
    for star in range(1, 6):
        m = a == star
        assert s["by_star"]["n"][star - 1] == m.sum()
        if m.any():
            assert s["by_star"]["bias"][star - 1] == pytest.approx((p[m] - a[m]).mean())
    assert pooled_summary(rating_pool(np.array([5.0]), np.array([5])))["calibration"]["n"][-1] == 1   # 5.0 in the top bin


def test_paired_lower_is_better_counts_lower_errors_as_wins():
    model, base = np.array([0.5, 0.8, 0.6, 1.0]), np.array([0.7, 0.8, 0.9, 0.9])
    c = paired(model, base, n_boot=200, lower_is_better=True)
    assert (c["win"], c["tie"], c["loss"]) == (0.5, 0.25, 0.25)
    assert c["mean_diff"] == pytest.approx((model - base).mean())           # still model - baseline
    assert c["lift"] == pytest.approx((model - base).mean() / base.mean())
    assert c["median_gain_when_win"] == pytest.approx(np.median([0.2, 0.3]))  # reductions, positive


def test_guardrail_blocks_only_a_significantly_worse_mae():
    from goodrec.eval.run import rating_guardrail_ok
    assert rating_guardrail_ok(None)
    assert rating_guardrail_ok({"n": 100, "ci95": [-0.01, 0.02]})       # not significantly worse
    assert rating_guardrail_ok({"n": 100, "ci95": [-0.03, -0.01]})      # better
    assert not rating_guardrail_ok({"n": 100, "ci95": [0.001, 0.02]})   # worse


# ---------------------------------------------------------------- settings and scale

def test_rating_settings_parse_and_cannot_be_promoted():
    assert RatingSettings.parse(None).is_default
    assert RatingSettings.parse("item_means=global;calibration=none") == RatingSettings("global", "none")
    assert RatingSettings.parse("calibration=full") == RatingSettings("goodreads", "full")
    with pytest.raises(SystemExit):
        RatingSettings.parse("calibration=sometimes")
    with pytest.raises(SystemExit):
        RatingSettings.parse("shrink=1")
    from goodrec.eval.run import main
    with pytest.raises(SystemExit, match="can't be promoted"):
        main(promote=True, rating="calibration=none")


def test_user_sigma_shrinks_toward_the_population_and_has_a_floor():
    sd = 1.0
    assert user_sigma(np.array([4.0]), sd) == pytest.approx(1.0)              # one rating: the population's
    assert user_sigma(np.array([4.0] * 200), sd) == pytest.approx(0.5)        # all the same: the floor
    wide = np.array([1.0, 5.0] * 50)
    assert user_sigma(wide, sd) == pytest.approx(np.sqrt((wide.var(ddof=0) * 100 + 5) / (99 + 5)))
    assert population_sd([0, 0, 0, 0, 1]) == 0.0
    assert list(rating_style(np.array([0.6, 0.9, 1.2, 0.833]), [0.833, 1.015])) == [0, 1, 2, 1]


def test_global_item_means_hand_computed():
    R = sparse.csr_matrix(np.array([[5, 0, 1], [3, 0, 0]], dtype=np.float32))   # global mean 3
    got = global_item_means(R, bayes_m=2)
    np.testing.assert_allclose(got, [(8 + 6) / 4, 3.0, (1 + 6) / 3], rtol=1e-6)


# ---------------------------------------------------------------- built data

@pytest.fixture(scope="module")
def built():
    import orjson

    from goodrec.core.artifacts import load_artifacts
    art = load_artifacts(with_readers=False)
    prior = np.asarray(orjson.loads((ARTIFACTS_DIR / "population_stats.json").read_bytes())["rating_dist"])
    return art, prior


@needs_data
def test_reconstruction_switches(built):
    from goodrec.core.scoring import calibration, predict_ratings, shown_predictions
    from goodrec.eval.rating import model_ratings, rating_artifacts
    art, prior = built
    user = UserInput(ratings={0: 5, 1: 4, 2: 2, 50: 5, 70: 3, 90: 4, 120: 1})
    items = np.arange(200, 260)
    today = model_ratings(art, user, items, RatingSettings(), prior)
    np.testing.assert_array_equal(today, shown_predictions(art, user, items, calibration(art, user, prior)))
    np.testing.assert_array_equal(model_ratings(art, user, items, RatingSettings(calibration="none"), prior),
                                  predict_ratings(art, user, items))
    assert rating_artifacts(art, RatingSettings(), 50) is art
    g = rating_artifacts(art, RatingSettings(item_means="global"), load_config()["blend"]["bayes_m"])
    expect = global_item_means(sparse.load_npz(INTERIM_DIR / "R_train.npz"), load_config()["blend"]["bayes_m"])
    np.testing.assert_array_equal(g.meta.bayes, expect)
    assert art.meta.bayes is not g.meta.bayes and not np.allclose(art.meta.bayes, g.meta.bayes)
    full = model_ratings(art, user, items, RatingSettings(calibration="full"), prior)
    assert not np.allclose(full, today)


@needs_data
def test_style_cut_points_come_from_validation_users(built):
    from goodrec.eval.split import load_split
    _, prior = built
    cfg = load_config()["eval"]
    sd = population_sd(prior)
    val = load_split(cfg=cfg).select("validation", None, seed=cfg["seed"])
    sig = np.array([user_sigma(c.visible_r, sd) for c in val])
    np.testing.assert_allclose(cfg["rating_style_cuts"], np.quantile(sig, [1 / 3, 2 / 3]), atol=6e-4)


@needs_data
def test_hidden_ratings_never_reach_a_prediction(built):
    """Run the harness's worker on a case and on a copy with different hidden ratings: every rating model gets
    identical inputs and so makes identical predictions; only the metrics change."""
    from goodrec.eval import run
    from goodrec.eval.models import rating_baselines, shelf_life_models
    from goodrec.eval.split import load_split
    art, prior = built
    cfg = load_config()["eval"]
    case = load_split(cfg=cfg).select("validation", 5, seed=cfg["seed"])[0]
    flipped = dataclasses.replace(case, hidden_r=(6 - case.hidden_r).astype(case.hidden_r.dtype))
    seen = []

    def spy(rec):
        rate = rec.rate

        def wrapped(user, items):
            out = rate(user, items)
            seen.append((rec.key, sorted(user.ratings.items()), items.tolist(), np.asarray(out).tolist()))
            return out
        object.__setattr__(rec, "rate", wrapped)
        return rec

    recs = [spy(r) for r in shelf_life_models(art, prior) + rating_baselines(art) if "rating" in r.tracks]
    run._STATE.update(models=recs, cases=[case, flipped], buckets=[3, -1], rel_min=4, seed=0,
                      fallback=np.clip(art.meta.bayes, 1, 5), pop_sd=population_sd(prior))
    outs = []
    for mi in range(len(recs)):
        seen.clear()
        _, _, _, rmet, _ = run._work((mi, np.array([0, 1])))
        half = len(seen) // 2
        assert seen[:half] == seen[half:], recs[mi].key
        outs.append(rmet)
    assert any(not np.allclose(r[0, :, 0], r[1, :, 0]) for r in outs)    # the truth does differ
