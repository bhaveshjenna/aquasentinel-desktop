"""Forecaster, regimes, risk and federated aggregation: shapes, finiteness, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.runtime import load_config, set_seed
from src.federated.simulate import fedavg
from src.perception.forecaster import LSTMForecaster
from src.perception.regimes import RegimeModel
from src.perception.risk import RiskModel
from src.twin import risk_bands as RB


@pytest.fixture(autouse=True)
def _seeded():
    set_seed(20240101)


def synthetic_series(n=900, seed=0):
    """A diurnal-ish multivariate series, so a forecaster has something learnable to find."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * (5 / 60)
    do = 7 + 2 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.1, n)
    temp = 28 + 2 * np.sin(2 * np.pi * (t - 3) / 24) + rng.normal(0, 0.05, n)
    ph = 7.6 + 0.3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.02, n)
    tan = 0.4 + 0.05 * np.cos(2 * np.pi * t / 24) + rng.normal(0, 0.01, n)
    no2 = np.full(n, 0.08) + rng.normal(0, 0.002, n)
    turb = 25 + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)
    return np.column_stack([do, temp, ph, tan, no2, turb])


# ---------------------------------------------------------------------------- forecaster
def test_windows_have_the_declared_shapes():
    f = LSTMForecaster()
    X, Y, M = f.make_windows(synthetic_series(), stride=4)
    assert X.shape[1:] == (f.window, 2 * len(f.channels))
    assert Y.shape[1:] == (f.horizon, len(f.channels))
    assert M.shape == Y.shape and M.dtype == bool
    assert np.isfinite(X).all() and np.isfinite(Y).all()


def test_windows_survive_missing_channels():
    """Real sites lack channels entirely; that must not empty the training set."""
    f = LSTMForecaster()
    series = synthetic_series()
    series[:, 3] = np.nan          # no ammonia probe at this site
    mask = np.isfinite(series)
    X, Y, M = f.make_windows(series, mask, stride=8)
    assert len(X) > 0
    assert not M[:, :, 3].any(), "the absent channel must be masked out of the loss"
    assert np.isfinite(X).all()


def test_a_window_without_an_anchor_reading_is_skipped():
    f = LSTMForecaster()
    series = synthetic_series()
    series[:, 0] = np.nan          # no DO at all: DO is an anchor channel
    X, _, _ = f.make_windows(series, np.isfinite(series), stride=8)
    assert len(X) == 0


def test_prediction_shape_and_finiteness():
    f = LSTMForecaster()
    out = f.predict_array(synthetic_series(n=f.window))
    assert out.shape == (f.horizon, len(f.channels))
    assert np.isfinite(out).all()


def test_training_reduces_the_loss():
    f = LSTMForecaster()
    X, Y, M = f.make_windows(synthetic_series(n=2000), stride=2)
    before = f.evaluate(X, Y, M)["rmse_do"]
    f.fit(X, Y, M, epochs=6)
    after = f.evaluate(X, Y, M)["rmse_do"]
    assert after < before


def test_weights_round_trip():
    a, b = LSTMForecaster(), LSTMForecaster(seed=999)
    w = a.get_weights()
    b.set_weights(w)
    x = synthetic_series(n=a.window)
    assert np.allclose(a.predict_array(x), b.predict_array(x))


def test_save_and_load(tmp_path):
    f = LSTMForecaster()
    x = synthetic_series(n=f.window)
    path = f.save(tmp_path / "fc.pt")
    g = LSTMForecaster.load(path)
    assert np.allclose(f.predict_array(x), g.predict_array(x))
    assert g.channels == f.channels and g.horizon == f.horizon


def test_forecaster_is_deterministic():
    a = LSTMForecaster(seed=5)
    b = LSTMForecaster(seed=5)
    X, Y, M = a.make_windows(synthetic_series(n=1200), stride=4)
    a.fit(X, Y, M, epochs=2)
    b.fit(X, Y, M, epochs=2)
    x = synthetic_series(n=a.window, seed=3)
    assert np.allclose(a.predict_array(x), b.predict_array(x))


# ---------------------------------------------------------------------------- regimes
FULL_CHANNELS = ["do", "temp_c", "ph", "tan", "no2", "turbidity"]


def regime_series(model, n=1500, seed=0, temp_mean=28.0, do_scale=1.0):
    """A series ordered to match ``model.channels``, not the forecaster's channel list.

    The regime model deliberately uses a SUBSET of the forecaster's channels, so feeding it a
    six-column array would silently read nitrite as turbidity - a test that passes for the
    wrong reason.
    """
    full = synthetic_series(n=n, seed=seed)
    full[:, 0] *= do_scale
    full[:, 1] += temp_mean - 28.0
    idx = [FULL_CHANNELS.index(c) for c in model.channels]
    return full[:, idx]


def test_regime_model_clusters_and_assigns():
    model = RegimeModel()
    feats = np.concatenate(
        [model.featurize_series(regime_series(model, n=1500, seed=s), stride=3) for s in range(4)]
    )
    model.fit(feats)
    assert model.metrics_["n_clusters"] >= 1
    assert 0.0 <= model.metrics_["noise_fraction"] <= 1.0
    label, name, anomaly = model.assign_features(feats[0])
    assert isinstance(name, str) and isinstance(anomaly, bool)


def test_regime_features_have_the_declared_width():
    model = RegimeModel()
    f = model.featurize_window(regime_series(model, n=model.window))
    assert len(f) == len(model.feature_names())
    assert len(model.feature_names()) == len(model.features) * len(model.derived)


def test_derived_features_are_dimensionless_across_domains():
    """The design decision under test: a cold river and a warm pond at the same saturation
    must land in the same place, or every pond window is an anomaly against every river."""
    model = RegimeModel()
    river = regime_series(model, n=model.window, seed=1, temp_mean=12.0, do_scale=1.30)
    pond = regime_series(model, n=model.window, seed=1, temp_mean=28.0, do_scale=0.92)

    dr, dp = model.derive(river), model.derive(pond)
    i_sat = model.derived.index("do_sat")
    assert abs(np.nanmean(dr[:, i_sat]) - np.nanmean(dp[:, i_sat])) < 0.25,         "DO saturation must be comparable across a 16 C temperature gap"

    i_temp = model.derived.index("temp_anom")
    assert abs(np.nanmean(dr[:, i_temp])) < 1e-9, "temp_anom is a within-window anomaly"
    assert abs(np.nanmean(dp[:, i_temp])) < 1e-9

    # And the raw channels, for contrast, are nowhere near each other.
    assert abs(np.nanmean(river[:, model.channels.index("temp_c")])
               - np.nanmean(pond[:, model.channels.index("temp_c")])) > 10.0


def test_constant_derived_features_are_dropped_before_clustering():
    """temp_anom_mean is exactly zero by construction and must not occupy a dimension."""
    model = RegimeModel()
    feats = np.concatenate(
        [model.featurize_series(regime_series(model, n=1500, seed=s), stride=3) for s in range(3)]
    )
    model.fit(feats)
    assert model.keep_ is not None
    assert not model.keep_.all(), "at least the anomaly-channel means should be dropped"
    assert model.center_.shape[0] == int(model.keep_.sum())


def test_regime_model_round_trips(tmp_path):
    model = RegimeModel()
    feats = np.concatenate(
        [model.featurize_series(regime_series(model, n=1500, seed=s), stride=3) for s in range(3)]
    )
    model.fit(feats)
    path = model.save(tmp_path / "regimes.pkl")
    loaded = RegimeModel.load(path)
    assert loaded.derived == model.derived
    assert loaded.assign_features(feats[0]) == model.assign_features(feats[0])


# ---------------------------------------------------------------------------- risk bands
def test_risk_bands_classify_the_obvious_cases():
    assert RB.state_risk({"do": 1.0})[0] == "critical"
    assert RB.state_risk({"do": 3.0})[0] == "warning"
    assert RB.state_risk({"do": 7.0, "ph": 7.5, "nh3": 0.001, "no2": 0.1, "temp_c": 28})[0] == "safe"
    assert RB.state_risk({"nh3": 0.08})[0] == "critical"
    assert RB.state_risk({"ph": 10.0})[0] == "critical"


def test_worst_channel_wins():
    level, per = RB.state_risk({"do": 7.0, "nh3": 0.09})
    assert level == "critical" and per["nh3"] == "critical" and per["do"] == "safe"


def test_chloride_relaxes_the_nitrite_band_but_not_without_limit():
    plain = RB.state_risk({"no2": 1.0})[0]
    salted = RB.state_risk({"no2": 1.0}, chloride=200.0)[0]
    assert plain == "warning"
    assert salted == "safe"
    assert RB.state_risk({"no2": 8.0}, chloride=200.0)[0] == "critical"


def test_forecast_risk_uses_the_hurtful_extreme():
    assert RB.worst_forecast_risk({"fc_do_min": 1.5}) == "critical"
    assert RB.worst_forecast_risk({"fc_do_min": 7.0}) == "safe"
    assert RB.worst_forecast_risk({"fc_nh3_max": 0.06}) == "critical"


# ---------------------------------------------------------------------------- risk model
def test_risk_model_learns_the_rule():
    rows, labels = [], []
    rng = np.random.default_rng(0)
    for _ in range(900):
        do = float(rng.uniform(0.5, 12))
        fc_do = do + float(rng.normal(0, 0.6))
        nh3 = float(rng.uniform(0, 0.09))
        observed = {"do": do, "ph": 7.6, "temp_c": 28.0, "tan": 0.4, "nh3": nh3, "no2": 0.1,
                    "turbidity": 20.0}
        summary = {"fc_do_min": fc_do, "fc_do_max": fc_do + 1, "fc_nh3_max": nh3,
                   "fc_no2_max": 0.1, "fc_temp_c_max": 28.0, "fc_ph_min": 7.5, "fc_ph_max": 7.7}
        m = RiskModel()
        rows.append(m.features(observed, summary))
        labels.append(m.label_from_forecast(summary, observed))

    model = RiskModel()
    n = int(len(rows) * 0.7)
    model.fit(rows[:n], labels[:n])
    metrics = model.evaluate(rows[n:], labels[n:])
    assert metrics["balanced_accuracy"] > 0.8
    cls, score = model.predict({"do": 1.0, "nh3": 0.001}, {"fc_do_min": 1.0})
    assert cls == "critical" and score > 0.5


def test_risk_model_falls_back_to_the_rule_when_unfitted():
    m = RiskModel()
    cls, score = m.predict({"do": 1.0}, {"fc_do_min": 1.0})
    assert cls == "critical"


# ---------------------------------------------------------------------------- fedavg
def test_fedavg_is_the_plain_mean_when_unweighted():
    a = {"w": np.array([0.0, 2.0], dtype=np.float32)}
    b = {"w": np.array([2.0, 4.0], dtype=np.float32)}
    out = fedavg([a, b])
    assert np.allclose(out["w"], [1.0, 3.0])


def test_fedavg_weights_by_sample_count():
    a = {"w": np.array([0.0], dtype=np.float32)}
    b = {"w": np.array([10.0], dtype=np.float32)}
    out = fedavg([a, b], [1, 9])
    assert np.allclose(out["w"], [9.0])


def test_fedavg_preserves_dtype_and_shape():
    w = {"lstm.weight": np.random.randn(8, 4).astype(np.float32)}
    out = fedavg([w, w, w], [1, 1, 1])
    assert out["lstm.weight"].shape == (8, 4)
    assert out["lstm.weight"].dtype == np.float32
    assert np.allclose(out["lstm.weight"], w["lstm.weight"], atol=1e-6)


def test_dp_clipping_bounds_the_update_norm():
    glob = {"w": np.zeros(4, dtype=np.float32)}
    rogue = {"w": np.array([100.0, 0, 0, 0], dtype=np.float32)}
    out = fedavg([rogue], [1], dp_cfg={"enabled": True, "clip_norm": 1.0, "noise_multiplier": 0.0},
                 global_weights=glob, rng=np.random.default_rng(0))
    assert np.linalg.norm(out["w"]) <= 1.01, "a single client must not be able to dominate"
