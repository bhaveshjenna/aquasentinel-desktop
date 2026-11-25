"""Collector and harmonisation: schema, mask correctness, QC. No network in any test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.runtime import load_config
from src.common.schema import SENSOR_CHANNELS, Action, Observation, State
from src.data.harmonize import (HARMONIZED_CHANNELS, channels_present, derive_nh3,
                                harmonize_site, missing_fraction)


@pytest.fixture
def qc_cfg():
    return load_config("data")["qc"]


def _long(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "channel", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_harmonize_produces_the_full_channel_schema(qc_cfg):
    ts = pd.date_range("2024-03-01", periods=200, freq="15min", tz="UTC")
    rows = [(t, "do", 8.0 + 0.1 * i) for i, t in enumerate(ts)]
    values, mask, report = harmonize_site(_long(rows), 15, qc_cfg)
    assert list(values.columns) == list(HARMONIZED_CHANNELS)
    assert list(mask.columns) == list(HARMONIZED_CHANNELS)
    assert report["rows"] == len(ts)


def test_absent_channels_are_all_nan_and_masked_false(qc_cfg):
    ts = pd.date_range("2024-03-01", periods=100, freq="15min", tz="UTC")
    # Varying, not constant: a constant series is (correctly) treated as a stuck probe.
    rows = [(t, "do", 7.0 + 0.01 * i) for i, t in enumerate(ts)]
    values, mask, _ = harmonize_site(_long(rows), 15, qc_cfg)
    assert values["tan"].isna().all()
    assert not mask["tan"].any()
    assert mask["do"].all()
    assert channels_present(mask) == ["do"]


def test_impossible_values_are_dropped(qc_cfg):
    ts = pd.date_range("2024-03-01", periods=60, freq="15min", tz="UTC")
    rows = [(t, "do", 7.0) for t in ts]
    rows[10] = (ts[10], "do", 99.0)        # far outside the configured range
    rows[20] = (ts[20], "ph", 25.0)
    values, mask, report = harmonize_site(_long(rows), 15, qc_cfg)
    assert not mask["do"].iloc[10]
    assert report["impossible_dropped"] >= 1


def test_detection_limit_zeros_are_censored_not_believed(qc_cfg):
    ts = pd.date_range("2024-03-01", periods=60, freq="15min", tz="UTC")
    rows = [(t, "do", 0.0 if i in (5, 6) else 7.0) for i, t in enumerate(ts)]
    values, mask, report = harmonize_site(_long(rows), 15, qc_cfg)
    assert not mask["do"].iloc[5]
    assert report["detection_limit_censored"] == 2


def test_a_flatlined_probe_is_masked_out(qc_cfg):
    """Hours of an identical value is a stuck sensor, not a calm pond."""
    n_flat = int(qc_cfg["flatline_steps"]) + 10
    ts = pd.date_range("2024-03-01", periods=n_flat + 60, freq="15min", tz="UTC")
    rows = [(t, "do", 7.0 if i < n_flat else 7.0 + 0.01 * i) for i, t in enumerate(ts)]
    values, mask, report = harmonize_site(_long(rows), 15, qc_cfg)
    assert report["flatline_masked"] >= n_flat
    assert not mask["do"].iloc[: n_flat // 2].any()


def test_mask_records_observations_not_fills(qc_cfg):
    """Forward fill must never be reported as a real reading - the mask is the contract."""
    ts = pd.date_range("2024-03-01", periods=40, freq="15min", tz="UTC")
    rows = [(t, "do", 7.0 + 0.05 * i) for i, t in enumerate(ts) if i not in (10, 11)]
    values, mask, _ = harmonize_site(_long(rows), 15, qc_cfg)
    assert not mask["do"].iloc[10]
    assert np.isfinite(values["do"].iloc[10]), "the gap should be filled ..."
    assert not mask["do"].iloc[10], "... but not claimed as an observation"


def test_missing_fraction_matches_the_mask(qc_cfg):
    ts = pd.date_range("2024-03-01", periods=100, freq="15min", tz="UTC")
    rows = [(t, "do", 7.0) for i, t in enumerate(ts) if i % 2 == 0]
    _, mask, _ = harmonize_site(_long(rows), 15, qc_cfg)
    assert 0.4 < missing_fraction(mask, ["do"]) < 0.6


def test_duplicate_timestamps_are_averaged(qc_cfg):
    ts = pd.date_range("2024-03-01", periods=20, freq="15min", tz="UTC")
    rows = [(t, "do", 6.0) for t in ts] + [(t, "do", 8.0) for t in ts]
    values, _, _ = harmonize_site(_long(rows), 15, qc_cfg)
    assert values["do"].dropna().round(3).eq(7.0).all()


def test_empty_input_is_handled():
    empty = pd.DataFrame(columns=["timestamp", "channel", "value"])
    values, mask, report = harmonize_site(empty, 15, load_config("data")["qc"])
    assert values.empty and report["raw_rows"] == 0


# ------------------------------------------------------------------------- chemistry
def test_derive_nh3_matches_the_standard_relation():
    nh3 = derive_nh3(np.array([1.0]), np.array([7.0]), np.array([25.0]))
    assert 0.001 < nh3[0] < 0.02
    hot = derive_nh3(np.array([1.0]), np.array([9.0]), np.array([30.0]))
    assert hot[0] > nh3[0] * 10, "pH 9 at 30 C is dramatically more toxic than pH 7 at 25 C"


def test_derive_nh3_propagates_missing_values():
    out = derive_nh3(np.array([np.nan]), np.array([7.5]), np.array([28.0]))
    assert np.isnan(out[0])


# ------------------------------------------------------------------------- schema
def test_observation_round_trips_from_state():
    s = State(t_hours=1.0, do=7.0, temp_c=28.0, ph=7.5, tan=0.3, no2=0.05, turbidity=20.0)
    o = Observation.from_state(s)
    assert set(o.values) == set(SENSOR_CHANNELS)
    assert o.mask["do"] is True
    assert o.mask["conductivity"] is False      # not set on the State
    assert o.vector(["do", "ph"]).tolist() == [7.0, 7.5]


def test_action_labels_and_no_op():
    assert Action().is_no_op()
    assert Action().label() == "no_op"
    assert "aerator" in Action(aerator=0.5).label()
    assert not Action(feed="skip").is_no_op()
