"""Map each source's parameters onto the common channel schema, resample, QC, mask.

Every source lands here as a long frame ``[timestamp, channel, value]`` and leaves as a wide,
regularly-sampled frame plus a per-channel availability mask. Real sites differ in which
channels exist and when the probe was working; that missingness is *kept*, not papered over -
it is what makes the federated clients genuinely non-IID.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..common.schema import SENSOR_CHANNELS

# Channels we carry through harmonisation. `no3` / `chlorophyll` / `bga` exist only on some
# EA stations; they are kept in the parquet but are not part of the modelled channel set.
HARMONIZED_CHANNELS: Tuple[str, ...] = SENSOR_CHANNELS + ("no3", "chlorophyll", "bga")


def _drop_impossible(df: pd.DataFrame, ranges: Dict[str, List[float]]) -> Tuple[pd.DataFrame, int]:
    dropped = 0
    for channel, (lo, hi) in ranges.items():
        if channel not in df.columns:
            continue
        bad = df[channel].notna() & ((df[channel] < lo) | (df[channel] > hi))
        dropped += int(bad.sum())
        df.loc[bad, channel] = np.nan
    return df, dropped


def _flag_flatlines(df: pd.DataFrame, min_steps: int) -> Tuple[pd.DataFrame, int]:
    """A probe stuck on one value for hours is broken, not calm. Mask those runs out."""
    flagged = 0
    for channel in df.columns:
        col = df[channel]
        if col.notna().sum() == 0:
            continue
        # Identify runs of identical consecutive values (ignoring NaN gaps).
        run_id = (col != col.shift()).cumsum()
        sizes = col.groupby(run_id).transform("size")
        stuck = col.notna() & (sizes >= min_steps)
        flagged += int(stuck.sum())
        df.loc[stuck, channel] = np.nan
    return df, flagged


def _censor_detection_limit(df: pd.DataFrame, channels: List[str]) -> Tuple[pd.DataFrame, int]:
    """Exact 0.0 in DO or turbidity is a censored/at-detection-limit reading, not a value."""
    censored = 0
    for channel in channels:
        if channel not in df.columns:
            continue
        zeros = df[channel] == 0.0
        censored += int(zeros.sum())
        df.loc[zeros, channel] = np.nan
    return df, censored


def harmonize_site(
    long_df: pd.DataFrame,
    cadence_minutes: int,
    qc_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Long frame -> (wide values, boolean mask, QC report).

    The mask is True where the value is a genuine observation, False where it is missing or
    was produced by short-gap forward fill. Downstream code uses it as the channel-availability
    mask the spec calls for.
    """
    report: Dict[str, Any] = {"raw_rows": int(len(long_df))}
    if long_df.empty:
        empty = pd.DataFrame(columns=list(HARMONIZED_CHANNELS))
        return empty, empty.astype(bool), report

    df = long_df.copy()
    df = df[df["channel"].isin(HARMONIZED_CHANNELS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        empty = pd.DataFrame(columns=list(HARMONIZED_CHANNELS))
        return empty, empty.astype(bool), report

    # Duplicate timestamps happen when a source publishes several sensors per parameter.
    wide = (
        df.groupby(["timestamp", "channel"])["value"]
        .mean()
        .unstack("channel")
        .sort_index()
    )

    rule = f"{int(cadence_minutes)}min"
    wide = wide.resample(rule).mean()

    wide, n_impossible = _drop_impossible(wide, qc_cfg.get("ranges", {}))
    wide, n_censored = _censor_detection_limit(wide, qc_cfg.get("detection_limit_zero", []))
    wide, n_flat = _flag_flatlines(wide, int(qc_cfg.get("flatline_steps", 48)))

    # The mask records real observations; fill happens *after* the mask is taken.
    mask = wide.notna()
    filled = wide.ffill(limit=int(qc_cfg.get("max_gap_fill_steps", 4)))

    for channel in HARMONIZED_CHANNELS:
        if channel not in filled.columns:
            filled[channel] = np.nan
            mask[channel] = False
    filled = filled[list(HARMONIZED_CHANNELS)]
    mask = mask[list(HARMONIZED_CHANNELS)]

    report.update(
        {
            "rows": int(len(filled)),
            "impossible_dropped": n_impossible,
            "detection_limit_censored": n_censored,
            "flatline_masked": n_flat,
            "start": filled.index.min().isoformat() if len(filled) else None,
            "end": filled.index.max().isoformat() if len(filled) else None,
        }
    )
    return filled, mask, report


def channels_present(mask: pd.DataFrame, min_frac: float = 0.05) -> List[str]:
    """Channels a site genuinely reports (present in at least `min_frac` of samples)."""
    if mask.empty:
        return []
    frac = mask.mean(axis=0)
    return sorted([c for c, f in frac.items() if f >= min_frac])


def missing_fraction(mask: pd.DataFrame, channels: List[str]) -> float:
    if mask.empty or not channels:
        return 1.0
    return float(1.0 - mask[channels].to_numpy().mean())


def derive_nh3(tan: np.ndarray, ph: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
    """Un-ionised NH3-N from total ammonia N, pH and temperature.

    Emerson et al. (1975) pKa relation, as used throughout the aquaculture literature:
        pKa = 0.09018 + 2729.92 / T_K
        f   = 1 / (1 + 10**(pKa - pH))
    The toxic fraction is what the risk bands are written against.
    """
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    pka = 0.09018 + 2729.92 / np.where(t_k > 0, t_k, np.nan)
    frac = 1.0 / (1.0 + np.power(10.0, pka - np.asarray(ph, dtype=float)))
    return np.asarray(tan, dtype=float) * frac
