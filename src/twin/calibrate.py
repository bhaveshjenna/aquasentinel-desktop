"""Fit the twin's *structural* defaults to statistics of the collected real data.

This is the bridge between Section 8's real monitoring data and the simulated pond. What it
is honest to transfer, and what it is not:

  TRANSFERRED  short-timescale process-noise magnitude per channel (Ornstein-Uhlenbeck
               sigma, fitted from lag-1 residuals after removing the hour-of-day climatology);
               baseline turbidity; the pH the water relaxes toward. Stochastic structure at
               the 15-minute scale is a property of turbulence and instrumentation, and those
               do carry across from a stream to a pond.

  NOT TRANSFERRED  the diurnal cycle - amplitude, phase and mean - and every biological rate
               constant and stocking parameter. The diurnal terms are the tempting ones and
               they are exactly the ones that must not move: a river's daily temperature swing
               is set by its depth, flow and thermal mass, and the measured median across our
               sites is ~0.5 C, against 2-4 C for a 1.5 m earthen pond. Transferring it would
               flatten the pond into an isothermal tank and delete the dawn oxygen sag the
               whole project is about. The fitted diurnal values are still computed and
               recorded in the provenance block so a reader can see them and judge; applying
               them requires --adopt-diurnal, and the temperature mean additionally --adopt-mean.

Every value written back to ``twin.yaml`` is recorded in a ``calibration:`` provenance block
with the site count and channel it came from, so nothing is silently transcribed.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..common.runtime import load_config, save_config
from ..data import registry as R

DIURNAL_CHANNELS = ("temp_c", "do", "ph")
NOISE_TARGETS = ("do", "temp_c", "ph", "tan", "no2", "turbidity")


def _hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy() + index.minute.to_numpy() / 60.0


def diurnal_fit(series: pd.Series) -> Optional[Dict[str, float]]:
    """Least-squares fit of ``mean + A*cos(2pi(h - peak)/24)`` to the hour-of-day climatology."""
    s = series.dropna()
    if len(s) < 500:
        return None
    hours = _hour_of_day(s.index)
    clim = pd.Series(s.to_numpy(), index=np.round(hours * 4) / 4).groupby(level=0).mean()
    h = clim.index.to_numpy(dtype=float)
    y = clim.to_numpy(dtype=float)
    if len(h) < 24 or not np.isfinite(y).all():
        return None
    # y = m + a*cos(w h) + b*sin(w h);  A = hypot(a, b), peak = atan2(b, a) / w
    w = 2.0 * np.pi / 24.0
    X = np.column_stack([np.ones_like(h), np.cos(w * h), np.sin(w * h)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    m, a, b = coef
    amp = float(np.hypot(a, b))
    peak = float((np.arctan2(b, a) / w) % 24.0)
    return {"mean": float(m), "amplitude": amp, "peak_hour": peak}


def ou_sigma(series: pd.Series, cadence_minutes: float) -> Optional[float]:
    """OU sigma in channel-units per sqrt(hour), from residuals about the daily climatology.

    Removing the hour-of-day climatology first matters: otherwise the diurnal signal is
    counted as noise and the twin ends up jittering instead of cycling.
    """
    s = series.dropna()
    if len(s) < 500:
        return None
    hours = np.round(_hour_of_day(s.index) * 4) / 4
    clim = pd.Series(s.to_numpy(), index=hours).groupby(level=0).mean()
    resid = s.to_numpy() - clim.reindex(hours).to_numpy()
    resid = resid[np.isfinite(resid)]
    if len(resid) < 100:
        return None
    increments = np.diff(resid)
    dt_h = cadence_minutes / 60.0
    sigma = float(np.nanstd(increments) / np.sqrt(dt_h))
    return sigma if np.isfinite(sigma) and sigma > 0 else None


def collect_statistics(max_sites: Optional[int] = None) -> Dict[str, Any]:
    """Per-channel statistics pooled across every cached real site."""
    sites: List[R.SiteDataset] = R.load_all()
    if max_sites:
        sites = sites[:max_sites]
    if not sites:
        raise RuntimeError("no cached sites; run scripts/collect_data.py first")

    diurnal: Dict[str, List[Dict[str, float]]] = {c: [] for c in DIURNAL_CHANNELS}
    sigmas: Dict[str, List[float]] = {c: [] for c in NOISE_TARGETS}
    levels: Dict[str, List[float]] = {c: [] for c in NOISE_TARGETS}

    for site in sites:
        for channel in set(DIURNAL_CHANNELS) | set(NOISE_TARGETS):
            if channel not in site.frame.columns:
                continue
            col = site.frame[channel].where(site.mask[channel]) if channel in site.mask else site.frame[channel]
            if col.notna().sum() < 500:
                continue
            if channel in DIURNAL_CHANNELS:
                fit = diurnal_fit(col)
                if fit:
                    diurnal[channel].append(fit)
            if channel in NOISE_TARGETS:
                sig = ou_sigma(col, site.cadence_minutes)
                if sig:
                    sigmas[channel].append(sig)
                levels[channel].append(float(col.median()))

    def med(xs):
        return float(np.median(xs)) if xs else None

    return {
        "n_sites": len(sites),
        "diurnal": {
            c: {
                "amplitude": med([f["amplitude"] for f in fits]),
                "peak_hour": med([f["peak_hour"] for f in fits]),
                "mean": med([f["mean"] for f in fits]),
                "n": len(fits),
            }
            for c, fits in diurnal.items()
            if fits
        },
        "ou_sigma": {c: med(v) for c, v in sigmas.items() if v},
        "median_level": {c: med(v) for c, v in levels.items() if v},
        "sigma_site_counts": {c: len(v) for c, v in sigmas.items() if v},
    }


def calibrate(
    stats: Optional[Dict[str, Any]] = None,
    adopt_mean: bool = False,
    adopt_diurnal: bool = False,
    write: bool = True,
) -> Dict[str, Any]:
    """Apply the transferable statistics to ``twin.yaml``; return the change log."""
    stats = stats or collect_statistics()
    cfg = dict(load_config("twin"))
    changes: Dict[str, Any] = {}

    def setval(path: str, value, note: str):
        if value is None or not np.isfinite(value):
            return
        node = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            node = node[p]
        changes[path] = {"from": node[parts[-1]], "to": round(float(value), 5), "source": note}
        node[parts[-1]] = round(float(value), 5)

    d = stats.get("diurnal", {})
    if "temp_c" in d and adopt_diurnal:
        setval("temperature.amplitude_c", d["temp_c"]["amplitude"],
               f"median diurnal amplitude across {d['temp_c']['n']} real sites - OPT-IN; "
               "a river's swing is not a pond's")
        setval("temperature.peak_hour", d["temp_c"]["peak_hour"],
               f"median diurnal phase across {d['temp_c']['n']} real sites - OPT-IN")
    if "temp_c" in d and adopt_mean:
        setval("temperature.mean_c", d["temp_c"]["mean"],
               "median real-site temperature - NOT a warm-water pond, opt-in only")

    for channel, sigma in (stats.get("ou_sigma") or {}).items():
        if channel in cfg["noise"]:
            n = stats["sigma_site_counts"].get(channel, 0)
            setval(f"noise.{channel}", sigma, f"OU sigma from lag-1 residuals, {n} real sites")

    lvl = stats.get("median_level", {})
    if lvl.get("ph") is not None:
        setval("ph.equilibrium", lvl["ph"], "median real-site pH")
    if lvl.get("turbidity") is not None:
        setval("turbidity.base_ntu", lvl["turbidity"], "median real-site turbidity")

    cfg["calibration"] = {
        "calibrated_at": dt.datetime.now().replace(microsecond=0).isoformat(),
        "n_real_sites": stats["n_sites"],
        "adopt_mean_temperature": bool(adopt_mean),
        "adopt_diurnal": bool(adopt_diurnal),
        "fitted_diurnal_not_applied": (
            {} if adopt_diurnal else {
                ch: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}
                for ch, f in d.items()
            }
        ),
        "note": (
            "Fitted from public river/lake monitoring data: per-channel process-noise "
            "magnitude, baseline turbidity, relaxation pH. NOT fitted: the diurnal cycle "
            "(amplitude/phase/mean - set by depth and thermal mass, see "
            "fitted_diurnal_not_applied for the measured values), biological rate constants, "
            "stocking density. Those remain warm-water-pond defaults from FAO/Boyd guidance. "
            "Rivers are not ponds."
        ),
        "changes": {k: v for k, v in changes.items()},
    }

    if write:
        save_config(
            "twin",
            cfg,
            header=(
                "# AquaSentinel pond digital twin.\n"
                "#\n"
                "# EVERY number here is a CALIBRATABLE DEFAULT, not a measured fact.\n"
                "# The `calibration:` block at the end records which values were fitted to real\n"
                "# monitoring data (src/twin/calibrate.py) and which remain domain defaults.\n"
                "# Calibrate against FAO pond-aquaculture guidance, Boyd 'Water Quality in Ponds\n"
                "# for Aquaculture', and APHA Standard Methods before treating output as physical."
            ),
        )
    return changes
