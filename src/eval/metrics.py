"""Metrics and the statistics that turn them into claims.

The rule this module enforces: **never report a bare mean**. Every controller comparison is a
paired Wilcoxon signed-rank test across seeds - paired because the same seed gives every
controller the identical perturbation schedule and noise realisation, so the pairing is real
and the test is the right one. ``p`` and ``n`` are reported alongside the effect, always, and
a comparison with too few seeds to be tested says so rather than quietly printing a mean.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MIN_N_FOR_TEST = 6  # below this, Wilcoxon has no power worth reporting


# ======================================================================================
# Forecasting
# ======================================================================================
def forecast_skill(rmse_model: float, rmse_baseline: float) -> float:
    """1 - RMSE_model / RMSE_baseline. Positive means better than the baseline."""
    if not np.isfinite(rmse_baseline) or rmse_baseline <= 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_baseline)


def per_horizon_rmse(pred: np.ndarray, true: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """RMSE at each step ahead - the curve that shows where a forecast stops being useful."""
    err = pred - true
    if mask is None:
        return np.sqrt((err**2).mean(axis=(0, 2)))
    out = []
    for h in range(err.shape[1]):
        m = mask[:, h, :]
        out.append(float(np.sqrt((err[:, h, :][m] ** 2).mean())) if m.sum() else np.nan)
    return np.asarray(out)


# ======================================================================================
# Clustering
# ======================================================================================
def anomaly_recall(flags: Sequence[bool], truth_windows: Sequence[Tuple[float, float]],
                   times: Sequence[float], tolerance_h: float = 1.0) -> Dict[str, float]:
    """Did the anomaly flag fire during each injected fault window?

    Scored per *event*, not per tick: a fault lasting six hours that is flagged once is a
    detection, and counting ticks would let a detector that fires constantly look excellent.
    """
    flags = np.asarray(flags, dtype=bool)
    times = np.asarray(times, dtype=float)
    if len(truth_windows) == 0:
        return {"events": 0, "detected": 0, "recall": float("nan"),
                "false_positive_rate": float(flags.mean()) if len(flags) else float("nan")}

    detected = 0
    covered = np.zeros(len(times), dtype=bool)
    for start, end in truth_windows:
        window = (times >= start - tolerance_h) & (times <= end + tolerance_h)
        covered |= window
        if flags[window].any():
            detected += 1
    outside = flags[~covered]
    return {
        "events": len(truth_windows),
        "detected": detected,
        "recall": detected / len(truth_windows),
        "false_positive_rate": float(outside.mean()) if len(outside) else 0.0,
    }


# ======================================================================================
# Controller comparison
# ======================================================================================
def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank on (a - b). Honest about when it cannot be run."""
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(min(len(a), len(b)))
    a, b = a[:n], b[:n]
    diff = a - b
    nonzero = int((diff != 0).sum())

    out: Dict[str, Any] = {
        "n": n,
        "n_nonzero": nonzero,
        "median_a": float(np.median(a)) if n else np.nan,
        "median_b": float(np.median(b)) if n else np.nan,
        "median_diff": float(np.median(diff)) if n else np.nan,
        "mean_diff": float(np.mean(diff)) if n else np.nan,
    }
    if n < MIN_N_FOR_TEST:
        out.update({"p": np.nan, "statistic": np.nan,
                    "note": f"n={n} < {MIN_N_FOR_TEST}: too few seeds to test, effect reported only"})
        return out
    if nonzero == 0:
        out.update({"p": 1.0, "statistic": 0.0, "note": "all differences are exactly zero"})
        return out
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    out.update({
        "p": float(p),
        "statistic": float(stat),
        # Matched-pairs rank-biserial correlation: an effect size the test actually implies.
        "effect_size_rb": _rank_biserial(diff),
        "note": "",
    })
    return out


def _rank_biserial(diff: np.ndarray) -> float:
    from scipy import stats

    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    total = ranks.sum()
    return float((ranks[d < 0].sum() - ranks[d > 0].sum()) / total)


def compare_controllers(
    runs: pd.DataFrame,
    metric: str = "risk_h_do",
    pairs: Optional[List[Tuple[str, str]]] = None,
    by_scenario: bool = True,
) -> pd.DataFrame:
    """Paired comparison across seeds for each (scenario, controller-pair).

    Lower is better for every metric used here, so a negative median difference means the
    first controller wins.
    """
    from .baselines import COMPARISON_PAIRS

    pairs = pairs or COMPARISON_PAIRS
    rows: List[Dict[str, Any]] = []
    groups = runs.groupby("scenario") if by_scenario else [("all", runs)]

    for scenario, grp in groups:
        wide = grp.pivot_table(index="seed", columns="controller", values=metric)
        for a, b in pairs:
            if a not in wide.columns or b not in wide.columns:
                continue
            paired = wide[[a, b]].dropna()
            res = wilcoxon(paired[a].to_numpy(), paired[b].to_numpy())
            rows.append({
                "scenario": scenario,
                "metric": metric,
                "controller_a": a,
                "controller_b": b,
                "better": a if res["median_diff"] < 0 else (b if res["median_diff"] > 0 else "tie"),
                **res,
            })
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Median (not mean) per scenario x controller, with the IQR, so outliers cannot hide."""
    metrics = [c for c in runs.columns if c.startswith("risk_h_")] + [
        "time_in_risk_h", "time_critical_h", "interventions", "intervention_cost",
        "unsafe_applied", "min_do", "max_nh3",
    ]
    metrics = [m for m in metrics if m in runs.columns]
    agg = runs.groupby(["scenario", "controller"])[metrics].agg(["median", _iqr, "count"])
    agg.columns = [f"{m}_{s}" if s != "_iqr" else f"{m}_iqr" for m, s in agg.columns]
    return agg.reset_index()


def _iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) else np.nan


def safety_table(runs: pd.DataFrame) -> pd.DataFrame:
    """The safety claim, in one table: unsafe-applied per controller across the whole grid."""
    return (
        runs.groupby("controller")
        .agg(
            runs_n=("seed", "count"),
            unsafe_applied_total=("unsafe_applied", "sum"),
            unsafe_applied_max=("unsafe_applied", "max"),
            guardrail_events_total=("guardrail_events", "sum"),
            scenarios=("scenario", "nunique"),
        )
        .reset_index()
    )
