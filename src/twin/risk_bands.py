"""Rule-derived risk from the domain bands in ``twin.yaml``.

This is deliberately NOT learned. The bands come from warm-water pond guidance (FAO / Boyd)
and are calibratable per species. The LightGBM risk model in ``perception/risk.py`` is trained
to reproduce *this* function over the forecast horizon - which is why the project never claims
to have learned water chemistry, only to have learned to anticipate a known rule.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..common.runtime import load_config
from ..common.schema import RISK_LEVELS

_LEVEL_INDEX = {name: i for i, name in enumerate(RISK_LEVELS)}


def bands(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return (cfg or load_config("twin"))["risk_bands"]


def channel_risk(channel: str, value: float, b: Dict[str, Any]) -> str:
    """Risk level contributed by one channel: safe | warning | critical."""
    if value is None or not np.isfinite(value) or channel not in b:
        return "safe"
    spec = b[channel]
    lo_c, hi_c = spec.get("critical_below"), spec.get("critical_above")
    lo_w, hi_w = spec.get("warning_below"), spec.get("warning_above")

    if lo_c is not None and value < lo_c:
        return "critical"
    if hi_c is not None and value > hi_c:
        return "critical"
    if lo_w is not None and value < lo_w:
        return "warning"
    if hi_w is not None and value > hi_w:
        return "warning"

    healthy = spec.get("healthy")
    if healthy is not None and not (healthy[0] <= value <= healthy[1]):
        return "warning"
    sup = spec.get("supersaturation_above")
    if sup is not None and value > sup:
        return "warning"
    return "safe"


def state_risk(
    values: Dict[str, float],
    cfg: Optional[Dict[str, Any]] = None,
    chloride: float = 0.0,
) -> Tuple[str, Dict[str, str]]:
    """Worst-channel risk for one instant, plus the per-channel breakdown.

    Chloride from salt dosing genuinely protects against nitrite toxicity (the classic
    "brown blood disease" mitigation), so the NO2 band is relaxed in proportion to the
    chloride:nitrite ratio - capped, because salt is not a licence to ignore nitrite.
    """
    b = dict(bands(cfg))
    if chloride > 0 and "no2" in b:
        ratio = chloride / max(values.get("no2", 0.0) or 0.0, 1e-6)
        relief = float(np.clip(ratio / 20.0, 1.0, 3.0))  # up to 3x tolerance at Cl:NO2 >= 60
        b["no2"] = {k: (v * relief if isinstance(v, (int, float)) else v) for k, v in b["no2"].items()}

    per_channel = {c: channel_risk(c, values.get(c, np.nan), b) for c in b}
    worst = max(per_channel.values(), key=lambda lvl: _LEVEL_INDEX[lvl], default="safe")
    return worst, per_channel


def risk_score(level: str, cfg: Optional[Dict[str, Any]] = None) -> float:
    weights = load_config("models")["risk"]["score_weights"]
    return float(weights.get(level, 0.0))


def state_risk_score(values: Dict[str, float], cfg=None, chloride: float = 0.0) -> float:
    level, _ = state_risk(values, cfg, chloride)
    return risk_score(level, cfg)


def series_risk(frame, cfg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Risk level per row of a DataFrame with channel columns. Returns an object array."""
    b = bands(cfg)
    channels = [c for c in b if c in frame.columns]
    if not channels:
        return np.array(["safe"] * len(frame), dtype=object)
    levels = np.full(len(frame), 0, dtype=int)
    for c in channels:
        col = frame[c].to_numpy(dtype=float)
        for i, v in enumerate(col):
            levels[i] = max(levels[i], _LEVEL_INDEX[channel_risk(c, v, b)])
    return np.array([RISK_LEVELS[i] for i in levels], dtype=object)


def time_in_risk(frame, cfg: Optional[Dict[str, Any]] = None, step_hours: float = 5 / 60) -> Dict[str, float]:
    """Hours spent in each risk level - the agent's headline evaluation metric."""
    levels = series_risk(frame, cfg)
    return {
        lvl: float((levels == lvl).sum() * step_hours) for lvl in RISK_LEVELS
    }


def worst_forecast_risk(forecast_summary: Dict[str, float], cfg=None) -> str:
    """Risk implied by the forecast horizon: the worst any channel is predicted to reach.

    Uses the extreme of each channel in the direction that hurts - min DO, max NH3 - which is
    what makes the risk model *preemptive* rather than merely descriptive.
    """
    b = bands(cfg)
    worst = 0
    probe = {
        "do": forecast_summary.get("fc_do_min"),
        "ph": None,
        "nh3": forecast_summary.get("fc_nh3_max"),
        "no2": forecast_summary.get("fc_no2_max"),
        "temp_c": forecast_summary.get("fc_temp_c_max"),
    }
    for c, v in probe.items():
        if v is not None:
            worst = max(worst, _LEVEL_INDEX[channel_risk(c, v, b)])
    # pH can fail in either direction, so both tails are checked.
    for key in ("fc_ph_min", "fc_ph_max"):
        v = forecast_summary.get(key)
        if v is not None:
            worst = max(worst, _LEVEL_INDEX[channel_risk("ph", v, b)])
    return RISK_LEVELS[worst]


def time_in_risk_by_channel(frame, cfg=None, step_hours: float = 5 / 60) -> Dict[str, float]:
    """Hours each individual channel spent out of its safe band.

    Without this the aggregate is unreadable: a pond parked one degree above its temperature
    window shows the same "time in risk" as one drowning, and the agent gets no credit for
    fixing the half it can actually act on.
    """
    b = bands(cfg)
    out: Dict[str, float] = {}
    for c in b:
        if c not in frame.columns:
            continue
        col = frame[c].to_numpy(dtype=float)
        bad = sum(1 for v in col if channel_risk(c, v, b) != "safe")
        out[c] = float(bad * step_hours)
    return out
