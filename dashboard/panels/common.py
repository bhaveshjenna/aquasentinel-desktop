"""Shared plotting helpers: risk-band shading, the colour vocabulary, small formatters."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

# One colour vocabulary for the whole dashboard, so "amber" means the same thing everywhere.
COLORS = {
    "safe": "#1a9e5c",
    "warning": "#d98324",
    "critical": "#c1372c",
    "agent": "#2166ac",
    "shadow": "#8a8a8a",
    "forecast": "#7b3fa0",
    "grid": "rgba(128,128,128,0.18)",
}
BAND_FILL = {
    "critical": "rgba(193,55,44,0.13)",
    "warning": "rgba(217,131,36,0.11)",
    "healthy": "rgba(26,158,92,0.08)",
}

CHANNEL_LABEL = {
    "do": "Dissolved oxygen (mg/L)",
    "temp_c": "Temperature (°C)",
    "ph": "pH",
    "tan": "Total ammonia N (mg/L)",
    "nh3": "Un-ionised NH₃-N (mg/L)",
    "no2": "Nitrite (mg/L)",
    "turbidity": "Turbidity (NTU)",
    "conductivity": "Conductivity (µS/cm)",
}


def label(channel: str) -> str:
    return CHANNEL_LABEL.get(channel, channel)


def y_range(series: np.ndarray, channel: str, bands: Dict[str, Any]):
    """Keep the relevant thresholds in frame even when the pond is comfortably safe.

    Every panel that calls ``add_risk_bands`` must also set this. The bands are drawn as
    ``add_hrect`` shapes spanning +/-1e3 so they reach the edge of any sensible axis, and
    plotly folds shape extents into its autorange - so a panel that shades bands and then
    leaves the y-axis to autoscale renders every trace as a flat line at the bottom of a
    2000-unit window. Passing an explicit range is what stops that.
    """
    finite = series[np.isfinite(series)]
    if len(finite) == 0:
        return None
    lo, hi = float(np.min(finite)), float(np.max(finite))
    spec = bands.get(channel, {})
    for key in ("critical_below", "warning_below"):
        if spec.get(key) is not None:
            lo = min(lo, float(spec[key]))
    for key in ("critical_above", "warning_above", "supersaturation_above"):
        if spec.get(key) is not None:
            hi = max(hi, float(spec[key]) * 1.02)
    pad = max((hi - lo) * 0.08, 1e-6)
    return [lo - pad, hi + pad]


def add_risk_bands(fig: go.Figure, channel: str, bands: Dict[str, Any], row=None, col=None) -> None:
    """Shade the safe / warning / critical regions behind a channel trace.

    The bands are the *same* dict the risk model and the guardrails read, so what the operator
    sees shaded is exactly what the agent is scored against - there is no second definition of
    "critical" living in the presentation layer.
    """
    spec = bands.get(channel)
    if not spec:
        return
    kw = {"row": row, "col": col} if row is not None else {}
    lo_c, hi_c = spec.get("critical_below"), spec.get("critical_above")
    lo_w, hi_w = spec.get("warning_below"), spec.get("warning_above")
    healthy = spec.get("healthy")
    sup = spec.get("supersaturation_above")

    def band(y0, y1, kind):
        if y0 is None or y1 is None or y1 <= y0:
            return
        fig.add_hrect(y0=y0, y1=y1, fillcolor=BAND_FILL[kind], line_width=0, layer="below", **kw)

    floor = -1e3
    ceil = 1e3
    if lo_c is not None:
        band(floor, lo_c, "critical")
    if hi_c is not None:
        band(hi_c, ceil, "critical")
    if lo_w is not None:
        band(lo_c if lo_c is not None else floor, lo_w, "warning")
    if hi_w is not None:
        band(hi_w, hi_c if hi_c is not None else ceil, "warning")
    if healthy:
        band(healthy[0], healthy[1], "healthy")
        # Only shade outside the healthy window where no explicit band already covers it,
        # otherwise the warning fill is drawn twice and reads as a darker, false severity.
        if lo_w is None and lo_c is None:
            band(floor, healthy[0], "warning")
        if hi_w is None:
            band(healthy[1], sup if sup is not None else ceil, "warning")
    if sup is not None:
        band(sup, ceil, "warning")


def base_layout(fig: go.Figure, height: int = 300, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=52, r=16, t=34 if title else 10, b=34),
        title=dict(text=title, font=dict(size=14)) if title else None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


def risk_color(level: str) -> str:
    return COLORS.get(level, COLORS["safe"])


def fmt_hours(h: float) -> str:
    if not np.isfinite(h):
        return "-"
    if h < 1:
        return f"{h * 60:.0f} min"
    return f"{h:.1f} h"


def clock(t_hours: float, start_hour: float) -> str:
    hod = (start_hour + t_hours) % 24.0
    day = int((start_hour + t_hours) // 24) + 1
    return f"day {day}  {int(hod):02d}:{int((hod % 1) * 60):02d}"
