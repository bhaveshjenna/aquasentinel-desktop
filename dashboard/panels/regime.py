"""Regime + anomaly panel: which operating regime HDBSCAN puts this window in, and whether
it belongs to any cluster at all.

The **anomaly flag is the deliverable here**, not the regime name. Water-quality windows turn
out to form a continuum rather than separated clusters (DBCV ~ 0 across the whole sweep in
outputs/regime_sweep.csv), so the regime label is a coarse descriptive split and the panel
says so rather than implying a taxonomy the data does not support.

The anomaly flag means "this window resembles nothing in the training distribution" - which
covers a broken probe and a genuine novel excursion equally, and cannot tell them apart. The
sensor_fault scenario exists precisely to expose that limit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

from .common import COLORS, base_layout


def render(regime_name: str, regime_id: int, anomaly: bool, history: Optional[List[Dict]] = None) -> None:
    left, right = st.columns([1, 1.6])

    with left:
        if anomaly:
            st.markdown(
                f"<div style='padding:14px;border-radius:8px;background:{COLORS['critical']}22;"
                f"border-left:4px solid {COLORS['critical']}'>"
                f"<div style='font-size:12px;opacity:.75'>REGIME</div>"
                f"<div style='font-size:20px;font-weight:600'>Anomaly</div>"
                f"<div style='font-size:12px;opacity:.8;margin-top:6px'>This window matches no "
                f"learned regime. Could be a probe fault or a genuinely novel state - HDBSCAN "
                f"cannot tell you which.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='padding:14px;border-radius:8px;background:{COLORS['agent']}18;"
                f"border-left:4px solid {COLORS['agent']}'>"
                f"<div style='font-size:12px;opacity:.75'>REGIME #{regime_id}</div>"
                f"<div style='font-size:20px;font-weight:600'>{regime_name}</div>"
                f"<div style='font-size:12px;opacity:.8;margin-top:6px'>Coarse unsupervised "
                f"split, named after the fact from its centroid. No labels were used. The "
                f"clusters are weakly separated (DBCV ~ 0) - read this as a description, not "
                f"as a claim that the pond has discrete states.</div></div>",
                unsafe_allow_html=True,
            )

    with right:
        if not history:
            st.caption("Regime history builds up as the run proceeds.")
            return
        ts = [h["t_hours"] for h in history]
        ids = [h["regime"] for h in history]
        anom = [h["anomaly"] for h in history]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ts, y=ids, mode="markers", name="regime",
                marker=dict(
                    size=6,
                    color=[COLORS["critical"] if a else COLORS["agent"] for a in anom],
                ),
                hovertemplate="t=%{x:.1f} h<br>regime %{y}<extra></extra>",
            )
        )
        fig.update_yaxes(title="cluster (-1 = anomaly)", dtick=1)
        fig.update_xaxes(title="hours")
        base_layout(fig, height=190, title="Regime history")
        st.plotly_chart(fig, width="stretch", key="regime_hist")
