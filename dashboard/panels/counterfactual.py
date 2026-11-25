"""Counterfactual panel - the money shot.

A shadow pond was forked at t=0 with the same seed, the same scenario schedule and the same
noise draws, and has been running with the agent disabled ever since. Every gap between the
two lines is caused by an action and nothing else. This is the only comparison in the project
that isolates the agent's contribution exactly, which is why it gets the most screen space.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .common import COLORS, add_risk_bands, base_layout, label, y_range


def render(
    live,
    shadow,
    bands: Optional[Dict[str, Any]] = None,
    channels: Optional[List[str]] = None,
    height: int = 300,
) -> None:
    channels = channels or ["do", "nh3"]
    bands = bands or {}
    if live is None or len(live) == 0 or shadow is None or len(shadow) == 0:
        st.info("The counterfactual appears once the run starts.")
        return

    n = min(len(live), len(shadow))
    t = live["t_hours"].to_numpy()[:n]

    cols = st.columns(len(channels))
    for i, ch in enumerate(channels):
        if ch not in live.columns:
            continue
        with cols[i]:
            fig = go.Figure()
            add_risk_bands(fig, ch, bands)
            y_live = live[ch].to_numpy()[:n]
            y_shadow = shadow[ch].to_numpy()[:n]

            fig.add_trace(
                go.Scatter(x=t, y=y_shadow, name="no action (shadow pond)", mode="lines",
                           line=dict(color=COLORS["shadow"], width=2, dash="dash"))
            )
            fig.add_trace(
                go.Scatter(x=t, y=y_live, name="with agent", mode="lines",
                           line=dict(color=COLORS["agent"], width=2.4), fill="tonexty",
                           fillcolor="rgba(33,102,172,0.13)")
            )
            # Both ponds share one axis, so the range has to span both traces - and it has
            # to be set at all, or the risk-band rects flatten the plot (see `y_range`).
            fig.update_yaxes(range=y_range(np.concatenate([y_live, y_shadow]), ch, bands))
            fig.update_xaxes(title="hours")
            base_layout(fig, height=height, title=label(ch))
            st.plotly_chart(fig, width="stretch", key=f"cf_{ch}")

    # A single honest number for the headline channel.
    if "do" in live.columns:
        gap = float(np.nanmin(live["do"].to_numpy()[:n])) - float(np.nanmin(shadow["do"].to_numpy()[:n]))
        st.caption(
            f"Minimum DO with the agent is **{gap:+.2f} mg/L** relative to the identical "
            f"no-action pond over the same {t[-1]:.1f} simulated hours "
            f"(same seed, same perturbations, same sensor noise)."
        )
