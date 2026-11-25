"""Live time-series panel: measured channel, LSTM forecast overlay, shaded risk bands."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .common import COLORS, add_risk_bands, base_layout, label, y_range

DEFAULT_CHANNELS = ["do", "ph", "nh3", "temp_c", "no2", "turbidity"]

#: How much history the live charts show. The run keeps every sample, but the x-axis has only
#: ever shown the last day, so shipping the whole trace every playback frame grows the payload
#: without changing the picture - and a payload that grows is how the browser ends up minutes
#: of simulated time behind the server.
WINDOW_H = 24.0


def render(
    truth,
    observed,
    forecast=None,
    bands: Optional[Dict[str, Any]] = None,
    channels: Optional[List[str]] = None,
    show_truth: bool = False,
    height: int = 240,
) -> None:
    """One chart per channel, stacked. Probe reading solid, forecast dashed ahead of `now`."""
    channels = channels or DEFAULT_CHANNELS
    bands = bands or {}
    if truth is None or len(truth) == 0:
        st.info("Press **Play** to start the pond.")
        return

    now = float(truth["t_hours"].to_numpy()[-1])
    lo = max(0.0, now - WINDOW_H)
    truth = truth[truth["t_hours"] >= lo]
    if observed is not None and len(observed):
        observed = observed[observed["t_hours"] >= lo]
    t = truth["t_hours"].to_numpy()

    cols = st.columns(2)
    for i, ch in enumerate(channels):
        with cols[i % 2]:
            fig = go.Figure()
            add_risk_bands(fig, ch, bands)

            # What the operator actually has: the probe, not the pond.
            if observed is not None and ch in observed.columns:
                fig.add_trace(
                    go.Scatter(
                        x=observed["t_hours"], y=observed[ch], name="probe",
                        mode="lines", line=dict(color=COLORS["agent"], width=1.8),
                        connectgaps=False,
                    )
                )
            elif ch in truth.columns:
                fig.add_trace(
                    go.Scatter(x=t, y=truth[ch], name="probe", mode="lines",
                               line=dict(color=COLORS["agent"], width=1.8))
                )

            # Ground truth is available in the simulation and nowhere else - it is opt-in, and
            # labelled as such, so nobody mistakes it for something a real farm would have.
            if show_truth and ch in truth.columns:
                fig.add_trace(
                    go.Scatter(x=t, y=truth[ch], name="twin truth", mode="lines",
                               line=dict(color=COLORS["shadow"], width=1.2, dash="dot"))
                )

            if forecast is not None and ch in forecast.channels:
                step_h = forecast.step_minutes / 60.0
                fx = now + np.arange(1, forecast.horizon + 1) * step_h
                fig.add_trace(
                    go.Scatter(x=fx, y=forecast.channel(ch), name="LSTM forecast",
                               mode="lines", line=dict(color=COLORS["forecast"], width=2, dash="dash"))
                )
            elif forecast is not None and ch == "nh3" and {"tan", "ph", "temp_c"} <= set(forecast.channels):
                from src.data.harmonize import derive_nh3

                step_h = forecast.step_minutes / 60.0
                fx = now + np.arange(1, forecast.horizon + 1) * step_h
                fy = derive_nh3(forecast.channel("tan"), forecast.channel("ph"), forecast.channel("temp_c"))
                fig.add_trace(
                    go.Scatter(x=fx, y=fy, name="LSTM forecast (derived)",
                               mode="lines", line=dict(color=COLORS["forecast"], width=2, dash="dash"))
                )

            fig.add_vline(x=now, line_width=1, line_dash="dot", line_color="rgba(120,120,120,0.6)")
            series = truth[ch].to_numpy() if ch in truth.columns else np.array([np.nan])
            fig.update_yaxes(range=y_range(series, ch, bands))
            fig.update_xaxes(title="hours", range=[lo, now + 3])
            base_layout(fig, height=height, title=label(ch))
            # The key must NOT vary with the tick. Streamlit identifies a component by its
            # key, so a key carrying the sample count makes every frame a *different* chart:
            # the frontend tears the plot down and calls newPlot instead of updating the
            # traces in place. Six of those per frame is what puts playback behind the sim.
            st.plotly_chart(fig, width="stretch", key=f"ts_{ch}")
