"""Risk gauge + SHAP drivers.

The gauge shows a probability-weighted severity, not a hard class, so the operator can see
risk building before it crosses a threshold - which is the entire argument for a preemptive
agent rather than a thermostat.

The SHAP bars answer "what is pushing this toward critical", the same question at every tick,
so the bars are comparable across time. They are attributions of the *model's* output, and
since the model's target is the rule in twin.yaml, they are also readable as attributions of
the rule. That equivalence is why this panel is allowed to look like an explanation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

from .common import COLORS, base_layout, risk_color


def render(risk_class: str, risk_score: float, drivers: Optional[List[Tuple[str, float]]] = None) -> None:
    left, right = st.columns([1, 1.7])

    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(risk_score),
                number=dict(font=dict(size=30), valueformat=".2f"),
                gauge=dict(
                    axis=dict(range=[0, 1], tickwidth=1, tickvals=[0, 0.5, 1]),
                    bar=dict(color=risk_color(risk_class), thickness=0.72),
                    steps=[
                        dict(range=[0, 0.34], color="rgba(26,158,92,0.16)"),
                        dict(range=[0.34, 0.67], color="rgba(217,131,36,0.18)"),
                        dict(range=[0.67, 1.0], color="rgba(193,55,44,0.18)"),
                    ],
                    borderwidth=0,
                ),
            )
        )
        fig.update_layout(height=190, margin=dict(l=10, r=10, t=8, b=4))
        st.plotly_chart(fig, width="stretch", key="risk_gauge")
        st.markdown(
            f"<div style='text-align:center;margin-top:-14px;font-weight:600;"
            f"color:{risk_color(risk_class)}'>{risk_class.upper()}</div>",
            unsafe_allow_html=True,
        )

    with right:
        if not drivers:
            st.caption("Risk drivers appear once the risk model and SHAP are loaded.")
            return
        names = [d[0] for d in drivers][::-1]
        vals = [d[1] for d in drivers][::-1]
        fig = go.Figure(
            go.Bar(
                x=vals, y=names, orientation="h",
                marker=dict(color=[COLORS["critical"] if v > 0 else COLORS["safe"] for v in vals]),
                hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_width=1, line_color="rgba(120,120,120,0.5)")
        fig.update_xaxes(title="SHAP contribution to CRITICAL")
        base_layout(fig, height=200, title="Why the risk is what it is")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch", key="shap_drivers")
