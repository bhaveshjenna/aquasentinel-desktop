"""Running scorecard: time-in-risk, interventions, unsafe-applied, agent vs. no-action.

``unsafe applied`` is the number that has to stay at zero. It is displayed even when nothing
is wrong, because a safety claim that only appears when it is violated is not a safety claim.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from .common import COLORS, fmt_hours


def render(metrics: Dict[str, Any], shadow_metrics: Optional[Dict[str, Any]] = None) -> None:
    c = st.columns(5)

    risk_h = metrics.get("time_in_risk_h", 0.0)
    shadow_h = metrics.get("shadow_time_in_risk_h")
    delta = None
    if shadow_h is not None and shadow_h > 0:
        delta = f"{risk_h - shadow_h:+.1f} h vs no action"
    c[0].metric("Time in risk", fmt_hours(risk_h), delta, delta_color="inverse")

    crit = metrics.get("time_critical_h", 0.0)
    c[1].metric("Time critical", fmt_hours(crit))

    c[2].metric("Interventions", int(metrics.get("interventions", 0)),
                f"cost {metrics.get('intervention_cost', 0):.0f}")

    unsafe = int(metrics.get("unsafe_applied", 0))
    c[3].metric("Unsafe applied", unsafe, "target 0", delta_color="off")
    if unsafe > 0:
        c[3].markdown(
            f"<span style='color:{COLORS['critical']};font-size:12px'>guardrail breach</span>",
            unsafe_allow_html=True,
        )

    c[4].metric("Guardrail events", int(metrics.get("guardrail_events", 0)),
                f"min DO {metrics.get('min_do', float('nan')):.2f}")

    per_channel = {k[len("risk_h_"):]: v for k, v in metrics.items() if k.startswith("risk_h_")}
    if per_channel:
        parts = [f"{k}: {fmt_hours(v)}" for k, v in sorted(per_channel.items(), key=lambda kv: -kv[1]) if v > 0]
        st.caption(
            "Time out of band, per channel — "
            + (", ".join(parts) if parts else "every channel inside its band")
            + ". Aggregated time-in-risk mixes channels the agent can act on with ones it cannot."
        )
