"""Agent panel: proposed action, rationale, guardrail verdict, applied action, and the log.

The proposed/applied split is the point. Anywhere they differ, a hard rule fired, and the rule
that fired is named. A demo that only showed the applied action would be hiding the one thing
this project is actually claiming.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import streamlit as st

from src.agent.actuators import describe
from src.common.schema import Decision

from .common import COLORS

SEVERITY_COLOR = {"block": COLORS["critical"], "clamp": COLORS["warning"], "override": COLORS["critical"]}


def render(decision: Optional[Decision], decisions: Optional[List[Decision]] = None) -> None:
    if decision is None:
        st.info("No decision yet.")
        return

    changed = decision.proposed != decision.applied
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Proposed**")
        st.markdown(
            f"<div style='padding:10px;border-radius:6px;background:rgba(128,128,128,.10)'>"
            f"{describe(decision.proposed)}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("**Applied**" + (" — modified by guardrails" if changed else ""))
        border = COLORS["warning"] if changed else COLORS["safe"]
        st.markdown(
            f"<div style='padding:10px;border-radius:6px;background:rgba(128,128,128,.10);"
            f"border-left:4px solid {border}'>{describe(decision.applied)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("**Rationale**")
    st.caption(decision.rationale or "-")

    if decision.shap_top:
        drivers = ", ".join(f"{n} ({v:+.2f})" for n, v in decision.shap_top[:3])
        st.caption(f"Top risk drivers: {drivers}")

    st.markdown("**Guardrails**")
    if not decision.guardrail_events:
        st.markdown(
            f"<span style='color:{COLORS['safe']}'>✓ all checks passed — proposal applied unchanged</span>",
            unsafe_allow_html=True,
        )
    else:
        for e in decision.guardrail_events:
            color = SEVERITY_COLOR.get(e.severity, COLORS["warning"])
            st.markdown(
                f"<div style='margin:3px 0'><span style='color:{color};font-weight:600'>"
                f"{e.severity.upper()}</span> <code>{e.rule}</code> — {e.detail}</div>",
                unsafe_allow_html=True,
            )

    if decisions:
        st.markdown("**Intervention log**")
        rows = [
            {
                "t (h)": round(d.t_hours, 2),
                "applied": describe(d.applied),
                "risk": d.risk_class,
                "regime": d.regime_name,
                "guardrails": "; ".join(e.rule for e in d.guardrail_events) or "-",
            }
            for d in decisions
            if not d.applied.is_no_op() or d.guardrail_events
        ]
        if rows:
            st.dataframe(
                pd.DataFrame(rows[::-1]).head(40),
                width="stretch",
                hide_index=True,
                height=240,
            )
        else:
            st.caption("No interventions yet — the pond has not needed one.")
