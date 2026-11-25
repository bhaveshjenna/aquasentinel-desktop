"""Running scorecard: time-in-risk, interventions, unsafe-applied, agent vs. no-action.

``unsafe applied`` is the number that has to stay at zero. It is displayed even when nothing
is wrong, because a safety claim that only appears when it is violated is not a safety claim.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ..theme import COLORS, INK_DIM
from ..widgets import MetricTile, SafetyTile, caption, fmt_hours

TILES = ("Time in risk", "Time critical", "Interventions", "Unsafe applied", "Guardrail events")


class ScorecardPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        strip = QWidget()
        strip_lay = QHBoxLayout(strip)
        strip_lay.setContentsMargins(0, 0, 0, 0)
        strip_lay.setSpacing(9)
        self.tiles = {}
        for name in TILES:
            tile = MetricTile(name)
            self.tiles[name] = tile
            strip_lay.addWidget(tile)
        self.safety = SafetyTile()
        strip_lay.addWidget(self.safety)

        self.per_channel = caption("")
        lay.addWidget(strip)
        lay.addWidget(self.per_channel)

    def update_view(self, metrics: Dict[str, Any], risk_class: Optional[str] = None) -> None:
        if not metrics:
            for tile in self.tiles.values():
                tile.set("-", "")
            self.safety.set("STANDBY", "PRESS PLAY TO START", COLORS["shadow"])
            self.per_channel.setText("")
            return

        # One word for "is the pond safe right now". An applied unsafe action outranks
        # everything, then the current risk class.
        if int(metrics.get("unsafe_applied", 0)) > 0:
            self.safety.set("BREACH", "UNSAFE ACTION APPLIED", COLORS["critical"])
        elif risk_class == "critical":
            self.safety.set("CRITICAL", "PARAMS: OUT_OF_RANGE", COLORS["critical"])
        elif risk_class == "warning":
            self.safety.set("WATCH", "PARAMS: NEAR_LIMIT", COLORS["warning"])
        else:
            self.safety.set("OPTIMAL", "PARAMS: SAFE_RANGE", COLORS["safe"])

        risk_h = float(metrics.get("time_in_risk_h", 0.0))
        shadow_h = metrics.get("shadow_time_in_risk_h")
        note, note_color = "", None
        if shadow_h is not None and float(shadow_h) > 0:
            delta = risk_h - float(shadow_h)
            # Inverted on purpose: for time in risk, less is better, so a negative delta is
            # the good news and gets the green.
            note = f"{delta:+.1f} h vs no action"
            # Darker green than the signal colour: this note sits on a cream tile.
            note_color = "#1a8c3c" if delta <= 0 else COLORS["critical"]
        self.tiles["Time in risk"].set(fmt_hours(risk_h), note, note_color)

        self.tiles["Time critical"].set(fmt_hours(float(metrics.get("time_critical_h", 0.0))))

        self.tiles["Interventions"].set(
            str(int(metrics.get("interventions", 0))),
            f"cost {float(metrics.get('intervention_cost', 0)):.0f}",
        )

        unsafe = int(metrics.get("unsafe_applied", 0))
        self.tiles["Unsafe applied"].set(
            str(unsafe),
            "guardrail breach" if unsafe else "target 0",
            COLORS["critical"] if unsafe else INK_DIM,
            COLORS["critical"] if unsafe else None,
        )

        min_do = float(metrics.get("min_do", float("nan")))
        self.tiles["Guardrail events"].set(
            str(int(metrics.get("guardrail_events", 0))),
            "min DO -" if math.isnan(min_do) else f"min DO {min_do:.2f}",
        )

        per = {k[len("risk_h_"):]: float(v) for k, v in metrics.items() if k.startswith("risk_h_")}
        if per:
            parts = [f"{k}: {fmt_hours(v)}"
                     for k, v in sorted(per.items(), key=lambda kv: -kv[1]) if v > 0]
            self.per_channel.setText(
                "Time out of band, per channel — "
                + (", ".join(parts) if parts else "every channel inside its band")
                + ". Aggregated time-in-risk mixes channels the agent can act on with ones it cannot."
            )
        else:
            self.per_channel.setText("")
