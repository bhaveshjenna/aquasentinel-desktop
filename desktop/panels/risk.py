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

from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ..engine import RunEngine
from ..plotting import make_plot
from ..theme import COLORS
from ..widgets import RiskGauge, caption


class RiskPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(2)
        self.gauge = RiskGauge()
        left_lay.addWidget(self.gauge)
        left_lay.addStretch(1)
        left.setFixedWidth(200)

        self.plot = make_plot("Why the risk is what it is", height=196)
        pi = self.plot.getPlotItem()
        pi.setLabel("bottom", "SHAP contribution to CRITICAL")
        self._yaxis = pi.getAxis("left")
        self._yaxis.setWidth(132)

        # One BarGraphItem, re-fed via setOpts. Bars are drawn from x0=0 so the sign of a
        # contribution reads off the direction it points, not off its colour alone.
        self.bars = pg.BarGraphItem(x0=[], y=[], height=0.62, width=[], brushes=[])
        pi.addItem(self.bars)
        self.zero = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#787878", width=1))
        pi.getViewBox().addItem(self.zero, ignoreBounds=True)

        self.note = caption("Risk drivers appear once the risk model and SHAP are loaded.")

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(3)
        right_lay.addWidget(self.plot, 1)
        right_lay.addWidget(self.note)

        lay.addWidget(left)
        lay.addWidget(right, 1)

        self._last_names: List[str] = []

    def clear(self) -> None:
        self.bars.setOpts(x0=[], y=[], width=[], brushes=[])
        self._last_names = []

    def update_view(self, engine: RunEngine) -> None:
        percept = (engine.run.last_perception or {}) if engine.run else {}
        self.gauge.set(percept.get("risk_class", "safe"), percept.get("risk_score", 0.0))

        decisions = engine.run.result.decisions if engine.run else []
        drivers: List[Tuple[str, float]] = list(decisions[-1].shap_top) if decisions else []
        if not drivers:
            self.note.setVisible(True)
            self.clear()
            return
        self.note.setVisible(False)

        # Reversed so the strongest driver sits at the top of the axis rather than the bottom.
        names = [d[0] for d in drivers][::-1]
        vals = np.asarray([float(d[1]) for d in drivers][::-1], dtype=float)
        ys = np.arange(len(vals), dtype=float)

        self.bars.setOpts(
            x0=np.zeros_like(vals), y=ys, width=vals, height=0.62,
            brushes=[pg.mkBrush(COLORS["critical"] if v > 0 else COLORS["safe"]) for v in vals],
        )
        # Tick labels are only rewritten when the driver set actually changes; SHAP returns a
        # stable top-k for long stretches, and setTicks forces an axis relayout.
        if names != self._last_names:
            self._yaxis.setTicks([[(y, n) for y, n in zip(ys, names)]])
            self._last_names = names

        span = float(np.max(np.abs(vals))) if vals.size else 1.0
        span = max(span, 1e-3) * 1.25
        self.plot.setXRange(-span, span, padding=0)
        self.plot.setYRange(-0.6, len(vals) - 0.4, padding=0)
