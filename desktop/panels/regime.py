"""Regime + anomaly panel: which operating regime HDBSCAN puts this window in, and whether
it belongs to any cluster at all.

The **anomaly flag is the deliverable here**, not the regime name. Water-quality windows turn
out to form a continuum rather than separated clusters (DBCV ~ 0 across the whole sweep in
outputs/regime_sweep.csv), so the regime label is a coarse descriptive split and the panel
says so rather than implying a taxonomy the data does not support.

The anomaly flag means "this window resembles nothing in the training distribution", which
covers a broken probe and a genuine novel excursion equally and cannot tell them apart. The
sensor_fault scenario exists precisely to expose that limit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QWidget

from ..engine import RunEngine
from ..plotting import make_plot
from ..theme import COLORS
from ..widgets import Banner

ANOMALY_BODY = (
    "This window matches no learned regime. Could be a probe fault or a genuinely novel "
    "state - HDBSCAN cannot tell you which."
)
REGIME_BODY = (
    "Coarse unsupervised split, named after the fact from its centroid. No labels were used. "
    "The clusters are weakly separated (DBCV ~ 0) - read this as a description, not as a "
    "claim that the pond has discrete states."
)


class RegimePanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.banner = Banner()
        self.banner.setFixedWidth(320)

        self.plot = make_plot("Regime history", height=178)
        pi = self.plot.getPlotItem()
        pi.setLabel("bottom", "hours")
        pi.setLabel("left", "cluster (-1 = anomaly)")
        pi.getAxis("left").setTickSpacing(major=1, minor=1)

        # Two scatter items rather than a per-point brush list: recolouring N points every
        # frame would put an O(N) Python loop back into the render path that the buffers just
        # took out of it.
        self.normal = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None),
                                         brush=pg.mkBrush(COLORS["agent"]))
        self.anomalous = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None),
                                            brush=pg.mkBrush(COLORS["critical"]))
        pi.addItem(self.normal)
        pi.addItem(self.anomalous)

        lay.addWidget(self.banner)
        lay.addWidget(self.plot, 1)

    def clear(self) -> None:
        self.normal.setData([], [])
        self.anomalous.setData([], [])

    def update_view(self, engine: RunEngine) -> None:
        percept = (engine.run.last_perception or {}) if engine.run else {}
        anomaly = bool(percept.get("anomaly", False))
        name = percept.get("regime_name", "warming up")
        rid = percept.get("regime", -1)

        if anomaly:
            self.banner.set("regime", "Anomaly", ANOMALY_BODY, COLORS["critical"])
        else:
            self.banner.set(f"regime #{rid}", str(name), REGIME_BODY, COLORS["agent"])

        hist = engine.regime_history
        if not hist:
            self.clear()
            return
        t = np.fromiter((h["t_hours"] for h in hist), dtype=float, count=len(hist))
        ids = np.fromiter((h["regime"] for h in hist), dtype=float, count=len(hist))
        flag = np.fromiter((bool(h["anomaly"]) for h in hist), dtype=bool, count=len(hist))

        self.normal.setData(t[~flag], ids[~flag])
        self.anomalous.setData(t[flag], ids[flag])
