"""Counterfactual panel - the money shot.

A shadow pond was forked at t=0 with the same seed, the same scenario schedule and the same
noise draws, and has been running with the agent disabled ever since. Every gap between the
two lines is caused by an action and nothing else. This is the only comparison in the project
that isolates the agent's contribution exactly, which is why it gets the most screen space.

These charts are the one place mouse interaction is left on: they show settled history rather
than a moving window, so zooming into a divergence does not fight the next frame's axes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ..engine import RunEngine
from ..plotting import add_risk_bands, curve, make_plot, y_range
from ..theme import COLORS, label
from ..widgets import LegendStrip, caption


class _Pair:
    """One channel, agent vs. shadow, with the divergence filled in between."""

    def __init__(self, channel: str, bands: Dict) -> None:
        self.channel = channel
        self.plot = make_plot(label(channel), height=300, interactive=True)
        add_risk_bands(self.plot, channel, bands)

        self.shadow = curve(self.plot, COLORS["shadow"], 2.0, dash=True,
                            name="no action (shadow pond)")
        self.live = curve(self.plot, COLORS["agent"], 2.4, name="with agent")

        fill = pg.FillBetweenItem(self.shadow, self.live,
                                  brush=pg.mkBrush(33, 102, 172, 33))
        fill.setZValue(-50)
        self.plot.getPlotItem().getViewBox().addItem(fill, ignoreBounds=True)
        self.plot.getPlotItem().setLabel("bottom", "hours")

    def clear(self) -> None:
        self.shadow.setData([], [])
        self.live.setData([], [])

    def update(self, engine: RunEngine) -> None:
        live = engine.truth.full(self.channel)
        shadow = engine.shadow.full(self.channel)
        if live is None or shadow is None or live.size == 0 or shadow.size == 0:
            return
        n = min(live.size, shadow.size)
        t = engine.truth.t[:n]
        y_live, y_shadow = live[:n], shadow[:n]

        self.shadow.setData(t, y_shadow)
        self.live.setData(t, y_live)

        # Both ponds share one axis, so the range has to span both traces.
        rng = y_range(np.concatenate([y_live, y_shadow]), self.channel, engine.bands)
        if rng is not None:
            self.plot.setYRange(rng[0], rng[1], padding=0)
        self.plot.setXRange(float(t[0]), float(t[n - 1]) + 0.1, padding=0)


class CounterfactualPanel(QWidget):
    def __init__(self, bands: Dict, channels: Optional[List[str]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.channels = channels or ["do", "nh3"]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.legend = LegendStrip((
            ("with agent", COLORS["agent"], False, 2.4),
            ("no action (shadow pond)", COLORS["shadow"], True, 2.0),
        ))
        lay.addWidget(self.legend)

        charts = QWidget()
        charts_lay = QHBoxLayout(charts)
        charts_lay.setContentsMargins(0, 0, 0, 0)
        charts_lay.setSpacing(9)
        self.pairs = []
        for ch in self.channels:
            pair = _Pair(ch, bands)
            self.pairs.append(pair)
            charts_lay.addWidget(pair.plot)

        self.headline = caption("")
        self.footnote = caption(
            "Both ponds were forked at t=0 from the same state with the same seed, and receive "
            "the same perturbation schedule and the same sensor-noise realisation. The only "
            "difference between them is the agent."
        )

        lay.addWidget(charts, 1)
        lay.addWidget(self.headline)
        lay.addWidget(self.footnote)

    def clear(self) -> None:
        for pair in self.pairs:
            pair.clear()
        self.headline.setText("")

    def update_view(self, engine: RunEngine) -> None:
        if engine.truth.n == 0 or engine.shadow.n == 0:
            self.headline.setText("The counterfactual appears once the run starts.")
            return
        for pair in self.pairs:
            pair.update(engine)

        # A single honest number for the headline channel.
        live_do = engine.truth.full("do")
        shadow_do = engine.shadow.full("do")
        if live_do is None or shadow_do is None or live_do.size == 0 or shadow_do.size == 0:
            return
        n = min(live_do.size, shadow_do.size)
        gap = float(np.nanmin(live_do[:n])) - float(np.nanmin(shadow_do[:n]))
        hours = float(engine.truth.t[n - 1])
        self.headline.setText(
            f"Minimum DO with the agent is {gap:+.2f} mg/L relative to the identical no-action "
            f"pond over the same {hours:.1f} simulated hours (same seed, same perturbations, "
            f"same sensor noise)."
        )
