"""Chart primitives shared by the panels.

Every chart in this client is built once and then fed with ``setData``. Nothing here creates
a plot item inside a render loop, which is the whole point of the port: the Streamlit version
rebuilt six Plotly figures per frame (~60 ms and 67 KB of JSON before the browser drew a
pixel), and this one moves two numpy views into curves that already exist.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from .theme import BAND_FILL, BORDER, COLORS, TEXT, TEXT_DIM, label


def make_plot(title: Optional[str] = None, height: int = 210, interactive: bool = False) -> pg.PlotWidget:
    """A configured PlotWidget.

    ``interactive`` is off for the live charts on purpose. They follow a moving window, so a
    user pan would be fought by the next frame's ``setXRange``; the counterfactual charts show
    a settled history and do allow zooming.
    """
    pw = pg.PlotWidget()
    pw.setMinimumHeight(height)
    pi = pw.getPlotItem()
    pi.showGrid(x=True, y=True, alpha=0.16)
    pi.getAxis("bottom").setPen(pg.mkPen(BORDER))
    pi.getAxis("left").setPen(pg.mkPen(BORDER))
    pi.getAxis("bottom").setTextPen(pg.mkPen(TEXT_DIM))
    pi.getAxis("left").setTextPen(pg.mkPen(TEXT_DIM))
    pi.setContentsMargins(2, 2, 8, 2)
    if title:
        pi.setTitle(f"<span style='font-family:Consolas;font-weight:700'>{title.upper()}</span>",
                    size="9pt", color=TEXT)
    if not interactive:
        pw.setMouseEnabled(x=False, y=False)
        pi.hideButtons()
        pi.setMenuEnabled(False)
    pw.setAntialiasing(True)
    return pw


def curve(pw: pg.PlotWidget, color: str, width: float = 1.8, dash: bool = False,
          name: Optional[str] = None, clip: bool = True) -> pg.PlotDataItem:
    """A persistent line. Feed it with ``setData`` - never recreate it."""
    pen = pg.mkPen(color=color, width=width,
                   style=Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine)
    # `connect="finite"` is the pyqtgraph spelling of Plotly's connectgaps=False: a dropped
    # probe reading leaves a hole in the trace instead of a straight line across the outage,
    # which is the difference between showing a sensor fault and hiding one.
    item = pw.plot([], [], pen=pen, name=name, connect="finite")
    if clip:
        item.setClipToView(True)
        item.setDownsampling(auto=True, method="peak")
    return item


def add_risk_bands(pw: pg.PlotWidget, channel: str, bands: Dict[str, Any]) -> None:
    """Shade the safe / warning / critical regions behind a channel trace.

    The bands are the *same* dict the risk model and the guardrails read, so what the operator
    sees shaded is exactly what the agent is scored against - there is no second definition of
    "critical" living in the presentation layer.

    Regions are added with ``ignoreBounds=True`` so their +/-1e3 extent cannot drag the view
    range; the visible window comes from :func:`y_range` instead.
    """
    spec = bands.get(channel)
    if not spec:
        return
    vb = pw.getPlotItem().getViewBox()

    def band(y0, y1, kind):
        if y0 is None or y1 is None or y1 <= y0:
            return
        region = pg.LinearRegionItem(
            values=(y0, y1), orientation="horizontal",
            brush=pg.mkBrush(BAND_FILL[kind]), pen=pg.mkPen(None), movable=False,
        )
        region.setZValue(-100)
        vb.addItem(region, ignoreBounds=True)

    lo_c, hi_c = spec.get("critical_below"), spec.get("critical_above")
    lo_w, hi_w = spec.get("warning_below"), spec.get("warning_above")
    healthy = spec.get("healthy")
    sup = spec.get("supersaturation_above")
    floor, ceil = -1e3, 1e3

    if lo_c is not None:
        band(floor, lo_c, "critical")
    if hi_c is not None:
        band(hi_c, ceil, "critical")
    if lo_w is not None:
        band(lo_c if lo_c is not None else floor, lo_w, "warning")
    if hi_w is not None:
        band(hi_w, hi_c if hi_c is not None else ceil, "warning")
    if healthy:
        band(healthy[0], healthy[1], "healthy")
        # Only shade outside the healthy window where no explicit band already covers it,
        # otherwise the warning fill is drawn twice and reads as a darker, false severity.
        if lo_w is None and lo_c is None:
            band(floor, healthy[0], "warning")
        if hi_w is None:
            band(healthy[1], sup if sup is not None else ceil, "warning")
    if sup is not None:
        band(sup, ceil, "warning")


def y_range(series: Sequence[float], channel: str, bands: Dict[str, Any]):
    """Keep the relevant thresholds in frame even when the pond is comfortably safe.

    Autoscaling to the trace alone would push a band off-screen exactly when the operator
    most wants to see how much headroom is left, so the thresholds are folded into the range.
    """
    arr = np.asarray(series, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    lo, hi = float(np.min(finite)), float(np.max(finite))
    spec = bands.get(channel, {})
    for key in ("critical_below", "warning_below"):
        if spec.get(key) is not None:
            lo = min(lo, float(spec[key]))
    for key in ("critical_above", "warning_above", "supersaturation_above"):
        if spec.get(key) is not None:
            hi = max(hi, float(spec[key]) * 1.02)
    pad = max((hi - lo) * 0.08, 1e-6)
    return lo - pad, hi + pad


def vline(pw: pg.PlotWidget, color: str = "#6a6a6a", dash: bool = True) -> pg.InfiniteLine:
    line = pg.InfiniteLine(
        pos=0, angle=90,
        pen=pg.mkPen(color=color, width=1,
                     style=Qt.PenStyle.DotLine if dash else Qt.PenStyle.SolidLine),
    )
    pw.getPlotItem().getViewBox().addItem(line, ignoreBounds=True)
    return line


def channel_title(channel: str) -> str:
    return label(channel)


def mono_font(size: int = 10) -> QFont:
    f = QFont("Consolas")
    f.setPointSize(size)
    return f
