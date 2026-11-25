"""Live time-series panel: measured channel, LSTM forecast overlay, shaded risk bands.

Each channel is a "stream" card - a numbered tag, the channel name and a status pill that
reads the latest probe value against the same risk bands the agent is scored on - and a cream
channel list beside the grid gives every channel's status and value at a glance.

Six charts, built once. Each frame writes two numpy views into curves that already exist,
sets the axis ranges and moves the "now" marker. Nothing is constructed, serialised or
re-keyed, which is the whole reason playback here is not fighting the renderer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)
from PySide6.QtGui import QColor

from src.twin.risk_bands import channel_risk

from ..engine import RunEngine
from ..plotting import add_risk_bands, curve, make_plot, vline, y_range
from ..theme import COLORS, INK, label, short
from ..widgets import LegendStrip, Pill

DEFAULT_CHANNELS = ["do", "ph", "nh3", "temp_c", "no2", "turbidity"]

#: How much history the live charts show. The run keeps every sample; the x-axis has only ever
#: shown the last day. Slicing to the window is a binary search into the engine's buffers, so
#: the drawn payload stays flat no matter how long the run has been going.
WINDOW_H = 24.0

STATUS = {
    "safe": ("NORMAL", COLORS["safe"]),
    "warning": ("ELEVATED", COLORS["warning"]),
    "critical": ("CRITICAL", COLORS["critical"]),
}


def _last_finite(arr) -> float:
    if arr is None or len(arr) == 0:
        return float("nan")
    finite = np.flatnonzero(np.isfinite(arr))
    return float(arr[finite[-1]]) if finite.size else float("nan")


class _ChannelChart:
    """One channel: probe trace, optional twin truth, optional forecast, risk bands."""

    def __init__(self, index: int, channel: str, bands: Dict) -> None:
        self.channel = channel
        self.bands = bands
        self.level = "safe"
        self.value = float("nan")

        # The card around the chart.
        self.frame = QFrame()
        self.frame.setObjectName("Stream")
        lay = QVBoxLayout(self.frame)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(6)

        tag = QLabel(f"STREAM_{index:02d}")
        tag.setObjectName("StreamTag")
        tag_row = QHBoxLayout()
        tag_row.setContentsMargins(0, 0, 0, 0)
        tag_row.addWidget(tag)
        tag_row.addStretch(1)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        title = QLabel(label(channel).upper())
        title.setObjectName("StreamTitle")
        self.live = Pill("LIVE", COLORS["agent"])
        self.status = Pill("NORMAL", COLORS["safe"])
        self.reading = QLabel("-")
        self.reading.setStyleSheet("font-family:Consolas;font-size:15px;font-weight:800")
        head.addWidget(title)
        head.addWidget(self.live)
        head.addStretch(1)
        head.addWidget(self.reading)
        head.addWidget(self.status)

        self.plot = make_plot(None, height=170)
        add_risk_bands(self.plot, channel, bands)

        # Order matters: the probe is the operator's reality and must sit on top.
        self.truth = curve(self.plot, COLORS["shadow"], 1.2, dash=True, name="twin truth")
        self.forecast = curve(self.plot, COLORS["forecast"], 2.0, dash=True, name="LSTM forecast")
        self.probe = curve(self.plot, COLORS["agent"], 1.9, name="probe")
        self.now = vline(self.plot)
        self.truth.setVisible(False)
        self.forecast.setVisible(False)

        lay.addLayout(tag_row)
        lay.addLayout(head)
        lay.addWidget(self.plot, 1)

    def clear(self) -> None:
        for c in (self.truth, self.forecast, self.probe):
            c.setData([], [])
        self.reading.setText("-")
        self.status.set("NORMAL", COLORS["safe"])
        self.level, self.value = "safe", float("nan")

    def update(self, engine: RunEngine, now: float, lo: float, show_truth: bool,
               forecast) -> None:
        truth_buf, obs_buf = engine.truth, engine.observed

        # What the operator actually has is the probe, not the pond. NH3 is the exception:
        # it is derived from TAN/pH/temp rather than measured, so no probe series exists and
        # the twin trace stands in - exactly as the Streamlit panel does it.
        ch = self.channel
        t_obs = obs_buf.times(lo)
        y_obs = obs_buf.series(ch, lo)
        if y_obs is not None and t_obs.size:
            self.probe.setData(t_obs, y_obs)
            span = y_obs
        else:
            t_tr = truth_buf.times(lo)
            y_tr = truth_buf.series(ch, lo)
            if y_tr is None or not t_tr.size:
                return
            self.probe.setData(t_tr, y_tr)
            span = y_tr

        if show_truth and truth_buf.has(ch):
            self.truth.setVisible(True)
            self.truth.setData(truth_buf.times(lo), truth_buf.series(ch, lo))
        elif self.truth.isVisible():
            self.truth.setVisible(False)
            self.truth.setData([], [])

        fx, fy = _forecast_series(forecast, ch, now)
        if fx is not None:
            self.forecast.setVisible(True)
            self.forecast.setData(fx, fy)
        elif self.forecast.isVisible():
            self.forecast.setVisible(False)
            self.forecast.setData([], [])

        rng = y_range(span, ch, engine.bands)
        if rng is not None:
            self.plot.setYRange(rng[0], rng[1], padding=0)
        self.plot.setXRange(lo, now + 3, padding=0)
        self.now.setPos(now)

        # The status pill reads the latest value against the very bands the agent is scored on.
        self.value = _last_finite(span)
        self.level = channel_risk(ch, self.value, self.bands)
        text, color = STATUS[self.level]
        self.status.set(text, color)
        self.reading.setText("-" if not np.isfinite(self.value) else _fmt(self.value))
        self.reading.setStyleSheet(
            f"font-family:Consolas;font-size:15px;font-weight:800;color:{color}")


def _fmt(v: float) -> str:
    return f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"


def _forecast_series(forecast, channel: str, now: float):
    """Forecast x/y for one channel, deriving NH3 from forecast TAN/pH/temp when needed."""
    if forecast is None:
        return None, None
    step_h = forecast.step_minutes / 60.0
    fx = now + np.arange(1, forecast.horizon + 1) * step_h
    if channel in forecast.channels:
        return fx, forecast.channel(channel)
    if channel == "nh3" and {"tan", "ph", "temp_c"} <= set(forecast.channels):
        from src.data.harmonize import derive_nh3

        fy = derive_nh3(forecast.channel("tan"), forecast.channel("ph"),
                        forecast.channel("temp_c"))
        return fx, fy
    return None, None


class ChannelList(QWidget):
    """The cream 'subject list': every channel, its status and its latest value."""

    def __init__(self, channels: List[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(270)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head = QFrame()
        head.setObjectName("Tile")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(14, 12, 14, 12)
        name = QLabel("CHANNEL LIST")
        name.setStyleSheet(f"color:{INK};font-size:15px;font-weight:800")
        hl.addWidget(name)
        hl.addStretch(1)
        count = QLabel(f"{len(channels)} STREAMS")
        count.setStyleSheet(f"color:{INK};font-family:Consolas;font-size:10px;font-weight:700")
        hl.addWidget(count)

        self.table = QTableWidget(len(channels), 3)
        self.table.setHorizontalHeaderLabels(("ID", "STATUS", "VALUE"))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setShowGrid(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Show every channel without scrolling: header + one 36 px row per channel.
        self.table.setFixedHeight(40 + 36 * len(channels))
        self.channels = channels
        for r, ch in enumerate(channels):
            self.table.setItem(r, 0, QTableWidgetItem(f"■  {short(ch)}"))
            self.table.setItem(r, 1, QTableWidgetItem("-"))
            self.table.setItem(r, 2, QTableWidgetItem("-"))

        lay.addWidget(head)
        lay.addWidget(self.table, 1)
        self._last: Dict[str, tuple] = {}

    def update_rows(self, charts: Dict[str, "_ChannelChart"]) -> None:
        for r, ch in enumerate(self.channels):
            c = charts.get(ch)
            if c is None:
                continue
            text, color = STATUS[c.level]
            val = "-" if not np.isfinite(c.value) else _fmt(c.value)
            state = (text, val)
            if self._last.get(ch) == state:
                continue
            self._last[ch] = state
            idc = self.table.item(r, 0)
            idc.setForeground(QColor(color if c.level != "safe" else "#1a8c3c"))
            st = self.table.item(r, 1)
            st.setText(f" {text[:4]} ")
            st.setBackground(QColor(color))
            st.setForeground(QColor(INK))
            st.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            v = self.table.item(r, 2)
            v.setText(val)
            v.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def clear(self) -> None:
        self._last.clear()
        for r in range(len(self.channels)):
            for c in (1, 2):
                item = self.table.item(r, c)
                item.setText("-")
                item.setBackground(QColor(0, 0, 0, 0))


class TimeSeriesPanel(QWidget):
    def __init__(self, bands: Dict, channels: Optional[List[str]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.channels = channels or DEFAULT_CHANNELS

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        left = QWidget()
        lay = QVBoxLayout(left)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.legend = LegendStrip((
            ("probe", COLORS["agent"], False, 1.8),
            ("LSTM forecast", COLORS["forecast"], True, 2.0),
            ("twin truth", COLORS["shadow"], True, 1.4),
        ))
        lay.addWidget(self.legend)

        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)

        self.charts: Dict[str, _ChannelChart] = {}
        for i, ch in enumerate(self.channels):
            chart = _ChannelChart(i + 1, ch, bands)
            self.charts[ch] = chart
            grid.addWidget(chart.frame, i // 2, i % 2)
        lay.addWidget(grid_holder, 1)

        self.channel_list = ChannelList(self.channels)
        outer.addWidget(left, 1)
        outer.addWidget(self.channel_list, 0, Qt.AlignmentFlag.AlignTop)

    def clear(self) -> None:
        for chart in self.charts.values():
            chart.clear()
        self.channel_list.clear()

    def update_view(self, engine: RunEngine, show_truth: bool) -> None:
        if engine.truth.n == 0:
            return
        now = engine.t_now()
        lo = max(0.0, now - WINDOW_H)
        forecast = (engine.run.last_perception or {}).get("forecast") if engine.run else None
        for chart in self.charts.values():
            chart.update(engine, now, lo, show_truth, forecast)
        self.channel_list.update_rows(self.charts)

        # The legend describes what is actually on screen, so an entry disappears with its
        # series rather than promising a trace the operator cannot find.
        self.legend.set_visible("twin truth", bool(show_truth))
        self.legend.set_visible("LSTM forecast", forecast is not None)
