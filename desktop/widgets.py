"""Small Qt building blocks: cards, metric tiles, the risk gauge, banners.

These replace the HTML-in-``st.markdown`` blocks the Streamlit panels used. Same information,
same colour vocabulary, drawn by Qt instead of shipped as a string for a browser to parse.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from .theme import BORDER, COLORS, INK, INK_DIM, TEXT, TEXT_DIM, risk_color


def h1(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("H1")
    return lab


def h2(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("H2")
    return lab


def section(text: str) -> QLabel:
    """A small mono upper-case divider label, e.g. SYSTEM_CONTROL."""
    lab = QLabel(text.upper())
    lab.setObjectName("Section")
    return lab


class Pill(QLabel):
    """A solid status tag (LIVE / NORMAL / ELEVATED / CRITICAL) - black ink on a signal colour."""

    def __init__(self, text: str = "", color: str = COLORS["agent"]) -> None:
        super().__init__(text)
        self.setObjectName("Pill")
        self._state = None
        self.set(text, color)

    def set(self, text: str, color: str) -> None:
        if self._state == (text, color):
            return
        self._state = (text, color)
        self.setText(text)
        self.setStyleSheet(f"background:{color};")


def caption(text: str, wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Caption")
    lab.setWordWrap(wrap)
    return lab


class Card(QFrame):
    """A bordered panel with a vertical layout, used to group a chart with its label."""

    def __init__(self, parent: Optional[QWidget] = None, margin: int = 10, spacing: int = 6) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(margin, margin, margin, margin)
        self.body.setSpacing(spacing)


class MetricTile(QFrame):
    """Name, big value, and an optional coloured note underneath.

    The Streamlit ``st.metric`` equivalent, including the inverted delta colouring: for
    time-in-risk, *less* is better, so a negative delta is drawn green.
    """

    def __init__(self, name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Tile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(2)

        self._name = QLabel(name.upper())
        self._name.setObjectName("TileName")
        self._value = QLabel("-")
        self._value.setObjectName("TileValue")
        self._note = QLabel("")
        self._note.setObjectName("TileNote")

        lay.addWidget(self._name)
        lay.addWidget(self._value)
        lay.addWidget(self._note)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(120)
        for lab in (self._name, self._note):
            lab.setMinimumWidth(1)

    def set(self, value: str, note: str = "", note_color: Optional[str] = None,
            value_color: Optional[str] = None) -> None:
        if self._value.text() != value:
            self._value.setText(value)
        self._value.setStyleSheet(f"color:{value_color or INK}")
        if self._note.text() != note:
            self._note.setText(note)
        self._note.setStyleSheet(f"color:{note_color or INK_DIM}")


class SafetyTile(QFrame):
    """The odd one out in the tile row: black, outlined in the colour of the current state.

    It answers one question - is the pond safe right now - in one word, so it is the first
    thing the eye lands on.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SafetyTile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(2)
        self._name = QLabel("SAFETY STATUS")
        self._name.setObjectName("TileName")
        self._value = QLabel("-")
        self._value.setStyleSheet("font-size:27px;font-weight:900;font-style:italic")
        self._note = QLabel("")
        self._note.setObjectName("TileNote")
        lay.addWidget(self._name)
        lay.addWidget(self._value)
        lay.addWidget(self._note)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(150)
        self._note.setMinimumWidth(1)
        self._state = None

    def set(self, word: str, note: str, color: str) -> None:
        if self._state == (word, note, color):
            return
        self._state = (word, note, color)
        self._value.setText(word)
        self._note.setText(note)
        self._name.setStyleSheet(f"color:{color}")
        self._note.setStyleSheet(f"color:{color};font-family:Consolas;font-size:10px")
        self._value.setStyleSheet(
            f"color:{TEXT};font-size:27px;font-weight:900;font-style:italic")
        self.setStyleSheet(
            f"QFrame#SafetyTile{{background:#0b0b0b;border:2px solid {color};border-radius:2px}}")


class Banner(QFrame):
    """A tinted block with a coloured left edge - the regime card, the action cards."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Banner")
        self._accent = QColor(COLORS["agent"])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 12, 11)
        lay.setSpacing(3)

        self._kicker = QLabel("")
        self._kicker.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;letter-spacing:.5px")
        self._title = QLabel("")
        self._title.setStyleSheet("font-size:17px;font-weight:600")
        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setStyleSheet(f"color:{TEXT_DIM};font-size:11px")

        lay.addWidget(self._kicker)
        lay.addWidget(self._title)
        lay.addWidget(self._body)

    def set(self, kicker: str, title: str, body: str, accent: str) -> None:
        self._kicker.setText(kicker.upper())
        self._title.setText(title)
        self._body.setText(body)
        # An empty body still reserves its line, which reads as a stray bar under the title.
        self._body.setVisible(bool(body))
        self._accent = QColor(accent)
        tint = QColor(accent)
        tint.setAlpha(20)
        # Scoped to this frame: an unscoped rule would also paint every child label.
        self.setStyleSheet(
            f"QFrame#Banner {{ background: rgba({tint.red()},{tint.green()},{tint.blue()},0.09);"
            f" border-radius: 2px; }}"
        )
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._accent)
        p.drawRoundedRect(QRectF(0, 0, 4, self.height()), 2, 2)
        p.end()


class RiskGauge(QWidget):
    """A 240-degree arc gauge for the probability-weighted risk score.

    Plotly's ``go.Indicator`` was one more figure to build and serialise every frame. This is
    a QPainter arc: the repaint is a handful of drawing calls and no allocation.

    The gauge shows a weighted severity rather than a hard class on purpose - the operator can
    watch risk build *before* it crosses a threshold, which is the entire argument for a
    preemptive agent over a thermostat.
    """

    SPAN = 240.0     # degrees of sweep
    START = 210.0    # degrees, measured counter-clockwise from 3 o'clock

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._score = 0.0
        self._level = "safe"
        self.setMinimumSize(170, 132)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set(self, level: str, score: float) -> None:
        score = 0.0 if score is None or math.isnan(score) else max(0.0, min(1.0, float(score)))
        if abs(score - self._score) < 1e-4 and level == self._level:
            return
        self._score, self._level = score, level
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height() * 1.42)
        thickness = max(11.0, side * 0.105)
        rect = QRectF(
            (self.width() - side) / 2 + thickness / 2,
            thickness / 2 + 4,
            side - thickness,
            side - thickness,
        )

        # Three severity arcs behind the needle, tinted the same as the chart bands.
        for frm, to, key in ((0.0, 0.34, "safe"), (0.34, 0.67, "warning"), (0.67, 1.0, "critical")):
            col = QColor(COLORS[key])
            col.setAlpha(46)
            pen = QPen(col, thickness, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.drawArc(
                rect,
                int((self.START - frm * self.SPAN) * 16),
                int(-(to - frm) * self.SPAN * 16),
            )

        # The filled portion.
        pen = QPen(QColor(risk_color(self._level)), thickness,
                   Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, int(self.START * 16), int(-self._score * self.SPAN * 16))

        # Value and class, centred inside the arc.
        f = QFont(self.font())
        f.setPointSize(21)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(TEXT))
        num = QRectF(rect.left(), rect.center().y() - rect.height() * 0.30,
                     rect.width(), rect.height() * 0.42)
        p.drawText(num, Qt.AlignmentFlag.AlignCenter, f"{self._score:.2f}")

        f.setPointSize(10)
        p.setFont(f)
        p.setPen(QColor(risk_color(self._level)))
        cls = QRectF(rect.left(), rect.center().y() + rect.height() * 0.10,
                     rect.width(), rect.height() * 0.30)
        p.drawText(cls, Qt.AlignmentFlag.AlignCenter, self._level.upper())
        p.end()


class _Swatch(QWidget):
    """A short line in a series' colour and dash pattern."""

    def __init__(self, color: str, dash: bool, width: float) -> None:
        super().__init__()
        self._color, self._dash, self._width = QColor(color), dash, width
        self.setFixedSize(22, 12)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self._color, self._width,
                   Qt.PenStyle.DashLine if self._dash else Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        y = self.height() / 2
        p.drawLine(1, int(y), self.width() - 1, int(y))
        p.end()


class LegendStrip(QWidget):
    """One legend for a whole grid of charts.

    pyqtgraph's per-plot LegendItem sits inside the view box, so on six stacked charts it
    covered the top-right of every one of them - which is exactly where a DO trace peaks -
    and drew an empty sample box for any series that was currently hidden. The colour
    vocabulary is identical across all the charts, so one strip above the grid says the same
    thing without occluding anything.
    """

    def __init__(self, entries, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 2)
        lay.setSpacing(7)
        self._labels = {}
        for name, color, dash, width in entries:
            item = QWidget()
            ilay = QHBoxLayout(item)
            ilay.setContentsMargins(0, 0, 0, 0)
            ilay.setSpacing(5)
            ilay.addWidget(_Swatch(color, dash, width))
            text = QLabel(name)
            text.setStyleSheet(f"color:{TEXT_DIM};font-size:11px")
            ilay.addWidget(text)
            self._labels[name] = item
            lay.addWidget(item)
        lay.addStretch(1)

    def set_visible(self, name: str, visible: bool) -> None:
        """Hide the entry for a series that is not currently drawn."""
        item = self._labels.get(name)
        if item is not None and item.isVisible() != visible:
            item.setVisible(visible)


class Divider(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet(f"color:{BORDER};background:{BORDER};max-height:1px;border:none")


def row(*widgets: QWidget, spacing: int = 10, margins: tuple = (0, 0, 0, 0)) -> QWidget:
    """Horizontal container - the layout idiom that replaces ``st.columns``."""
    holder = QWidget()
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for w in widgets:
        lay.addWidget(w)
    return holder


def fmt_hours(h: float) -> str:
    if h is None or not math.isfinite(h):
        return "-"
    if h < 1:
        return f"{h * 60:.0f} min"
    return f"{h:.1f} h"


def clock(t_hours: float, start_hour: float) -> str:
    hod = (start_hour + t_hours) % 24.0
    day = int((start_hour + t_hours) // 24) + 1
    return f"day {day}  {int(hod):02d}:{int((hod % 1) * 60):02d}"
