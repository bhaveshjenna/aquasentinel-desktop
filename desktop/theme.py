"""One colour and type vocabulary for the desktop client.

The look is a dark "control room" console: near-black surfaces, cream stat tiles with black
ink, monospace upper-case labels, and a small set of neon signal colours. Each colour keeps one
meaning everywhere - green is safe, yellow is warning, red is critical, cyan is the agent's own
pond / the live probe - so a reading can be judged by colour before it is read.
"""

from __future__ import annotations

from typing import Dict

import pyqtgraph as pg
from PySide6.QtGui import QColor

COLORS: Dict[str, str] = {
    "safe": "#39e66b",
    "warning": "#ffd60a",
    "critical": "#ff3b30",
    "agent": "#00e5ff",
    "shadow": "#8a8a8a",
    "forecast": "#c77dff",
    "grid": "#2a2a2a",
}

#: Band fills carry their alpha here rather than in an rgba() string, because Qt wants a
#: QColor and pyqtgraph wants a brush. Kept faint: on black, a strong fill drowns the trace.
BAND_FILL = {
    "critical": QColor(255, 59, 48, 34),
    "warning": QColor(255, 214, 10, 22),
    "healthy": QColor(57, 230, 107, 14),
}

CHANNEL_LABEL = {
    "do": "Dissolved oxygen (mg/L)",
    "temp_c": "Temperature (°C)",
    "ph": "pH",
    "tan": "Total ammonia N (mg/L)",
    "nh3": "Un-ionised NH₃-N (mg/L)",
    "no2": "Nitrite (mg/L)",
    "turbidity": "Turbidity (NTU)",
    "conductivity": "Conductivity (µS/cm)",
}

CHANNEL_SHORT = {
    "do": "DO", "temp_c": "TEMP", "ph": "PH", "tan": "TAN", "nh3": "NH3",
    "no2": "NO2", "turbidity": "TURB", "conductivity": "COND",
}

# Surfaces.
BG = "#0b0b0b"          # window
PANEL_BG = "#121212"    # sidebar and dark cards
BORDER = "#2c2c2c"      # dark-on-dark rules
FRAME = "#e9e7df"       # the light outline of a stream card
TEXT = "#f2f2ee"
TEXT_DIM = "#8f8f88"

# Cream tiles, as in a printed instrument panel.
CREAM = "#efede4"
INK = "#111111"
INK_DIM = "#5d5c56"

MONO = '"Consolas", "JetBrains Mono", "Courier New", monospace'
SANS = '"Segoe UI", system-ui, sans-serif'


def label(channel: str) -> str:
    return CHANNEL_LABEL.get(channel, channel)


def short(channel: str) -> str:
    return CHANNEL_SHORT.get(channel, channel.upper())


def risk_color(level: str) -> str:
    return COLORS.get(level, COLORS["safe"])


def configure_pyqtgraph() -> None:
    """Global pyqtgraph defaults. Call once, before any PlotWidget is constructed."""
    pg.setConfigOption("background", BG)
    pg.setConfigOption("foreground", TEXT_DIM)
    pg.setConfigOptions(antialias=True)


STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: {SANS};
    font-size: 12px;
}}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}

/* ---------------------------------------------------------------- type */
QLabel#H1 {{ font-size: 19px; font-weight: 700; }}
QLabel#H2 {{
    font-family: {MONO}; font-size: 11px; font-weight: 700; letter-spacing: 1px;
    color: {TEXT}; padding-top: 6px;
}}
QLabel#Caption {{ color: {TEXT_DIM}; font-size: 11px; }}
QLabel#Title {{
    font-size: 27px; font-weight: 900; font-style: italic; letter-spacing: 0.5px;
}}
QLabel#Brand {{ font-family: {MONO}; font-size: 17px; font-weight: 800; letter-spacing: 1px; }}
QLabel#BrandSub {{ font-family: {MONO}; font-size: 10px; color: {TEXT_DIM}; }}
QLabel#Logo {{
    background: {CREAM}; color: {INK}; font-size: 20px; font-weight: 900;
    border-radius: 3px; min-width: 38px; max-width: 38px; min-height: 38px; max-height: 38px;
}}
QLabel#Section {{
    font-family: {MONO}; font-size: 10px; color: {TEXT_DIM}; letter-spacing: 1px;
    padding-top: 10px;
}}
QLabel#Readout {{ font-family: {MONO}; font-size: 13px; font-weight: 700; color: {COLORS['agent']}; }}
QLabel#ReadoutSub {{ font-family: {MONO}; font-size: 10px; color: {TEXT_DIM}; }}
QLabel#Meta {{ font-family: {MONO}; font-size: 11px; color: {TEXT_DIM}; }}

/* ---------------------------------------------------------------- frames */
QFrame#Sidebar {{ background: {BG}; border-right: 1px solid {BORDER}; }}
QFrame#Card {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 2px; }}
QFrame#Card > QWidget {{ background: transparent; }}
QFrame#Tile {{ background: {CREAM}; border: 1px solid {INK}; border-radius: 2px; }}
QFrame#Tile QLabel {{ color: {INK}; }}
QFrame#SafetyTile {{ background: {BG}; border: 2px solid {COLORS['safe']}; border-radius: 2px; }}
QFrame#Stream {{ background: {BG}; border: 1px solid {FRAME}; border-radius: 1px; }}
QFrame#Stream > QWidget {{ background: transparent; }}

QLabel#TileName {{ font-family: {MONO}; font-size: 10px; font-weight: 700; letter-spacing: 1px; }}
QLabel#TileValue {{ font-family: {MONO}; font-size: 26px; font-weight: 800; }}
QLabel#TileNote {{ font-family: {MONO}; font-size: 10px; color: {INK_DIM}; }}

QFrame#Stream QLabel#StreamTag {{
    font-family: {MONO}; font-size: 10px; font-weight: 700; color: {INK};
    background: {CREAM}; padding: 2px 6px;
}}
QLabel#StreamTitle {{ font-family: {MONO}; font-size: 13px; font-weight: 800; }}
QLabel#Pill {{
    font-family: {MONO}; font-size: 10px; font-weight: 800; color: {INK};
    padding: 2px 6px; border-radius: 1px;
}}
QLabel#Badge {{
    font-family: {MONO}; font-size: 10px; font-weight: 800; color: {INK};
    padding: 5px 9px; border: 1px solid {INK};
}}

/* ---------------------------------------------------------------- buttons */
QPushButton {{
    background: {BG}; color: {TEXT}; border: 1px solid {TEXT};
    border-radius: 1px; padding: 8px 14px;
    font-family: {MONO}; font-size: 11px; font-weight: 700; letter-spacing: 1px;
}}
QPushButton:hover {{ border-color: {COLORS['agent']}; color: {COLORS['agent']}; }}
QPushButton:pressed {{ background: {PANEL_BG}; }}
QPushButton#Primary {{
    background: {COLORS['agent']}; color: {INK}; border: 1px solid {COLORS['agent']};
}}
QPushButton#Primary:hover {{ background: #7ff2ff; color: {INK}; }}
QPushButton#Nav {{
    text-align: left; border: none; padding: 10px 12px; color: {TEXT};
    font-size: 11px; background: transparent;
}}
QPushButton#Nav:hover {{ color: {COLORS['agent']}; }}
QPushButton#Nav:checked {{ background: {TEXT}; color: {INK}; }}

/* ---------------------------------------------------------------- inputs */
QComboBox, QSpinBox {{
    background: {PANEL_BG}; color: {TEXT}; border: 1px solid #3a3a3a; border-radius: 1px;
    padding: 5px 8px; font-family: {MONO}; font-size: 11px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL_BG}; color: {TEXT}; selection-background-color: {COLORS['agent']};
    selection-color: {INK}; border: 1px solid #3a3a3a;
}}
QRadioButton, QCheckBox {{ padding: 3px 0; spacing: 8px; font-size: 11px; }}
QRadioButton::indicator, QCheckBox::indicator {{
    width: 11px; height: 11px; border: 1px solid #8a8a8a; background: {BG};
}}
QRadioButton::indicator:checked, QCheckBox::indicator:checked {{
    border: 1px solid {COLORS['agent']}; background: {COLORS['agent']}; image: none;
}}
QRadioButton::indicator:hover, QCheckBox::indicator:hover {{ border-color: {COLORS['agent']}; }}
QRadioButton:disabled, QCheckBox:disabled {{ color: {TEXT_DIM}; }}

QSlider::groove:horizontal {{ height: 3px; background: #3a3a3a; }}
QSlider::handle:horizontal {{ background: {COLORS['agent']}; width: 12px; margin: -5px 0; }}

/* ---------------------------------------------------------------- tables (cream, like the
   subject list) */
QTableWidget {{
    background: {CREAM}; color: {INK}; gridline-color: #cfccc1; border: 1px solid {INK};
    font-family: {MONO}; font-size: 11px;
}}
QTableWidget::item {{ color: {INK}; padding: 3px; }}
QHeaderView::section {{
    background: {CREAM}; color: {INK}; border: none; border-bottom: 1px solid {INK};
    padding: 6px; font-family: {MONO}; font-size: 10px; font-weight: 800;
}}
QTableCornerButton::section {{ background: {CREAM}; border: none; }}

QScrollBar:vertical {{ background: {BG}; width: 9px; }}
QScrollBar::handle:vertical {{ background: #3a3a3a; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: {CREAM}; color: {INK}; border: 1px solid {INK}; }}
"""
