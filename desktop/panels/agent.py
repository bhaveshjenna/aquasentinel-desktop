"""Agent panel: proposed action, rationale, guardrail verdict, applied action, and the log.

The proposed/applied split is the point. Anywhere they differ, a hard rule fired, and the rule
that fired is named. A demo that only showed the applied action would be hiding the one thing
this project is actually claiming.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.agent.actuators import describe
from src.common.schema import Decision

from ..engine import RunEngine
from ..theme import COLORS, TEXT_DIM
from ..widgets import Banner, Card, caption, h2

SEVERITY_COLOR = {"block": COLORS["critical"], "clamp": COLORS["warning"],
                  "override": COLORS["critical"]}

LOG_COLUMNS = ("t (h)", "applied", "risk", "regime", "guardrails")
LOG_LIMIT = 40


class AgentPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        actions = QWidget()
        actions_lay = QHBoxLayout(actions)
        actions_lay.setContentsMargins(0, 0, 0, 0)
        actions_lay.setSpacing(10)
        self.proposed = Banner()
        self.applied = Banner()
        actions_lay.addWidget(self.proposed)
        actions_lay.addWidget(self.applied)

        self.rationale = caption("-")
        self.drivers = caption("")

        guard_card = Card(spacing=3)
        guard_card.body.addWidget(h2("Guardrails"))
        self.guardrails = QLabel("")
        self.guardrails.setWordWrap(True)
        self.guardrails.setTextFormat(Qt.TextFormat.RichText)
        guard_card.body.addWidget(self.guardrails)

        self.log = QTableWidget(0, len(LOG_COLUMNS))
        self.log.setHorizontalHeaderLabels(LOG_COLUMNS)
        self.log.verticalHeader().setVisible(False)
        self.log.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.log.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.log.setAlternatingRowColors(False)
        self.log.setMinimumHeight(220)
        header = self.log.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        self.log_note = caption("No interventions yet — the pond has not needed one.")

        lay.addWidget(actions)
        lay.addWidget(h2("Rationale"))
        lay.addWidget(self.rationale)
        lay.addWidget(self.drivers)
        lay.addWidget(guard_card)
        lay.addWidget(h2("Intervention log"))
        lay.addWidget(self.log, 1)
        lay.addWidget(self.log_note)

        self._logged = -1

    def clear(self) -> None:
        self.log.setRowCount(0)
        self._logged = -1
        self.rationale.setText("-")
        self.drivers.setText("")
        self.guardrails.setText("")
        self.proposed.set("proposed", "-", "", COLORS["shadow"])
        self.applied.set("applied", "-", "", COLORS["shadow"])

    def update_view(self, engine: RunEngine) -> None:
        decisions: List[Decision] = engine.run.result.decisions if engine.run else []
        if not decisions:
            self.clear()
            return
        d = decisions[-1]
        changed = d.proposed != d.applied

        self.proposed.set("proposed", describe(d.proposed), "", COLORS["shadow"])
        self.applied.set(
            "applied",
            describe(d.applied),
            "modified by guardrails" if changed else "",
            COLORS["warning"] if changed else COLORS["safe"],
        )

        self.rationale.setText(d.rationale or "-")
        if d.shap_top:
            drivers = ", ".join(f"{n} ({v:+.2f})" for n, v in d.shap_top[:3])
            self.drivers.setText(f"Top risk drivers: {drivers}")
        else:
            self.drivers.setText("")

        if not d.guardrail_events:
            self.guardrails.setText(
                f"<span style='color:{COLORS['safe']}'>&#10003; all checks passed "
                f"&mdash; proposal applied unchanged</span>"
            )
        else:
            rows = "".join(
                f"<div style='margin:3px 0'>"
                f"<span style='color:{SEVERITY_COLOR.get(e.severity, COLORS['warning'])};"
                f"font-weight:600'>{e.severity.upper()}</span> "
                f"<code>{e.rule}</code> &mdash; "
                f"<span style='color:{TEXT_DIM}'>{e.detail}</span></div>"
                for e in d.guardrail_events
            )
            self.guardrails.setText(rows)

        self._refresh_log(decisions)

    def _refresh_log(self, decisions: List[Decision]) -> None:
        """Rebuild the table only when a new row would actually appear in it.

        The log shows interventions and guardrail events, which are rare relative to ticks, so
        rebuilding 40 QTableWidgetItems every frame would be paying a per-frame cost for a
        view that changes a handful of times per run.
        """
        rows = [d for d in decisions if not d.applied.is_no_op() or d.guardrail_events]
        if len(rows) == self._logged:
            return
        self._logged = len(rows)

        if not rows:
            self.log.setRowCount(0)
            self.log_note.setVisible(True)
            return
        self.log_note.setVisible(False)

        newest = rows[::-1][:LOG_LIMIT]
        self.log.setRowCount(len(newest))
        for r, d in enumerate(newest):
            cells = (
                f"{d.t_hours:.2f}",
                describe(d.applied),
                d.risk_class,
                d.regime_name,
                "; ".join(e.rule for e in d.guardrail_events) or "-",
            )
            for c, text in enumerate(cells):
                self.log.setItem(r, c, QTableWidgetItem(text))
        self.log.resizeColumnsToContents()
        header = self.log.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
