"""The application window: nav + control sidebar, console header, and one view at a time.

Only the visible view is updated. That is the native counterpart of the Streamlit app's move
away from ``st.tabs`` - there, every tab body re-ran on every rerun, so the counterfactual
charts and the metrics table were rebuilt four times a second while nobody was looking at
them. Here a hidden panel costs nothing at all.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QPushButton, QRadioButton, QScrollArea, QSlider,
    QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.twin.scenarios import load_scenarios

from .engine import RunEngine, available_sources, data_summary, load_models
from .panels.agent import AgentPanel
from .panels.counterfactual import CounterfactualPanel
from .panels.regime import RegimePanel
from .panels.risk import RiskPanel
from .panels.scorecard import ScorecardPanel
from .panels.timeseries import TimeSeriesPanel
from .theme import COLORS
from .widgets import Divider, caption, clock, h2, section

VIEWS = ("Live", "Counterfactual", "Agent", "Scorecard")
NAV_LABELS = {
    "Live": "▣   LIVE DASHBOARD",
    "Counterfactual": "◫   COUNTERFACTUAL",
    "Agent": "◉   AGENT",
    "Scorecard": "▤   SCORECARD",
}

CONTROLLERS = {
    "predictive": "Agent — predictive (learned MPC)",
    "reactive": "Agent — reactive rules",
    "static_threshold": "Baseline — static DO threshold",
    "do_nothing": "Baseline — do nothing",
    "oracle": "Reference — oracle (reads latent state)",
}
PLANNERS = ("predictive", "reactive")
BASELINES = ("do_nothing", "static_threshold", "oracle")

SPEEDS = (1, 2, 4, 8, 16, 32)

APP_TITLE = "Safe Predictive Control and Federated Forecasting for Aquaculture Water Quality Simulation"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(min(1560, avail.width()), min(980, avail.height()))
        else:
            self.resize(1280, 670)

        self.engine = RunEngine(self)
        self.scenarios = load_scenarios()
        self.show_truth = False
        self._view = "Live"

        root = QWidget()
        root_lay = QHBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)
        root_lay.addWidget(self._build_sidebar())
        root_lay.addWidget(self._build_main(), 1)
        self.setCentralWidget(root)

        self.engine.advanced.connect(self._refresh)
        self.engine.runReset.connect(self._on_reset)
        self.engine.playingChanged.connect(self._on_playing)
        self.engine.finished.connect(self._on_finished)

        self._bind_shortcuts()
        self.engine.rebuild()

    # ------------------------------------------------------------------ sidebar
    def _build_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Sidebar")
        panel.setFixedWidth(300)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # -- brand
        brand = QWidget()
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(16, 18, 16, 14)
        bl.setSpacing(10)
        logo = QLabel("≋")
        logo.setObjectName("Logo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(logo)
        names = QVBoxLayout()
        names.setSpacing(0)
        n1 = QLabel("AQUASENTINEL")
        n1.setObjectName("Brand")
        n2 = QLabel("POND MONITOR · NG-NSUKKA")
        n2.setObjectName("BrandSub")
        names.addWidget(n1)
        names.addWidget(n2)
        bl.addLayout(names, 1)
        outer.addWidget(brand)

        # -- navigation (the view switcher)
        nav = QWidget()
        nl = QVBoxLayout(nav)
        nl.setContentsMargins(10, 6, 10, 10)
        nl.setSpacing(3)
        self.view_buttons: Dict[str, QPushButton] = {}
        group = QButtonGroup(nav)
        group.setExclusive(True)
        for name in VIEWS:
            btn = QPushButton(NAV_LABELS[name])
            btn.setObjectName("Nav")
            btn.setCheckable(True)
            btn.setChecked(name == "Live")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, n=name: self._select_view(n))
            group.addButton(btn)
            self.view_buttons[name] = btn
            nl.addWidget(btn)
        outer.addWidget(nav)
        outer.addWidget(Divider())

        # -- system controls (scrolls)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(16, 4, 16, 16)
        lay.setSpacing(6)

        lay.addWidget(section("system_control"))
        lay.addWidget(h2("SCENARIO"))
        self.scenario_box = QComboBox()
        for key, spec in self.scenarios.items():
            self.scenario_box.addItem(spec.label, key)
        idx = list(self.scenarios).index(self.engine.scenario_id) \
            if self.engine.scenario_id in self.scenarios else 1
        self.scenario_box.setCurrentIndex(idx)
        self.scenario_box.currentIndexChanged.connect(self._on_scenario)
        lay.addWidget(self.scenario_box)
        self.scenario_note = caption("")
        lay.addWidget(self.scenario_note)

        lay.addWidget(h2("CONTROLLER"))
        self.agent_on = QCheckBox("Agent enabled")
        self.agent_on.setChecked(True)
        self.agent_on.toggled.connect(self._on_agent_toggle)
        lay.addWidget(self.agent_on)

        self.controller_stack = QStackedWidget()
        self.planner_group = self._radio_set(PLANNERS, "predictive")
        self.baseline_group = self._radio_set(BASELINES, "do_nothing")
        self.controller_stack.addWidget(self.planner_group[0])
        self.controller_stack.addWidget(self.baseline_group[0])
        lay.addWidget(self.controller_stack)

        self.guardrails_box = QCheckBox("Guardrails active")
        self.guardrails_box.setChecked(True)
        self.guardrails_box.setToolTip(
            "Hard safety rules. Turning them off is an ablation - the unsafe-applied counter "
            "is expected to leave zero."
        )
        self.guardrails_box.toggled.connect(
            lambda on: self.engine.configure(guardrails=bool(on))
        )
        lay.addWidget(self.guardrails_box)

        lay.addWidget(h2("PERCEPTION WEIGHTS"))
        sources = available_sources()
        self.source_group = self._radio_set(sources, self.engine.model_source, plain=True)
        self.source_group[0].setToolTip(
            "Which forecaster drives the agent: the federated global model, a single pond's "
            "solo model, or the centrally-pooled reference."
        )
        lay.addWidget(self.source_group[0])

        lay.addWidget(h2("PLAYBACK"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(len(SPEEDS) - 1)
        self.speed_slider.setValue(SPEEDS.index(self.engine.speed))
        self.speed_slider.setPageStep(1)
        self.speed_slider.valueChanged.connect(self._on_speed)
        lay.addWidget(self.speed_slider)
        self.speed_note = caption("")
        lay.addWidget(self.speed_note)

        seed_row = QWidget()
        slay = QHBoxLayout(seed_row)
        slay.setContentsMargins(0, 0, 0, 0)
        seed_lab = QLabel("SEED")
        seed_lab.setObjectName("Meta")
        slay.addWidget(seed_lab)
        self.seed_box = QSpinBox()
        self.seed_box.setRange(0, 10_000_000)
        self.seed_box.setValue(self.engine.seed)
        self.seed_box.editingFinished.connect(
            lambda: self.engine.configure(seed=int(self.seed_box.value()))
        )
        slay.addWidget(self.seed_box, 1)
        lay.addWidget(seed_row)

        self.truth_box = QCheckBox("Show twin ground truth")
        self.truth_box.setToolTip("Simulation only. A real farm sees the probe trace, never this.")
        self.truth_box.toggled.connect(self._on_truth)
        lay.addWidget(self.truth_box)
        lay.addWidget(caption("Space play/pause · S step · R reset"))

        lay.addWidget(section("provenance"))
        self.notes = caption("")
        lay.addWidget(self.notes)
        lay.addWidget(caption(
            "The pond being controlled is a simulation. The perception models are trained on "
            "real catfish-pond sensor data from Nsukka, Nigeria; every number here is relative, "
            "not a real-world guarantee."
        ))
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)
        return panel

    def _radio_set(self, keys, selected: str, plain: bool = False):
        """A vertical radio group. Returns (container, QButtonGroup, {key: button})."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        group = QButtonGroup(box)
        buttons: Dict[str, QRadioButton] = {}
        for key in keys:
            btn = QRadioButton(key if plain else CONTROLLERS[key])
            btn.setChecked(key == selected)
            group.addButton(btn)
            buttons[key] = btn
            lay.addWidget(btn)
            btn.toggled.connect(
                lambda on, k=key, p=plain: self._on_choice(k, p) if on else None
            )
        return box, group, buttons

    # ------------------------------------------------------------------ main area
    def _build_main(self) -> QWidget:
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(12)

        # header ------------------------------------------------------------------
        header = QWidget()
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(0, 0, 0, 0)
        hlay.setSpacing(16)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        t1 = QLabel("REAL-TIME")
        t1.setObjectName("Title")
        t2 = QLabel("MONITORING")
        t2.setObjectName("Title")
        titles.addWidget(t1)
        titles.addWidget(t2)
        hlay.addLayout(titles)

        self.badge = QLabel("●  SYSTEM IDLE")
        self.badge.setObjectName("Badge")
        hlay.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignVCenter)

        session = QVBoxLayout()
        session.setSpacing(1)
        self.session_id = QLabel("SESSION_ID::-")
        self.session_id.setObjectName("Readout")
        self.session_clock = QLabel("T+ -")
        self.session_clock.setObjectName("ReadoutSub")
        session.addWidget(self.session_id)
        session.addWidget(self.session_clock)
        hlay.addLayout(session)
        hlay.addStretch(1)

        self.reset_btn = QPushButton("RESET")
        self.reset_btn.clicked.connect(self.engine.rebuild)
        self.step_btn = QPushButton("STEP")
        self.step_btn.clicked.connect(self.engine.step)
        self.play_btn = QPushButton("▶  PLAY")
        self.play_btn.setObjectName("Primary")
        self.play_btn.setMinimumWidth(120)
        self.play_btn.clicked.connect(self.engine.toggle)
        for b in (self.reset_btn, self.step_btn, self.play_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            hlay.addWidget(b, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(header)

        # meta line: scenario, progress, controller -------------------------------
        meta = QWidget()
        mlay = QHBoxLayout(meta)
        mlay.setContentsMargins(0, 0, 0, 0)
        mlay.setSpacing(22)
        self.readouts: Dict[str, QLabel] = {}
        for name in ("Scenario", "Simulated clock", "Progress", "Controller"):
            lab = QLabel(f"{name.upper()}: -")
            lab.setObjectName("Meta")
            self.readouts[name] = lab
            mlay.addWidget(lab)
        mlay.addStretch(1)
        lay.addWidget(meta)
        lay.addWidget(Divider())

        self.status = QLabel("")
        self.status.setVisible(False)
        lay.addWidget(self.status)

        # views -------------------------------------------------------------------
        bands = self.engine.bands
        self.stack = QStackedWidget()

        self.scorecard = ScorecardPanel()
        self.timeseries = TimeSeriesPanel(bands)
        self.regime = RegimePanel()
        self.risk = RiskPanel()
        self.counterfactual = CounterfactualPanel(bands)
        self.agent = AgentPanel()

        self.stack.addWidget(self._live_view())
        self.stack.addWidget(self._wrap(self.counterfactual,
                                        "AGENT VS. AN IDENTICAL POND WITH THE AGENT SWITCHED OFF"))
        self.stack.addWidget(self._wrap(self.agent, None))
        self.stack.addWidget(self._scorecard_view())
        lay.addWidget(self.stack, 1)
        return holder

    def _live_view(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(10)
        lay.addWidget(self.scorecard)
        lay.addWidget(h2("LIVE CHANNELS · FORECAST OVERLAY · RISK BANDS"))
        lay.addWidget(self.timeseries, 1)
        lay.addWidget(h2("REGIME AND ANOMALY"))
        lay.addWidget(self.regime)
        lay.addWidget(h2("RISK AND ITS DRIVERS"))
        lay.addWidget(self.risk)
        return self._scrolled(page)

    def _scorecard_view(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.scorecard_tiles = ScorecardPanel()
        lay.addWidget(self.scorecard_tiles)
        lay.addWidget(h2("FULL METRICS"))
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(("METRIC", "VALUE"))
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.metrics_table.horizontalHeader()
        # Metric names run to `shadow_time_in_risk_h`; a fixed first column clips exactly the
        # comparison the scorecard exists to show.
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.metrics_table, 1)
        return page

    def _wrap(self, widget: QWidget, heading: Optional[str]) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        if heading:
            lay.addWidget(h2(heading))
        lay.addWidget(widget, 1)
        return page

    def _scrolled(self, widget: QWidget) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.engine.toggle)
        QShortcut(QKeySequence("S"), self, activated=self.engine.step)
        QShortcut(QKeySequence("R"), self, activated=self.engine.rebuild)

    # ------------------------------------------------------------------ handlers
    def _on_scenario(self, index: int) -> None:
        key = self.scenario_box.itemData(index)
        spec = self.scenarios[key]
        self.scenario_note.setText(spec.description)
        # A scenario carries its own default seed, and reusing the previous scenario's seed
        # silently changes what the demo shows. Follow the scenario unless the user has
        # deliberately typed a seed since the last switch.
        self.seed_box.blockSignals(True)
        self.seed_box.setValue(int(spec.seed))
        self.seed_box.blockSignals(False)
        self.engine.configure(scenario_id=key, seed=int(spec.seed))

    def _on_agent_toggle(self, on: bool) -> None:
        self.controller_stack.setCurrentIndex(0 if on else 1)
        buttons = (self.planner_group if on else self.baseline_group)[2]
        for key, btn in buttons.items():
            if btn.isChecked():
                self.engine.configure(controller=key)
                return
        first = next(iter(buttons))
        buttons[first].setChecked(True)

    def _on_choice(self, key: str, plain: bool) -> None:
        if plain:
            self.engine.configure(model_source=key)
        else:
            self.engine.configure(controller=key)

    def _on_speed(self, index: int) -> None:
        self.engine.speed = SPEEDS[index]
        self._update_speed_note()

    def _on_truth(self, on: bool) -> None:
        self.show_truth = bool(on)
        self._refresh()

    def _select_view(self, name: str) -> None:
        self._view = name
        for key, btn in self.view_buttons.items():
            btn.setChecked(key == name)
        self.stack.setCurrentIndex(VIEWS.index(name))
        self._refresh()

    def _set_badge(self, text: str, color: str) -> None:
        self.badge.setText(f"●  {text}")
        self.badge.setStyleSheet(f"background:{color};")

    def _on_playing(self, playing: bool) -> None:
        self.play_btn.setText("❚❚  PAUSE" if playing else "▶  PLAY")
        if playing:
            self._set_badge("SYSTEM ACTIVE", COLORS["safe"])
        elif self.engine.run is not None and self.engine.run.done():
            self._set_badge("RUN COMPLETE", COLORS["agent"])
        else:
            self._set_badge("SYSTEM IDLE", "#bdbbb2")

    def _on_finished(self) -> None:
        self._set_badge("RUN COMPLETE", COLORS["agent"])
        self.status.setText(
            "SCENARIO COMPLETE — change the controller or the model source and reset to compare."
        )
        self.status.setStyleSheet(
            f"color:{COLORS['safe']};border:1px solid {COLORS['safe']};"
            f"padding:7px 10px;font-family:Consolas;font-size:11px;font-weight:700"
        )
        self.status.setVisible(True)

    def _on_reset(self) -> None:
        self.status.setVisible(False)
        self._set_badge("SYSTEM IDLE", "#bdbbb2")
        self.timeseries.clear()
        self.regime.clear()
        self.risk.clear()
        self.counterfactual.clear()
        self.agent.clear()
        self.metrics_table.setRowCount(0)
        self._update_notes()
        self._update_speed_note()
        key = self.scenario_box.currentData()
        if key in self.scenarios:
            self.scenario_note.setText(self.scenarios[key].description)

    def _update_speed_note(self) -> None:
        self.speed_note.setText(
            f"{self.engine.speed} ticks/s  ≈  {self.engine.sim_hours_per_second():.1f} "
            f"simulated hours per real second."
        )

    def _update_notes(self) -> None:
        lines: List[str] = []
        for note in load_models(self.engine.model_source)["notes"]:
            lines.append(f"⚠ {note}")
        info = data_summary()
        if info:
            lines.append(
                f"Training data: {info['n_sites']} real fish ponds, "
                f"{info['n_rows']:,} rows (Nsukka, Nigeria)."
            )
        self.notes.setText("\n".join(lines))

    # ------------------------------------------------------------------ per-frame
    def _refresh(self) -> None:
        engine = self.engine
        run = engine.run
        if run is None:
            return

        scenario = run.scenario
        t_clock = clock(engine.t_now(), scenario.start_hour_of_day)
        self.readouts["Scenario"].setText(f"SCENARIO: {scenario.label.upper()}")
        self.readouts["Simulated clock"].setText(f"CLOCK: {t_clock.upper()}")
        self.readouts["Progress"].setText(f"PROGRESS: {run.tick_index}/{run.n_ticks}")
        self.readouts["Controller"].setText(
            "CONTROLLER: " + CONTROLLERS[run.controller].split("—")[-1].strip().upper()
        )
        self.session_id.setText(f"SESSION_ID::{run.seed}")
        remaining_h = max(run.n_ticks - run.tick_index, 0) * engine.sample_minutes / 60.0
        self.session_clock.setText(f"T-MINUS {int(remaining_h):02d}:{int(remaining_h % 1 * 60):02d}:00")

        risk_class = (run.last_perception or {}).get("risk_class")

        # Only the visible view is drawn. A hidden panel costs nothing.
        if self._view == "Live":
            self.scorecard.update_view(engine.metrics(), risk_class)
            self.timeseries.update_view(engine, self.show_truth)
            self.regime.update_view(engine)
            self.risk.update_view(engine)
        elif self._view == "Counterfactual":
            self.counterfactual.update_view(engine)
        elif self._view == "Agent":
            self.agent.update_view(engine)
        elif self._view == "Scorecard":
            metrics = engine.metrics()
            self.scorecard_tiles.update_view(metrics, risk_class)
            self._fill_metrics(metrics)

    def _fill_metrics(self, metrics: Dict) -> None:
        if not metrics:
            self.metrics_table.setRowCount(0)
            return
        items = list(metrics.items())
        if self.metrics_table.rowCount() != len(items):
            self.metrics_table.setRowCount(len(items))
        for r, (k, v) in enumerate(items):
            key_item = self.metrics_table.item(r, 0)
            if key_item is None or key_item.text() != k:
                self.metrics_table.setItem(r, 0, QTableWidgetItem(str(k)))
            text = str(v)
            val_item = self.metrics_table.item(r, 1)
            if val_item is None:
                self.metrics_table.setItem(r, 1, QTableWidgetItem(text))
            elif val_item.text() != text:
                val_item.setText(text)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.engine.pause()
        super().closeEvent(event)
