"""Run lifecycle and playback clock for the desktop client.

Two things here are the reason the native client is cheap where the Streamlit one was not.

**Incremental buffers.** ``RunResult.frame()`` rebuilds a pandas DataFrame from every state
in the run each time it is called, so in the browser version the cost of drawing a frame grew
with the length of the run. Here each tick appends one row to preallocated numpy arrays and
the panels read numpy views, so drawing frame 2000 costs what drawing frame 10 cost.

**A real clock.** Playback is a ``QTimer``, not a rerun-the-whole-script loop. Advancing the
simulation and redrawing are separate steps, so a slow paint slows the paint, not the pond.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from src.agent.loop import AgentRun, Perception
from src.common.runtime import MODELS_DIR, load_config, set_seed

#: Channels held as ground truth by the twin.
TRUTH_CHANNELS = ("do", "temp_c", "ph", "tan", "nh3", "no2", "turbidity", "algae", "alkalinity")

#: Channels the probe actually reports. NH3 is absent by design - it is derived, not measured,
#: which is why the NH3 chart falls back to the twin trace exactly as the Streamlit panel does.
OBS_CHANNELS = ("do", "temp_c", "ph", "tan", "no2", "turbidity", "conductivity")

SHADOW_CHANNELS = ("do", "nh3", "ph", "tan", "no2", "temp_c", "turbidity")

#: Paint target. Qt coalesces paint events, so this is a target rather than the hard ceiling
#: the Streamlit MAX_FPS had to be - overshooting costs frames, not correctness.
RENDER_HZ = 30

#: metrics() walks every state in the run, so it is throttled rather than run per frame.
METRICS_INTERVAL_S = 0.4

_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def load_models(model_source: str) -> Dict[str, Any]:
    """Load the pretrained stack for one weight source. Missing pieces degrade, never crash.

    Mirrors the Streamlit ``@st.cache_resource`` loader, including its failure text, so the
    two clients report an incomplete ``models/`` directory the same way.
    """
    if model_source in _MODEL_CACHE:
        return _MODEL_CACHE[model_source]

    from src.perception.explain import RiskExplainer
    from src.perception.forecaster import LSTMForecaster
    from src.perception.regimes import RegimeModel
    from src.perception.risk import RiskModel

    out: Dict[str, Any] = {"forecaster": None, "regimes": None, "risk": None,
                           "explainer": None, "notes": []}

    path = {"federated": MODELS_DIR / "forecaster_federated.pt",
            "central": MODELS_DIR / "forecaster_central.pt"}.get(model_source)
    if path is None:  # independent-local: any one client's solo model
        locals_ = sorted(MODELS_DIR.glob("forecaster_local_*.pt"))
        path = locals_[0] if locals_ else None
        if path is not None:
            out["notes"].append(f"independent-local model: {path.stem[len('forecaster_local_'):]}")

    if path is not None and path.exists():
        try:
            out["forecaster"] = LSTMForecaster.load(path)
        except Exception as exc:
            out["notes"].append(f"forecaster failed to load: {exc}")
    else:
        out["notes"].append("no forecaster weights - run scripts/pretrain_federated.py")

    rp = MODELS_DIR / "regimes.pkl"
    if rp.exists():
        try:
            out["regimes"] = RegimeModel.load(rp)
        except Exception as exc:
            out["notes"].append(f"regime model failed to load: {exc}")

    kp = MODELS_DIR / "risk_model.pkl"
    if kp.exists():
        try:
            out["risk"] = RiskModel.load(kp)
            out["explainer"] = RiskExplainer(out["risk"])
        except Exception as exc:
            out["notes"].append(f"risk model failed to load: {exc}")
    else:
        out["notes"].append("no risk model - falling back to the rule-based risk bands")

    _MODEL_CACHE[model_source] = out
    return out


def available_sources() -> List[str]:
    out = []
    if (MODELS_DIR / "forecaster_federated.pt").exists():
        out.append("federated")
    if list(MODELS_DIR.glob("forecaster_local_*.pt")):
        out.append("independent-local")
    if (MODELS_DIR / "forecaster_central.pt").exists():
        out.append("central")
    return out or ["federated"]


def data_summary() -> Optional[Dict[str, Any]]:
    try:
        from src.data import registry as R

        return R.summary()
    except Exception:
        return None


class Columns:
    """Growable column store: one float64 array per channel, doubling on overflow.

    A dict of numpy arrays rather than a DataFrame because the panels want ``x`` and ``y``
    vectors and nothing else, and because rebuilding a frame per tick is exactly the cost
    this client exists to remove.
    """

    def __init__(self, channels, capacity: int = 4096) -> None:
        self._channels = tuple(channels)
        self._cap = int(capacity)
        self.n = 0
        self.t = np.empty(self._cap, dtype=float)
        self._c = {c: np.empty(self._cap, dtype=float) for c in self._channels}

    def _grow(self) -> None:
        self._cap *= 2
        self.t = np.resize(self.t, self._cap)
        for c in self._channels:
            self._c[c] = np.resize(self._c[c], self._cap)

    def append(self, t_hours: float, values: Dict[str, float]) -> None:
        if self.n >= self._cap:
            self._grow()
        i = self.n
        self.t[i] = t_hours
        for c in self._channels:
            v = values.get(c, np.nan)
            self._c[c][i] = np.nan if v is None else v
        self.n = i + 1

    def clear(self) -> None:
        self.n = 0

    def has(self, channel: str) -> bool:
        return channel in self._c

    def times(self, since: float = 0.0) -> np.ndarray:
        """Times from the first sample at or after `since`. A view, never a copy."""
        return self.t[self._start(since):self.n]

    def series(self, channel: str, since: float = 0.0) -> Optional[np.ndarray]:
        if channel not in self._c:
            return None
        return self._c[channel][self._start(since):self.n]

    def full(self, channel: str) -> Optional[np.ndarray]:
        if channel not in self._c:
            return None
        return self._c[channel][:self.n]

    def _start(self, since: float) -> int:
        if since <= 0.0 or self.n == 0:
            return 0
        # t increases monotonically, so a binary search is exact and O(log n).
        return int(np.searchsorted(self.t[:self.n], since, side="left"))


class RunEngine(QObject):
    """Owns the AgentRun, the playback timer and the display buffers."""

    #: Emitted after the simulation advances, at most once per painted frame.
    advanced = Signal()
    #: Emitted when a new run replaces the old one. Panels drop their state on this.
    runReset = Signal()
    #: Emitted when play/pause changes, so the toolbar button can relabel itself.
    playingChanged = Signal(bool)
    #: Emitted once the scenario reaches its final tick.
    finished = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.sample_minutes = float(load_config("sensors")["sample_every_minutes"])
        self.bands = load_config("twin")["risk_bands"]

        self.scenario_id = "night_do_crash"
        self.controller = "predictive"
        self.seed = 1002
        self.model_source = available_sources()[0]
        self.guardrails = True
        self.speed = 8  # ticks per real second

        self.run: Optional[AgentRun] = None
        self.truth = Columns(TRUTH_CHANNELS)
        self.observed = Columns(OBS_CHANNELS)
        self.shadow = Columns(SHADOW_CHANNELS)
        self.regime_history: List[Dict[str, Any]] = []

        self._playing = False
        self._last_frame: Optional[float] = None
        #: Fractional ticks carried between frames. Without it, rounding each frame's share to
        #: a whole tick and flooring at 1 makes every speed below the frame rate run at the
        #: frame rate: at 30 Hz, "8 ticks per second" played back at 20+.
        self._tick_debt = 0.0
        self._metrics: Dict[str, Any] = {}
        self._metrics_at = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / RENDER_HZ))
        self._timer.timeout.connect(self._on_timer)

    # ------------------------------------------------------------------ configuration
    def configure(self, **kwargs: Any) -> bool:
        """Apply settings. Returns True if the change forced a new run."""
        rebuild_keys = ("scenario_id", "controller", "seed", "model_source", "guardrails")
        dirty = False
        for k, v in kwargs.items():
            if getattr(self, k, None) != v:
                setattr(self, k, v)
                dirty = dirty or k in rebuild_keys
        if dirty:
            self.rebuild()
        return dirty

    def rebuild(self) -> None:
        """Start a fresh run from the current settings."""
        self.pause()
        set_seed(self.seed)
        models = load_models(self.model_source)
        perception = Perception(
            forecaster=models["forecaster"],
            regimes=models["regimes"],
            risk=models["risk"],
            explainer=models["explainer"],
        )
        self.run = AgentRun(
            self.scenario_id,
            controller=self.controller,
            seed=self.seed,
            perception=perception,
            guardrails_enabled=self.guardrails,
        )
        self.truth.clear()
        self.observed.clear()
        self.shadow.clear()
        self.regime_history.clear()
        self._metrics = {}
        self._metrics_at = 0.0
        self._last_frame = None
        self._tick_debt = 0.0
        self.runReset.emit()
        self.advanced.emit()

    def ensure(self) -> AgentRun:
        if self.run is None:
            self.rebuild()
        return self.run  # type: ignore[return-value]

    # ------------------------------------------------------------------ playback
    @property
    def playing(self) -> bool:
        return self._playing

    def play(self) -> None:
        run = self.ensure()
        if run.done():
            return
        self._playing = True
        self._last_frame = None  # pace from now, not from whenever playback last ran
        self._tick_debt = 0.0
        self._timer.start()
        self.playingChanged.emit(True)

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self._playing:
            self._playing = False
            self.playingChanged.emit(False)

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def step(self) -> None:
        """One decision interval, paused. The keyboard-driven way to inspect a transition."""
        self.pause()
        run = self.ensure()
        self._advance(run.ticks_per_decision)
        self.advanced.emit()

    def _on_timer(self) -> None:
        run = self.ensure()
        if run.done():
            self.pause()
            self.finished.emit()
            return
        now = time.perf_counter()
        elapsed = (now - self._last_frame) if self._last_frame else (1.0 / RENDER_HZ)
        self._last_frame = now

        self._tick_debt += elapsed * self.speed
        n = int(self._tick_debt)
        if n == 0:
            # This frame's share of a tick has not accumulated yet. Repainting an unchanged
            # pond would burn a frame to draw the same picture, so skip it.
            return
        # Never leap more than a second of simulated time in one frame: a slow frame must not
        # be repaid by skipping the pond past the excursion you were watching for.
        capped = min(n, max(1, int(self.speed)))
        self._tick_debt -= n  # drop what the cap refused rather than owing it forever

        self._advance(capped)
        self.advanced.emit()
        if run.done():
            self.pause()
            self.finished.emit()

    def _advance(self, n: int) -> None:
        run = self.ensure()
        for _ in range(n):
            if run.done():
                break
            self._capture(run.tick())

    def _capture(self, decision) -> None:
        """Pull the newest sample into the display buffers. O(1) per tick."""
        run = self.run
        assert run is not None
        state = run.result.states[-1]
        self.truth.append(state.t_hours, {c: getattr(state, c, np.nan) for c in TRUTH_CHANNELS})

        obs = run.result.observations[-1]
        self.observed.append(obs.t_hours, obs.values)

        if run.result.shadow_states:
            s = run.result.shadow_states[-1]
            self.shadow.append(s.t_hours, {c: getattr(s, c, np.nan) for c in SHADOW_CHANNELS})

        self.regime_history.append(
            {"t_hours": decision.t_hours, "regime": decision.regime, "anomaly": decision.anomaly}
        )

    # ------------------------------------------------------------------ readouts
    def metrics(self) -> Dict[str, Any]:
        """Throttled: metrics() walks the whole run, so it is not paid per frame."""
        run = self.run
        if run is None or not run.result.states:
            return {}
        now = time.perf_counter()
        if self._metrics and (now - self._metrics_at) < METRICS_INTERVAL_S:
            return self._metrics
        self._metrics = run.result.metrics()
        self._metrics_at = now
        return self._metrics

    def t_now(self) -> float:
        return float(self.truth.t[self.truth.n - 1]) if self.truth.n else 0.0

    def sim_hours_per_second(self) -> float:
        return self.speed * self.sample_minutes / 60.0
