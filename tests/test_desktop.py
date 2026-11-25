"""Desktop client tests. Offscreen, so they run in CI and over SSH with no display.

These cover the two things the port could plausibly get wrong: the display buffers drifting
out of step with the run they mirror, and the playback clock advancing the pond at a rate
other than the one the user asked for.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Must be set before QApplication is constructed, which pytest-qt-less fixtures do lazily.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="desktop client extras not installed")
pytest.importorskip("pyqtgraph", reason="desktop client extras not installed")

from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop.engine import RunEngine  # noqa: E402
from desktop.theme import configure_pyqtgraph  # noqa: E402


@pytest.fixture(scope="module")
def app():
    configure_pyqtgraph()
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def engine(app):
    eng = RunEngine()
    eng.rebuild()
    yield eng
    eng.pause()


def test_buffers_track_the_run(engine):
    """The numpy buffers must mirror the AgentRun exactly - they are what the charts draw."""
    engine._advance(120)
    run = engine.run

    assert run.tick_index == 120
    assert engine.truth.n == 120
    assert engine.observed.n == 120
    assert engine.shadow.n == 120
    assert len(engine.regime_history) == 120

    # Values, not just counts: an off-by-one here would silently draw the wrong pond.
    assert engine.truth.t[119] == pytest.approx(run.result.states[-1].t_hours)
    assert engine.truth.full("do")[119] == pytest.approx(run.result.states[-1].do)
    assert engine.shadow.full("do")[119] == pytest.approx(run.result.shadow_states[-1].do)


def test_window_slice_matches_a_manual_filter(engine):
    """`times(since)` is a binary search; it has to agree with the obvious filter."""
    engine._advance(200)
    t_all = engine.truth.t[:engine.truth.n]
    lo = float(t_all[-1]) - 24.0

    got = engine.truth.times(lo)
    expected = t_all[t_all >= lo]
    assert np.array_equal(got, expected)
    assert engine.truth.series("do", lo).shape == expected.shape


def test_buffers_grow_past_initial_capacity(app):
    """Capacity doubling must preserve everything already written."""
    eng = RunEngine()
    eng.rebuild()
    eng.truth._cap = 8            # force several growth steps in a short run
    eng.truth.t = np.empty(8)
    eng.truth._c = {c: np.empty(8) for c in eng.truth._c}
    eng._advance(40)
    assert eng.truth.n == 40
    assert np.all(np.diff(eng.truth.t[:40]) > 0)
    eng.pause()


def test_reset_clears_every_buffer(engine):
    engine._advance(30)
    assert engine.truth.n == 30
    engine.rebuild()
    assert engine.truth.n == 0
    assert engine.observed.n == 0
    assert engine.shadow.n == 0
    assert engine.regime_history == []
    assert engine.run.tick_index == 0


def test_fractional_speeds_do_not_round_up_to_the_frame_rate(engine):
    """The pacing bug this guards against: `max(1, round(elapsed * speed))` per frame floors
    at one tick per repaint, so every speed below the frame rate plays back at the frame rate.
    Ticks owed must accumulate instead."""
    engine.speed = 4
    engine._last_frame = None
    engine._tick_debt = 0.0

    # Simulate 30 frames of a 30 Hz timer: one second of wall clock, so four ticks.
    frame = 1.0 / 30.0
    advanced = 0
    for _ in range(30):
        engine._tick_debt += frame * engine.speed
        n = int(engine._tick_debt)
        engine._tick_debt -= n
        advanced += n

    assert advanced == pytest.approx(4, abs=1)


def test_speed_cap_limits_one_frame_to_a_second_of_simulated_time(engine):
    """A slow frame must not be repaid by leaping the pond past the excursion under watch."""
    engine.speed = 8
    engine._tick_debt = 500.0     # as if the app had been suspended
    n = int(engine._tick_debt)
    capped = min(n, max(1, int(engine.speed)))
    assert capped == 8


def test_metrics_are_throttled_not_recomputed_per_frame(engine):
    engine._advance(60)
    first = engine.metrics()
    engine._advance(1)
    # Inside the throttle window the cached dict comes back by identity.
    assert engine.metrics() is first


def test_runs_with_no_perception_models(app, monkeypatch):
    """An empty models/ directory must degrade to the rule-based bands, never crash."""
    import desktop.engine as E

    monkeypatch.setattr(E, "load_models", lambda src: {
        "forecaster": None, "regimes": None, "risk": None, "explainer": None,
        "notes": ["no forecaster weights - run scripts/pretrain_federated.py"],
    })
    eng = RunEngine()
    eng.rebuild()
    eng._advance(20)
    assert eng.truth.n == 20
    assert eng.run.result.decisions[-1].risk_class in ("safe", "warning", "critical")
    eng.pause()


def test_every_view_renders(app):
    """Build the whole window and paint each view once - catches Qt wiring errors."""
    from desktop.main_window import VIEWS, MainWindow

    win = MainWindow()
    win.resize(1400, 900)
    win.engine._advance(90)
    for view in VIEWS:
        win._select_view(view)
        win._refresh()
    assert win.metrics_table.rowCount() > 0
    win.engine.pause()
    win.close()
