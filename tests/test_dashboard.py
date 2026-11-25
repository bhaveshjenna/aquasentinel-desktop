"""Dashboard tests: verify Streamlit modules, models loader, and runtime advance functions."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
import numpy as np

from dashboard.app import load_models, available_sources, advance, CONTROLLERS
from dashboard.panels.common import COLORS, fmt_hours, label, y_range
from src.agent.loop import AgentRun, Perception
from src.twin.scenarios import load_scenarios


def test_dashboard_imports_and_controllers():
    """Verify that dashboard module loads and defines supported controllers."""
    assert "predictive" in CONTROLLERS
    assert "reactive" in CONTROLLERS
    assert "do_nothing" in CONTROLLERS


def test_dashboard_available_sources():
    """Verify weight sources detection."""
    sources = available_sources()
    assert isinstance(sources, list)
    assert len(sources) > 0


def test_dashboard_load_models():
    """Verify model loader loads forecaster, regimes, and risk models."""
    models = load_models("federated")
    assert isinstance(models, dict)
    assert "forecaster" in models
    assert "regimes" in models
    assert "risk" in models
    assert "explainer" in models


def test_dashboard_advance_loop(monkeypatch):
    """Verify that advance() advances the run and logs regime history."""
    scenarios = load_scenarios()
    scenario_id = list(scenarios.keys())[0]

    models = load_models("federated")
    perception = Perception(
        forecaster=models["forecaster"],
        regimes=models["regimes"],
        risk=models["risk"],
        explainer=models["explainer"],
    )
    run = AgentRun(scenario_id, controller="reactive", perception=perception, seed=42)

    fake_session_state = MagicMock()
    fake_session_state.playing = True
    fake_session_state.regime_history = []

    import streamlit as st
    monkeypatch.setattr(st, "session_state", fake_session_state)

    advance(run, 3)
    assert len(fake_session_state.regime_history) == 3
    assert "t_hours" in fake_session_state.regime_history[0]
    assert "regime" in fake_session_state.regime_history[0]


def test_dashboard_panel_helpers():
    """Verify common layout and formatting helpers in panels/common.py."""
    assert fmt_hours(0.5) == "30 min"
    assert fmt_hours(2.5) == "2.5 h"
    assert label("do") == "Dissolved oxygen (mg/L)"
    yr = y_range(np.array([4.0, 6.0, 8.0]), "do", {"do": {"critical_below": 3.0}})
    assert isinstance(yr, list) and len(yr) == 2
    assert "safe" in COLORS
