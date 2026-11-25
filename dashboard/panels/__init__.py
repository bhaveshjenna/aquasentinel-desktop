"""Dashboard panels. Each renders one region of the app from a live AgentRun."""

from . import agent, counterfactual, regime, risk_shap, scorecard, timeseries

__all__ = ["timeseries", "regime", "risk_shap", "agent", "counterfactual", "scorecard"]
